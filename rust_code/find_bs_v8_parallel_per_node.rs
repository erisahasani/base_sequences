/// BS(n+1, n) search - V8 Per-Node CD-Tree Sharding
/// Based on V7 (bitwise autocorr, 400-angle spectral, lower-bound pruning,
/// arc-consistent AB backtracking, lex-leader symmetry, CPU pinning) with:
///   - --claim-dir / --shard-depth shard the CD-GENERATION TREE across nodes,
///     not the spectral-pass boundary. Multiple nodes on a shared filesystem
///     atomically partition the inner CD backtrack so no spectral work is
///     duplicated. Correct for the CD-bound regime at n >= ~35 where 99.99%
///     of wall-clock is in spectral filtering, not AB search.
///
/// Usage: cargo run --release --bin find_bs_v8_parallel_per_node -- <n> [opts]
///
/// Options:
///   --instance X/Y      Split tuples across Y instances, run instance X
///   --tuple-range S-E   Manual tuple range (alternative to --instance)
///   --tuple N           Run just the tuple at rank N (0-indexed by score_tuple)
///   --top K             Shortcut for --tuple-range 0-K
///   --timeout SECS      Abort after SECS wall-clock seconds (saves timeout
///                       record file when combined with --tuple)
///   --ab-limit N        AB backtrack node limit (default: unlimited)
///   --ab-limit unlimited  Explicit unlimited AB backtracking
///   --claim-dir <path>  Shared FS dir for per-node CD-tree claim files.
///                       When set, each node claims subtrees by atomic file
///                       create; non-owning nodes skip the subtree entirely.
///   --shard-depth N     Number of CD-tree pair-choice levels in the claim
///                       prefix. Default 1 (≤16 claims per (tuple, m3m6_pair)).
///                       Must be identical across all nodes in a run.
///   --reset-claims      Wipe --claim-dir contents at startup.
///   --found-flag <path> Cross-node sentinel; touched on solution to terminate
///                       peer nodes; polled every 5s.
///
/// Implements the 5-step algorithm from Wang & Zhu (2025):
/// 1. Tuple discovery (Theorem 2.1 sum constraints)
/// 2. Mod-3 partial sum enumeration (Theorem 2.3, m=3)
/// 3. Mod-6 CD refinement (Theorem 2.3, m=6)
/// 4. CD generation + spectral filter (Theorems 2.2 + 2.4)
/// 5. AB search via backtracking (Theorem 2.2)

use std::time::Instant;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};
use std::fs::File;
use std::io::{Write, BufReader, BufWriter};
use std::path::Path;
use std::env;
use serde::{Serialize, Deserialize};
use std::sync::{Mutex, OnceLock};

// Optional log file (set via --log <path>). Mirrors selected stdout output;
// progress lines are written once per hour rather than every 30s.
static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

fn log_write(msg: &str) {
    if let Some(m) = LOG_FILE.get() {
        if let Ok(mut f) = m.lock() {
            let _ = writeln!(f, "{}", msg);
            let _ = f.flush();
        }
    }
}

macro_rules! log_println {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        println!("{}", msg);
        $crate::log_write(&msg);
    }};
}

// ============================================================================
// Inlined from lib.rs: Sequence, BaseSequence, SumTuple, AltSumTuple
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Sequence {
    values: Vec<i32>,
}

impl Sequence {
    fn new(values: Vec<i32>) -> Self {
        Sequence { values }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn autocorrelation(&self, shift: usize) -> i32 {
        let n = self.values.len();
        if shift >= n { return 0; }
        if n <= 128 {
            // Use fast bitwise XOR + POPCNT path
            let bs = BitSequence::from_values(&self.values);
            bs.autocorrelation(shift)
        } else {
            // Fallback for very long sequences
            let mut sum = 0;
            for j in 0..(n - shift) {
                sum += self.values[j] * self.values[j + shift];
            }
            sum
        }
    }
}

#[derive(Clone, Debug)]
struct BaseSequence {
    a: Sequence,
    b: Sequence,
    c: Sequence,
    d: Sequence,
}

impl BaseSequence {
    fn new(a: Sequence, b: Sequence, c: Sequence, d: Sequence) -> Self {
        BaseSequence { a, b, c, d }
    }

    fn is_valid(&self) -> bool {
        let m = self.a.len();
        let n = self.c.len();
        if self.b.len() != m || self.d.len() != n { return false; }
        let ac_0 = self.a.autocorrelation(0) + self.b.autocorrelation(0)
            + self.c.autocorrelation(0) + self.d.autocorrelation(0);
        if ac_0 != 2 * (m as i32 + n as i32) { return false; }
        for i in 1..=n {
            let ac_i = self.a.autocorrelation(i) + self.b.autocorrelation(i)
                + self.c.autocorrelation(i) + self.d.autocorrelation(i);
            if ac_i != 0 { return false; }
        }
        true
    }
}

// ============================================================================
// Bitwise sequence representation for fast autocorrelation via XOR + POPCNT
// Encodes {+1,-1} as bits: +1 -> 0, -1 -> 1
// Autocorrelation at shift s = (len-s) - 2*popcount(bits XOR (bits >> s))
// Hardware POPCNT executes in 1 cycle; processes 64 elements per operation.
// ============================================================================

/// Bitwise representation of a {+1, -1} sequence for O(n/64) autocorrelation.
/// Supports sequences up to length 128 (two u64 words).
#[derive(Clone, Debug)]
struct BitSequence {
    words: [u64; 2],  // words[0] = bits 0..63, words[1] = bits 64..127
    len: usize,
}

impl BitSequence {
    /// Pack a {+1, -1} slice into bitwise representation.
    fn from_values(values: &[i32]) -> Self {
        assert!(values.len() <= 128, "BitSequence supports up to 128 elements");
        let mut words = [0u64; 2];
        for (i, &v) in values.iter().enumerate() {
            if v == -1 {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        BitSequence { words, len: values.len() }
    }

    /// Fast autocorrelation using XOR + POPCNT.
    /// N_X(shift) = sum_{i=0}^{len-shift-1} x[i] * x[i+shift]
    #[inline]
    fn autocorrelation(&self, shift: usize) -> i32 {
        if shift >= self.len { return 0; }
        let overlap = self.len - shift;

        if self.len <= 64 {
            // Single-word fast path (covers n up to 63, which includes n=44)
            let mask = if overlap >= 64 { u64::MAX } else { (1u64 << overlap) - 1 };
            let xor = (self.words[0] & mask) ^ ((self.words[0] >> shift) & mask);
            let diff = xor.count_ones() as i32;
            (overlap as i32) - 2 * diff
        } else {
            // Two-word path for sequences 65..128
            // Build a virtual 128-bit shift and XOR
            let (a0, a1) = (self.words[0], self.words[1]);
            let (b0, b1) = if shift < 64 {
                if shift == 0 {
                    (a0, a1)
                } else {
                    ((a0 >> shift) | (a1 << (64 - shift)), a1 >> shift)
                }
            } else {
                let s = shift - 64;
                if s == 0 { (a1, 0u64) } else { (a1 >> s, 0u64) }
            };
            // XOR with overlap masking
            let x0 = a0 ^ b0;
            let x1 = a1 ^ b1;
            // Mask out bits beyond overlap
            let (m0, m1) = if overlap <= 64 {
                (if overlap == 64 { u64::MAX } else { (1u64 << overlap) - 1 }, 0u64)
            } else {
                let rem = overlap - 64;
                (u64::MAX, if rem >= 64 { u64::MAX } else { (1u64 << rem) - 1 })
            };
            let diff = (x0 & m0).count_ones() + (x1 & m1).count_ones();
            (overlap as i32) - 2 * (diff as i32)
        }
    }
}

/// Compute all autocorrelations for C+D combined using bitwise operations.
/// Returns vec where result[shift] = N_C(shift) + N_D(shift) for shift=0..=max_shift.
#[inline]
fn bitwise_cd_autocorrelations(c: &[i32], d: &[i32], max_shift: usize) -> Vec<i32> {
    let bc = BitSequence::from_values(c);
    let bd = BitSequence::from_values(d);
    let mut result = vec![0i32; max_shift + 1];
    for s in 0..=max_shift {
        result[s] = bc.autocorrelation(s) + bd.autocorrelation(s);
    }
    result
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SumTuple { a: i32, b: i32, c: i32, d: i32 }

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AltSumTuple { a_star: i32, b_star: i32, c_star: i32, d_star: i32 }

// ============================================================================
// Inlined from fast_tuple_search_v2.rs
// ============================================================================

#[inline]
fn mod_positive(a: i32, m: i32) -> i32 {
    ((a % m) + m) % m
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct Mod4Signature {
    a_mod4: i32,
    b_mod4: i32,
    c_mod4: i32,
    d_mod4: i32,
}

impl Mod4Signature {
    fn from_sum_tuple(st: &SumTuple) -> Self {
        Mod4Signature {
            a_mod4: mod_positive(st.a, 4),
            b_mod4: mod_positive(st.b, 4),
            c_mod4: mod_positive(st.c, 4),
            d_mod4: mod_positive(st.d, 4),
        }
    }

    fn required_for_alt_tuple(at: &AltSumTuple, n: usize) -> Self {
        match n % 4 {
            0 => Mod4Signature {
                a_mod4: mod_positive(at.a_star, 4),
                b_mod4: mod_positive(at.b_star, 4),
                c_mod4: mod_positive(at.c_star, 4),
                d_mod4: mod_positive(at.d_star, 4),
            },
            1 => Mod4Signature {
                a_mod4: mod_positive(at.a_star + 2, 4),
                b_mod4: mod_positive(at.b_star + 2, 4),
                c_mod4: mod_positive(at.c_star, 4),
                d_mod4: mod_positive(at.d_star, 4),
            },
            2 => Mod4Signature {
                a_mod4: mod_positive(at.a_star + 2, 4),
                b_mod4: mod_positive(at.b_star + 2, 4),
                c_mod4: mod_positive(at.c_star + 2, 4),
                d_mod4: mod_positive(at.d_star + 2, 4),
            },
            3 => Mod4Signature {
                a_mod4: mod_positive(at.a_star, 4),
                b_mod4: mod_positive(at.b_star, 4),
                c_mod4: mod_positive(at.c_star + 2, 4),
                d_mod4: mod_positive(at.d_star + 2, 4),
            },
            _ => unreachable!(),
        }
    }
}

fn find_valid_sum_tuples_fast_v2(n: usize) -> Vec<(SumTuple, AltSumTuple)> {
    let target = (4 * n + 2) as i32;
    let max_sum_m = (n + 1) as i32;
    let max_sum_n = n as i32;

    log_println!("  Phase 1: Finding valid (a,b,c,d) tuples...");
    let sum_tuples = find_sum_tuples(n, target, max_sum_m, max_sum_n);
    log_println!("  Found {} valid (a,b,c,d) tuples", sum_tuples.len());

    log_println!("  Phase 2: Finding valid (a*,b*,c*,d*) tuples...");
    let alt_tuples = find_alt_tuples(n, target, max_sum_m, max_sum_n);
    log_println!("  Found {} valid (a*,b*,c*,d*) tuples", alt_tuples.len());

    log_println!("  Phase 3: HashMap-based matching (Equation 2.4)...");
    let mut alt_by_signature: HashMap<Mod4Signature, Vec<AltSumTuple>> = HashMap::new();
    for at in alt_tuples {
        let sig = Mod4Signature::required_for_alt_tuple(&at, n);
        alt_by_signature.entry(sig).or_default().push(at);
    }

    let mut valid_pairs = Vec::new();
    for st in &sum_tuples {
        let sig = Mod4Signature::from_sum_tuple(st);
        if let Some(matching_alts) = alt_by_signature.get(&sig) {
            for at in matching_alts {
                valid_pairs.push((st.clone(), at.clone()));
            }
        }
    }

    log_println!("  Found {} valid tuple pairs", valid_pairs.len());
    valid_pairs
}

fn build_perfect_square_map(max_val: i32) -> HashMap<i32, i32> {
    let mut map = HashMap::new();
    for i in 0..=max_val {
        map.insert(i * i, i);
    }
    map
}

#[inline]
fn first_with_parity(max_val: i32, parity: i32) -> i32 {
    let start = -max_val;
    if mod_positive(start, 2) == parity { start } else { start + 1 }
}

fn find_sum_tuples(n: usize, target: i32, max_sum_m: i32, max_sum_n: i32) -> Vec<SumTuple> {
    let mut sum_tuples = Vec::new();
    let perfect_squares = build_perfect_square_map(max_sum_n);
    let ab_parity = ((n + 1) % 2) as i32;
    let cd_parity = (n % 2) as i32;
    let n_even = n % 2 == 0;

    let mut a = first_with_parity(max_sum_m, ab_parity);
    while a <= max_sum_m {
        let a_sq = a * a;
        if a_sq > target { a += 2; continue; }
        let mut b = first_with_parity(max_sum_m, ab_parity);
        while b <= max_sum_m {
            let ab_sum_sq = a_sq + b * b;
            if ab_sum_sq > target { b += 2; continue; }
            if !n_even && mod_positive(a, 4) != mod_positive(b + 2, 4) { b += 2; continue; }
            let remaining = target - ab_sum_sq;
            let mut c = first_with_parity(max_sum_n, cd_parity);
            while c <= max_sum_n {
                let c_sq = c * c;
                let d_sq_needed = remaining - c_sq;
                if d_sq_needed < 0 {
                    if c >= 0 { break; }
                    c += 2; continue;
                }
                if let Some(&d_abs) = perfect_squares.get(&d_sq_needed) {
                    for &d in &[d_abs, -d_abs] {
                        if d < -max_sum_n || d > max_sum_n { continue; }
                        if mod_positive(d, 2) != cd_parity { continue; }
                        if n_even && mod_positive(c, 4) != mod_positive(d, 4) { continue; }
                        sum_tuples.push(SumTuple { a, b, c, d });
                    }
                }
                c += 2;
            }
            b += 2;
        }
        a += 2;
    }
    sum_tuples
}

fn find_alt_tuples(n: usize, target: i32, max_sum_m: i32, max_sum_n: i32) -> Vec<AltSumTuple> {
    let mut alt_tuples = Vec::new();
    let perfect_squares = build_perfect_square_map(max_sum_n);
    let ab_parity = ((n + 1) % 2) as i32;
    let cd_parity = (n % 2) as i32;
    let n_even = n % 2 == 0;

    let mut a_star = first_with_parity(max_sum_m, ab_parity);
    while a_star <= max_sum_m {
        let a_sq = a_star * a_star;
        if a_sq > target { a_star += 2; continue; }
        let mut b_star = first_with_parity(max_sum_m, ab_parity);
        while b_star <= max_sum_m {
            let ab_sum_sq = a_sq + b_star * b_star;
            if ab_sum_sq > target { b_star += 2; continue; }
            if !n_even && mod_positive(a_star, 4) != mod_positive(b_star + 2, 4) { b_star += 2; continue; }
            let remaining = target - ab_sum_sq;
            let mut c_star = first_with_parity(max_sum_n, cd_parity);
            while c_star <= max_sum_n {
                let c_sq = c_star * c_star;
                let d_sq_needed = remaining - c_sq;
                if d_sq_needed < 0 {
                    if c_star >= 0 { break; }
                    c_star += 2; continue;
                }
                if let Some(&d_abs) = perfect_squares.get(&d_sq_needed) {
                    for &d_star in &[d_abs, -d_abs] {
                        if d_star < -max_sum_n || d_star > max_sum_n { continue; }
                        if mod_positive(d_star, 2) != cd_parity { continue; }
                        if n_even && mod_positive(c_star, 4) != mod_positive(d_star, 4) { continue; }
                        alt_tuples.push(AltSumTuple { a_star, b_star, c_star, d_star });
                    }
                }
                c_star += 2;
            }
            b_star += 2;
        }
        a_star += 2;
    }
    alt_tuples
}

// ============================================================================
// 5-class isomorphic equivalence (Paper Section 2, Definition 1.5)
// Transformations (i)-(v) at the tuple (a,b,c,d,a*,b*,c*,d*) level:
//   (i)   Negation: flip sign of any subset of sums/alt_sums
//   (ii)  Reversal: sum unchanged, alt_sum *= (-1)^{L-1} where L is seq length
//         A,B have length n+1, C,D have length n
//   (iii) Interchange: swap (a,a*)<->(b,b*) and/or (c,c*)<->(d,d*)
//   (iv)  Alternation: swap sums<->alt_sums: (a,b,c,d)<->(a*,b*,c*,d*)
//   (v)   CD matrix replacement: sequence-level only, no effect on tuple
// ============================================================================

fn tuple_key(st: &SumTuple, at: &AltSumTuple) -> [i32; 8] {
    [st.a, st.b, st.c, st.d, at.a_star, at.b_star, at.c_star, at.d_star]
}

/// Generate all equivalent tuples under the 5 isomorphic transformations.
/// We compose independent generators: negation × interchange × reversal × alternation.
fn generate_equivalent_tuples(st: &SumTuple, at: &AltSumTuple, n: usize) -> Vec<[i32; 8]> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();

    // Reversal flips: for length L, alt_sum -> (-1)^{L-1} * alt_sum
    // A,B have length n+1: flip_ab = if n%2==0 { 1 } else { -1 }  ((-1)^n)
    // C,D have length n:   flip_cd = if (n-1)%2==0 { 1 } else { -1 }  ((-1)^{n-1})
    let flip_ab: i32 = if n % 2 == 0 { 1 } else { -1 };
    let flip_cd: i32 = if n <= 1 { 1 } else if (n - 1) % 2 == 0 { 1 } else { -1 };

    // Start with base tuple
    let base = [st.a, st.b, st.c, st.d, at.a_star, at.b_star, at.c_star, at.d_star];

    // Apply all combinations of transformations
    // We generate: for each alternation choice × each interchange choice × each reversal choice × each negation choice
    let mut tuples_after_alt = Vec::with_capacity(2);
    // Alternation: identity or swap sums<->alt_sums
    tuples_after_alt.push(base);
    tuples_after_alt.push([base[4], base[5], base[6], base[7], base[0], base[1], base[2], base[3]]);

    let mut tuples_after_swap = Vec::with_capacity(8);
    for t in &tuples_after_alt {
        // Interchange: swap A<->B? swap C<->D?
        let [a, b, c, d, as_, bs, cs, ds] = *t;
        tuples_after_swap.push([a, b, c, d, as_, bs, cs, ds]);
        tuples_after_swap.push([b, a, c, d, bs, as_, cs, ds]); // swap AB
        tuples_after_swap.push([a, b, d, c, as_, bs, ds, cs]); // swap CD
        tuples_after_swap.push([b, a, d, c, bs, as_, ds, cs]); // swap both
    }

    let mut tuples_after_rev = Vec::with_capacity(128);
    for t in &tuples_after_swap {
        // Reversal: independently reverse any subset of {A,B,C,D}
        // Reversing X: sum_X unchanged, alt_sum_X *= flip factor
        let [a, b, c, d, as_, bs, cs, ds] = *t;
        for rev_mask in 0u8..16 {
            let ra = if rev_mask & 1 != 0 { flip_ab } else { 1 };
            let rb = if rev_mask & 2 != 0 { flip_ab } else { 1 };
            let rc = if rev_mask & 4 != 0 { flip_cd } else { 1 };
            let rd = if rev_mask & 8 != 0 { flip_cd } else { 1 };
            tuples_after_rev.push([a, b, c, d, ra * as_, rb * bs, rc * cs, rd * ds]);
        }
    }

    // Negation: independently negate any subset of {A,B,C,D}
    for t in &tuples_after_rev {
        let [a, b, c, d, as_, bs, cs, ds] = *t;
        for neg_mask in 0u8..16 {
            let na = if neg_mask & 1 != 0 { -1 } else { 1 };
            let nb = if neg_mask & 2 != 0 { -1 } else { 1 };
            let nc = if neg_mask & 4 != 0 { -1 } else { 1 };
            let nd = if neg_mask & 8 != 0 { -1 } else { 1 };
            let key = [na*a, nb*b, nc*c, nd*d, na*as_, nb*bs, nc*cs, nd*ds];
            seen.insert(key);
        }
    }

    seen.into_iter().collect()
}

fn filter_to_canonical_5class(tuples: Vec<(SumTuple, AltSumTuple)>, n: usize) -> Vec<(SumTuple, AltSumTuple)> {
    let original_count = tuples.len();
    let canonical: Vec<_> = tuples
        .into_iter()
        .filter(|(st, at)| {
            let self_key = tuple_key(st, at);
            let equivalents = generate_equivalent_tuples(st, at, n);
            // Keep this tuple only if it's the lexicographic minimum of its orbit
            for equiv in &equivalents {
                if *equiv < self_key {
                    return false;
                }
            }
            true
        })
        .collect();
    let filtered_count = canonical.len();
    let reduction_factor = if filtered_count > 0 {
        original_count as f64 / filtered_count as f64
    } else {
        1.0
    };
    log_println!("  5-class isomorphic filtering: {} -> {} tuples ({:.1}x reduction)",
             original_count, filtered_count, reduction_factor);
    canonical
}

// ============================================================================
// Inlined from spectral_filter.rs: hall_polynomial, passes_spectral_bound, compute_ab_headroom
// ============================================================================

#[inline]
fn hall_polynomial(values: &[i32], theta: f64) -> f64 {
    let mut real_sum = 0.0;
    let mut imag_sum = 0.0;
    for (j, &val) in values.iter().enumerate() {
        let angle = (j as f64) * theta;
        let (sin_a, cos_a) = angle.sin_cos();
        real_sum += (val as f64) * cos_a;
        imag_sum += (val as f64) * sin_a;
    }
    real_sum * real_sum + imag_sum * imag_sum
}

/// Number of spectral sample angles.
/// For real sequences, |H(θ)|² = |H(2π-θ)|² (conjugate symmetry), so angles above
/// π are redundant with angles below π. Paper formulas θ = jπ/200 for j=1..200
/// fully cover the distinct spectral information at the same density the previous
/// 400-angle config used over [π/200, 2π].
const NUM_SPECTRAL_ANGLES: usize = 200;

/// Spectral filter per paper Theorem 2.4, enhanced with 400 angles.
/// Two-tier: fast pre-check at 50 angles, then full check at 400.
/// θ = jπ/200 for j=1,...,400 (twice the paper's density)
fn passes_spectral_bound(c: &Sequence, d: &Sequence, margin: f64) -> bool {
    let n = c.len();
    let target = 4.0 * (n as f64) + 2.0;
    let threshold = target + margin;
    // Tier 1: Fast pre-check at 50 evenly-spaced angles (every 8th angle)
    for j in (1..=NUM_SPECTRAL_ANGLES).step_by(8) {
        let theta = (j as f64) * PI / (NUM_SPECTRAL_ANGLES as f64);
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);
        if fc + fd > threshold { return false; }
    }
    // Tier 2: Full check at all 400 angles
    for j in 1..=NUM_SPECTRAL_ANGLES {
        let theta = (j as f64) * PI / (NUM_SPECTRAL_ANGLES as f64);
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);
        if fc + fd > threshold { return false; }
    }
    true
}

fn compute_ab_headroom(c: &Sequence, d: &Sequence) -> f64 {
    let n = c.len();
    let target = 4.0 * (n as f64) + 2.0;
    let mut min_headroom = f64::MAX;
    // V7: Match enhanced spectral filter with 400 angles
    for j in 1..=NUM_SPECTRAL_ANGLES {
        let theta = (j as f64) * PI / (NUM_SPECTRAL_ANGLES as f64);
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);
        let headroom = target - fc - fd;
        min_headroom = min_headroom.min(headroom);
    }
    min_headroom
}

// ============================================================================
// Theorem 2.2: Paired position constraints (Equation 2.7)
// ============================================================================

/// For AB pairs: a_i + b_i + a_{n+2-i} + b_{n+2-i} = target_mod (mod 4)
/// where target_mod = 2 for i=1, 0 for i>=2.
/// Returns all valid (a_i, b_i, a_j, b_j) combinations from {-1, +1}.
fn valid_symmetric_pairs_ab(i: usize) -> Vec<(i32, i32, i32, i32)> {
    let target_mod = if i == 1 { 2 } else { 0 };
    let mut valid = Vec::new();
    for a_i in [-1i32, 1] {
        for b_i in [-1i32, 1] {
            for a_j in [-1i32, 1] {
                for b_j in [-1i32, 1] {
                    let sum = a_i + b_i + a_j + b_j;
                    let sum_mod4 = ((sum % 4) + 4) % 4;
                    if sum_mod4 == target_mod {
                        valid.push((a_i, b_i, a_j, b_j));
                    }
                }
            }
        }
    }
    valid
}

/// For CD pairs: c_i + d_i + c_{n+1-i} + d_{n+1-i} = 0 (mod 4) for i>=2.
/// Returns all valid (c_i, d_i, c_j, d_j) combinations from {-1, +1}.
fn valid_symmetric_pairs_cd() -> Vec<(i32, i32, i32, i32)> {
    let mut valid = Vec::new();
    for c_i in [-1i32, 1] {
        for d_i in [-1i32, 1] {
            for c_j in [-1i32, 1] {
                for d_j in [-1i32, 1] {
                    let sum = c_i + d_i + c_j + d_j;
                    let sum_mod4 = ((sum % 4) + 4) % 4;
                    if sum_mod4 == 0 {
                        valid.push((c_i, d_i, c_j, d_j));
                    }
                }
            }
        }
    }
    valid
}

/// Precompute valid symmetric pair constraints for A,B positions.
/// Returns one entry per symmetric pair (i, m+1-i) where i < m+1-i,
/// plus optionally a middle position entry for odd m.
fn precompute_symmetric_constraints_ab(n: usize) -> Vec<Vec<(i32, i32, i32, i32)>> {
    let m = n + 1; // Length of A and B
    let mut constraints = Vec::new();
    for i in 1..=((m + 1) / 2) {
        let j = m + 1 - i;
        if i < j {
            constraints.push(valid_symmetric_pairs_ab(i));
        } else if i == j {
            // Middle position (odd m): no pairing constraint, just (a_i, b_i) combos
            constraints.push(vec![
                (1, 1, 0, 0), (-1, 1, 0, 0), (1, -1, 0, 0), (-1, -1, 0, 0),
            ]);
        }
    }
    constraints
}





// ============================================================================
// Theorem 2.3: Modular decomposition (Steps 2-3)
// ============================================================================

/// Modular partial sums solution for m=3
#[derive(Clone, Debug)]
struct Mod3Solution {
    /// k_i3 for i=1,2,3 (partial sums of A at residue classes mod 3)
    k: [i32; 3],
    /// r_i3 for i=1,2,3 (partial sums of B)
    r: [i32; 3],
    /// p_i3 for i=1,2,3 (partial sums of C)
    p: [i32; 3],
    /// q_i3 for i=1,2,3 (partial sums of D)
    q: [i32; 3],
}

/// Modular partial sums solution for m=6 (CD part only)
#[derive(Clone, Debug)]
struct Mod6CDSolution {
    /// p_i6 for i=1,...,6 (partial sums of C at residue classes mod 6)
    p: [i32; 6],
    /// q_i6 for i=1,...,6 (partial sums of D)
    q: [i32; 6],
    /// The parent mod-3 solution index
    _mod3_idx: usize,
}

/// Compute range bound for partial sum k_{i,m} of a sequence of length len.
/// |k_{i,m}| <= floor((len - i) / m) + 1, and parity = (floor((len-i)/m) + 1) % 2
/// i is 1-indexed.
fn partial_sum_bound(len: usize, i: usize, m: usize) -> (i32, i32, i32) {
    let count = (len - i) / m + 1; // number of positions in residue class i mod m
    let max_val = count as i32;
    let parity = (count % 2) as i32;
    (-max_val, max_val, parity)
}

/// Compute N_X(s) = sum_{i=1}^{m-s} x_i * x_{i+s} for partial sum vector x of length m.
fn partial_autocorr(x: &[i32], s: usize) -> i32 {
    let m = x.len();
    let mut sum = 0i32;
    for i in 0..m.saturating_sub(s) {
        sum += x[i] * x[i + s];
    }
    sum
}

// Collecting wrappers for diagnostic modes (debug pipeline, stats)
fn collect_mod3_solutions(n: usize, st: &SumTuple, at: &AltSumTuple, max: usize) -> Vec<Mod3Solution> {
    let mut results = Vec::new();
    enumerate_mod3_solutions(n, st, at, &mut |sol| {
        results.push(sol.clone());
        results.len() < max
    });
    results
}

fn collect_mod6_cd_solutions(n: usize, mod3_sol: &Mod3Solution, mod3_idx: usize, st: &SumTuple, at: &AltSumTuple, max: usize) -> Vec<Mod6CDSolution> {
    let mut results = Vec::new();
    enumerate_mod6_cd_solutions(n, mod3_sol, mod3_idx, st, at, &mut |sol| {
        results.push(sol.clone());
        results.len() < max
    });
    results
}

fn collect_cd_from_mod6(n: usize, mod6_sol: &Mod6CDSolution, spectral_margin: f64, max: usize) -> (Vec<(Sequence, Sequence)>, u64) {
    let mut results = Vec::new();
    let counter = AtomicU64::new(0);
    backtrack_cd_from_mod6(n, mod6_sol, spectral_margin, &mut |c, d| {
        results.push((Sequence::new(c.to_vec()), Sequence::new(d.to_vec())));
        results.len() < max
    }, &counter, ShardCtx::none());
    (results, counter.load(Ordering::Relaxed))
}

/// Enumerate all valid mod-3 partial sum solutions for a given tuple.
/// Implements Step 2 of the paper's algorithm.
fn enumerate_mod3_solutions<F: FnMut(&Mod3Solution) -> bool>(
    n: usize,
    st: &SumTuple,
    _at: &AltSumTuple,
    callback: &mut F,
) {
    let m = 3;
    let target_sq = (4 * n + 2) as i32;
    let m_ab = n + 1; // length of A, B
    let m_cd = n;     // length of C, D

    // Compute bounds for each residue class
    // k_{i,3} bounds (A has length n+1, 1-indexed positions)
    let (k1_lo, k1_hi, k1_par) = partial_sum_bound(m_ab, 1, m);
    let (k2_lo, k2_hi, k2_par) = partial_sum_bound(m_ab, 2, m);
    let (k3_lo, k3_hi, k3_par) = partial_sum_bound(m_ab, 3, m);

    // r_{i,3} bounds (B has length n+1)
    let (r1_lo, r1_hi, r1_par) = partial_sum_bound(m_ab, 1, m);
    let (r2_lo, r2_hi, r2_par) = partial_sum_bound(m_ab, 2, m);
    let (r3_lo, r3_hi, r3_par) = partial_sum_bound(m_ab, 3, m);

    // p_{i,3} bounds (C has length n)
    let (p1_lo, p1_hi, p1_par) = partial_sum_bound(m_cd, 1, m);
    let (p2_lo, p2_hi, p2_par) = partial_sum_bound(m_cd, 2, m);
    let (p3_lo, p3_hi, p3_par) = partial_sum_bound(m_cd, 3, m);

    // q_{i,3} bounds (D has length n)
    let (q1_lo, q1_hi, q1_par) = partial_sum_bound(m_cd, 1, m);
    let (q2_lo, q2_hi, q2_par) = partial_sum_bound(m_cd, 2, m);
    let (q3_lo, q3_hi, q3_par) = partial_sum_bound(m_cd, 3, m);

    // Sum constraints: k1+k2+k3 = a, r1+r2+r3 = b, p1+p2+p3 = c, q1+q2+q3 = d
    let a = st.a;
    let b = st.b;
    let c = st.c;
    let d = st.d;

    // Enumerate k1, k2 (derive k3 = a - k1 - k2)
    let mut k1 = k1_lo + ((k1_par - ((k1_lo % 2 + 2) % 2)) % 2 + 2) % 2;
    while k1 <= k1_hi {
        let mut k2 = k2_lo + ((k2_par - ((k2_lo % 2 + 2) % 2)) % 2 + 2) % 2;
        while k2 <= k2_hi {
            let k3 = a - k1 - k2;
            if k3 < k3_lo || k3 > k3_hi || ((k3 % 2 + 2) % 2) != k3_par {
                k2 += 2; continue;
            }
            let k = [k1, k2, k3];
            let k_sq: i32 = k.iter().map(|x| x * x).sum();
            if k_sq > target_sq { k2 += 2; continue; }

            // Enumerate r1, r2 (derive r3 = b - r1 - r2)
            let mut r1 = r1_lo + ((r1_par - ((r1_lo % 2 + 2) % 2)) % 2 + 2) % 2;
            while r1 <= r1_hi {
                let mut r2 = r2_lo + ((r2_par - ((r2_lo % 2 + 2) % 2)) % 2 + 2) % 2;
                while r2 <= r2_hi {
                    let r3 = b - r1 - r2;
                    if r3 < r3_lo || r3 > r3_hi || ((r3 % 2 + 2) % 2) != r3_par {
                        r2 += 2; continue;
                    }
                    let r = [r1, r2, r3];
                    let kr_sq: i32 = k_sq + r.iter().map(|x| x * x).sum::<i32>();
                    if kr_sq > target_sq { r2 += 2; continue; }

                    // Check Eq 2.12 for AB (paired mod-4 constraints)
                    // k_{1,3}+r_{1,3}+k_{n+1,3}+r_{n+1,3}
                    // Residue of position n+1 in 1-indexed mod 3: ((n+1)-1)%3+1 = n%3+1
                    // But Eq 2.12 uses k_{j,m} notation differently:
                    // k_{j,m} = sum of a_t where t = j (mod m)
                    // So k_{n+1,m} is the partial sum for residue class (n+1) mod m
                    // For m=3: residues are 1,2,3 (or 0,1,2 depending on convention)
                    //
                    // Using 1-indexed convention: positions 1,2,...,n+1
                    // k_{1,3} = sum of a_j where j = 1 mod 3 (positions 1,4,7,...)
                    // k_{2,3} = sum of a_j where j = 2 mod 3 (positions 2,5,8,...)
                    // k_{3,3} = sum of a_j where j = 0 mod 3 (positions 3,6,9,...)
                    //
                    // Position n+1: its residue class is ((n+1)-1)%3 = n%3, mapped to class (n%3)+1
                    // But in 0-indexed residue: n%3, so k_{(n%3)+1, 3} or k_{n+1,3}
                    // Actually k_{n+1,m} in the paper notation is k_{((n+1 mod m)),m}
                    // where positions 1..m map to classes 1..m

                    // Eq 2.12 first line (j=1 for AB):
                    // k_{1,m} + r_{1,m} + k_{n+1,m} + r_{n+1,m} = 2 (mod 4) if n != 0 (mod m)
                    //                                              = 0 (mod 4) if n = 0 (mod m)
                    let res_n1 = if (n + 1) % m == 0 { m } else { (n + 1) % m }; // 1-indexed residue of position n+1
                    let k_n1 = k[res_n1 - 1];
                    let r_n1 = r[res_n1 - 1];
                    let ab_pair1_sum = k[0] + r[0] + k_n1 + r_n1;
                    let ab_pair1_target = if n % m != 0 { 2 } else { 0 };
                    if ((ab_pair1_sum % 4) + 4) % 4 != ab_pair1_target {
                        r2 += 2; continue;
                    }

                    // Eq 2.12 second line: for j=2,...,m
                    // k_{j,m} + r_{j,m} + k_{n+2-j,m} + r_{n+2-j,m} = 0 (mod 4)
                    // Skip j that creates the same class pair as the first line
                    let overlap_j_ab = if (n + 1) % m == 0 { m } else { (n + 1) % m };
                    let mut ab_mod4_ok = true;
                    for j in 2..=m {
                        if j == overlap_j_ab { continue; }
                        let res_pair = if (n + 2 - j) % m == 0 { m } else { (n + 2 - j) % m };
                        let sum_j = k[j - 1] + r[j - 1] + k[res_pair - 1] + r[res_pair - 1];
                        if ((sum_j % 4) + 4) % 4 != 0 {
                            ab_mod4_ok = false;
                            break;
                        }
                    }
                    if !ab_mod4_ok { r2 += 2; continue; }

                    // Now enumerate p1, p2, derive p3
                    let mut p1 = p1_lo + ((p1_par - ((p1_lo % 2 + 2) % 2)) % 2 + 2) % 2;
                    while p1 <= p1_hi {
                        let mut p2 = p2_lo + ((p2_par - ((p2_lo % 2 + 2) % 2)) % 2 + 2) % 2;
                        while p2 <= p2_hi {
                            let p3 = c - p1 - p2;
                            if p3 < p3_lo || p3 > p3_hi || ((p3 % 2 + 2) % 2) != p3_par {
                                p2 += 2; continue;
                            }
                            let p = [p1, p2, p3];
                            let p_sq: i32 = p.iter().map(|x| x * x).sum();
                            if kr_sq + p_sq > target_sq { p2 += 2; continue; }

                            // q_sq budget pruning: q1²+q2²+q3²= target_sq - kr_sq - p_sq
                            let q_sq_needed = target_sq - kr_sq - p_sq;
                            // Cauchy-Schwarz: min sq for sum d with 3 values is d²/3
                            // (if d² > 3*q_sq_needed, no solution exists)
                            if d * d > 3 * q_sq_needed { p2 += 2; continue; }
                            // Max achievable q_sq with current bounds
                            let max_q_abs = q1_hi.max(q2_hi).max(q3_hi);
                            if q_sq_needed > 3 * max_q_abs * max_q_abs { p2 += 2; continue; }

                            // Enumerate q1, q2, derive q3
                            let mut q1 = q1_lo + ((q1_par - ((q1_lo % 2 + 2) % 2)) % 2 + 2) % 2;
                            while q1 <= q1_hi {
                                // Prune: q1² already exceeds budget (can skip but not break since q1² isn't monotonic)
                                if q1 * q1 > q_sq_needed { q1 += 2; continue; }
                                let mut q2 = q2_lo + ((q2_par - ((q2_lo % 2 + 2) % 2)) % 2 + 2) % 2;
                                while q2 <= q2_hi {
                                    let q3 = d - q1 - q2;
                                    if q3 < q3_lo || q3 > q3_hi || ((q3 % 2 + 2) % 2) != q3_par {
                                        q2 += 2; continue;
                                    }
                                    let q = [q1, q2, q3];

                                    // Check Eq 2.10 (sum-of-squares = 4n+2)
                                    let total_sq: i32 = kr_sq + p_sq + q.iter().map(|x| x * x).sum::<i32>();
                                    if total_sq != target_sq { q2 += 2; continue; }

                                    // Check Eq 2.10 (orthogonality) for s=1:
                                    // N_K(1)+N_R(1)+N_P(1)+N_Q(1)+N_K(2)+N_R(2)+N_P(2)+N_Q(2) = 0
                                    let ortho = partial_autocorr(&k, 1) + partial_autocorr(&r, 1)
                                        + partial_autocorr(&p, 1) + partial_autocorr(&q, 1)
                                        + partial_autocorr(&k, 2) + partial_autocorr(&r, 2)
                                        + partial_autocorr(&p, 2) + partial_autocorr(&q, 2);
                                    if ortho != 0 { q2 += 2; continue; }

                                    // Check Eq 2.12 for CD (third line):
                                    // p_{j,m} + q_{j,m} + p_{n+1-j,m} + q_{n+1-j,m} = 0 (mod 4) for j=1,...,m
                                    let mut cd_mod4_ok = true;
                                    for j in 1..=m {
                                        let res_pair_cd = if (n + 1 - j) % m == 0 { m } else { (n + 1 - j) % m };
                                        let sum_j = p[j - 1] + q[j - 1] + p[res_pair_cd - 1] + q[res_pair_cd - 1];
                                        if ((sum_j % 4) + 4) % 4 != 0 {
                                            cd_mod4_ok = false;
                                            break;
                                        }
                                    }
                                    if !cd_mod4_ok { q2 += 2; continue; }

                                    let sol = Mod3Solution { k, r, p, q };
                                    if !callback(&sol) { return; }

                                    q2 += 2;
                                }
                                q1 += 2;
                            }
                            p2 += 2;
                        }
                        p1 += 2;
                    }
                    r2 += 2;
                }
                r1 += 2;
            }
            k2 += 2;
        }
        k1 += 2;
    }
}

/// Enumerate valid mod-6 CD partial sum solutions that refine a mod-3 solution.
/// Implements Step 3 of the paper's algorithm.
fn enumerate_mod6_cd_solutions<F: FnMut(&Mod6CDSolution) -> bool>(
    n: usize,
    mod3_sol: &Mod3Solution,
    mod3_idx: usize,
    st: &SumTuple,
    at: &AltSumTuple,
    callback: &mut F,
) {
    let m = 6;
    let m_cd = n; // length of C, D

    // The mod-6 CD partials must satisfy BOTH:
    //   mod-3 refinement: p6[i]+p6[i+3] = p3[i] for i=0,1,2
    //   even/odd sums:    p6[0]+p6[2]+p6[4] = (st.c + at.c_star)/2
    //                     p6[1]+p6[3]+p6[5] = (st.c - at.c_star)/2
    // Together these leave only 2 free parameters (p6[0] and p6[2]), down from 3.
    //
    // Derivation (free: p6[0], p6[2]):
    //   p6[3] = p3[0] - p6[0]
    //   p6[4] = E_c - p6[0] - p6[2]
    //   p6[1] = p3[1] - p6[4] = p3[1] - E_c + p6[0] + p6[2]
    //   p6[5] = p3[2] - p6[2]
    // where E_c = (st.c + at.c_star) / 2, E_d = (st.d + at.d_star) / 2

    let sum_cp = st.c + at.c_star;
    let sum_cm = st.c - at.c_star;
    if sum_cp % 2 != 0 || sum_cm % 2 != 0 { return; }
    let e_c = sum_cp / 2; // p6[0]+p6[2]+p6[4]

    let sum_dp = st.d + at.d_star;
    let sum_dm = st.d - at.d_star;
    if sum_dp % 2 != 0 || sum_dm % 2 != 0 { return; }
    let e_d = sum_dp / 2; // q6[0]+q6[2]+q6[4]

    // Compute bounds for each mod-6 residue class of C and D (length n)
    let mut p_bounds = [(0i32, 0i32, 0i32); 6];
    let mut q_bounds = [(0i32, 0i32, 0i32); 6];
    for i in 1..=6 {
        p_bounds[i - 1] = partial_sum_bound(m_cd, i, m);
        q_bounds[i - 1] = partial_sum_bound(m_cd, i, m);
    }

    let (plo0, phi0, ppar0) = p_bounds[0];
    let (plo1, phi1, ppar1) = p_bounds[1];
    let (plo2, phi2, ppar2) = p_bounds[2];
    let (plo3, phi3, ppar3) = p_bounds[3];
    let (plo4, phi4, ppar4) = p_bounds[4];
    let (plo5, phi5, ppar5) = p_bounds[5];

    // Enumerate p6: 2 free params (p6[0] and p6[2])
    let mut p_candidates: Vec<[i32; 6]> = Vec::new();
    let mut p6_0 = plo0 + ((ppar0 - ((plo0 % 2 + 2) % 2)) % 2 + 2) % 2;
    while p6_0 <= phi0 {
        let p6_3 = mod3_sol.p[0] - p6_0;
        if p6_3 < plo3 || p6_3 > phi3 || ((p6_3 % 2 + 2) % 2) != ppar3 { p6_0 += 2; continue; }

        let mut p6_2 = plo2 + ((ppar2 - ((plo2 % 2 + 2) % 2)) % 2 + 2) % 2;
        while p6_2 <= phi2 {
            let p6_4 = e_c - p6_0 - p6_2;
            if p6_4 < plo4 || p6_4 > phi4 || ((p6_4 % 2 + 2) % 2) != ppar4 { p6_2 += 2; continue; }
            let p6_1 = mod3_sol.p[1] - p6_4;
            if p6_1 < plo1 || p6_1 > phi1 || ((p6_1 % 2 + 2) % 2) != ppar1 { p6_2 += 2; continue; }
            let p6_5 = mod3_sol.p[2] - p6_2;
            if p6_5 < plo5 || p6_5 > phi5 || ((p6_5 % 2 + 2) % 2) != ppar5 { p6_2 += 2; continue; }
            p_candidates.push([p6_0, p6_1, p6_2, p6_3, p6_4, p6_5]);
            p6_2 += 2;
        }
        p6_0 += 2;
    }

    let (qlo0, qhi0, qpar0) = q_bounds[0];
    let (qlo1, qhi1, qpar1) = q_bounds[1];
    let (qlo2, qhi2, qpar2) = q_bounds[2];
    let (qlo3, qhi3, qpar3) = q_bounds[3];
    let (qlo4, qhi4, qpar4) = q_bounds[4];
    let (qlo5, qhi5, qpar5) = q_bounds[5];

    // For each p6 candidate, enumerate q6: 2 free params (q6[0] and q6[2])
    for p_cand in &p_candidates {
        let mut q6_0 = qlo0 + ((qpar0 - ((qlo0 % 2 + 2) % 2)) % 2 + 2) % 2;
        while q6_0 <= qhi0 {
            let q6_3 = mod3_sol.q[0] - q6_0;
            if q6_3 < qlo3 || q6_3 > qhi3 || ((q6_3 % 2 + 2) % 2) != qpar3 { q6_0 += 2; continue; }

            let mut q6_2 = qlo2 + ((qpar2 - ((qlo2 % 2 + 2) % 2)) % 2 + 2) % 2;
            while q6_2 <= qhi2 {
                let q6_4 = e_d - q6_0 - q6_2;
                if q6_4 < qlo4 || q6_4 > qhi4 || ((q6_4 % 2 + 2) % 2) != qpar4 { q6_2 += 2; continue; }
                let q6_1 = mod3_sol.q[1] - q6_4;
                if q6_1 < qlo1 || q6_1 > qhi1 || ((q6_1 % 2 + 2) % 2) != qpar1 { q6_2 += 2; continue; }
                let q6_5 = mod3_sol.q[2] - q6_2;
                if q6_5 < qlo5 || q6_5 > qhi5 || ((q6_5 % 2 + 2) % 2) != qpar5 { q6_2 += 2; continue; }
                let q_cand = [q6_0, q6_1, q6_2, q6_3, q6_4, q6_5];

                // Check Eq 2.10 sum-of-squares feasibility
                let pq_sq: i32 = p_cand.iter().chain(q_cand.iter()).map(|x| x * x).sum();
                let target_sq = (4 * n + 2) as i32;
                if pq_sq > target_sq { q6_2 += 2; continue; }

                // Check Eq 2.12 for CD at m=6:
                // p_{j,6} + q_{j,6} + p_{n+1-j,6} + q_{n+1-j,6} = 0 (mod 4) for j=1,...,6
                let mut cd_mod4_ok = true;
                for j in 1..=m {
                    let res_pair = if (n + 1 - j) % m == 0 { m } else { (n + 1 - j) % m };
                    let sum_j = p_cand[j - 1] + q_cand[j - 1] + p_cand[res_pair - 1] + q_cand[res_pair - 1];
                    if ((sum_j % 4) + 4) % 4 != 0 {
                        cd_mod4_ok = false;
                        break;
                    }
                }
                if !cd_mod4_ok { q6_2 += 2; continue; }

                // Check that at least one valid AB mod-6 exists (feasibility check)
                let feasible = check_mod6_ab_feasible(n, mod3_sol, p_cand, &q_cand, target_sq - pq_sq);
                if feasible {
                    let sol = Mod6CDSolution {
                        p: *p_cand,
                        q: q_cand,
                        _mod3_idx: mod3_idx,
                    };
                    if !callback(&sol) { return; }
                }

                q6_2 += 2;
            }
            q6_0 += 2;
        }
    }
}

/// Check if at least one valid AB mod-6 solution exists given CD mod-6 partials.
/// This is a feasibility check, not full enumeration.
fn check_mod6_ab_feasible(
    n: usize,
    mod3_sol: &Mod3Solution,
    p6: &[i32; 6],
    q6: &[i32; 6],
    kr_sq_budget: i32,
) -> bool {
    let m: usize = 6;
    let m_ab = n + 1;

    // Compute AB bounds at mod-6
    let mut k_bounds = [(0i32, 0i32, 0i32); 6];
    let mut r_bounds = [(0i32, 0i32, 0i32); 6];
    for i in 1..=6 {
        k_bounds[i - 1] = partial_sum_bound(m_ab, i, m);
        r_bounds[i - 1] = partial_sum_bound(m_ab, i, m);
    }

    // k must refine mod3: k_{i,6} + k_{i+3,6} = mod3_sol.k[i] for i=0,1,2
    // Enumerate k values
    let mut k_candidates: Vec<[i32; 6]> = vec![[0; 6]];
    for i_mod3 in 0..3 {
        let target = mod3_sol.k[i_mod3];
        let (lo_a, hi_a, par_a) = k_bounds[i_mod3];
        let (lo_b, hi_b, par_b) = k_bounds[i_mod3 + 3];

        let mut new_candidates = Vec::new();
        for cand in &k_candidates {
            let mut ka = lo_a + ((par_a - ((lo_a % 2 + 2) % 2)) % 2 + 2) % 2;
            while ka <= hi_a {
                let kb = target - ka;
                if kb >= lo_b && kb <= hi_b && ((kb % 2 + 2) % 2) == par_b {
                    let k_sq: i32 = cand.iter().take(i_mod3).map(|x| x * x).sum::<i32>()
                        + cand.iter().skip(3).take(i_mod3).map(|x| x * x).sum::<i32>()
                        + ka * ka + kb * kb;
                    if k_sq <= kr_sq_budget {
                        let mut new_cand = *cand;
                        new_cand[i_mod3] = ka;
                        new_cand[i_mod3 + 3] = kb;
                        new_candidates.push(new_cand);
                    }
                }
                ka += 2;
            }
        }
        k_candidates = new_candidates;
        if k_candidates.is_empty() { return false; }
    }

    // For each k candidate, check if any r candidate exists
    for k_cand in &k_candidates {
        let k_sq: i32 = k_cand.iter().map(|x| x * x).sum();
        let r_sq_budget = kr_sq_budget - k_sq;
        if r_sq_budget < 0 { continue; }

        // Try to find at least one valid r
        let mut found_r = false;
        // Enumerate r similarly
        let mut r_candidates: Vec<[i32; 6]> = vec![[0; 6]];
        for i_mod3 in 0..3 {
            let target = mod3_sol.r[i_mod3];
            let (lo_a, hi_a, par_a) = r_bounds[i_mod3];
            let (lo_b, hi_b, par_b) = r_bounds[i_mod3 + 3];

            let mut new_candidates = Vec::new();
            for cand in &r_candidates {
                let mut ra = lo_a + ((par_a - ((lo_a % 2 + 2) % 2)) % 2 + 2) % 2;
                while ra <= hi_a {
                    let rb = target - ra;
                    if rb >= lo_b && rb <= hi_b && ((rb % 2 + 2) % 2) == par_b {
                        let r_sq: i32 = cand.iter().take(i_mod3).map(|x| x * x).sum::<i32>()
                            + cand.iter().skip(3).take(i_mod3).map(|x| x * x).sum::<i32>()
                            + ra * ra + rb * rb;
                        if r_sq <= r_sq_budget {
                            let mut new_cand = *cand;
                            new_cand[i_mod3] = ra;
                            new_cand[i_mod3 + 3] = rb;
                            new_candidates.push(new_cand);
                        }
                    }
                    ra += 2;
                }
            }
            r_candidates = new_candidates;
            if r_candidates.is_empty() { break; }
        }

        for r_cand in &r_candidates {
            // Check sum-of-squares
            let total_sq: i32 = k_sq + r_cand.iter().map(|x| x * x).sum::<i32>()
                + p6.iter().map(|x| x * x).sum::<i32>()
                + q6.iter().map(|x| x * x).sum::<i32>();
            let target_sq_full = (4 * n + 2) as i32;
            if total_sq != target_sq_full { continue; }

            // Check orthogonality at m=6: for s=1,...,3
            let mut ortho_ok = true;
            for s in 1..=3 {
                let ortho = partial_autocorr(k_cand, s) + partial_autocorr(r_cand, s)
                    + partial_autocorr(p6, s) + partial_autocorr(q6, s)
                    + partial_autocorr(k_cand, m - s) + partial_autocorr(r_cand, m - s)
                    + partial_autocorr(p6, m - s) + partial_autocorr(q6, m - s);
                if ortho != 0 {
                    ortho_ok = false;
                    break;
                }
            }
            if !ortho_ok { continue; }

            // Check Eq 2.12 for AB at m=6
            let res_n1 = if (n + 1) % m == 0 { m } else { (n + 1) % m };
            let ab_pair1_sum = k_cand[0] + r_cand[0] + k_cand[res_n1 - 1] + r_cand[res_n1 - 1];
            let ab_pair1_target = if n % m != 0 { 2 } else { 0 };
            if ((ab_pair1_sum % 4) + 4) % 4 != ab_pair1_target { continue; }

            // Skip j that creates same class pair as first line
            let overlap_j_ab6 = if (n + 1) % m == 0 { m } else { (n + 1) % m };
            let mut ab_mod4_ok = true;
            for j in 2..=m {
                if j == overlap_j_ab6 { continue; }
                let res_pair = if (n + 2 - j) % m == 0 { m } else { (n + 2 - j) % m };
                let sum_j = k_cand[j - 1] + r_cand[j - 1] + k_cand[res_pair - 1] + r_cand[res_pair - 1];
                if ((sum_j % 4) + 4) % 4 != 0 {
                    ab_mod4_ok = false;
                    break;
                }
            }
            if !ab_mod4_ok { continue; }

            found_r = true;
            break;
        }

        if found_r { return true; }
    }

    false
}

/// Enumerate ALL valid mod-6 AB partial sum solutions (k6, r6) given the mod-3 AB
/// partial sums from `mod3_sol` and the known CD mod-6 partials `p6`/`q6`.
/// Returns Vec of ([k6;6], [r6;6]) pairs that satisfy:
///   - Refinement of mod-3 sums
///   - Sum-of-squares budget (k² + r² = target - p² - q²)
///   - Orthogonality at m=6 (N_K(s)+N_R(s)+N_P(s)+N_Q(s)=0 for s=1..3 and 6-s)
///   - Eq 2.12 mod-4 constraints for AB
fn enumerate_mod6_ab_solutions(
    n: usize,
    st: &SumTuple,
    at: &AltSumTuple,
    p6: &[i32; 6],
    q6: &[i32; 6],
) -> Vec<([i32; 6], [i32; 6], u16)> {
    let mut results: Vec<([i32; 6], [i32; 6], u16)> = Vec::new();
    let m: usize = 6;
    let m_ab = n + 1;

    let target_sq = (4 * n + 2) as i32;
    let pq_sq: i32 = p6.iter().chain(q6.iter()).map(|x| x * x).sum();
    if pq_sq > target_sq { return results; }
    let kr_sq_budget = target_sq - pq_sq;

    let mut k_bounds = [(0i32, 0i32, 0i32); 6];
    let mut r_bounds = [(0i32, 0i32, 0i32); 6];
    for i in 1..=6 {
        k_bounds[i - 1] = partial_sum_bound(m_ab, i, m);
        r_bounds[i - 1] = partial_sum_bound(m_ab, i, m);
    }

    // Derive per-tuple even/odd sum constraints for k6:
    //   k6[0]+k6[2]+k6[4] = (st.a + at.a_star) / 2   (a[j] at even positions j)
    //   k6[1]+k6[3]+k6[5] = (st.a - at.a_star) / 2   (a[j] at odd positions j)
    // These are correct for ALL mod3_sols and all (c,d) pairs under this tuple.
    // Using mod3_sol.k here was the original bug: the same (c,d) can be reached from
    // multiple mod3_sols (with different k3/r3), so constraining to one mod3_sol's k3
    // falsely pruned AB paths found under the correct mod3_sol.
    let k_sum_a_plus = st.a + at.a_star;
    let k_sum_a_minus = st.a - at.a_star;
    if k_sum_a_plus % 2 != 0 || k_sum_a_minus % 2 != 0 { return results; }
    let k_even_sum = k_sum_a_plus / 2;   // k6[0]+k6[2]+k6[4]
    let k_odd_sum  = k_sum_a_minus / 2;  // k6[1]+k6[3]+k6[5]

    let r_sum_b_plus = st.b + at.b_star;
    let r_sum_b_minus = st.b - at.b_star;
    if r_sum_b_plus % 2 != 0 || r_sum_b_minus % 2 != 0 { return results; }
    let r_even_sum = r_sum_b_plus / 2;
    let r_odd_sum  = r_sum_b_minus / 2;

    // Enumerate k6 as two independent groups:
    //   Even group: k6[0], k6[2] free, k6[4] = k_even_sum - k6[0] - k6[2]
    //   Odd  group: k6[1], k6[3] free, k6[5] = k_odd_sum  - k6[1] - k6[3]
    // This has the same O(bound²) cost as the former mod3 approach.
    let (lo0, hi0, par0) = k_bounds[0];
    let (lo2, hi2, par2) = k_bounds[2];
    let (lo4, hi4, par4) = k_bounds[4];
    let (lo1, hi1, par1) = k_bounds[1];
    let (lo3, hi3, par3) = k_bounds[3];
    let (lo5, hi5, par5) = k_bounds[5];

    let mut k_candidates: Vec<[i32; 6]> = Vec::new();
    {
        let mut k0 = lo0 + ((par0 - ((lo0 % 2 + 2) % 2)) % 2 + 2) % 2;
        while k0 <= hi0 {
            let sq0 = k0 * k0;
            if sq0 > kr_sq_budget { k0 += 2; continue; }
            let mut k2 = lo2 + ((par2 - ((lo2 % 2 + 2) % 2)) % 2 + 2) % 2;
            while k2 <= hi2 {
                let sq02 = sq0 + k2 * k2;
                if sq02 > kr_sq_budget { k2 += 2; continue; }
                let k4 = k_even_sum - k0 - k2;
                if k4 < lo4 || k4 > hi4 || ((k4 % 2 + 2) % 2) != par4 { k2 += 2; continue; }
                let sq024 = sq02 + k4 * k4;
                if sq024 > kr_sq_budget { k2 += 2; continue; }
                // Odd group
                let mut k1 = lo1 + ((par1 - ((lo1 % 2 + 2) % 2)) % 2 + 2) % 2;
                while k1 <= hi1 {
                    let sq0241 = sq024 + k1 * k1;
                    if sq0241 > kr_sq_budget { k1 += 2; continue; }
                    let mut k3 = lo3 + ((par3 - ((lo3 % 2 + 2) % 2)) % 2 + 2) % 2;
                    while k3 <= hi3 {
                        let k5 = k_odd_sum - k1 - k3;
                        if k5 < lo5 || k5 > hi5 || ((k5 % 2 + 2) % 2) != par5 { k3 += 2; continue; }
                        let sq_total = sq0241 + k3 * k3 + k5 * k5;
                        if sq_total <= kr_sq_budget {
                            k_candidates.push([k0, k1, k2, k3, k4, k5]);
                        }
                        k3 += 2;
                    }
                    k1 += 2;
                }
                k2 += 2;
            }
            k0 += 2;
        }
    }
    if k_candidates.is_empty() { return results; }

    let overlap_j = if (n + 1) % m == 0 { m } else { (n + 1) % m };
    let ab_pair1_target = if n % m != 0 { 2i32 } else { 0i32 };

    // Precompute r_bounds for the odd/even sum enumeration
    let (rlo0, rhi0, rpar0) = r_bounds[0];
    let (rlo2, rhi2, rpar2) = r_bounds[2];
    let (rlo4, rhi4, rpar4) = r_bounds[4];
    let (rlo1, rhi1, rpar1) = r_bounds[1];
    let (rlo3, rhi3, rpar3) = r_bounds[3];
    let (rlo5, rhi5, rpar5) = r_bounds[5];

    for k_cand in &k_candidates {
        let k_sq: i32 = k_cand.iter().map(|x| x * x).sum();
        let r_sq_budget = kr_sq_budget - k_sq;
        if r_sq_budget < 0 { continue; }

        // Enumerate r6 using even/odd sum constraints, same structure as k6
        let mut r_candidates: Vec<[i32; 6]> = Vec::new();
        {
            let mut r0 = rlo0 + ((rpar0 - ((rlo0 % 2 + 2) % 2)) % 2 + 2) % 2;
            while r0 <= rhi0 {
                let sq0 = r0 * r0;
                if sq0 > r_sq_budget { r0 += 2; continue; }
                let mut r2 = rlo2 + ((rpar2 - ((rlo2 % 2 + 2) % 2)) % 2 + 2) % 2;
                while r2 <= rhi2 {
                    let sq02 = sq0 + r2 * r2;
                    if sq02 > r_sq_budget { r2 += 2; continue; }
                    let r4 = r_even_sum - r0 - r2;
                    if r4 < rlo4 || r4 > rhi4 || ((r4 % 2 + 2) % 2) != rpar4 { r2 += 2; continue; }
                    let sq024 = sq02 + r4 * r4;
                    if sq024 > r_sq_budget { r2 += 2; continue; }
                    let mut r1 = rlo1 + ((rpar1 - ((rlo1 % 2 + 2) % 2)) % 2 + 2) % 2;
                    while r1 <= rhi1 {
                        let sq0241 = sq024 + r1 * r1;
                        if sq0241 > r_sq_budget { r1 += 2; continue; }
                        let mut r3 = rlo3 + ((rpar3 - ((rlo3 % 2 + 2) % 2)) % 2 + 2) % 2;
                        while r3 <= rhi3 {
                            let r5 = r_odd_sum - r1 - r3;
                            if r5 < rlo5 || r5 > rhi5 || ((r5 % 2 + 2) % 2) != rpar5 { r3 += 2; continue; }
                            let sq_total = sq0241 + r3 * r3 + r5 * r5;
                            if sq_total <= r_sq_budget {
                                r_candidates.push([r0, r1, r2, r3, r4, r5]);
                            }
                            r3 += 2;
                        }
                        r1 += 2;
                    }
                    r2 += 2;
                }
                r0 += 2;
            }
        }

        for r_cand in &r_candidates {
            // Sum-of-squares exact check
            let total_sq: i32 = k_sq + r_cand.iter().map(|x| x * x).sum::<i32>() + pq_sq;
            if total_sq != target_sq { continue; }

            // Orthogonality at m=6: for s=1,2,3 check N_K(s)+N_R(s)+N_P(s)+N_Q(s)
            //                                          + N_K(6-s)+N_R(6-s)+N_P(6-s)+N_Q(6-s) = 0
            let mut ortho_ok = true;
            for s in 1..=3usize {
                let ortho = partial_autocorr(k_cand, s) + partial_autocorr(r_cand, s)
                    + partial_autocorr(p6, s) + partial_autocorr(q6, s)
                    + partial_autocorr(k_cand, m - s) + partial_autocorr(r_cand, m - s)
                    + partial_autocorr(p6, m - s) + partial_autocorr(q6, m - s);
                if ortho != 0 { ortho_ok = false; break; }
            }
            if !ortho_ok { continue; }

            // Eq 2.12 mod-4 for AB at m=6
            let res_n1 = if (n + 1) % m == 0 { m } else { (n + 1) % m };
            let sum1 = k_cand[0] + r_cand[0] + k_cand[res_n1 - 1] + r_cand[res_n1 - 1];
            if ((sum1 % 4) + 4) % 4 != ((ab_pair1_target % 4) + 4) % 4 { continue; }

            let mut mod4_ok = true;
            for j in 2..=m {
                if j == overlap_j { continue; }
                let res_pair = if (n + 2 - j) % m == 0 { m } else { (n + 2 - j) % m };
                let sum_j = k_cand[j-1] + r_cand[j-1] + k_cand[res_pair-1] + r_cand[res_pair-1];
                if ((sum_j % 4) + 4) % 4 != 0 { mod4_ok = false; break; }
            }
            if !mod4_ok { continue; }

            // Precompute 12-bit parity signature: low 6 bits = k6 parities, high 6 = r6 parities.
            // Lets backtrack_search_ab reject parity-incompatible sols with a single u16 compare.
            let mut sig: u16 = 0;
            for i in 0..6 {
                sig |= ((k_cand[i] & 1) as u16) << i;
                sig |= ((r_cand[i] & 1) as u16) << (i + 6);
            }
            results.push((*k_cand, *r_cand, sig));
        }
    }
    results
}

/// Multi-node CD-tree sharding context. When `claim_dir` is Some and
/// `shard_depth > 0`, the recursion claims subtrees by atomic file create at
/// `<claim_dir>/t{tuple_idx}_p{m3m6_pair_idx}_cd{prefix:x}.claimed`. Each
/// prefix represents a path of pair-choice indices in `backtrack_cd_from_mod6`
/// taken from depth 0 up to depth `shard_depth-1`. Exactly one node wins the
/// create per prefix; the rest skip the entire subtree. Sharing happens
/// BEFORE the spectral filter, so non-owning nodes do zero CD-generation work
/// on skipped subtrees.
#[derive(Clone, Copy)]
struct ShardCtx<'a> {
    tuple_idx: usize,
    m3m6_pair_idx: usize,
    claim_dir: Option<&'a str>,
    shard_depth: usize,
}

impl<'a> ShardCtx<'a> {
    /// No-shard context: behaves identically to v7 (no FS coordination).
    fn none() -> Self {
        Self { tuple_idx: 0, m3m6_pair_idx: 0, claim_dir: None, shard_depth: 0 }
    }
}

/// Atomic O_CREAT|O_EXCL on a per-prefix sentinel file. Returns true if this
/// process won the claim (or if no claim_dir is configured).
fn try_claim_cd_branch(dir: &str, tuple_idx: usize, m3m6_pair_idx: usize, branch_prefix: u64) -> bool {
    let path = format!("{}/t{}_p{}_cd{:x}.claimed", dir, tuple_idx, m3m6_pair_idx, branch_prefix);
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .is_ok()
}

/// Deterministic backtracking to construct C,D sequences satisfying both
/// Theorem 2.2 paired constraints AND target mod-6 partial sums.
/// Integrates spectral filtering: maintains Hall polynomial incrementally,
/// checks exact spectral bound at base case, prunes via lower bound near leaves.
/// Calls `callback` for each spectrally-valid CD pair. Callback returns true to continue, false to stop.
/// Returns total CDs that reached spectral check.
///
/// v8: When `shard.claim_dir` is set, the top `shard.shard_depth` levels of
/// pair-choice iteration are gated on atomic file claims, allowing multiple
/// nodes on a shared filesystem to disjointly partition the CD tree.
fn backtrack_cd_from_mod6(
    n: usize,
    mod6_sol: &Mod6CDSolution,
    spectral_margin: f64,
    callback: &mut dyn FnMut(&[i32], &[i32]) -> bool,
    cd_checked_counter: &AtomicU64,
    shard: ShardCtx<'_>,
) {
    let m = 6usize;
    let valid_pairs_cd = valid_symmetric_pairs_cd(); // 8 valid choices

    // V7: Enhanced symmetry breaking for CD generation
    // When C<->D are interchangeable (same mod-6 partial sum targets),
    // require (c[0], c[n-1]) <= (d[0], d[n-1]) lexicographically.
    let cd_symmetric = mod6_sol.p == mod6_sol.q;

    // All 16 choices for unconstrained pair (0, n-1), filtered by CD symmetry
    let all_16: Vec<(i32, i32, i32, i32)> = {
        let mut v = Vec::new();
        for &ci in &[-1i32, 1] {
            for &di in &[-1i32, 1] {
                for &cj in &[-1i32, 1] {
                    for &dj in &[-1i32, 1] {
                        // V7: CD lex-leader constraint
                        if cd_symmetric {
                            // Require (ci, cj) <= (di, dj) lexicographically
                            if ci > di || (ci == di && cj > dj) { continue; }
                        }
                        v.push((ci, di, cj, dj));
                    }
                }
            }
        }
        v
    };

    // Pair info: (pos_left, pos_right, class_left, class_right)
    let mut pair_positions: Vec<(usize, usize, usize, usize)> = Vec::new();

    // Pair (0, n-1): NO Theorem 2.2 constraint (i=1 in 1-indexed, only applies for i>=2)
    if n >= 2 {
        pair_positions.push((0, n - 1, 0 % m, (n - 1) % m));
    }

    // Constrained pairs: (1, n-2), (2, n-3), ..., up to left < right
    for i_0 in 1..(n / 2) {
        let j_0 = n - 1 - i_0;
        if i_0 >= j_0 { break; }
        pair_positions.push((i_0, j_0, i_0 % m, j_0 % m));
    }

    // Middle position (odd n only)
    let has_middle = n % 2 == 1;
    let mid_pos = n / 2;
    let mid_class = mid_pos % m;

    // Compute total positions per class
    let mut c_total = [0usize; 6];
    for pos in 0..n {
        c_total[pos % m] += 1;
    }

    // Sort pairs by minimum class size for tighter pruning (skip index 0 which is unconstrained)
    if pair_positions.len() > 1 {
        let mut indices: Vec<usize> = (1..pair_positions.len()).collect();
        indices.sort_by_key(|&i| {
            let (_, _, lc, rc) = pair_positions[i];
            std::cmp::min(c_total[lc], c_total[rc])
        });
        let old_positions = pair_positions.clone();
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            pair_positions[new_idx + 1] = old_positions[old_idx];
        }
    }

    let target_p = mod6_sol.p;
    let target_q = mod6_sol.q;

    let mut stop = false;
    let mut c_vals = vec![0i32; n];
    let mut d_vals = vec![0i32; n];
    let mut c_running = [0i32; 6];
    let mut d_running = [0i32; 6];
    let mut filled = [0usize; 6];
    let num_pairs = pair_positions.len();

    // Precompute trig tables for incremental spectral check
    // V7: θ = jπ/200 for j=1,...,400 (400 angles, doubled from paper's 200)
    // Hoisted via OnceLock — same n across program means tables are computed exactly once.
    static SPECTRAL_TRIG: std::sync::OnceLock<(usize, Vec<f64>, Vec<f64>)> = std::sync::OnceLock::new();
    let num_angles: usize = NUM_SPECTRAL_ANGLES;
    let spectral_threshold = 4.0 * (n as f64) + 2.0 + spectral_margin;
    let angle_denom = NUM_SPECTRAL_ANGLES as f64; // = 200.0
    let trig_pair = SPECTRAL_TRIG.get_or_init(|| {
        let trig_cos: Vec<f64> = (0..n).flat_map(|pos| {
            (0..num_angles).map(move |k| {
                ((pos as f64) * ((k + 1) as f64) * PI / angle_denom).cos()
            })
        }).collect();
        let trig_sin: Vec<f64> = (0..n).flat_map(|pos| {
            (0..num_angles).map(move |k| {
                ((pos as f64) * ((k + 1) as f64) * PI / angle_denom).sin()
            })
        }).collect();
        (n, trig_cos, trig_sin)
    });
    debug_assert_eq!(trig_pair.0, n, "OnceLock initialized with different n");
    let trig_cos = &trig_pair.1[..];
    let trig_sin = &trig_pair.2[..];

    // Running Hall polynomial state: Re/Im of C and D at each spectral angle
    let mut real_c = vec![0.0f64; num_angles];
    let mut imag_c = vec![0.0f64; num_angles];
    let mut real_d = vec![0.0f64; num_angles];
    let mut imag_d = vec![0.0f64; num_angles];
    #[allow(clippy::too_many_arguments)]
    fn recurse(
        pair_idx: usize,
        num_pairs: usize,
        pair_positions: &[(usize, usize, usize, usize)],
        valid_pairs_cd: &[(i32, i32, i32, i32)],
        all_16: &[(i32, i32, i32, i32)],
        c_vals: &mut Vec<i32>,
        d_vals: &mut Vec<i32>,
        c_running: &mut [i32; 6],
        d_running: &mut [i32; 6],
        filled: &mut [usize; 6],
        c_total: &[usize; 6],
        target_p: &[i32; 6],
        target_q: &[i32; 6],
        has_middle: bool,
        mid_pos: usize,
        mid_class: usize,
        callback: &mut dyn FnMut(&[i32], &[i32]) -> bool,
        stop: &mut bool,
        trig_cos: &[f64],
        trig_sin: &[f64],
        num_angles: usize,
        real_c: &mut [f64],
        imag_c: &mut [f64],
        real_d: &mut [f64],
        imag_d: &mut [f64],
        spectral_threshold: f64,
        cd_checked: &AtomicU64,
        shard: &ShardCtx<'_>,
        prefix_so_far: u64,
    ) {
        if *stop { return; }
        let na = num_angles;

        if pair_idx >= num_pairs {
            // All pairs filled; handle middle if needed
            if has_middle {
                for &cm in &[-1i32, 1] {
                    for &dm in &[-1i32, 1] {
                        let mut sums_ok = true;
                        for cls in 0..6 {
                            let c_sum = c_running[cls] + if cls == mid_class { cm } else { 0 };
                            let d_sum = d_running[cls] + if cls == mid_class { dm } else { 0 };
                            if c_sum != target_p[cls] || d_sum != target_q[cls] {
                                sums_ok = false; break;
                            }
                        }
                        if sums_ok {
                            // Update spectral for middle position
                            let cm_f = cm as f64;
                            let dm_f = dm as f64;
                            let mid_off = mid_pos * na;
                            for k in 0..na {
                                let c = trig_cos[mid_off + k];
                                let s = trig_sin[mid_off + k];
                                real_c[k] += cm_f * c;
                                imag_c[k] += cm_f * s;
                                real_d[k] += dm_f * c;
                                imag_d[k] += dm_f * s;
                            }

                            // Exact spectral check
                            cd_checked.fetch_add(1, Ordering::Relaxed);
                            let mut spectral_ok = true;
                            for k in 0..na {
                                let fc = real_c[k] * real_c[k] + imag_c[k] * imag_c[k];
                                let fd = real_d[k] * real_d[k] + imag_d[k] * imag_d[k];
                                if fc + fd > spectral_threshold {
                                    spectral_ok = false;
                                    break;
                                }
                            }

                            if spectral_ok {
                                c_vals[mid_pos] = cm;
                                d_vals[mid_pos] = dm;
                                if !callback(c_vals, d_vals) { *stop = true; }
                            }

                            // Undo spectral for middle
                            for k in 0..na {
                                let c = trig_cos[mid_off + k];
                                let s = trig_sin[mid_off + k];
                                real_c[k] -= cm_f * c;
                                imag_c[k] -= cm_f * s;
                                real_d[k] -= dm_f * c;
                                imag_d[k] -= dm_f * s;
                            }

                            if *stop { return; }
                        }
                    }
                }
            } else {
                let mut sums_ok = true;
                for cls in 0..6 {
                    if c_running[cls] != target_p[cls] || d_running[cls] != target_q[cls] {
                        sums_ok = false; break;
                    }
                }
                if sums_ok {
                    // Exact spectral check (no middle position)
                    cd_checked.fetch_add(1, Ordering::Relaxed);
                    let mut spectral_ok = true;
                    for k in 0..na {
                        let fc = real_c[k] * real_c[k] + imag_c[k] * imag_c[k];
                        let fd = real_d[k] * real_d[k] + imag_d[k] * imag_d[k];
                        if fc + fd > spectral_threshold {
                            spectral_ok = false;
                            break;
                        }
                    }
                    if spectral_ok {
                        if !callback(c_vals, d_vals) { *stop = true; }
                    }
                }
            }
            return;
        }

        let (left, right, lc, rc) = pair_positions[pair_idx];
        let choices: &[(i32, i32, i32, i32)] = if pair_idx == 0 { all_16 } else { valid_pairs_cd };

        // Precompute trig sums for this pair's two positions (same for all options)
        let left_off = left * na;
        let right_off = right * na;

        // Precompute sum/diff trig arrays (stack allocated, no heap).
        // For options with ci==cj: ci_f * cl + cj_f * cr = ci_f * (cl+cr) = ci_f * p_cos[k]
        // For options with ci!=cj: ci_f * cl + cj_f * cr = ci_f * (cl-cr) = ci_f * m_cos[k]
        // This halves the number of muls in the update/undo loops (1 FMA per coord instead of 2).
        let mut p_cos = [0.0f64; NUM_SPECTRAL_ANGLES];
        let mut m_cos = [0.0f64; NUM_SPECTRAL_ANGLES];
        let mut p_sin = [0.0f64; NUM_SPECTRAL_ANGLES];
        let mut m_sin = [0.0f64; NUM_SPECTRAL_ANGLES];
        for k in 0..na {
            let cl = trig_cos[left_off + k];
            let cr = trig_cos[right_off + k];
            let sl = trig_sin[left_off + k];
            let sr = trig_sin[right_off + k];
            p_cos[k] = cl + cr;
            m_cos[k] = cl - cr;
            p_sin[k] = sl + sr;
            m_sin[k] = sl - sr;
        }

        for (choice_idx, &(ci, di, cj, dj)) in choices.iter().enumerate() {
            // v8 CD-tree shard: compute the prefix this choice extends. Encoded
            // in a u64 with radix 16 (max choices at any pair). pair_idx 0 has
            // up to 16 options (`all_16` post-symmetry filter); deeper pairs
            // have ≤ 8 (`valid_pairs_cd`). Radix-16 keeps the encoding
            // injective without per-level bookkeeping.
            let new_prefix = prefix_so_far
                .wrapping_mul(16)
                .wrapping_add(choice_idx as u64);

            // Multi-node sharding: at the deepest configured shard level
            // (pair_idx + 1 == shard_depth), attempt to claim this prefix.
            // Skip the entire subtree if another node already owns it.
            // Shallower depths (pair_idx + 1 < shard_depth) recurse
            // unconditionally — the claim contention happens once per
            // top-level prefix path, deeper levels inherit ownership.
            if let Some(dir) = shard.claim_dir {
                if shard.shard_depth > 0 && pair_idx + 1 == shard.shard_depth {
                    if !try_claim_cd_branch(dir, shard.tuple_idx, shard.m3m6_pair_idx, new_prefix) {
                        continue;
                    }
                }
            }

            // Apply mod-6 running sums
            c_running[lc] += ci;
            d_running[lc] += di;
            c_running[rc] += cj;
            d_running[rc] += dj;
            filled[lc] += 1;
            filled[rc] += 1;

            // Feasibility check (mod-6 partial sums)
            let mut feasible = true;
            for cls in 0..6 {
                let remaining = c_total[cls] - filled[cls];
                let c_need = target_p[cls] - c_running[cls];
                let d_need = target_q[cls] - d_running[cls];
                if c_need.unsigned_abs() as usize > remaining
                    || (c_need + remaining as i32) % 2 != 0
                {
                    feasible = false; break;
                }
                if d_need.unsigned_abs() as usize > remaining
                    || (d_need + remaining as i32) % 2 != 0
                {
                    feasible = false; break;
                }
            }

            if feasible {
                let ci_f = ci as f64;
                let di_f = di as f64;

                // Quick spectral pre-screen: check 8 evenly-spaced angles before
                // doing the full O(400) update. Computes what magnitude WOULD be
                // and applies lower-bound check. Avoids expensive update+undo for
                // obviously doomed options.
                let unfilled_pre = 2 * (num_pairs - pair_idx - 1)
                    + if has_middle { 1 } else { 0 };
                let mut pre_ok = true;
                if unfilled_pre > 0 && unfilled_pre <= 16 {
                    let u_pre = unfilled_pre as f64;
                    let st_pre = spectral_threshold;
                    let cc_pre = if ci == cj { &p_cos } else { &m_cos };
                    let cs_pre = if ci == cj { &p_sin } else { &m_sin };
                    let dc_pre = if di == dj { &p_cos } else { &m_cos };
                    let ds_pre = if di == dj { &p_sin } else { &m_sin };
                    // Check every (na/8)th angle (8 angles out of na)
                    let stride = na / 8;
                    for ki in 0..8 {
                        let k = ki * stride;
                        if k >= na { break; }
                        let new_rc = real_c[k] + ci_f * cc_pre[k];
                        let new_ic = imag_c[k] + ci_f * cs_pre[k];
                        let new_rd = real_d[k] + di_f * dc_pre[k];
                        let new_id = imag_d[k] + di_f * ds_pre[k];
                        let rc2 = new_rc * new_rc + new_ic * new_ic;
                        let rd2 = new_rd * new_rd + new_id * new_id;
                        let mag_c_v = rc2.sqrt();
                        let mag_d_v = rd2.sqrt();
                        let lb_c = if mag_c_v > u_pre { mag_c_v - u_pre } else { 0.0 };
                        let lb_d = if mag_d_v > u_pre { mag_d_v - u_pre } else { 0.0 };
                        if lb_c * lb_c + lb_d * lb_d > st_pre {
                            pre_ok = false;
                            break;
                        }
                    }
                }
                if !pre_ok {
                    // Undo mod-6 running sums and skip
                    c_running[lc] -= ci;
                    d_running[lc] -= di;
                    c_running[rc] -= cj;
                    d_running[rc] -= dj;
                    filled[lc] -= 1;
                    filled[rc] -= 1;
                    continue;
                }

                c_vals[left] = ci;
                d_vals[left] = di;
                c_vals[right] = cj;
                d_vals[right] = dj;

                // Update spectral state incrementally using precomputed sum/diff trig arrays.
                // For ci==cj: combined trig = p_cos/p_sin; for ci!=cj: m_cos/m_sin.
                // Update uses 4 FMAs per 4 angles (one per coord) instead of 8.
                let c_cos_arr = if ci == cj { &p_cos } else { &m_cos };
                let c_sin_arr = if ci == cj { &p_sin } else { &m_sin };
                let d_cos_arr = if di == dj { &p_cos } else { &m_cos };
                let d_sin_arr = if di == dj { &p_sin } else { &m_sin };
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
                unsafe {
                    use std::arch::x86_64::*;
                    let ci_v = _mm256_set1_pd(ci_f);
                    let di_v = _mm256_set1_pd(di_f);
                    let k_end = na & !3;
                    let mut k = 0;
                    while k < k_end {
                        let cc_v = _mm256_loadu_pd(c_cos_arr.as_ptr().add(k));
                        let cs_v = _mm256_loadu_pd(c_sin_arr.as_ptr().add(k));
                        let dc_v = _mm256_loadu_pd(d_cos_arr.as_ptr().add(k));
                        let ds_v = _mm256_loadu_pd(d_sin_arr.as_ptr().add(k));
                        let rc_v = _mm256_loadu_pd(real_c.as_ptr().add(k));
                        let ic_v = _mm256_loadu_pd(imag_c.as_ptr().add(k));
                        let rd_v = _mm256_loadu_pd(real_d.as_ptr().add(k));
                        let id_v = _mm256_loadu_pd(imag_d.as_ptr().add(k));
                        let rc_new = _mm256_fmadd_pd(ci_v, cc_v, rc_v);
                        let ic_new = _mm256_fmadd_pd(ci_v, cs_v, ic_v);
                        let rd_new = _mm256_fmadd_pd(di_v, dc_v, rd_v);
                        let id_new = _mm256_fmadd_pd(di_v, ds_v, id_v);
                        _mm256_storeu_pd(real_c.as_mut_ptr().add(k), rc_new);
                        _mm256_storeu_pd(imag_c.as_mut_ptr().add(k), ic_new);
                        _mm256_storeu_pd(real_d.as_mut_ptr().add(k), rd_new);
                        _mm256_storeu_pd(imag_d.as_mut_ptr().add(k), id_new);
                        k += 4;
                    }
                    while k < na {
                        real_c[k] += ci_f * c_cos_arr[k];
                        imag_c[k] += ci_f * c_sin_arr[k];
                        real_d[k] += di_f * d_cos_arr[k];
                        imag_d[k] += di_f * d_sin_arr[k];
                        k += 1;
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
                for k in 0..na {
                    real_c[k] += ci_f * c_cos_arr[k];
                    imag_c[k] += ci_f * c_sin_arr[k];
                    real_d[k] += di_f * d_cos_arr[k];
                    imag_d[k] += di_f * d_sin_arr[k];
                }

                // V7: Improved spectral lower-bound pruning
                // Key insight: unfilled positions contribute at most u_c to C and u_d to D
                // where u_c + u_d = unfilled. The JOINT bound is tighter than assuming
                // both can reduce by the full u.
                //
                // Three pruning levels:
                // 1. Quick cutoff (no sqrt): if partial magnitude² already exceeds
                //    (sqrt(threshold) + u)², prune immediately
                // 2. Standard lb: (|H_C| - u)² + (|H_D| - u)² > threshold
                // 3. Tight joint lb: unfilled shared between C,D, so minimize
                //    (|H_C| - uc)² + (|H_D| - ud)² subject to uc + ud = u, uc,ud >= 0
                //    Optimal: uc = min(u, max(0, |H_C| - |H_D| + u)/2 ... but
                //    simpler: (max(0, |H_C|+|H_D| - u))² / 2 (Cauchy-Schwarz)
                let unfilled = 2 * (num_pairs - pair_idx - 1)
                    + if has_middle { 1 } else { 0 };
                let mut spectral_feasible = true;
                if unfilled > 0 {
                    let u = unfilled as f64;
                    // Scaled tiers relative to na.
                    // At medium depths (unfilled 4-8), pruning is effective and saves
                    // exponential work — use more angles. At shallow depths (unfilled > 16),
                    // lower bounds are zero so more angles don't help.
                    let check_angles = if unfilled <= 4 { na }
                        else if unfilled <= 8 { na * 3 / 4 }
                        else if unfilled <= 16 { na / 4 }
                        else { na / 8 };
                    let st = spectral_threshold;
                    let cutoff_mag = u + st.sqrt();
                    let cutoff_sq = cutoff_mag * cutoff_mag;
                    // Precompute tight bound threshold:
                    // sum_lb = max(0, mag_c + mag_d - u); if sum_lb² > 2*st, prune
                    let tight_2st = 2.0 * st;
                    // AVX2 Level 1 bulk prefilter: compute 4 rc2/rd2 at once, skip chunks where all pass.
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
                    {
                        use std::arch::x86_64::*;
                        let cutoff_v = unsafe { _mm256_set1_pd(cutoff_sq) };
                        let k_end_simd = check_angles & !3;
                        let mut k = 0;
                        while k < k_end_simd {
                            let (rc2_arr, rd2_arr, any_over) = unsafe {
                                let rc_v = _mm256_loadu_pd(real_c.as_ptr().add(k));
                                let ic_v = _mm256_loadu_pd(imag_c.as_ptr().add(k));
                                let rd_v = _mm256_loadu_pd(real_d.as_ptr().add(k));
                                let id_v = _mm256_loadu_pd(imag_d.as_ptr().add(k));
                                let rc2_v = _mm256_fmadd_pd(ic_v, ic_v, _mm256_mul_pd(rc_v, rc_v));
                                let rd2_v = _mm256_fmadd_pd(id_v, id_v, _mm256_mul_pd(rd_v, rd_v));
                                let rc_over = _mm256_cmp_pd::<_CMP_GT_OQ>(rc2_v, cutoff_v);
                                let rd_over = _mm256_cmp_pd::<_CMP_GT_OQ>(rd2_v, cutoff_v);
                                let any_over_v = _mm256_or_pd(rc_over, rd_over);
                                let mask = _mm256_movemask_pd(any_over_v);
                                let mut rc2_a = [0f64; 4];
                                let mut rd2_a = [0f64; 4];
                                _mm256_storeu_pd(rc2_a.as_mut_ptr(), rc2_v);
                                _mm256_storeu_pd(rd2_a.as_mut_ptr(), rd2_v);
                                (rc2_a, rd2_a, mask)
                            };
                            if any_over != 0 {
                                // Scalar handling for this 4-chunk
                                for j in 0..4 {
                                    let rc2 = rc2_arr[j];
                                    let rd2 = rd2_arr[j];
                                    if rc2 <= cutoff_sq && rd2 <= cutoff_sq { continue; }
                                    let mag_c = rc2.sqrt();
                                    let mag_d = rd2.sqrt();
                                    let lb_c = if mag_c > u { mag_c - u } else { 0.0 };
                                    let lb_d = if mag_d > u { mag_d - u } else { 0.0 };
                                    if lb_c * lb_c + lb_d * lb_d > st {
                                        spectral_feasible = false;
                                        break;
                                    }
                                    let sum_lb = mag_c + mag_d - 2.0 * u;
                                    if sum_lb > 0.0 && sum_lb * sum_lb > tight_2st {
                                        spectral_feasible = false;
                                        break;
                                    }
                                }
                                if !spectral_feasible { break; }
                            }
                            k += 4;
                        }
                        // Scalar tail
                        if spectral_feasible {
                            while k < check_angles {
                                let rc2 = real_c[k] * real_c[k] + imag_c[k] * imag_c[k];
                                let rd2 = real_d[k] * real_d[k] + imag_d[k] * imag_d[k];
                                if rc2 <= cutoff_sq && rd2 <= cutoff_sq { k += 1; continue; }
                                let mag_c = rc2.sqrt();
                                let mag_d = rd2.sqrt();
                                let lb_c = if mag_c > u { mag_c - u } else { 0.0 };
                                let lb_d = if mag_d > u { mag_d - u } else { 0.0 };
                                if lb_c * lb_c + lb_d * lb_d > st {
                                    spectral_feasible = false;
                                    break;
                                }
                                let sum_lb = mag_c + mag_d - 2.0 * u;
                                if sum_lb > 0.0 && sum_lb * sum_lb > tight_2st {
                                    spectral_feasible = false;
                                    break;
                                }
                                k += 1;
                            }
                        }
                    }
                    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
                    for k in 0..check_angles {
                        let rc2 = real_c[k] * real_c[k] + imag_c[k] * imag_c[k];
                        let rd2 = real_d[k] * real_d[k] + imag_d[k] * imag_d[k];
                        if rc2 <= cutoff_sq && rd2 <= cutoff_sq { continue; }
                        let mag_c = rc2.sqrt();
                        let mag_d = rd2.sqrt();
                        let lb_c = if mag_c > u { mag_c - u } else { 0.0 };
                        let lb_d = if mag_d > u { mag_d - u } else { 0.0 };
                        if lb_c * lb_c + lb_d * lb_d > st {
                            spectral_feasible = false;
                            break;
                        }
                        let sum_lb = mag_c + mag_d - 2.0 * u;
                        if sum_lb > 0.0 && sum_lb * sum_lb > tight_2st {
                            spectral_feasible = false;
                            break;
                        }
                    }
                }

                if spectral_feasible {
                    recurse(
                        pair_idx + 1, num_pairs, pair_positions,
                        valid_pairs_cd, all_16,
                        c_vals, d_vals, c_running, d_running, filled,
                        c_total, target_p, target_q,
                        has_middle, mid_pos, mid_class,
                        callback, stop,
                        trig_cos, trig_sin, num_angles,
                        real_c, imag_c, real_d, imag_d,
                        spectral_threshold, cd_checked,
                        shard, new_prefix,
                    );
                }

                // Undo spectral state using same precomputed sum/diff arrays (fnmadd).
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
                unsafe {
                    use std::arch::x86_64::*;
                    let ci_v = _mm256_set1_pd(ci_f);
                    let di_v = _mm256_set1_pd(di_f);
                    let k_end = na & !3;
                    let mut k = 0;
                    while k < k_end {
                        let cc_v = _mm256_loadu_pd(c_cos_arr.as_ptr().add(k));
                        let cs_v = _mm256_loadu_pd(c_sin_arr.as_ptr().add(k));
                        let dc_v = _mm256_loadu_pd(d_cos_arr.as_ptr().add(k));
                        let ds_v = _mm256_loadu_pd(d_sin_arr.as_ptr().add(k));
                        let rc_v = _mm256_loadu_pd(real_c.as_ptr().add(k));
                        let ic_v = _mm256_loadu_pd(imag_c.as_ptr().add(k));
                        let rd_v = _mm256_loadu_pd(real_d.as_ptr().add(k));
                        let id_v = _mm256_loadu_pd(imag_d.as_ptr().add(k));
                        let rc_new = _mm256_fnmadd_pd(ci_v, cc_v, rc_v);
                        let ic_new = _mm256_fnmadd_pd(ci_v, cs_v, ic_v);
                        let rd_new = _mm256_fnmadd_pd(di_v, dc_v, rd_v);
                        let id_new = _mm256_fnmadd_pd(di_v, ds_v, id_v);
                        _mm256_storeu_pd(real_c.as_mut_ptr().add(k), rc_new);
                        _mm256_storeu_pd(imag_c.as_mut_ptr().add(k), ic_new);
                        _mm256_storeu_pd(real_d.as_mut_ptr().add(k), rd_new);
                        _mm256_storeu_pd(imag_d.as_mut_ptr().add(k), id_new);
                        k += 4;
                    }
                    while k < na {
                        real_c[k] -= ci_f * c_cos_arr[k];
                        imag_c[k] -= ci_f * c_sin_arr[k];
                        real_d[k] -= di_f * d_cos_arr[k];
                        imag_d[k] -= di_f * d_sin_arr[k];
                        k += 1;
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
                for k in 0..na {
                    real_c[k] -= ci_f * c_cos_arr[k];
                    imag_c[k] -= ci_f * c_sin_arr[k];
                    real_d[k] -= di_f * d_cos_arr[k];
                    imag_d[k] -= di_f * d_sin_arr[k];
                }

                if *stop {
                    c_running[lc] -= ci;
                    d_running[lc] -= di;
                    c_running[rc] -= cj;
                    d_running[rc] -= dj;
                    filled[lc] -= 1;
                    filled[rc] -= 1;
                    return;
                }
            }

            // Undo mod-6 running sums
            c_running[lc] -= ci;
            d_running[lc] -= di;
            c_running[rc] -= cj;
            d_running[rc] -= dj;
            filled[lc] -= 1;
            filled[rc] -= 1;
        }
    }

    recurse(
        0, num_pairs, &pair_positions,
        &valid_pairs_cd, &all_16,
        &mut c_vals, &mut d_vals, &mut c_running, &mut d_running, &mut filled,
        &c_total, &target_p, &target_q,
        has_middle, mid_pos, mid_class,
        callback, &mut stop,
        &trig_cos, &trig_sin, num_angles,
        &mut real_c, &mut imag_c, &mut real_d, &mut imag_d,
        spectral_threshold, cd_checked_counter,
        &shard, 0,
    );
}

/// Backtracking search for A,B sequences using Theorem 2.2 constraints.
/// Uses incremental autocorrelation tracking: O(n) per node instead of O(n²).
/// Positions filled outside-in; after pair k, shift n-k becomes fully determined.
/// Enhanced with parity pruning, next-shift pre-check, and cross-term look-ahead.

/// Convenience wrapper: computes constraints inline (for non-hot-path calls)
fn backtrack_search_ab_simple(
    n: usize, c: &[i32], d: &[i32], st: &SumTuple, at: &AltSumTuple, max_nodes: u64,
) -> (Option<(Sequence, Sequence)>, u64) {
    let constraints = precompute_symmetric_constraints_ab(n);
    let m = n + 1;
    let max_contrib: Vec<Vec<i32>> = (0..constraints.len()).map(|pidx| {
        let ready_shift = n - pidx;
        let uf_lo = pidx + 1;
        (0..=n).map(|s| {
            if s == 0 || s >= ready_shift || pidx + 2 > m { return 0; }
            let uf_hi = m - 2 - pidx;
            if uf_hi < uf_lo { return 0; }
            let uf_len = uf_hi - uf_lo + 1;
            let both_unfilled = if s < uf_len { (uf_len - s) as i32 } else { 0 };
            let left_lo = std::cmp::max(s, uf_lo);
            let left_hi = std::cmp::min(s + uf_lo - 1, uf_hi);
            let left_count = if left_hi >= left_lo { (left_hi - left_lo + 1) as i32 } else { 0 };
            let right_lo = std::cmp::max(if uf_hi + 1 >= s { uf_hi + 1 - s } else { 0 }, uf_lo);
            let right_hi = std::cmp::min(if m > s { m - 1 - s } else { 0 }, uf_hi);
            let right_count = if m > s && right_hi >= right_lo { (right_hi - right_lo + 1) as i32 } else { 0 };
            2 * (both_unfilled + left_count + right_count)
        }).collect()
    }).collect();
    backtrack_search_ab(n, c, d, st, at, max_nodes, &constraints, &max_contrib, &[])
}

fn backtrack_search_ab(
    n: usize,
    c: &[i32],
    d: &[i32],
    st: &SumTuple,
    at: &AltSumTuple,
    max_nodes: u64,
    cached_constraints: &[Vec<(i32, i32, i32, i32)>],
    cached_max_contrib: &[Vec<i32>],
    ab_mod6_sols: &[([i32; 6], [i32; 6], u16)],
) -> (Option<(Sequence, Sequence)>, u64) {
    let m = n + 1;

    // Precompute CD autocorrelations using bitwise XOR + POPCNT (O(n/64) per shift)
    let cd_autocorr = bitwise_cd_autocorrelations(c, d, n);

    let constraints = cached_constraints;
    let max_contrib_table = cached_max_contrib;

    // Precompute Σ max_contrib[s]² per pair_idx via OnceLock — saves O(n) per AB node.
    // Hoisted to a slice we pass down so backtrack() avoids the per-node atomic load.
    static MAX_ENERGY_PRECOMP: std::sync::OnceLock<Vec<i64>> = std::sync::OnceLock::new();
    let max_energy_table: &[i64] = MAX_ENERGY_PRECOMP.get_or_init(|| {
        (0..max_contrib_table.len()).map(|pidx| {
            let ready = n.saturating_sub(pidx);
            (1..ready).map(|s| {
                let mc = max_contrib_table[pidx][s] as i64;
                mc * mc
            }).sum()
        }).collect()
    });

    let mut a = vec![0i32; m];
    let mut b = vec![0i32; m];
    let mut nodes_visited = 0u64;

    // Feature 3: Mod-6 AB partial-sum feasibility.
    // mod6_rem[pidx][i] = count of positions j in [pidx, m-1-pidx] with j%6==i.
    // After filling pair pidx, remaining unfilled range becomes [pidx+1, m-2-pidx],
    // so we use mod6_rem[pidx+1] for the feasibility check.
    let mod6_rem: Vec<[i32; 6]> = (0..=m).map(|pidx| {
        let mut cnt = [0i32; 6];
        let lo = pidx;
        let hi = if m >= 1 + pidx { m - 1 - pidx } else { 0 };
        if hi >= lo {
            for j in lo..=hi { cnt[j % 6] += 1; }
        }
        cnt
    }).collect();
    // Running mod-6 partial sums of A and B (updated as pairs are filled)
    let mut k_run = [0i32; 6];
    let mut r_run = [0i32; 6];

    // Incremental state: partial_ac[s] = N_C(s) + N_D(s) + (AB terms for filled pairs)
    let mut partial_ac = cd_autocorr.clone();
    let mut a_sum = 0i32;
    let mut b_sum = 0i32;
    let mut a_alt_sum = 0i32; // running alternating sum of a
    let mut b_alt_sum = 0i32; // running alternating sum of b

    // Add contributions of newly-filled position `pos` to shifts 1..=max_shift.
    // Keep branch (skips unfilled positions efficiently) but remove bounds checks.
    #[inline(always)]
    fn update_partial_ac(
        partial_ac: &mut [i32], a: &[i32], b: &[i32],
        pos: usize, a_val: i32, b_val: i32, max_shift: usize, m: usize,
    ) {
        let left_end = max_shift.min(pos);
        let right_end = max_shift.min(m - 1 - pos);
        unsafe {
            for s in 1..=left_end {
                let nb = pos - s;
                let av = *a.get_unchecked(nb);
                if av != 0 {
                    *partial_ac.get_unchecked_mut(s) +=
                        av * a_val + *b.get_unchecked(nb) * b_val;
                }
            }
            for s in 1..=right_end {
                let nb = pos + s;
                let av = *a.get_unchecked(nb);
                if av != 0 {
                    *partial_ac.get_unchecked_mut(s) +=
                        a_val * av + b_val * *b.get_unchecked(nb);
                }
            }
        }
    }

    // Reverse the update for position `pos` (called during backtrack undo).
    #[inline(always)]
    fn undo_partial_ac(
        partial_ac: &mut [i32], a: &[i32], b: &[i32],
        pos: usize, a_val: i32, b_val: i32, max_shift: usize, m: usize,
    ) {
        let left_end = max_shift.min(pos);
        let right_end = max_shift.min(m - 1 - pos);
        unsafe {
            for s in 1..=left_end {
                let nb = pos - s;
                let av = *a.get_unchecked(nb);
                if av != 0 {
                    *partial_ac.get_unchecked_mut(s) -=
                        av * a_val + *b.get_unchecked(nb) * b_val;
                }
            }
            for s in 1..=right_end {
                let nb = pos + s;
                let av = *a.get_unchecked(nb);
                if av != 0 {
                    *partial_ac.get_unchecked_mut(s) -=
                        a_val * av + b_val * *b.get_unchecked(nb);
                }
            }
        }
    }

    // Precompute alternating signs for each position
    let alt_sign: Vec<i32> = (0..m).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

    // Symmetry breaking flags for AB backtracking (depth 0 only)
    // Reversal: reverse(A), reverse(B) is a same-tuple symmetry only when n is even
    let use_reversal = n % 2 == 0;
    // A<->B swap: swapping A,B gives same tuple only when sum and alt-sum match
    let ab_symmetric = st.a == st.b && at.a_star == at.b_star;

    // max_contrib_table is now thread-local cached (see above)

    fn backtrack(
        pair_idx: usize,
        a: &mut Vec<i32>,
        b: &mut Vec<i32>,
        m: usize,
        n: usize,
        constraints: &[Vec<(i32, i32, i32, i32)>],
        partial_ac: &mut Vec<i32>,
        a_sum: &mut i32,
        b_sum: &mut i32,
        a_alt_sum: &mut i32,
        b_alt_sum: &mut i32,
        alt_sign: &[i32],
        st: &SumTuple,
        at: &AltSumTuple,
        nodes_visited: &mut u64,
        max_nodes: u64,
        use_reversal: bool,
        ab_symmetric: bool,
        max_contrib_table: &[Vec<i32>],
        max_energy_table: &[i64],
        k_run: &mut [i32; 6],
        r_run: &mut [i32; 6],
        ab_mod6_sols: &[([i32; 6], [i32; 6], u16)],
        mod6_rem: &[[i32; 6]],
    ) -> bool {
        if *nodes_visited >= max_nodes { return false; }

        let num_pairs = constraints.len();
        if pair_idx >= num_pairs {
            if *a_sum != st.a || *b_sum != st.b { return false; }
            if *a_alt_sum != at.a_star || *b_alt_sum != at.b_star { return false; }
            for s in 1..=(n.saturating_sub(num_pairs)) {
                if partial_ac[s] != 0 { return false; }
            }
            return true;
        }

        let pos_left = pair_idx;
        let pos_right = m - 1 - pair_idx;
        let is_middle = pos_left == pos_right;
        let ready_shift = n - pair_idx;
        let remaining_positions = if is_middle { 0 } else { m - 2 * (pair_idx + 1) };
        let rp = remaining_positions as i32;

        // Hoist alt_sign lookups out of the option loop (was 2 array reads per option).
        let eps_l = alt_sign[pos_left];
        let eps_r = if !is_middle { alt_sign[pos_right] } else { 0 };

        // --- Pre-filter coefficients for ready_shift (computed once per depth) ---
        let cross_dist = if !is_middle { pos_right - pos_left } else { 0 };

        macro_rules! compute_shift_coeffs {
            ($s:expr) => {{
                let s = $s;
                let mut cla = 0i32; let mut clb = 0i32;
                let mut cra = 0i32; let mut crb = 0i32;
                if pos_left >= s {
                    let av = a[pos_left - s];
                    if av != 0 { cla += av; clb += b[pos_left - s]; }
                }
                if pos_left + s < m {
                    let av = a[pos_left + s];
                    if av != 0 { cla += av; clb += b[pos_left + s]; }
                }
                if !is_middle {
                    if pos_right >= s {
                        let av = a[pos_right - s];
                        if av != 0 { cra += av; crb += b[pos_right - s]; }
                    }
                    if pos_right + s < m {
                        let av = a[pos_right + s];
                        if av != 0 { cra += av; crb += b[pos_right + s]; }
                    }
                }
                (cla, clb, cra, crb, cross_dist == s)
            }};
        }

        let (cl_a, cl_b, cr_a, cr_b, has_cross_rs) = compute_shift_coeffs!(ready_shift);
        let required_delta = -partial_ac[ready_shift];

        // Pre-check coefficients for shifts ready_shift-1 through ready_shift-6
        let ns1 = ready_shift.wrapping_sub(1);
        let do_ns1 = ns1 >= 1 && !is_middle && remaining_positions > 0;
        let (ns1_cla, ns1_clb, ns1_cra, ns1_crb, ns1_cross) = if do_ns1 { compute_shift_coeffs!(ns1) } else { (0,0,0,0,false) };
        let ns1_mc = if do_ns1 { max_contrib_table[pair_idx][ns1] } else { 0 };

        let ns2 = ready_shift.wrapping_sub(2);
        let do_ns2 = ns2 >= 1 && !is_middle && remaining_positions > 0;
        let (ns2_cla, ns2_clb, ns2_cra, ns2_crb, ns2_cross) = if do_ns2 { compute_shift_coeffs!(ns2) } else { (0,0,0,0,false) };
        let ns2_mc = if do_ns2 { max_contrib_table[pair_idx][ns2] } else { 0 };

        let ns3 = ready_shift.wrapping_sub(3);
        let do_ns3 = ns3 >= 1 && !is_middle && remaining_positions > 0;
        let (ns3_cla, ns3_clb, ns3_cra, ns3_crb, ns3_cross) = if do_ns3 { compute_shift_coeffs!(ns3) } else { (0,0,0,0,false) };
        let ns3_mc = if do_ns3 { max_contrib_table[pair_idx][ns3] } else { 0 };

        // ns4-ns6 removed: 3 shift checks + Level 3 look-ahead is the sweet spot

        // Theorem 2.2 parity is preserved by every option (a_i+a_j is always even, ditto
        // for b and the alt-sums since alt_sign is ±1). Combined with the invariant that
        // st.a, st.b, at.a_star, at.b_star all have parity m%2, the parity portion of the
        // sum/alt-sum feasibility check is provably ALWAYS true — we removed it from the
        // option loop below.

        for &(a_i, b_i, a_j, b_j) in &constraints[pair_idx] {
            *nodes_visited += 1;
            if *nodes_visited >= max_nodes { return false; }

            // Symmetry breaking: at depth 0, only explore canonical orbit representative
            if pair_idx == 0 {
                let option = (a_i, b_i, a_j, b_j);
                let mut canonical = option;
                if use_reversal {
                    let rev = (a_j, b_j, a_i, b_i);
                    if rev < canonical { canonical = rev; }
                }
                if ab_symmetric {
                    let swp = (b_i, a_i, b_j, a_j);
                    if swp < canonical { canonical = swp; }
                }
                if use_reversal && ab_symmetric {
                    let rev_swp = (b_j, a_j, b_i, a_i);
                    if rev_swp < canonical { canonical = rev_swp; }
                }
                if option != canonical {
                    continue;
                }
            }

            // 1. Ready-shift pre-filter: O(1), skip options that can't zero ready shift
            let cross_ab = if !is_middle { a_i * a_j + b_i * b_j } else { 0 };
            let delta = a_i * cl_a + b_i * cl_b + a_j * cr_a + b_j * cr_b
                      + if has_cross_rs { cross_ab } else { 0 };
            if delta != required_delta { continue; }

            // 2. Sum/alt-sum feasibility: O(1), check without modifying state
            let da_alt;
            let db_alt;
            if is_middle {
                da_alt = eps_l * a_i;
                db_alt = eps_l * b_i;
                if *a_sum + a_i != st.a || *b_sum + b_i != st.b { continue; }
                if *a_alt_sum + da_alt != at.a_star || *b_alt_sum + db_alt != at.b_star { continue; }
            } else {
                da_alt = eps_l * a_i + eps_r * a_j;
                db_alt = eps_l * b_i + eps_r * b_j;
                let ar = st.a - *a_sum - a_i - a_j;
                let br = st.b - *b_sum - b_i - b_j;
                let aar = at.a_star - *a_alt_sum - da_alt;
                let bar = at.b_star - *b_alt_sum - db_alt;
                // Parity checks are provably dead by construction (st.a, etc. all have
                // parity m%2, and option deltas are always even). Only magnitudes remain.
                if ar.abs() > rp || br.abs() > rp
                    || aar.abs() > rp || bar.abs() > rp
                { continue; }
            }

            // 2b. Multi-shift pre-check: 6 shifts (ready_shift-1 through -6)
            //     Each O(1), avoids expensive O(n) update for infeasible options.
            macro_rules! check_shift {
                ($do:expr, $cla:expr, $clb:expr, $cra:expr, $crb:expr, $cross:expr, $mc:expr, $s:expr) => {
                    if $do {
                        let d = a_i * $cla + b_i * $clb + a_j * $cra + b_j * $crb
                              + if $cross { cross_ab } else { 0 };
                        let np = partial_ac[$s] + d;
                        if (np & 1) != ($mc & 1) || np.abs() > $mc { continue; }
                    }
                };
            }
            check_shift!(do_ns1, ns1_cla, ns1_clb, ns1_cra, ns1_crb, ns1_cross, ns1_mc, ns1);
            check_shift!(do_ns2, ns2_cla, ns2_clb, ns2_cra, ns2_crb, ns2_cross, ns2_mc, ns2);
            check_shift!(do_ns3, ns3_cla, ns3_clb, ns3_cra, ns3_crb, ns3_cross, ns3_mc, ns3);

            // 3. Update: O(n-pair_idx) — only reached by options passing all pre-checks
            if is_middle {
                a[pos_left] = a_i;
                b[pos_left] = b_i;
                *a_sum += a_i;
                *b_sum += b_i;
                *a_alt_sum += da_alt;
                *b_alt_sum += db_alt;
                update_partial_ac(partial_ac, a, b, pos_left, a_i, b_i, n - pair_idx, m);
                // Feature 3: update mod-6 running sums
                k_run[pos_left % 6] += a_i;
                r_run[pos_left % 6] += b_i;
            } else {
                a[pos_left] = a_i;
                b[pos_left] = b_i;
                update_partial_ac(partial_ac, a, b, pos_left, a_i, b_i, n - pair_idx, m);
                a[pos_right] = a_j;
                b[pos_right] = b_j;
                update_partial_ac(partial_ac, a, b, pos_right, a_j, b_j, n - pair_idx, m);
                *a_sum += a_i + a_j;
                *b_sum += b_i + b_j;
                *a_alt_sum += da_alt;
                *b_alt_sum += db_alt;
                // Feature 3: update mod-6 running sums
                k_run[pos_left % 6] += a_i;
                k_run[pos_right % 6] += a_j;
                r_run[pos_left % 6] += b_i;
                r_run[pos_right % 6] += b_j;
            }

            // 4a. Mod-6 AB feasibility (cheap; parity-sig short-circuits most sols) runs
            //     FIRST so a mismatch skips the expensive tight-bound + doubly-constrained
            //     scan below.
            let mut shift_ok = true;
            if !ab_mod6_sols.is_empty() {
                let next_rem = if pair_idx + 1 <= mod6_rem.len().saturating_sub(1) {
                    &mod6_rem[pair_idx + 1]
                } else {
                    &[0i32; 6]
                };
                let mut req_sig: u16 = 0;
                for i in 0..6 {
                    let kb = ((k_run[i] ^ next_rem[i]) & 1) as u16;
                    let rb = ((r_run[i] ^ next_rem[i]) & 1) as u16;
                    req_sig |= kb << i;
                    req_sig |= rb << (i + 6);
                }
                let feasible = ab_mod6_sols.iter().any(|(k6, r6, sig)| {
                    if *sig != req_sig { return false; }
                    for i in 0..6 {
                        let rem = next_rem[i];
                        let k_need = k6[i] - k_run[i];
                        if k_need.abs() > rem { return false; }
                        let r_need = r6[i] - r_run[i];
                        if r_need.abs() > rem { return false; }
                    }
                    true
                });
                if !feasible { shift_ok = false; }
            }

            // 4b. Tight bound check on ALL shifts (catches failures not covered by 3-shift pre-filter)
            //    Fused with sum-of-squares energy accumulation so we only walk partial_ac once.
            if shift_ok && remaining_positions > 0 {
                let mct = unsafe { max_contrib_table.get_unchecked(pair_idx) };
                let max_energy = max_energy_table[pair_idx];
                let mut remaining_energy: i64 = 0;
                for s in (1..ready_shift).rev() {
                    let pac = unsafe { *partial_ac.get_unchecked(s) };
                    let max_c = unsafe { *mct.get_unchecked(s) };
                    if (pac & 1) != (max_c & 1) || pac.abs() > max_c {
                        shift_ok = false;
                        break;
                    }
                    let v = pac as i64;
                    remaining_energy += v * v;
                }
                // Sum-of-squares bound: each unfilled position can't fix all shifts
                // independently. max_energy precomputed via slice parameter (no per-node load).
                if shift_ok && remaining_energy > max_energy {
                    shift_ok = false;
                }

                // Feature 1: Doubly-constrained position tightening.
                // An unfilled position k with BOTH shift-s neighbors already filled
                // contributes at most |a[k-s]+a[k+s]| + |b[k-s]+b[k+s]| ≤ 4
                // to partial_ac[s], instead of the 4 the static table assumes.
                // We only run this scan for shifts whose headroom is < 4 (meaning
                // even a single doubly-constrained position saving 4 could cause pruning).
                if shift_ok {
                    let uf_lo = pair_idx + 1;
                    if m >= 2 + pair_idx {
                        let uf_hi = m - 2 - pair_idx;
                        if uf_hi >= uf_lo {
                            for s in (1..ready_shift).rev() {
                                if !shift_ok { break; }
                                let pac = partial_ac[s];
                                if pac == 0 { continue; }
                                let coarse = max_contrib_table[pair_idx][s];
                                if coarse - pac.abs() >= 4 { continue; }
                                let mut tight = coarse;
                                for k in uf_lo..=uf_hi {
                                    if k >= s && k + s < m {
                                        unsafe {
                                            let al = *a.get_unchecked(k - s);
                                            if al != 0 {
                                                let ar = *a.get_unchecked(k + s);
                                                if ar != 0 {
                                                    let sa = al + ar;
                                                    let sb = *b.get_unchecked(k - s)
                                                           + *b.get_unchecked(k + s);
                                                    tight += sa.abs() + sb.abs() - 4;
                                                }
                                            }
                                        }
                                    }
                                }
                                if pac.abs() > tight { shift_ok = false; }
                            }
                        }
                    }
                }
            }

            // 4c. (The original 4b block below is now dead — kept structurally to avoid
            //      reshuffling the remaining code paths.)
            if false && shift_ok && !ab_mod6_sols.is_empty() {
                let next_rem = if pair_idx + 1 <= mod6_rem.len().saturating_sub(1) {
                    &mod6_rem[pair_idx + 1]
                } else {
                    &[0i32; 6]
                };
                // Compute required 12-bit parity signature from k_run, r_run, next_rem.
                // Required: k6[i] has parity (k_run[i] ^ next_rem[i]) & 1, same for r6.
                // If sol's precomputed sig differs in any bit, parity fails → skip sol
                // without doing any bound math (saves most work for large ab_mod6_sols).
                let mut req_sig: u16 = 0;
                for i in 0..6 {
                    let kb = ((k_run[i] ^ next_rem[i]) & 1) as u16;
                    let rb = ((r_run[i] ^ next_rem[i]) & 1) as u16;
                    req_sig |= kb << i;
                    req_sig |= rb << (i + 6);
                }
                let feasible = ab_mod6_sols.iter().any(|(k6, r6, sig)| {
                    if *sig != req_sig { return false; }
                    for i in 0..6 {
                        let rem = next_rem[i];
                        let k_need = k6[i] - k_run[i];
                        if k_need.abs() > rem { return false; }
                        let r_need = r6[i] - r_run[i];
                        if r_need.abs() > rem { return false; }
                    }
                    true
                });
                if !feasible { shift_ok = false; }
            }

            // 5. V7: Multi-shift look-ahead with arc consistency
            //    Level 1: Can ANY next-depth option zero shift ready_shift-1?
            //    Level 2: For each such option, is shift ready_shift-2 still achievable?
            //    This catches options that zero one shift but make the next impossible.
            if shift_ok && pair_idx + 1 < num_pairs {
                let next_ready = ready_shift - 1;
                if next_ready >= 1 {
                    let next_left = pair_idx + 1;
                    let next_right = m - 2 - pair_idx;
                    let req = -partial_ac[next_ready];

                    let mut nla = 0i32;
                    let mut nlb = 0i32;
                    if next_left >= next_ready && a[next_left - next_ready] != 0 {
                        nla += a[next_left - next_ready];
                        nlb += b[next_left - next_ready];
                    }
                    if next_left + next_ready < m && a[next_left + next_ready] != 0 {
                        nla += a[next_left + next_ready];
                        nlb += b[next_left + next_ready];
                    }

                    let next_is_middle = next_left == next_right;
                    let next_remaining = if next_is_middle { 0 } else { m - 2 * (pair_idx + 2) };
                    let next_rp = next_remaining as i32;

                    // V7: Precompute 2nd-shift look-ahead coefficients
                    let s2 = next_ready.wrapping_sub(1); // = ready_shift - 2
                    let do_s2_check = s2 >= 1 && !next_is_middle && pair_idx + 1 < max_contrib_table.len();
                    let s2_mc = if do_s2_check { max_contrib_table[pair_idx + 1][s2] } else { 0 };

                    if next_is_middle {
                        let mut any_ok = false;
                        for &nai in &[-1i32, 1] {
                            for &nbi in &[-1i32, 1] {
                                if nai * nla + nbi * nlb != req { continue; }
                                // Sum check at middle
                                if *a_sum + nai != st.a || *b_sum + nbi != st.b { continue; }
                                let da = alt_sign[next_left] * nai;
                                let db = alt_sign[next_left] * nbi;
                                if *a_alt_sum + da != at.a_star || *b_alt_sum + db != at.b_star { continue; }
                                any_ok = true; break;
                            }
                            if any_ok { break; }
                        }
                        if !any_ok { shift_ok = false; }
                    } else {
                        let mut nra = 0i32;
                        let mut nrb = 0i32;
                        if next_right >= next_ready && a[next_right - next_ready] != 0 {
                            nra += a[next_right - next_ready];
                            nrb += b[next_right - next_ready];
                        }
                        if next_right + next_ready < m && a[next_right + next_ready] != 0 {
                            nra += a[next_right + next_ready];
                            nrb += b[next_right + next_ready];
                        }

                        let has_next_cross = (next_right >= next_ready && next_right - next_ready == next_left)
                            || (next_right + next_ready < m && next_right + next_ready == next_left);

                        // V7: Precompute s2 coefficients for 2nd-shift look-ahead
                        let mut s2_nla = 0i32;
                        let mut s2_nlb = 0i32;
                        let mut s2_nra = 0i32;
                        let mut s2_nrb = 0i32;
                        let mut s2_has_cross = false;
                        if do_s2_check {
                            if next_left >= s2 && a[next_left - s2] != 0 {
                                s2_nla += a[next_left - s2]; s2_nlb += b[next_left - s2];
                            }
                            if next_left + s2 < m && next_left + s2 != next_right && a[next_left + s2] != 0 {
                                s2_nla += a[next_left + s2]; s2_nlb += b[next_left + s2];
                            }
                            if next_right >= s2 && next_right - s2 != next_left && a[next_right - s2] != 0 {
                                s2_nra += a[next_right - s2]; s2_nrb += b[next_right - s2];
                            }
                            if next_right + s2 < m && a[next_right + s2] != 0 {
                                s2_nra += a[next_right + s2]; s2_nrb += b[next_right + s2];
                            }
                            s2_has_cross = next_right - next_left == s2;
                        }

                        // Precompute s3 coefficients for 3rd-shift look-ahead
                        let s3_la = if next_ready >= 3 { next_ready - 2 } else { 0 };
                        let do_s3_la = s3_la >= 1 && !next_is_middle && pair_idx + 1 < max_contrib_table.len();
                        let s3_la_mc = if do_s3_la { max_contrib_table[pair_idx + 1][s3_la] } else { 0 };
                        let mut s3_nla = 0i32;
                        let mut s3_nlb = 0i32;
                        let mut s3_nra = 0i32;
                        let mut s3_nrb = 0i32;
                        let mut s3_la_has_cross = false;
                        if do_s3_la {
                            if next_left >= s3_la && a[next_left - s3_la] != 0 {
                                s3_nla += a[next_left - s3_la]; s3_nlb += b[next_left - s3_la];
                            }
                            if next_left + s3_la < m && next_left + s3_la != next_right && a[next_left + s3_la] != 0 {
                                s3_nla += a[next_left + s3_la]; s3_nlb += b[next_left + s3_la];
                            }
                            if next_right >= s3_la && next_right - s3_la != next_left && a[next_right - s3_la] != 0 {
                                s3_nra += a[next_right - s3_la]; s3_nrb += b[next_right - s3_la];
                            }
                            if next_right + s3_la < m && a[next_right + s3_la] != 0 {
                                s3_nra += a[next_right + s3_la]; s3_nrb += b[next_right + s3_la];
                            }
                            s3_la_has_cross = next_right - next_left == s3_la;
                        }

                        let mut any_ok = false;
                        for &(nai, nbi, naj, nbj) in &constraints[pair_idx + 1] {
                            let d = nai * nla + nbi * nlb + naj * nra + nbj * nrb
                                + if has_next_cross { nai * naj + nbi * nbj } else { 0 };
                            if d != req { continue; }
                            // Sum/alt-sum feasibility
                            let ar = st.a - *a_sum - nai - naj;
                            let br = st.b - *b_sum - nbi - nbj;
                            if ar.abs() > next_rp || br.abs() > next_rp
                                || (ar + next_rp) % 2 != 0 || (br + next_rp) % 2 != 0 { continue; }
                            let da = alt_sign[next_left] * nai + alt_sign[next_right] * naj;
                            let db = alt_sign[next_left] * nbi + alt_sign[next_right] * nbj;
                            let aar = at.a_star - *a_alt_sum - da;
                            let bar = at.b_star - *b_alt_sum - db;
                            if aar.abs() > next_rp || bar.abs() > next_rp
                                || (aar + next_rp) % 2 != 0 || (bar + next_rp) % 2 != 0 { continue; }

                            // Level 2: check shift s2 = ready_shift-2
                            if do_s2_check {
                                let s2_delta = nai * s2_nla + nbi * s2_nlb
                                    + naj * s2_nra + nbj * s2_nrb
                                    + if s2_has_cross { nai * naj + nbi * nbj } else { 0 };
                                let new_pac_s2 = (partial_ac[s2] + s2_delta).abs();
                                if new_pac_s2 > s2_mc || (new_pac_s2 & 1) != (s2_mc & 1) {
                                    continue;
                                }
                            }

                            // Level 3: check shift s3 = ready_shift-3
                            if do_s3_la {
                                let s3_delta = nai * s3_nla + nbi * s3_nlb
                                    + naj * s3_nra + nbj * s3_nrb
                                    + if s3_la_has_cross { nai * naj + nbi * nbj } else { 0 };
                                let new_pac_s3 = (partial_ac[s3_la] + s3_delta).abs();
                                if new_pac_s3 > s3_la_mc || (new_pac_s3 & 1) != (s3_la_mc & 1) {
                                    continue;
                                }
                            }

                            any_ok = true;
                            break;
                        }
                        if !any_ok { shift_ok = false; }
                    }
                }
            }

            if shift_ok {
                if backtrack(pair_idx + 1, a, b, m, n, constraints,
                             partial_ac, a_sum, b_sum, a_alt_sum, b_alt_sum,
                             alt_sign, st, at,
                             nodes_visited, max_nodes,
                             use_reversal, ab_symmetric,
                             max_contrib_table, max_energy_table,
                             k_run, r_run, ab_mod6_sols, mod6_rem,
) {
                    return true;
                }
            }

            // 6. Undo
            if is_middle {
                undo_partial_ac(partial_ac, a, b, pos_left, a_i, b_i, n - pair_idx, m);
                a[pos_left] = 0;
                b[pos_left] = 0;
                *a_sum -= a_i;
                *b_sum -= b_i;
                *a_alt_sum -= da_alt;
                *b_alt_sum -= db_alt;
                // Feature 3: undo mod-6 running sums
                k_run[pos_left % 6] -= a_i;
                r_run[pos_left % 6] -= b_i;
            } else {
                undo_partial_ac(partial_ac, a, b, pos_right, a_j, b_j, n - pair_idx, m);
                a[pos_right] = 0;
                b[pos_right] = 0;
                undo_partial_ac(partial_ac, a, b, pos_left, a_i, b_i, n - pair_idx, m);
                a[pos_left] = 0;
                b[pos_left] = 0;
                *a_sum -= a_i + a_j;
                *b_sum -= b_i + b_j;
                *a_alt_sum -= da_alt;
                *b_alt_sum -= db_alt;
                // Feature 3: undo mod-6 running sums
                k_run[pos_left % 6] -= a_i;
                k_run[pos_right % 6] -= a_j;
                r_run[pos_left % 6] -= b_i;
                r_run[pos_right % 6] -= b_j;
            }
        }

        false
    }

    let found = backtrack(0, &mut a, &mut b, m, n, &constraints,
                 &mut partial_ac, &mut a_sum, &mut b_sum,
                 &mut a_alt_sum, &mut b_alt_sum, &alt_sign, st, at,
                 &mut nodes_visited, max_nodes,
                 use_reversal, ab_symmetric,
                 &max_contrib_table, max_energy_table,
                 &mut k_run, &mut r_run, ab_mod6_sols, &mod6_rem);

    if found {
        (Some((Sequence::new(a), Sequence::new(b))), nodes_visited)
    } else {
        (None, nodes_visited)
    }
}

// ============================================================================
// Shared: precompute_binomials (used by both cd_optimized and exhaustive_ab)
// ============================================================================

fn precompute_binomials(max_n: usize, max_k: usize) -> Vec<Vec<u64>> {
    let mut cache = vec![vec![0u64; max_k + 1]; max_n + 1];
    for n in 0..=max_n {
        cache[n][0] = 1;
        for k in 1..=max_k.min(n) {
            cache[n][k] = cache[n-1][k-1].saturating_add(cache[n-1][k]);
        }
    }
    cache
}

// ============================================================================
// Inlined from cd_optimized.rs
// ============================================================================

#[inline]
fn constraints_feasible(n: usize, target_sum: i32, target_alt_sum: i32) -> bool {
    let sum_plus_n = n as i32 + target_sum;
    if sum_plus_n < 0 || sum_plus_n > 2 * n as i32 || sum_plus_n % 2 != 0 {
        return false;
    }
    let num_plus = (sum_plus_n / 2) as usize;
    let n_even = (n + 1) / 2;
    let n_odd = n / 2;
    let numerator = target_alt_sum + (n_even as i32) + 2 * (num_plus as i32) - (n_odd as i32);
    if numerator % 4 != 0 { return false; }
    let e_plus = numerator / 4;
    if e_plus < 0 || e_plus > n_even as i32 { return false; }
    let o_plus = num_plus as i32 - e_plus;
    if o_plus < 0 || o_plus > n_odd as i32 { return false; }
    true
}




fn count_cd_pairs(n: usize, c_sum: i32, d_sum: i32, c_alt: i32, d_alt: i32) -> u64 {
    if !constraints_feasible(n, c_sum, c_alt) || !constraints_feasible(n, d_sum, d_alt) {
        return 0;
    }
    let n_even = (0..n).filter(|&i| i % 2 == 0).count();
    let n_odd = (0..n).filter(|&i| i % 2 == 1).count();

    let c_num_plus = ((n as i32 + c_sum) / 2) as usize;
    let c_numerator = c_alt + (n_even as i32) + 2 * (c_num_plus as i32) - (n_odd as i32);
    let c_k_even = (c_numerator / 4) as usize;
    let c_k_odd = c_num_plus - c_k_even;

    let d_num_plus = ((n as i32 + d_sum) / 2) as usize;
    let d_numerator = d_alt + (n_even as i32) + 2 * (d_num_plus as i32) - (n_odd as i32);
    let d_k_even = (d_numerator / 4) as usize;
    let d_k_odd = d_num_plus - d_k_even;

    let max_n = n_even.max(n_odd) + 1;
    let max_k = c_k_even.max(c_k_odd).max(d_k_even).max(d_k_odd) + 1;
    let binom_cache = precompute_binomials(max_n, max_k);

    let c_total = binom_cache[n_even][c_k_even] * binom_cache[n_odd][c_k_odd];
    let d_total = binom_cache[n_even][d_k_even] * binom_cache[n_odd][d_k_odd];
    c_total.saturating_mul(d_total)
}

/// Score a tuple by search difficulty (lower = try first).
///
/// Three-feature linear score, chosen by exhaustive search prioritising large-n
/// performance (n>=33) over medium-n (n=28..32). Where each n>=34 tuple costs
/// hours of AB backtracking but small/medium n run in seconds, this trade is
/// what we want.
///
/// Empirical ranks of the 16 known BS(n+1,n) solutions:
///   - n>=33 in top-10:  4/5 (was 1/5 with the prior heuristic)
///   - n>=33 in top-5:   3/5 (was 1/5)
///   - n=33: 4, n=34: 2, n=35: 2, n=36: 7, n=37: 13
///   - Cost: medium n=28..32 mostly drop out of top-5
///   - All-n top-5: 5/16 (was 11/16) — but those were cheap n's anyway
///
/// Pattern: large-n solutions tend to have a single very large |sum_X|, the
/// other three small but non-zero, and balanced |a*| ≈ |b*|. The min_sum term
/// pushes tuples with at least one zero in the sum tuple LATER, which is why
/// the prior n=36 solution (which has sum_c=sum_d=0) drops to rank 7.
fn score_tuple(sum_tuple: &SumTuple, alt_tuple: &AltSumTuple, _n: usize) -> i64 {
    let aa = sum_tuple.a.abs();
    let ab = sum_tuple.b.abs();
    let ac = sum_tuple.c.abs();
    let ad = sum_tuple.d.abs();
    let aas = alt_tuple.a_star.abs();
    let abss = alt_tuple.b_star.abs();

    let max_sum = aa.max(ab).max(ac).max(ad);
    let min_sum = aa.min(ab).min(ac).min(ad);
    let gap_aab = (aas - abss).abs();

    (-2 * max_sum - min_sum + gap_aab) as i64
}
// ============================================================================
// Debug pipeline mode: trace through each step for a known n=10 solution
// ============================================================================
fn run_debug_pipeline(n: usize) {
    println!("============================================================");
    println!("  DEBUG PIPELINE TRACE for BS({},{})  n={}", n + 1, n, n);
    println!("============================================================\n");

    // Known n=10 solution data
    let known_st = SumTuple { a: -5, b: -3, c: -2, d: -2 };
    let known_at = AltSumTuple { a_star: 1, b_star: -5, c_star: -4, d_star: 0 };
    let known_mod3_p = [-2, 1, -1]; // C partials mod 3
    let known_mod3_q = [2, -1, -3]; // D partials mod 3
    let known_mod3_k = [-2, 0, -3]; // A partials mod 3
    let known_mod3_r = [0, 0, -3];  // B partials mod 3
    let known_mod6_p = [-2, 2, 0, 0, -1, -1]; // C partials mod 6
    let known_mod6_q = [2, 0, -2, 0, -1, -1]; // D partials mod 6
    let known_c = vec![-1, 1, -1, 1, -1, -1, -1, 1, 1, -1];
    let known_d = vec![1, -1, -1, -1, -1, -1, 1, 1, -1, 1];

    if n != 10 {
        println!("WARNING: Known solution data is for n=10 only.");
        println!("Running pipeline with the known n=10 tuples against n={} constraints.", n);
        println!("Results may not be meaningful.\n");
    }

    // ================================================================
    // STEP 0: Verify tuple is valid
    // ================================================================
    println!("--- STEP 0: Tuple Validation ---\n");
    let target = (4 * n + 2) as i32;
    let sq_sum = known_st.a * known_st.a + known_st.b * known_st.b
               + known_st.c * known_st.c + known_st.d * known_st.d;
    println!("  Sum tuple:     (a={}, b={}, c={}, d={})", known_st.a, known_st.b, known_st.c, known_st.d);
    println!("  Alt-sum tuple: (a*={}, b*={}, c*={}, d*={})", known_at.a_star, known_at.b_star, known_at.c_star, known_at.d_star);
    println!("  a^2+b^2+c^2+d^2 = {} (target = {}): {}",
             sq_sum, target, if sq_sum == target { "PASS" } else { "FAIL" });

    let alt_sq_sum = known_at.a_star * known_at.a_star + known_at.b_star * known_at.b_star
                   + known_at.c_star * known_at.c_star + known_at.d_star * known_at.d_star;
    println!("  a*^2+b*^2+c*^2+d*^2 = {} (target = {}): {}",
             alt_sq_sum, target, if alt_sq_sum == target { "PASS" } else { "FAIL" });

    // Check Eq 2.4 mod-4 compatibility
    let st_sig = Mod4Signature::from_sum_tuple(&known_st);
    let at_sig = Mod4Signature::required_for_alt_tuple(&known_at, n);
    println!("  Mod-4 signature match (Eq 2.4): {}", if st_sig == at_sig { "PASS" } else { "FAIL" });
    println!("    sum mod4: ({},{},{},{})", st_sig.a_mod4, st_sig.b_mod4, st_sig.c_mod4, st_sig.d_mod4);
    println!("    alt mod4: ({},{},{},{})", at_sig.a_mod4, at_sig.b_mod4, at_sig.c_mod4, at_sig.d_mod4);

    // Check it appears in tuple enumeration
    let all_tuples = find_valid_sum_tuples_fast_v2(n);
    let tuple_found = all_tuples.iter().any(|(st, at)|
        st.a == known_st.a && st.b == known_st.b && st.c == known_st.c && st.d == known_st.d
        && at.a_star == known_at.a_star && at.b_star == known_at.b_star
        && at.c_star == known_at.c_star && at.d_star == known_at.d_star
    );
    println!("  Tuple found in enumeration ({} total): {}",
             all_tuples.len(), if tuple_found { "PASS" } else { "FAIL" });

    // Check if tuple survives 5-class filtering
    let tuple_vec = vec![(known_st.clone(), known_at.clone())];
    let canonical = filter_to_canonical_5class(tuple_vec, n);
    let survives_5class = !canonical.is_empty();
    println!("  Survives 5-class canonical filter: {}", if survives_5class { "PASS" } else { "FAIL (not canonical representative)" });
    if !survives_5class {
        // Find which equivalent tuple IS the canonical representative
        let self_key = tuple_key(&known_st, &known_at);
        let equivalents = generate_equivalent_tuples(&known_st, &known_at, n);
        let min_key = equivalents.iter().min().unwrap();
        println!("    Our key:      {:?}", self_key);
        println!("    Canonical key: {:?}", min_key);
        println!("    (We are an equivalent of the canonical representative, so the tuple");
        println!("     IS processed, just under a different representative.)");

        // Find the canonical representative in the full enumeration
        let canonical_st = SumTuple { a: min_key[0], b: min_key[1], c: min_key[2], d: min_key[3] };
        let canonical_at = AltSumTuple { a_star: min_key[4], b_star: min_key[5], c_star: min_key[6], d_star: min_key[7] };
        println!("    Canonical rep: ({},{},{},{}) / ({},{},{},{})",
                 canonical_st.a, canonical_st.b, canonical_st.c, canonical_st.d,
                 canonical_at.a_star, canonical_at.b_star, canonical_at.c_star, canonical_at.d_star);
    }

    println!();

    // ================================================================
    // STEP 1: Verify known C,D produce correct partial sums
    // ================================================================
    println!("--- STEP 1: Verify Known C,D Partial Sums ---\n");

    // Compute mod-3 partial sums of C
    let mut computed_p3 = [0i32; 3];
    for (pos, &val) in known_c.iter().enumerate() {
        let cls = pos % 3; // 0-indexed: class 0,1,2 maps to p[1],p[2],p[3] in 1-indexed
        // In the code, 1-indexed class: position j (1-indexed) has class ((j-1) % 3)
        // But p[0] = class 1 (positions 1,4,7,10 -> 0-indexed 0,3,6,9)
        // Position 0 (0-indexed) = position 1 (1-indexed) -> class 1 -> index 0
        // Position 1 (0-indexed) = position 2 (1-indexed) -> class 2 -> index 1
        // Position 2 (0-indexed) = position 3 (1-indexed) -> class 3 -> index 2
        // So: 0-indexed pos -> 1-indexed (pos+1) -> class = if (pos+1)%3==0 {3} else {(pos+1)%3} -> index class-1
        let cls_1indexed = if (pos + 1) % 3 == 0 { 3 } else { (pos + 1) % 3 };
        computed_p3[cls_1indexed - 1] += val;
        let _ = cls; // suppress warning
    }
    println!("  C mod-3 partial sums: computed={:?}, known={:?}: {}",
             computed_p3, known_mod3_p,
             if computed_p3 == known_mod3_p { "MATCH" } else { "MISMATCH" });

    let mut computed_q3 = [0i32; 3];
    for (pos, &val) in known_d.iter().enumerate() {
        let cls_1indexed = if (pos + 1) % 3 == 0 { 3 } else { (pos + 1) % 3 };
        computed_q3[cls_1indexed - 1] += val;
    }
    println!("  D mod-3 partial sums: computed={:?}, known={:?}: {}",
             computed_q3, known_mod3_q,
             if computed_q3 == known_mod3_q { "MATCH" } else { "MISMATCH" });

    // Compute mod-6 partial sums
    let mut computed_p6 = [0i32; 6];
    for (pos, &val) in known_c.iter().enumerate() {
        let cls_1indexed = if (pos + 1) % 6 == 0 { 6 } else { (pos + 1) % 6 };
        computed_p6[cls_1indexed - 1] += val;
    }
    println!("  C mod-6 partial sums: computed={:?}, known={:?}: {}",
             computed_p6, known_mod6_p,
             if computed_p6 == known_mod6_p { "MATCH" } else { "MISMATCH" });

    let mut computed_q6 = [0i32; 6];
    for (pos, &val) in known_d.iter().enumerate() {
        let cls_1indexed = if (pos + 1) % 6 == 0 { 6 } else { (pos + 1) % 6 };
        computed_q6[cls_1indexed - 1] += val;
    }
    println!("  D mod-6 partial sums: computed={:?}, known={:?}: {}",
             computed_q6, known_mod6_q,
             if computed_q6 == known_mod6_q { "MATCH" } else { "MISMATCH" });

    // Verify mod-6 refines mod-3: p_{i,3} = p_{i,6} + p_{i+3,6}
    println!("\n  Refinement check (p_{{i,3}} = p_{{i,6}} + p_{{i+3,6}}):");
    for i in 0..3 {
        let refined = computed_p6[i] + computed_p6[i + 3];
        println!("    p_{{{},3}} = {} = {} + {} (p_{{{},6}} + p_{{{},6}}): {}",
                 i + 1, computed_p3[i], computed_p6[i], computed_p6[i + 3], i + 1, i + 4,
                 if refined == computed_p3[i] { "OK" } else { "FAIL" });
    }
    println!("  Refinement check (q_{{i,3}} = q_{{i,6}} + q_{{i+3,6}}):");
    for i in 0..3 {
        let refined = computed_q6[i] + computed_q6[i + 3];
        println!("    q_{{{},3}} = {} = {} + {} (q_{{{},6}} + q_{{{},6}}): {}",
                 i + 1, computed_q3[i], computed_q6[i], computed_q6[i + 3], i + 1, i + 4,
                 if refined == computed_q3[i] { "OK" } else { "FAIL" });
    }
    println!();

    // ================================================================
    // STEP 2: enumerate_mod3_solutions - check known partials appear
    // ================================================================
    println!("--- STEP 2: Mod-3 Enumeration (enumerate_mod3_solutions) ---\n");

    let mod3_solutions = collect_mod3_solutions(n, &known_st, &known_at, 100_000);
    println!("  Total mod-3 solutions found: {}", mod3_solutions.len());

    let mut known_mod3_found = false;
    let mut known_mod3_idx: Option<usize> = None;
    for (i, sol) in mod3_solutions.iter().enumerate() {
        if sol.p == known_mod3_p && sol.q == known_mod3_q
            && sol.k == known_mod3_k && sol.r == known_mod3_r
        {
            known_mod3_found = true;
            known_mod3_idx = Some(i);
            println!("  >>> KNOWN mod-3 solution found at index {} <<<", i);
            println!("      k={:?}, r={:?}, p={:?}, q={:?}", sol.k, sol.r, sol.p, sol.q);
        }
    }
    if !known_mod3_found {
        println!("  !!! KNOWN mod-3 solution NOT FOUND in enumeration !!!");
        // Check if the p,q (CD part) appears with any k,r
        let pq_found = mod3_solutions.iter().any(|s| s.p == known_mod3_p && s.q == known_mod3_q);
        println!("  p={:?}, q={:?} with any k,r: {}", known_mod3_p, known_mod3_q,
                 if pq_found { "found" } else { "NOT found" });
        // Print first few solutions for reference
        println!("  First 10 mod-3 solutions:");
        for (i, sol) in mod3_solutions.iter().take(10).enumerate() {
            println!("    [{}] k={:?} r={:?} p={:?} q={:?}", i, sol.k, sol.r, sol.p, sol.q);
        }
    }
    println!();

    // ================================================================
    // STEP 3: enumerate_mod6_cd_solutions - check known mod-6 partials appear
    // ================================================================
    println!("--- STEP 3: Mod-6 CD Enumeration (enumerate_mod6_cd_solutions) ---\n");

    if let Some(mod3_idx) = known_mod3_idx {
        let mod3_sol = &mod3_solutions[mod3_idx];
        let mod6_solutions = collect_mod6_cd_solutions(n, mod3_sol, mod3_idx, &known_st, &known_at, 100_000);
        println!("  Total mod-6 CD solutions from known mod-3 (index {}): {}", mod3_idx, mod6_solutions.len());

        let mut known_mod6_found = false;
        let mut known_mod6_idx: Option<usize> = None;
        for (i, sol) in mod6_solutions.iter().enumerate() {
            if sol.p == known_mod6_p && sol.q == known_mod6_q {
                known_mod6_found = true;
                known_mod6_idx = Some(i);
                println!("  >>> KNOWN mod-6 CD solution found at index {} <<<", i);
                println!("      p6={:?}, q6={:?}", sol.p, sol.q);
            }
        }
        if !known_mod6_found {
            println!("  !!! KNOWN mod-6 CD solution NOT FOUND !!!");
            println!("  Looking for p6={:?}, q6={:?}", known_mod6_p, known_mod6_q);
            println!("  First 10 mod-6 solutions:");
            for (i, sol) in mod6_solutions.iter().take(10).enumerate() {
                println!("    [{}] p6={:?} q6={:?}", i, sol.p, sol.q);
            }

            // Manual check: does the known mod-6 satisfy the refinement?
            println!("\n  Manual refinement check from known mod-3:");
            for i in 0..3 {
                println!("    p_{{{},3}}={} -> p_{{{},6}}={} + p_{{{},6}}={} = {}  {}",
                         i + 1, known_mod3_p[i],
                         i + 1, known_mod6_p[i], i + 4, known_mod6_p[i + 3],
                         known_mod6_p[i] + known_mod6_p[i + 3],
                         if known_mod6_p[i] + known_mod6_p[i + 3] == known_mod3_p[i] { "OK" } else { "FAIL" });
            }
            for i in 0..3 {
                println!("    q_{{{},3}}={} -> q_{{{},6}}={} + q_{{{},6}}={} = {}  {}",
                         i + 1, known_mod3_q[i],
                         i + 1, known_mod6_q[i], i + 4, known_mod6_q[i + 3],
                         known_mod6_q[i] + known_mod6_q[i + 3],
                         if known_mod6_q[i] + known_mod6_q[i + 3] == known_mod3_q[i] { "OK" } else { "FAIL" });
            }

            // Check Eq 2.12 for CD at m=6 manually
            let m = 6usize;
            println!("\n  Manual Eq 2.12 CD check at m=6:");
            for j in 1..=m {
                let res_pair = if (n + 1 - j) % m == 0 { m } else { (n + 1 - j) % m };
                let sum_j = known_mod6_p[j - 1] + known_mod6_q[j - 1]
                    + known_mod6_p[res_pair - 1] + known_mod6_q[res_pair - 1];
                let mod4 = ((sum_j % 4) + 4) % 4;
                println!("    j={}: p[{}]+q[{}]+p[{}]+q[{}] = {} mod 4 = {} {}",
                         j, j, j, res_pair, res_pair, sum_j, mod4,
                         if mod4 == 0 { "OK" } else { "FAIL" });
            }

            // Check bounds
            let m_cd = n;
            println!("\n  Bounds check for known mod-6 partials:");
            for i in 1..=6 {
                let (lo, hi, par) = partial_sum_bound(m_cd, i, 6);
                let p_val = known_mod6_p[i - 1];
                let q_val = known_mod6_q[i - 1];
                let p_ok = p_val >= lo && p_val <= hi && ((p_val % 2 + 2) % 2) == par;
                let q_ok = q_val >= lo && q_val <= hi && ((q_val % 2 + 2) % 2) == par;
                println!("    class {}: lo={}, hi={}, par={} | p={} {} | q={} {}",
                         i, lo, hi, par, p_val, if p_ok { "OK" } else { "FAIL" }, q_val, if q_ok { "OK" } else { "FAIL" });
            }
        }

        // ================================================================
        // STEP 4: backtrack_cd_from_mod6 - check known C,D appear
        // ================================================================
        println!("\n--- STEP 4: CD Backtracking (backtrack_cd_from_mod6) ---\n");

        if let Some(mod6_idx) = known_mod6_idx {
            let mod6_sol = &mod6_solutions[mod6_idx];
            let (cd_pairs, cd_checked) = collect_cd_from_mod6(n, mod6_sol, f64::MAX, 10000);
            println!("  Total C,D pairs from known mod-6 (index {}): {} (checked {})", mod6_idx, cd_pairs.len(), cd_checked);

            let mut known_cd_found = false;
            for (i, (c, d)) in cd_pairs.iter().enumerate() {
                if c.values == known_c && d.values == known_d {
                    known_cd_found = true;
                    println!("  >>> KNOWN C,D found at index {} <<<", i);
                }
            }
            if !known_cd_found {
                println!("  !!! KNOWN C,D NOT FOUND in backtracking output !!!");
                println!("  Looking for C={:?}", known_c);
                println!("              D={:?}", known_d);
                if cd_pairs.len() <= 20 {
                    println!("  All {} generated C,D pairs:", cd_pairs.len());
                    for (i, (c, d)) in cd_pairs.iter().enumerate() {
                        println!("    [{}] C={:?}", i, c.values);
                        println!("         D={:?}", d.values);
                    }
                } else {
                    println!("  First 10 generated C,D pairs:");
                    for (i, (c, d)) in cd_pairs.iter().take(10).enumerate() {
                        println!("    [{}] C={:?}", i, c.values);
                        println!("         D={:?}", d.values);
                    }
                }

                // Check Theorem 2.2 for known C,D
                println!("\n  Manual Theorem 2.2 check on known C,D:");
                println!("  CD symmetric pair constraint (c_i+d_i+c_{{n+1-i}}+d_{{n+1-i}} = 0 mod 4 for i>=2):");
                for i in 2..=(n / 2) {
                    let j = n + 1 - i; // 1-indexed
                    let i0 = i - 1; // 0-indexed
                    let j0 = j - 1;
                    let sum = known_c[i0] + known_d[i0] + known_c[j0] + known_d[j0];
                    let mod4 = ((sum % 4) + 4) % 4;
                    println!("    i={}: c[{}]+d[{}]+c[{}]+d[{}] = {}+{}+{}+{} = {} mod4={} {}",
                             i, i0, i0, j0, j0,
                             known_c[i0], known_d[i0], known_c[j0], known_d[j0],
                             sum, mod4, if mod4 == 0 { "OK" } else { "FAIL" });
                }
            }

            // ================================================================
            // STEP 5: Spectral filter
            // ================================================================
            println!("\n--- STEP 5: Spectral Filter ---\n");

            let known_c_seq = Sequence::new(known_c.clone());
            let known_d_seq = Sequence::new(known_d.clone());

            // Test with various margins
            for margin in [0.0, 0.5, 1.0, 2.0, 3.0] {
                let passes = passes_spectral_bound(&known_c_seq, &known_d_seq, margin);
                println!("  passes_spectral_bound(margin={:.1}): {}", margin, if passes { "PASS" } else { "FAIL" });
            }

            let headroom = compute_ab_headroom(&known_c_seq, &known_d_seq);
            println!("  AB headroom (min over all theta): {:.4}", headroom);
            println!("  headroom >= 0: {}", if headroom >= 0.0 { "PASS" } else { "FAIL" });

            // Find worst-case theta
            let target_f = 4.0 * (n as f64) + 2.0;
            let mut worst_j = 0;
            let mut worst_surplus = f64::NEG_INFINITY;
            let angle_denom = NUM_SPECTRAL_ANGLES as f64;
            for j in 1..=NUM_SPECTRAL_ANGLES {
                let theta = (j as f64) * PI / angle_denom;
                let fc = hall_polynomial(&known_c_seq.values, theta);
                let fd = hall_polynomial(&known_d_seq.values, theta);
                let surplus = fc + fd - target_f;
                if surplus > worst_surplus {
                    worst_surplus = surplus;
                    worst_j = j;
                }
            }
            println!("  Worst theta: j={} (theta={:.4}), f(C)+f(D)-target = {:.4}",
                     worst_j, (worst_j as f64) * PI / angle_denom, worst_surplus);

            // Count how many generated CDs pass spectral
            let spectral_margin = 0.5_f64;
            let pass_count = cd_pairs.iter()
                .filter(|(c, d)| passes_spectral_bound(c, d, spectral_margin))
                .count();
            println!("\n  Of {} generated CD pairs, {} pass spectral (margin={:.1}) = {:.1}%",
                     cd_pairs.len(), pass_count, spectral_margin,
                     if cd_pairs.is_empty() { 0.0 } else { pass_count as f64 / cd_pairs.len() as f64 * 100.0 });
        } else {
            // Try constructing from known mod-6 directly
            println!("  (Skipping backtracking since known mod-6 was not found in enumeration)");
            println!("  Testing backtrack_cd_from_mod6 with known mod-6 partials directly...\n");

            let direct_mod6 = Mod6CDSolution {
                p: known_mod6_p,
                q: known_mod6_q,
                _mod3_idx: 0,
            };
            let (cd_pairs, cd_checked) = collect_cd_from_mod6(n, &direct_mod6, f64::MAX, 10000);
            println!("  Total C,D pairs from known mod-6 partials: {} (checked {})", cd_pairs.len(), cd_checked);

            let mut known_cd_found = false;
            for (i, (c, d)) in cd_pairs.iter().enumerate() {
                if c.values == known_c && d.values == known_d {
                    known_cd_found = true;
                    println!("  >>> KNOWN C,D found at index {} <<<", i);
                }
            }
            if !known_cd_found {
                println!("  !!! KNOWN C,D NOT FOUND !!!");
                if cd_pairs.len() <= 20 {
                    for (i, (c, d)) in cd_pairs.iter().enumerate() {
                        println!("    [{}] C={:?}  D={:?}", i, c.values, d.values);
                    }
                } else {
                    println!("  (showing first 10 of {}):", cd_pairs.len());
                    for (i, (c, d)) in cd_pairs.iter().take(10).enumerate() {
                        println!("    [{}] C={:?}  D={:?}", i, c.values, d.values);
                    }
                }
            }

            // Spectral on known C,D regardless
            println!("\n--- STEP 5: Spectral Filter (using known C,D directly) ---\n");
            let known_c_seq = Sequence::new(known_c.clone());
            let known_d_seq = Sequence::new(known_d.clone());
            for margin in [0.0, 0.5, 1.0, 2.0, 3.0] {
                let passes = passes_spectral_bound(&known_c_seq, &known_d_seq, margin);
                println!("  passes_spectral_bound(margin={:.1}): {}", margin, if passes { "PASS" } else { "FAIL" });
            }
            let headroom = compute_ab_headroom(&known_c_seq, &known_d_seq);
            println!("  AB headroom: {:.4}", headroom);

            let spectral_margin = 0.5_f64;
            let pass_count = cd_pairs.iter()
                .filter(|(c, d)| passes_spectral_bound(c, d, spectral_margin))
                .count();
            println!("\n  Of {} generated CD pairs, {} pass spectral (margin={:.1}) = {:.1}%",
                     cd_pairs.len(), pass_count, spectral_margin,
                     if cd_pairs.is_empty() { 0.0 } else { pass_count as f64 / cd_pairs.len() as f64 * 100.0 });
        }
    } else {
        // No matching mod-3 found, still test steps 4-5 with known data directly
        println!("--- STEP 3: Skipped (no matching mod-3 found) ---");
        println!("--- STEP 4: CD Backtracking (using known mod-6 directly) ---\n");

        let direct_mod6 = Mod6CDSolution {
            p: known_mod6_p,
            q: known_mod6_q,
            _mod3_idx: 0,
        };
        let (cd_pairs, cd_checked) = collect_cd_from_mod6(n, &direct_mod6, f64::MAX, 10000);
        println!("  Total C,D pairs from known mod-6 partials: {} (checked {})", cd_pairs.len(), cd_checked);

        let mut known_cd_found = false;
        for (i, (c, d)) in cd_pairs.iter().enumerate() {
            if c.values == known_c && d.values == known_d {
                known_cd_found = true;
                println!("  >>> KNOWN C,D found at index {} <<<", i);
            }
        }
        if !known_cd_found {
            println!("  !!! KNOWN C,D NOT FOUND !!!");
            if cd_pairs.len() <= 20 {
                for (i, (c, d)) in cd_pairs.iter().enumerate() {
                    println!("    [{}] C={:?}  D={:?}", i, c.values, d.values);
                }
            }
        }

        println!("\n--- STEP 5: Spectral Filter ---\n");
        let known_c_seq = Sequence::new(known_c.clone());
        let known_d_seq = Sequence::new(known_d.clone());
        for margin in [0.0, 0.5, 1.0, 2.0, 3.0] {
            let passes = passes_spectral_bound(&known_c_seq, &known_d_seq, margin);
            println!("  passes_spectral_bound(margin={:.1}): {}", margin, if passes { "PASS" } else { "FAIL" });
        }
        let headroom = compute_ab_headroom(&known_c_seq, &known_d_seq);
        println!("  AB headroom: {:.4}", headroom);
    }

    // ================================================================
    // STEP 6: Full validation - verify known ABCD is a valid base sequence
    // ================================================================
    println!("\n--- STEP 6: Full Solution Validation ---\n");

    // The known A,B sequences for n=10 aren't provided, but we can verify C,D autocorrelation contribution
    let known_c_seq = Sequence::new(known_c.clone());
    let known_d_seq = Sequence::new(known_d.clone());
    println!("  C,D autocorrelation contributions (should be small):");
    for shift in 1..=n {
        let ac_c = known_c_seq.autocorrelation(shift);
        let ac_d = known_d_seq.autocorrelation(shift);
        println!("    shift={}: AC_C={:>3}, AC_D={:>3}, sum={:>3}", shift, ac_c, ac_d, ac_c + ac_d);
    }

    // Test AB backtracking with known C,D
    println!("\n--- STEP 7: AB Backtracking Test ---\n");
    {
        let c_seq = Sequence::new(known_c.clone());
        let d_seq = Sequence::new(known_d.clone());
        let (ab_result, _nodes) = backtrack_search_ab_simple(n, &c_seq.values, &d_seq.values, &known_st, &known_at, 10_000_000);
        match ab_result {
            Some((a, b)) => {
                println!("  backtrack_search_ab: FOUND");
                println!("    A = {:?}", a.values);
                println!("    B = {:?}", b.values);
                let base = BaseSequence::new(a, b, c_seq.clone(), d_seq.clone());
                println!("    Valid: {}", base.is_valid());
            }
            None => {
                println!("  backtrack_search_ab: NOT FOUND (BUG!)");
            }
        }
    }

    // Summary
    println!("\n============================================================");
    println!("  DEBUG PIPELINE SUMMARY");
    println!("============================================================");
    println!("  Tuple valid: {}", if sq_sum == target && alt_sq_sum == target { "YES" } else { "NO" });
    println!("  Tuple found in enumeration: {}", if tuple_found { "YES" } else { "NO" });
    println!("  Mod-3 solution found: {}", if known_mod3_found { "YES" } else { "NO" });
    if known_mod3_idx.is_some() {
        let mod3_sol = &mod3_solutions[known_mod3_idx.unwrap()];
        let mod6_solutions = collect_mod6_cd_solutions(n, mod3_sol, known_mod3_idx.unwrap(), &known_st, &known_at, 100_000);
        let mod6_found = mod6_solutions.iter().any(|s| s.p == known_mod6_p && s.q == known_mod6_q);
        println!("  Mod-6 CD solution found: {}", if mod6_found { "YES" } else { "NO" });

        if mod6_found {
            let mod6_idx = mod6_solutions.iter().position(|s| s.p == known_mod6_p && s.q == known_mod6_q).unwrap();
            let (cd_pairs, _) = collect_cd_from_mod6(n, &mod6_solutions[mod6_idx], f64::MAX, 10000);
            let cd_found = cd_pairs.iter().any(|(c, d)| c.values == known_c && d.values == known_d);
            println!("  Known C,D from backtracking: {}", if cd_found { "YES" } else { "NO" });
        }
    }
    let known_c_seq2 = Sequence::new(known_c);
    let known_d_seq2 = Sequence::new(known_d);
    let spectral_pass = passes_spectral_bound(&known_c_seq2, &known_d_seq2, 0.0);
    println!("  Spectral filter passes: {}", if spectral_pass { "YES" } else { "NO" });
    println!("============================================================\n");
}

fn run_pipeline_stats(n: usize) {
    println!("============================================================");
    println!("  PIPELINE STATS for BS({},{})  n={}", n + 1, n, n);
    println!("============================================================\n");

    let start = Instant::now();

    // Step 1: Tuples
    println!("Step 1: Tuple discovery...");
    let all_tuples = find_valid_sum_tuples_fast_v2(n);
    let canonical = filter_to_canonical_5class(all_tuples, n);
    let mut sorted: Vec<(SumTuple, AltSumTuple)> = canonical.into_iter().collect();
    sorted.sort_by_key(|(st, at)| score_tuple(st, at, n));
    println!("  {} canonical tuples\n", sorted.len());

    let mut total_mod3 = 0u64;
    let mut total_mod6 = 0u64;
    let mut total_cd = 0u64;
    let mut total_spectral_pass = 0u64;
    let mut total_spectral_fail = 0u64;
    let mut best_headroom = f64::NEG_INFINITY;
    let mut found_solution = false;

    let max_tuples = sorted.len().min(5); // Process first 5 tuples for stats
    let max_cd_per_mod6: usize = 100;

    for (t_idx, (st, at)) in sorted.iter().take(max_tuples).enumerate() {
        let tuple_start = Instant::now();
        println!("  Tuple {}/{}: ({},{},{},{}) | ({},{},{},{})",
            t_idx, sorted.len(), st.a, st.b, st.c, st.d,
            at.a_star, at.b_star, at.c_star, at.d_star);

        // Step 2: Mod-3
        let mod3_sols = collect_mod3_solutions(n, st, at, 100_000);
        total_mod3 += mod3_sols.len() as u64;
        println!("    Mod-3 solutions: {}", mod3_sols.len());

        let mut tuple_mod6 = 0u64;
        let mut tuple_cd = 0u64;
        let mut tuple_pass = 0u64;
        let mut tuple_fail = 0u64;

        // Process a subset of mod-3 solutions for stats
        let max_mod3 = mod3_sols.len().min(50);
        for (m3_idx, mod3_sol) in mod3_sols.iter().take(max_mod3).enumerate() {
            // Step 3: Mod-6
            let mod6_sols = collect_mod6_cd_solutions(n, mod3_sol, m3_idx, st, at, 100_000);
            tuple_mod6 += mod6_sols.len() as u64;

            for mod6_sol in &mod6_sols {
                // Step 4: CD generation with integrated spectral filter (exact threshold)
                let (cd_pairs, cd_checked) = collect_cd_from_mod6(n, mod6_sol, 0.0, max_cd_per_mod6);
                tuple_cd += cd_checked;
                tuple_pass += cd_pairs.len() as u64;
                tuple_fail += cd_checked - cd_pairs.len() as u64;

                for (c, d) in &cd_pairs {
                    let headroom = compute_ab_headroom(c, d);
                    if headroom > best_headroom {
                        best_headroom = headroom;
                    }

                    // Step 5: Try AB backtracking
                    if let Some((_a, _b)) = backtrack_search_ab_simple(n, &c.values, &d.values, st, at, 10_000_000).0 {
                        println!("    >>> SOLUTION FOUND! <<<");
                        found_solution = true;
                    }
                }
            }
        }

        total_mod6 += tuple_mod6;
        total_cd += tuple_cd;
        total_spectral_pass += tuple_pass;
        total_spectral_fail += tuple_fail;

        let pass_rate = if tuple_pass + tuple_fail > 0 {
            tuple_pass as f64 / (tuple_pass + tuple_fail) as f64 * 100.0
        } else { 0.0 };

        println!("    Mod-6 solutions: {} (from {} mod-3)", tuple_mod6, max_mod3);
        println!("    CD pairs generated: {}", tuple_cd);
        println!("    Spectral: {} pass / {} fail ({:.1}%)",
            tuple_pass, tuple_fail, pass_rate);
        println!("    Time: {:.1}s\n", tuple_start.elapsed().as_secs_f64());
    }

    let total = total_spectral_pass + total_spectral_fail;
    let overall_rate = if total > 0 {
        total_spectral_pass as f64 / total as f64 * 100.0
    } else { 0.0 };

    println!("============================================================");
    println!("  PIPELINE SUMMARY for n={}", n);
    println!("============================================================");
    println!("  Tuples processed: {} / {}", max_tuples, sorted.len());
    println!("  Total mod-3 solutions: {}", total_mod3);
    println!("  Total mod-6 solutions: {}", total_mod6);
    println!("  Total CD pairs: {}", total_cd);
    println!("  Spectral pass: {} / {} ({:.1}%)", total_spectral_pass, total, overall_rate);
    println!("  Best AB headroom: {:.4}", best_headroom);
    println!("  Solution found: {}", if found_solution { "YES" } else { "NO" });
    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());
    println!("============================================================\n");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = if args.len() > 1 {
        args[1].parse().unwrap_or_else(|_| {
            eprintln!("Usage: {} <n> [--instance X/Y] [--claim-dir <path>] [--shard-depth N] [--ab-limit N|unlimited]", args[0]);
            std::process::exit(1);
        })
    } else {
        eprintln!("Usage: {} <n> [--instance X/Y] [--claim-dir <path>] [--shard-depth N] [--ab-limit N|unlimited]", args[0]);
        std::process::exit(1);
    };

    let debug_pipeline = args.iter().any(|a| a == "--debug-pipeline");
    let pipeline_stats = args.iter().any(|a| a == "--pipeline-stats");
    let list_tuples = args.iter().any(|a| a == "--list-tuples");

    // Log file: mirrors all init/result output and writes one progress line per
    // hour. Path comes from --log <PATH> if given, otherwise auto-generated
    // from n and (when --tuple is used) the tuple index.
    let explicit_log = args.iter().position(|a| a == "--log").and_then(|i| args.get(i + 1)).cloned();
    let tuple_for_log = args.iter().position(|a| a == "--tuple")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok());
    let log_path = explicit_log.unwrap_or_else(|| match tuple_for_log {
        Some(k) => format!("BS_{}_{}_V7Parallel_tuple{}_progress.log", n + 1, n, k),
        None => format!("BS_{}_{}_V7Parallel_progress.log", n + 1, n),
    });
    match std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(f) => { let _ = LOG_FILE.set(Mutex::new(f)); }
        Err(e) => { eprintln!("Warning: could not open log file {}: {}", log_path, e); }
    }

    if list_tuples {
        let all_tuples = find_valid_sum_tuples_fast_v2(n);
        let canonical = filter_to_canonical_5class(all_tuples, n);
        let sorted: Vec<(SumTuple, AltSumTuple)> = canonical.into_iter().collect();
        let no_sort = args.iter().any(|a| a == "--no-sort");
        let mut sorted = sorted;
        if !no_sort {
            sorted.sort_by_key(|(st, at)| score_tuple(st, at, n));
        }
        println!("# n={} canonical={}", n, sorted.len());
        for (idx, (st, at)) in sorted.iter().enumerate() {
            println!("{} {} {} {} {} {} {} {} {}",
                idx, st.a, st.b, st.c, st.d,
                at.a_star, at.b_star, at.c_star, at.d_star);
        }
        return;
    }

    // Parse --instance X/Y (e.g., --instance 1/3 for first of 3 instances)
    let instance_spec: Option<(usize, usize)> = args.iter()
        .position(|a| a == "--instance")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| {
            let parts: Vec<&str> = s.split('/').collect();
            if parts.len() == 2 {
                let inst: usize = parts[0].parse().ok()?;
                let total: usize = parts[1].parse().ok()?;
                if inst >= 1 && inst <= total && total >= 1 {
                    Some((inst, total))
                } else {
                    None
                }
            } else {
                None
            }
        });

    // Parse --tuple-range START-END (alternative to --instance)
    // Or --top K (shortcut for --tuple-range 0-K: search only the K
    // highest-ranked tuples per the score_tuple heuristic).
    let mut tuple_range: Option<(usize, usize)> = args.iter()
        .position(|a| a == "--tuple-range")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| {
            let parts: Vec<&str> = s.split('-').collect();
            if parts.len() == 2 {
                Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
            } else {
                None
            }
        });

    if let Some(top_k) = args.iter()
        .position(|a| a == "--top")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
    {
        if tuple_range.is_some() {
            eprintln!("Error: --top and --tuple-range are mutually exclusive");
            std::process::exit(1);
        }
        tuple_range = Some((0, top_k));
    }

    // --tuple N: search only tuple at rank N (0-indexed by score_tuple).
    if let Some(idx) = args.iter()
        .position(|a| a == "--tuple")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
    {
        if tuple_range.is_some() {
            eprintln!("Error: --tuple is mutually exclusive with --top / --tuple-range");
            std::process::exit(1);
        }
        tuple_range = Some((idx, idx + 1));
    }

    if instance_spec.is_some() && tuple_range.is_some() {
        eprintln!("Error: --instance and --tuple/--top/--tuple-range are mutually exclusive");
        std::process::exit(1);
    }

    // Parse --timeout SECS (wall-clock seconds; 0 or unset = no timeout)
    let timeout_secs: u64 = args.iter()
        .position(|a| a == "--timeout")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    // v8: --claim-dir <path>: enable filesystem-based work-stealing across
    // nodes at the CD-GENERATION TREE level (not the spectral-pass level as
    // in v7). Each node, when iterating choices in backtrack_cd_from_mod6 at
    // pair_idx == shard_depth - 1, atomically creates a per-prefix sentinel
    // file. Exactly one node wins per prefix; the rest skip the entire
    // subtree, doing zero CD generation / spectral filtering for it.
    //
    // This is correct for CD-bound workloads at large n (where 99.99% of
    // wall-clock is spent in CD-generation + spectral filtering, not AB
    // search). The v7 --claim-dir was wrong for that regime: it sharded
    // after the spectral filter, so every node still ran the full CD tree.
    let claim_dir: Option<String> = args.iter()
        .position(|a| a == "--claim-dir")
        .and_then(|i| args.get(i + 1))
        .cloned();

    // --shard-depth N: number of CD-tree pair-choice levels to use as the
    // claim prefix. Depth 1 = claim each top-level choice (≤16 claims per
    // (tuple, m3m6_pair)); depth 2 = claim each (top, second) pair (≤128
    // claims). Larger depth = finer load balance, more claim files, more FS
    // ops per backtrack_cd_from_mod6 invocation. Default 1. Must be the same
    // across all nodes in a multi-node run (the prefix encoding depends on
    // it). No effect unless --claim-dir is set.
    let shard_depth: usize = args.iter()
        .position(|a| a == "--shard-depth")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&d| d > 0)
        .unwrap_or(1);

    // --reset-claims: wipe --claim-dir contents at startup. Safe to pass from
    // every node in a multi-node job (races are benign: the worst outcome is
    // duplicate work on a chunk that was claimed-then-wiped), but cleaner if
    // the slurm script handles the cleanup once before srun.
    let reset_claims = args.iter().any(|a| a == "--reset-claims");

    // --found-flag <path>: shared sentinel file. A background thread polls it
    // every 5s; if it appears, sets the in-process `found` flag and the search
    // exits. The solution finder writes the file so peer nodes terminate.
    let found_flag: Option<String> = args.iter()
        .position(|a| a == "--found-flag")
        .and_then(|i| args.get(i + 1))
        .cloned();

    // Parse --ab-limit N or --ab-limit unlimited
    let backtrack_limit: u64 = args.iter()
        .position(|a| a == "--ab-limit")
        .and_then(|i| args.get(i + 1))
        .map(|s| {
            if s == "unlimited" || s == "0" {
                u64::MAX
            } else {
                s.parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("Error: --ab-limit must be a number or 'unlimited'");
                    std::process::exit(1);
                })
            }
        })
        .unwrap_or(u64::MAX);

    if debug_pipeline {
        run_debug_pipeline(n);
        return;
    }

    if pipeline_stats {
        run_pipeline_stats(n);
        return;
    }

    // V7: CPU pinning for cache locality
    // Pin each Rayon worker thread to a specific CPU core to prevent migration
    // and preserve L1/L2 cache contents. Gives ~10-20% improvement.
    let no_pin = args.iter().any(|a| a == "--no-pin");
    if !no_pin {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        if !core_ids.is_empty() {
            let num_cores = core_ids.len();
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_cores)
                .start_handler(move |thread_idx| {
                    let core_id = core_ids[thread_idx % core_ids.len()];
                    core_affinity::set_for_current(core_id);
                })
                .build_global()
                .unwrap_or_else(|_| {
                    eprintln!("Warning: Failed to set up pinned thread pool, using default");
                });
            log_println!("V7: CPU pinning enabled ({} cores)", num_cores);
        }
    }

    let instance_label = instance_spec.map(|(i, t)| format!("[Instance {}/{}] ", i, t)).unwrap_or_default();
    log_println!("{}BS({},{}) - V8 Per-Node CD-Tree Sharding", instance_label, n + 1, n);
    log_println!("==============================================\n");

    let num_threads = rayon::current_num_threads();
    log_println!("Threads: {} (rayon, pinned)", num_threads);

    let limit_str = if backtrack_limit == u64::MAX {
        "unlimited".to_string()
    } else {
        format!("{}M nodes", backtrack_limit / 1_000_000)
    };
    log_println!("Configuration for n={}:", n);
    log_println!("  Spectral margin: 1e-6 ({} angles)", NUM_SPECTRAL_ANGLES);
    log_println!("  AB backtrack limit: {}", limit_str);
    log_println!("  Two-level parallelism: tuples x mod-3 solutions (nested rayon)");
    #[cfg(feature = "gpu")]
    {
    }
    #[cfg(not(feature = "gpu"))]
    {
    }
    if let Some((inst, total)) = instance_spec {
        log_println!("  Instance: {}/{}", inst, total);
    }
    if let Some((start, end)) = tuple_range {
        log_println!("  Tuple range: {}-{}", start, end);
    }
    if let Some(ref dir) = claim_dir {
        if reset_claims {
            let _ = std::fs::remove_dir_all(dir);
        }
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("Error: failed to create --claim-dir {}: {}", dir, e);
            std::process::exit(1);
        }
        log_println!("  Claim dir: {} (CD-tree shard, depth {})", dir, shard_depth);
        if reset_claims {
            log_println!("  Reset claims: yes (existing claim files wiped at startup)");
        }
    }
    if let Some(ref f) = found_flag {
        log_println!("  Found-flag: {} (polled every 5s; written on solution)", f);
        // If sentinel already exists from a previous run, treat it as a hit and exit early.
        if std::path::Path::new(f).exists() {
            log_println!("  WARNING: found-flag already exists — exiting (delete it before rerun)");
            return;
        }
    }
    log_println!("");

    log_println!("Step 1: Find valid tuples...");
    let all_tuples = find_valid_sum_tuples_fast_v2(n);
    log_println!("  {} raw tuples found", all_tuples.len());

    log_println!("Step 2: Filter and sort by difficulty...");
    let canonical = filter_to_canonical_5class(all_tuples, n);
    let mut sorted: Vec<(SumTuple, AltSumTuple)> = canonical.into_iter().collect();
    sorted.sort_by_key(|(st, at)| score_tuple(st, at, n));
    log_println!("  {} canonical tuples", sorted.len());

    // Calculate total CD pairs
    log_println!("\nStep 3: Calculate search space...");
    let mut tuple_cd_counts: Vec<u64> = Vec::with_capacity(sorted.len());
    let mut total_cd_pairs: u64 = 0;

    for (st, at) in &sorted {
        let count = count_cd_pairs(n, st.c, st.d, at.c_star, at.d_star);
        tuple_cd_counts.push(count);
        total_cd_pairs = total_cd_pairs.saturating_add(count);
    }

    log_println!("  Total CD pairs: {:.2e}", total_cd_pairs as f64);
    log_println!("");

    // Convert --instance to tuple range now that we know the tuple count
    let num_tuples = sorted.len();
    let effective_range: Option<(usize, usize)> = if let Some((inst, total)) = instance_spec {
        let chunk = (num_tuples + total - 1) / total; // ceil division
        let start = (inst - 1) * chunk;
        let end = (inst * chunk).min(num_tuples);
        println!("Instance commands ({} tuples, {} instances):", num_tuples, total);
        for i in 1..=total {
            let s = (i - 1) * chunk;
            let e = (i * chunk).min(num_tuples);
            let marker = if i == inst { " <-- this instance" } else { "" };
            println!("  Instance {}/{}: ./find_bs_v7_gpu_parallel {} --instance {}/{}   (tuples {}-{}){}",
                i, total, n, i, total, s, e - 1, marker);
        }
        println!();
        Some((start, end))
    } else if let Some((s, e)) = tuple_range {
        Some((s, e.min(num_tuples)))
    } else {
        None
    };

    let prior_elapsed = 0.0_f64;
    let start = Instant::now();

    let found = Arc::new(AtomicBool::new(false));
    let tuples_done = Arc::new(AtomicUsize::new(0));
    let tuples_active = Arc::new(AtomicUsize::new(0));
    let cd_tried = Arc::new(AtomicU64::new(0));
    let cd_total = Arc::new(AtomicU64::new(0));

    let ab_timeouts = Arc::new(AtomicU64::new(0));
    let sorted_arc = Arc::new(sorted);

    // Mod-3/mod-6 counters (needed by both progress thread and search)
    let total_mod3_found = Arc::new(AtomicU64::new(0));
    let total_mod6_found = Arc::new(AtomicU64::new(0));
    // Signal for progress thread termination
    let search_done = Arc::new(AtomicBool::new(false));

    // Progress monitor thread
    let found_clone = Arc::clone(&found);
    let search_done_clone = Arc::clone(&search_done);
    let tuples_clone = Arc::clone(&tuples_done);
    let active_clone = Arc::clone(&tuples_active);
    let cd_clone = Arc::clone(&cd_tried);
    let cd_total_clone = Arc::clone(&cd_total);
    let ab_timeouts_clone = Arc::clone(&ab_timeouts);
    let total_tuples = effective_range.map(|(s, e)| e - s).unwrap_or(sorted_arc.len());
    let start_clone = start.clone();

    std::thread::spawn(move || {
        let mut last_tried = cd_clone.load(Ordering::Relaxed);
        let mut last_time = Instant::now();
        let mut last_log_time = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(30));
            if found_clone.load(Ordering::Relaxed) || search_done_clone.load(Ordering::Relaxed) { break; }

            let done = tuples_clone.load(Ordering::Relaxed);
            let active = active_clone.load(Ordering::Relaxed);
            let tried = cd_clone.load(Ordering::Relaxed);
            let total = cd_total_clone.load(Ordering::Relaxed);
            let elapsed = start_clone.elapsed().as_secs_f64() + prior_elapsed;

            let pass_rate = if total > 0 {
                tried as f64 / total as f64 * 100.0
            } else { 0.0 };

            let dt = last_time.elapsed().as_secs_f64();
            let cd_per_sec = if dt > 0.0 {
                (tried - last_tried) as f64 / dt
            } else { 0.0 };
            last_tried = tried;
            last_time = Instant::now();

            let timeouts = ab_timeouts_clone.load(Ordering::Relaxed);
            let timeout_rate = if tried > 0 {
                timeouts as f64 / tried as f64 * 100.0
            } else { 0.0 };

            let progress_line = format!(
                "  [{:>4}/{:>4} +{}] | {:.2e} pass / {:.2e} checked ({:.1}%) | {:.1}/s | {:.1}% tmout | {:.1}h",
                done, total_tuples, active,
                tried as f64,
                total as f64,
                pass_rate,
                cd_per_sec,
                timeout_rate,
                elapsed / 3600.0);
            println!("{}", progress_line);

            // Append a progress line to the log file once per hour.
            if last_log_time.elapsed().as_secs() >= 3600 {
                log_write(&progress_line);
                last_log_time = Instant::now();
            }
        }
    });

    // ========================================================================
    // Full paper algorithm (Wang & Zhu 2025, Steps 2-5)
    // Step 1: Tuple discovery (done above)
    // Step 2: Mod-3 partial sum enumeration (Theorem 2.3, m=3)
    // Step 3: Mod-6 CD refinement (Theorem 2.3, m=6)
    // Step 4: CD generation from mod-6 + spectral filter (Theorems 2.2 + 2.4)
    // Step 5: AB search via backtracking (Theorem 2.2)
    // ========================================================================

    log_println!("Step 4: Paper pipeline (Steps 2-5: mod-3 -> mod-6 -> CD+spectral -> AB)\n");

    // Wall-clock timeout: flips `found` to break the par_iter. The
    // `timed_out` flag disambiguates this from a real solution.
    let timed_out = Arc::new(AtomicBool::new(false));
    if timeout_secs > 0 {
        let found_to = Arc::clone(&found);
        let timed_out_to = Arc::clone(&timed_out);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(timeout_secs));
            if !found_to.load(Ordering::Relaxed) {
                timed_out_to.store(true, Ordering::Relaxed);
                found_to.store(true, Ordering::Relaxed);
            }
        });
    }

    // Cross-node early termination: poll the --found-flag sentinel file every
    // 5s. If it appears (written by a peer node that found a solution), flip
    // `found` so this process exits its par_iter and reports gracefully.
    let peer_found = Arc::new(AtomicBool::new(false));
    if let Some(ref flag_path) = found_flag {
        let path = flag_path.clone();
        let found_pf = Arc::clone(&found);
        let peer_found_pf = Arc::clone(&peer_found);
        let search_done_pf = Arc::clone(&search_done);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                if found_pf.load(Ordering::Relaxed) || search_done_pf.load(Ordering::Relaxed) { break; }
                if std::path::Path::new(&path).exists() {
                    peer_found_pf.store(true, Ordering::Relaxed);
                    found_pf.store(true, Ordering::Relaxed);
                    break;
                }
            }
        });
    }

    let range_start = effective_range.map(|(s, _)| s).unwrap_or(0);
    let range_end = effective_range.map(|(_, e)| e).unwrap_or(sorted_arc.len());

    let result: Option<(BaseSequence, usize, SumTuple, AltSumTuple)> = (range_start..range_end)
        .into_par_iter()
        .find_map_first(|tuple_idx| {
            if found.load(Ordering::Relaxed) { return None; }

            tuples_active.fetch_add(1, Ordering::Relaxed);
            let (st, at) = &sorted_arc[tuple_idx];

            // Pre-compute constraints + max_contrib once per tuple (NOT per CD pair)
            let ab_constraints = precompute_symmetric_constraints_ab(n);
            let m = n + 1;
            let num_pairs_pre = ab_constraints.len();
            let ab_max_contrib: Vec<Vec<i32>> = (0..num_pairs_pre).map(|pidx| {
                let ready_shift = n - pidx;
                let uf_lo = pidx + 1;
                (0..=n).map(|s| {
                    if s == 0 || s >= ready_shift || pidx + 2 > m { return 0; }
                    let uf_hi = m - 2 - pidx;
                    if uf_hi < uf_lo { return 0; }
                    let uf_len = uf_hi - uf_lo + 1;
                    let both_unfilled = if s < uf_len { (uf_len - s) as i32 } else { 0 };
                    let left_lo = std::cmp::max(s, uf_lo);
                    let left_hi = std::cmp::min(s + uf_lo - 1, uf_hi);
                    let left_count = if left_hi >= left_lo { (left_hi - left_lo + 1) as i32 } else { 0 };
                    let right_lo = std::cmp::max(if uf_hi + 1 >= s { uf_hi + 1 - s } else { 0 }, uf_lo);
                    let right_hi = std::cmp::min(if m > s { m - 1 - s } else { 0 }, uf_hi);
                    let right_count = if m > s && right_hi >= right_lo { (right_hi - right_lo + 1) as i32 } else { 0 };
                    2 * (both_unfilled + left_count + right_count)
                }).collect()
            }).collect();

            // Step 2: Collect mod-3 solutions (fast, ~48 bytes each)
            let mod3_sols = collect_mod3_solutions(n, st, at, usize::MAX);
            total_mod3_found.fetch_add(mod3_sols.len() as u64, Ordering::Relaxed);

            // INNER PARALLELISM: collect (mod3 × mod6) pairs and par_iter the combined set.
            // Adding the mod-6 level helps when a tuple has a small number of mod-3 sols
            // (which otherwise would underfeed the thread pool while one of them runs long).
            let spectral_margin = 1e-6;
            let mut m3_m6_pairs: Vec<(Mod3Solution, Mod6CDSolution)> = Vec::new();
            for (m3_idx, mod3_sol) in mod3_sols.iter().enumerate() {
                enumerate_mod6_cd_solutions(n, mod3_sol, m3_idx, st, at, &mut |mod6_sol| {
                    m3_m6_pairs.push((mod3_sol.clone(), mod6_sol.clone()));
                    true
                });
            }
            total_mod6_found.fetch_add(m3_m6_pairs.len() as u64, Ordering::Relaxed);

            let claim_dir_ref = claim_dir.as_deref();
            let result = m3_m6_pairs.into_par_iter().enumerate().find_map_first(|(pair_idx, (_mod3_sol, mod6_sol))| {
                if found.load(Ordering::Relaxed) { return None; }
                let mut local_result: Option<(BaseSequence, usize, SumTuple, AltSumTuple)> = None;

                let ab_mod6_sols = enumerate_mod6_ab_solutions(n, st, at, &mod6_sol.p, &mod6_sol.q);

                // v8: ownership is decided inside backtrack_cd_from_mod6 via
                // the ShardCtx (claim files at the CD-tree prefix level). By
                // the time the callback below fires, this node has already
                // won the prefix — no per-CD ownership check needed.
                let shard = ShardCtx {
                    tuple_idx,
                    m3m6_pair_idx: pair_idx,
                    claim_dir: claim_dir_ref,
                    shard_depth,
                };

                backtrack_cd_from_mod6(n, &mod6_sol, spectral_margin, &mut |c, d| {
                    if found.load(Ordering::Relaxed) { return false; }

                    cd_tried.fetch_add(1, Ordering::Relaxed);

                    let (ab_result, nodes) = backtrack_search_ab(n, c, d, st, at, backtrack_limit, &ab_constraints, &ab_max_contrib, &ab_mod6_sols);
                    if nodes >= backtrack_limit && backtrack_limit != u64::MAX {
                        ab_timeouts.fetch_add(1, Ordering::Relaxed);
                    }
                    if let Some((a, b)) = ab_result {
                        let base = BaseSequence::new(a, b, Sequence::new(c.to_vec()), Sequence::new(d.to_vec()));
                        if base.is_valid() {
                            found.store(true, Ordering::Relaxed);
                            local_result = Some((base, tuple_idx, st.clone(), at.clone()));
                            return false;
                        }
                    }
                    true // continue
                }, &cd_total, shard);

                local_result
            });

            tuples_active.fetch_sub(1, Ordering::Relaxed);

            if result.is_some() {
                return result;
            }

            tuples_done.fetch_add(1, Ordering::Relaxed);
            None
        });

    // Stop progress thread
    search_done.store(true, Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Cleanup GPU spectral filter
    #[cfg(feature = "gpu")]
    if let Some(ref filter) = gpu_spectral {
        filter.lock().unwrap().cleanup();
    }

    let elapsed_secs = start.elapsed().as_secs_f64() + prior_elapsed;
    let total_mod3 = total_mod3_found.load(Ordering::Relaxed);
    let total_mod6 = total_mod6_found.load(Ordering::Relaxed);

    println!("\n");

    if let Some((base, idx, st, at)) = result {
        // Write the cross-node sentinel so peer nodes terminate. Best-effort:
        // a write failure here doesn't change correctness, just delays peers.
        if let Some(ref flag_path) = found_flag {
            if let Err(e) = std::fs::write(flag_path, format!("n={} tuple={} elapsed={:.0}s\n", n, idx, elapsed_secs)) {
                eprintln!("Warning: failed to write --found-flag {}: {}", flag_path, e);
            }
        }
        print_solution(n, &base, &st, &at, idx, elapsed_secs,
            &tuples_done, &cd_tried, instance_spec);
    } else if peer_found.load(Ordering::Relaxed) {
        println!("============================================");
        println!("     PEER NODE FOUND SOLUTION - exiting     ");
        println!("============================================\n");
        println!("Elapsed: {:.2} hours", elapsed_secs / 3600.0);
        let tried = cd_tried.load(Ordering::Relaxed);
        let total_checked = cd_total.load(Ordering::Relaxed);
        println!("CD pairs checked: {:.2e}", total_checked as f64);
        println!("CD pairs searched (passed spectral): {:.2e}", tried as f64);
        if let Some(ref flag_path) = found_flag {
            println!("Found-flag: {}", flag_path);
        }
    } else if timed_out.load(Ordering::Relaxed) {
        // Timeout: record what we saw, including tuple identity if a single
        // tuple was targeted (--tuple K or --top 1).
        println!("============================================");
        println!("     TIMEOUT - no solution within {} s      ", timeout_secs);
        println!("============================================\n");

        println!("Elapsed: {:.2} hours", elapsed_secs / 3600.0);
        let tried = cd_tried.load(Ordering::Relaxed);
        let total_checked = cd_total.load(Ordering::Relaxed);
        println!("CD pairs checked: {:.2e}", total_checked as f64);
        println!("CD pairs searched (passed spectral): {:.2e}", tried as f64);

        // Only emit the per-tuple timeout file when we ran a single tuple.
        if range_end == range_start + 1 {
            let idx = range_start;
            let (st, at) = &sorted_arc[idx];
            let inst_str = instance_spec.map(|(i, t)| format!("_inst{}of{}", i, t)).unwrap_or_default();
            let filename = format!("BS_{}_{}_V7Parallel{}_tuple{}_{:.0}s.txt",
                n + 1, n, inst_str, idx, elapsed_secs);
            if let Ok(mut f) = File::create(&filename) {
                writeln!(f, "BS({},{}) TIMEOUT - V7 Optimized Parallel", n + 1, n).ok();
                writeln!(f, "====================================").ok();
                writeln!(f, "Time: {:.1}s ({:.2}h) -- TIMED OUT", elapsed_secs, elapsed_secs / 3600.0).ok();
                writeln!(f, "Timeout: {} s", timeout_secs).ok();
                writeln!(f, "CD pairs checked: {:.2e}", total_checked as f64).ok();
                writeln!(f, "CD pairs searched (passed spectral): {:.2e}", tried as f64).ok();
                writeln!(f, "").ok();
                writeln!(f, "Tuple #{}", idx).ok();
                writeln!(f, "Sum tuple:     ({:>3},{:>3},{:>3},{:>3})", st.a, st.b, st.c, st.d).ok();
                writeln!(f, "Alt-sum tuple: ({:>3},{:>3},{:>3},{:>3})", at.a_star, at.b_star, at.c_star, at.d_star).ok();
                writeln!(f, "").ok();
                writeln!(f, "Solution NOT found.").ok();
                println!("\nSaved to: {}", filename);
            }
        }
    } else if !found.load(Ordering::Relaxed) {
        println!("============================================");
        println!("     Search complete - no solution          ");
        println!("============================================\n");

        println!("Time: {:.2} hours", elapsed_secs / 3600.0);
        if let Some((inst, total)) = instance_spec {
            println!("Instance: {}/{}", inst, total);
        }
        println!("Tuples processed: {}/{}", tuples_done.load(Ordering::Relaxed), total_tuples);
        println!("Mod-3 solutions found: {}", total_mod3);
        println!("Mod-6 CD solutions found: {}", total_mod6);
        let tried = cd_tried.load(Ordering::Relaxed);
        let total_checked = cd_total.load(Ordering::Relaxed);
        let timeouts = ab_timeouts.load(Ordering::Relaxed);
        let pass_rate = if total_checked > 0 { tried as f64 / total_checked as f64 * 100.0 } else { 0.0 };
        let timeout_rate = if tried > 0 { timeouts as f64 / tried as f64 * 100.0 } else { 0.0 };
        println!("CD pairs checked: {:.2e} ({:.1}% passed spectral)", total_checked as f64, pass_rate);
        println!("CD pairs searched (passed spectral): {:.2e}", tried as f64);
        println!("AB timeouts: {} ({:.1}% of searched)", timeouts, timeout_rate);
        if backtrack_limit == u64::MAX {
            println!("\nNote: Unlimited AB backtracking. If no solution found,");
            println!("all CD pairs have been exhaustively searched.");
        } else {
            println!("\nNote: AB backtrack limit was {}M nodes. {} CD pairs timed out.",
                backtrack_limit / 1_000_000, timeouts);
            println!("Re-run with --ab-limit unlimited to exhaustively search timed-out pairs.");
        }
    }
}

fn print_solution(
    n: usize,
    base: &BaseSequence,
    st: &SumTuple,
    at: &AltSumTuple,
    idx: usize,
    elapsed_secs: f64,
    tuples_done: &AtomicUsize,
    cd_tried: &AtomicU64,
    instance_spec: Option<(usize, usize)>,
) {
    println!("============================================");
    println!("       SUCCESS! BS({},{}) FOUND          ", n + 1, n);
    println!("============================================\n");

    println!("Time: {:.2} hours", elapsed_secs / 3600.0);
    if let Some((inst, total)) = instance_spec {
        println!("Found by instance {}/{}", inst, total);
    }
    println!("Tuples checked: {}", tuples_done.load(Ordering::Relaxed));
    println!("CD pairs tried: {:.2e}", cd_tried.load(Ordering::Relaxed) as f64);
    println!("");

    println!("Solution at tuple #{}", idx);
    println!("Sum tuple:     ({:>3},{:>3},{:>3},{:>3})", st.a, st.b, st.c, st.d);
    println!("Alt-sum tuple: ({:>3},{:>3},{:>3},{:>3})", at.a_star, at.b_star, at.c_star, at.d_star);
    println!("");

    println!("A = {:?}", base.a.values);
    println!("B = {:?}", base.b.values);
    println!("C = {:?}", base.c.values);
    println!("D = {:?}", base.d.values);

    // Verify
    println!("\nVerification:");
    let mut valid = true;
    for shift in 1..=n {
        let ac = base.a.autocorrelation(shift) + base.b.autocorrelation(shift)
               + base.c.autocorrelation(shift) + base.d.autocorrelation(shift);
        if ac != 0 {
            println!("  WARNING: AC({}) = {} (should be 0)", shift, ac);
            valid = false;
        }
    }
    if valid {
        println!("  All {} autocorrelation checks passed!", n);
    }

    // Save
    let inst_str = instance_spec.map(|(i, t)| format!("_inst{}of{}", i, t)).unwrap_or_default();
    let gpu_str = "";
    let tuple_str = format!("_tuple{}", idx);
    let filename = format!("BS_{}_{}_V7Parallel{}{}{}_{:.0}s.txt", n + 1, n, gpu_str, inst_str, tuple_str, elapsed_secs);
    if let Ok(mut f) = File::create(&filename) {
        writeln!(f, "BS({},{}) Solution - V7 Optimized Parallel", n + 1, n).ok();
        writeln!(f, "====================================").ok();
        writeln!(f, "Time: {:.1}s ({:.2}h)", elapsed_secs, elapsed_secs / 3600.0).ok();
        if let Some((inst, total)) = instance_spec {
            writeln!(f, "Instance: {}/{}", inst, total).ok();
        }
        writeln!(f, "CD pairs tried: {:.2e}", cd_tried.load(Ordering::Relaxed) as f64).ok();
        writeln!(f, "").ok();
        writeln!(f, "Solution at tuple #{}", idx).ok();
        writeln!(f, "Sum tuple:     ({:>3},{:>3},{:>3},{:>3})", st.a, st.b, st.c, st.d).ok();
        writeln!(f, "Alt-sum tuple: ({:>3},{:>3},{:>3},{:>3})", at.a_star, at.b_star, at.c_star, at.d_star).ok();
        writeln!(f, "").ok();
        writeln!(f, "A = {:?}", base.a.values).ok();
        writeln!(f, "B = {:?}", base.b.values).ok();
        writeln!(f, "C = {:?}", base.c.values).ok();
        writeln!(f, "D = {:?}", base.d.values).ok();
        println!("\nSaved to: {}", filename);
    }
}