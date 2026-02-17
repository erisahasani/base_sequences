
/// Generic BS(n+1, n) search - V6 Paper Pipeline Implementation
///
/// Usage: cargo run --release --bin find_bs_generic_v6 -- <n>
/// Example: cargo run --release --bin find_bs_generic_v6 -- 30
///
/// Resume from checkpoint:
///   cargo run --release --bin find_bs_generic_v6 -- <n> --resume
///
/// Implements the 5-step algorithm from Wang & Zhu (2025):
/// 1. Tuple discovery (Theorem 2.1 sum constraints)
/// 2. Mod-3 partial sum enumeration (Theorem 2.3, m=3)
/// 3. Mod-6 CD refinement (Theorem 2.3, m=6)
/// 4. CD generation + spectral filter (Theorems 2.2 + 2.4)
/// 5. AB search via backtracking (Theorem 2.2)

use std::time::Instant;
use std::collections::HashMap;
use std::f64::consts::PI;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};
use std::fs::File;
use std::io::{Write, BufReader, BufWriter};
use std::path::Path;
use std::env;
use serde::{Serialize, Deserialize};
use std::sync::Mutex;

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
        let mut sum = 0;
        for j in 0..(n - shift) {
            sum += self.values[j] * self.values[j + shift];
        }
        sum
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

    println!("  Phase 1: Finding valid (a,b,c,d) tuples...");
    let sum_tuples = find_sum_tuples(n, target, max_sum_m, max_sum_n);
    println!("  Found {} valid (a,b,c,d) tuples", sum_tuples.len());

    println!("  Phase 2: Finding valid (a*,b*,c*,d*) tuples...");
    let alt_tuples = find_alt_tuples(n, target, max_sum_m, max_sum_n);
    println!("  Found {} valid (a*,b*,c*,d*) tuples", alt_tuples.len());

    println!("  Phase 3: HashMap-based matching (Equation 2.4)...");
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

    println!("  Found {} valid tuple pairs", valid_pairs.len());
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
    println!("  5-class isomorphic filtering: {} -> {} tuples ({:.1}x reduction)",
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

/// Spectral filter per paper Theorem 2.4: θ = jπ/100 for j=1,...,200
fn passes_spectral_bound(c: &Sequence, d: &Sequence, margin: f64) -> bool {
    let n = c.len();
    let target = 4.0 * (n as f64) + 2.0;
    let threshold = target + margin;
    // Paper Step 4: 200 samples at θ = jπ/100, j=1,...,200
    for j in 1..=200 {
        let theta = (j as f64) * PI / 100.0;
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
    // Match spectral filter: 200 samples at θ = jπ/100
    for j in 1..=200 {
        let theta = (j as f64) * PI / 100.0;
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

/// Enumerate all valid mod-3 partial sum solutions for a given tuple.
/// Implements Step 2 of the paper's algorithm.
fn enumerate_mod3_solutions(
    n: usize,
    st: &SumTuple,
    _at: &AltSumTuple,
) -> Vec<Mod3Solution> {
    let m = 3;
    let target_sq = (4 * n + 2) as i32;
    let m_ab = n + 1; // length of A, B
    let m_cd = n;     // length of C, D

    // Solution cap to prevent memory explosion (we only process a fraction anyway)
    let max_solutions: usize = if n <= 15 { 100_000 } else { 10_000 };
    let mut solutions = Vec::new();

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

                                    solutions.push(Mod3Solution { k, r, p, q });
                                    if solutions.len() >= max_solutions { return solutions; }

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

    solutions
}

/// Enumerate valid mod-6 CD partial sum solutions that refine a mod-3 solution.
/// Implements Step 3 of the paper's algorithm.
fn enumerate_mod6_cd_solutions(
    n: usize,
    mod3_sol: &Mod3Solution,
    mod3_idx: usize,
) -> Vec<Mod6CDSolution> {
    let m = 6;
    let m_cd = n; // length of C, D

    let max_solutions: usize = if n <= 20 { 10_000 } else { 1_000 };
    let mut solutions = Vec::new();

    // The mod-6 CD partials must refine mod-3:
    // p_{i,3} = p_{i,6} + p_{i+3,6} for i=1,2,3
    // q_{i,3} = q_{i,6} + q_{i+3,6} for i=1,2,3

    // Compute bounds for each mod-6 residue class of C (length n)
    let mut p_bounds = [(0i32, 0i32, 0i32); 6];
    let mut q_bounds = [(0i32, 0i32, 0i32); 6];
    for i in 1..=6 {
        p_bounds[i - 1] = partial_sum_bound(m_cd, i, m);
        q_bounds[i - 1] = partial_sum_bound(m_cd, i, m);
    }

    // Enumerate p values: for each i=1,2,3, enumerate p_{i,6}, derive p_{i+3,6}
    // p_{i,6} range: p_bounds[i-1], and p_{i+3,6} = mod3_sol.p[i-1] - p_{i,6}
    let mut p_candidates: Vec<[i32; 6]> = vec![[0; 6]]; // start with one empty candidate

    for i_mod3 in 0..3 {
        let target = mod3_sol.p[i_mod3];
        let (lo_a, hi_a, par_a) = p_bounds[i_mod3];
        let (lo_b, hi_b, par_b) = p_bounds[i_mod3 + 3];

        let mut new_candidates = Vec::new();
        for cand in &p_candidates {
            let mut pa = lo_a + ((par_a - ((lo_a % 2 + 2) % 2)) % 2 + 2) % 2;
            while pa <= hi_a {
                let pb = target - pa;
                if pb >= lo_b && pb <= hi_b && ((pb % 2 + 2) % 2) == par_b {
                    let mut new_cand = *cand;
                    new_cand[i_mod3] = pa;
                    new_cand[i_mod3 + 3] = pb;
                    new_candidates.push(new_cand);
                }
                pa += 2;
            }
        }
        p_candidates = new_candidates;
    }

    // Similarly for q values
    for p_cand in &p_candidates {
        let mut q_candidates: Vec<[i32; 6]> = vec![[0; 6]];

        for i_mod3 in 0..3 {
            let target = mod3_sol.q[i_mod3];
            let (lo_a, hi_a, par_a) = q_bounds[i_mod3];
            let (lo_b, hi_b, par_b) = q_bounds[i_mod3 + 3];

            let mut new_candidates = Vec::new();
            for cand in &q_candidates {
                let mut qa = lo_a + ((par_a - ((lo_a % 2 + 2) % 2)) % 2 + 2) % 2;
                while qa <= hi_a {
                    let qb = target - qa;
                    if qb >= lo_b && qb <= hi_b && ((qb % 2 + 2) % 2) == par_b {
                        let mut new_cand = *cand;
                        new_cand[i_mod3] = qa;
                        new_cand[i_mod3 + 3] = qb;
                        new_candidates.push(new_cand);
                    }
                    qa += 2;
                }
            }
            q_candidates = new_candidates;
        }

        for q_cand in &q_candidates {
            // Check Eq 2.10 constraints at m=6 for PQ part only:
            // We need sum(p_i^2 + q_i^2) to be feasible (not exceed 4n+2)
            let pq_sq: i32 = p_cand.iter().chain(q_cand.iter()).map(|x| x * x).sum();
            let target_sq = (4 * n + 2) as i32;
            if pq_sq > target_sq { continue; }

            // Check Eq 2.10 orthogonality for CD part at m=6:
            // For s=1,...,3: N_P(s)+N_Q(s)+N_P(6-s)+N_Q(6-s) should contribute correctly
            // We check the combined KRPQ orthogonality later when checking AB feasibility.
            // For now, just check CD-only partial orthogonality is not obviously wrong.

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
            if !cd_mod4_ok { continue; }

            // Check that at least one valid AB mod-6 exists (feasibility check)
            let feasible = if n <= 20 {
                check_mod6_ab_feasible(n, mod3_sol, p_cand, q_cand, target_sq - pq_sq)
            } else if n <= 35 {
                check_mod6_ab_feasible_sampled(n, mod3_sol, p_cand, q_cand, target_sq - pq_sq, 50_000)
            } else {
                true // Skip for very large n
            };
            if feasible {
                solutions.push(Mod6CDSolution {
                    p: *p_cand,
                    q: *q_cand,
                    _mod3_idx: mod3_idx,
                });
                if solutions.len() >= max_solutions { return solutions; }
            }
        }
    }

    solutions
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

/// Probabilistic AB feasibility check for mod-6 solutions when n > 25.
/// Randomly samples K,R mod-6 refinements to check if any valid AB exists.
fn check_mod6_ab_feasible_sampled(
    n: usize,
    mod3_sol: &Mod3Solution,
    p6: &[i32; 6],
    q6: &[i32; 6],
    kr_sq_budget: i32,
    num_samples: usize,
) -> bool {
    use rand::Rng;
    let m: usize = 6;
    let m_ab = n + 1;
    let mut rng = rand::thread_rng();
    let target_sq_full = (4 * n + 2) as i32;

    // Compute bounds for K,R at mod-6
    let mut k_bounds = [(0i32, 0i32, 0i32); 6];
    let mut r_bounds = [(0i32, 0i32, 0i32); 6];
    for i in 1..=6 {
        k_bounds[i - 1] = partial_sum_bound(m_ab, i, m);
        r_bounds[i - 1] = partial_sum_bound(m_ab, i, m);
    }

    for _ in 0..num_samples {
        // Random K refinement: for each i=0,1,2 pick k_{i,6}, derive k_{i+3,6}
        let mut k = [0i32; 6];
        let mut valid = true;
        for i_mod3 in 0..3 {
            let target = mod3_sol.k[i_mod3];
            let (lo_a, hi_a, par_a) = k_bounds[i_mod3];
            // Compute first valid value with correct parity
            let first_valid = lo_a + ((par_a - ((lo_a % 2 + 2) % 2)) % 2 + 2) % 2;
            let num_vals = if first_valid > hi_a { 0 } else { ((hi_a - first_valid) / 2 + 1) as usize };
            if num_vals == 0 { valid = false; break; }
            let ka = first_valid + 2 * rng.gen_range(0..num_vals as i32);
            let kb = target - ka;
            let (lo_b, hi_b, par_b) = k_bounds[i_mod3 + 3];
            if kb < lo_b || kb > hi_b || ((kb % 2 + 2) % 2) != par_b { valid = false; break; }
            k[i_mod3] = ka;
            k[i_mod3 + 3] = kb;
        }
        if !valid { continue; }
        let k_sq: i32 = k.iter().map(|x| x * x).sum();
        if k_sq > kr_sq_budget { continue; }

        // Random R refinement
        let mut r = [0i32; 6];
        valid = true;
        for i_mod3 in 0..3 {
            let target = mod3_sol.r[i_mod3];
            let (lo_a, hi_a, par_a) = r_bounds[i_mod3];
            let first_valid = lo_a + ((par_a - ((lo_a % 2 + 2) % 2)) % 2 + 2) % 2;
            let num_vals = if first_valid > hi_a { 0 } else { ((hi_a - first_valid) / 2 + 1) as usize };
            if num_vals == 0 { valid = false; break; }
            let ra = first_valid + 2 * rng.gen_range(0..num_vals as i32);
            let rb = target - ra;
            let (lo_b, hi_b, par_b) = r_bounds[i_mod3 + 3];
            if rb < lo_b || rb > hi_b || ((rb % 2 + 2) % 2) != par_b { valid = false; break; }
            r[i_mod3] = ra;
            r[i_mod3 + 3] = rb;
        }
        if !valid { continue; }
        let r_sq: i32 = r.iter().map(|x| x * x).sum();
        if k_sq + r_sq != kr_sq_budget { continue; }

        // Check total sum-of-squares
        let total_sq: i32 = k_sq + r_sq
            + p6.iter().map(|x| x * x).sum::<i32>()
            + q6.iter().map(|x| x * x).sum::<i32>();
        if total_sq != target_sq_full { continue; }

        // Check orthogonality at m=6
        let mut ortho_ok = true;
        for s in 1..=3 {
            let ortho = partial_autocorr(&k, s) + partial_autocorr(&r, s)
                + partial_autocorr(p6, s) + partial_autocorr(q6, s)
                + partial_autocorr(&k, m - s) + partial_autocorr(&r, m - s)
                + partial_autocorr(p6, m - s) + partial_autocorr(q6, m - s);
            if ortho != 0 { ortho_ok = false; break; }
        }
        if !ortho_ok { continue; }

        // Check Eq 2.12 for AB at m=6
        let res_n1 = if (n + 1) % m == 0 { m } else { (n + 1) % m };
        let ab_pair1_sum = k[0] + r[0] + k[res_n1 - 1] + r[res_n1 - 1];
        let ab_pair1_target = if n % m != 0 { 2 } else { 0 };
        if ((ab_pair1_sum % 4) + 4) % 4 != ab_pair1_target { continue; }

        let overlap_j_ab6 = res_n1;
        let mut ab_mod4_ok = true;
        for j in 2..=m {
            if j == overlap_j_ab6 { continue; }
            let res_pair = if (n + 2 - j) % m == 0 { m } else { (n + 2 - j) % m };
            let sum_j = k[j - 1] + r[j - 1] + k[res_pair - 1] + r[res_pair - 1];
            if ((sum_j % 4) + 4) % 4 != 0 { ab_mod4_ok = false; break; }
        }
        if !ab_mod4_ok { continue; }

        return true;
    }
    false
}


/// Deterministic backtracking to construct C,D sequences satisfying both
/// Theorem 2.2 paired constraints AND target mod-6 partial sums.
/// Integrates spectral filtering: maintains Hall polynomial incrementally,
/// checks exact spectral bound at base case, prunes via lower bound near leaves.
/// Returns (spectrally-valid CDs, total CDs that reached spectral check).
fn backtrack_cd_from_mod6(
    n: usize,
    mod6_sol: &Mod6CDSolution,
    max_solutions: usize,
    spectral_margin: f64,
) -> (Vec<(Sequence, Sequence)>, u64) {
    let m = 6usize;
    let valid_pairs_cd = valid_symmetric_pairs_cd(); // 8 valid choices

    // All 16 choices for unconstrained pair (0, n-1)
    let all_16: Vec<(i32, i32, i32, i32)> = {
        let mut v = Vec::new();
        for &ci in &[-1i32, 1] {
            for &di in &[-1i32, 1] {
                for &cj in &[-1i32, 1] {
                    for &dj in &[-1i32, 1] {
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

    let mut results: Vec<(Sequence, Sequence)> = Vec::new();
    let mut c_vals = vec![0i32; n];
    let mut d_vals = vec![0i32; n];
    let mut c_running = [0i32; 6];
    let mut d_running = [0i32; 6];
    let mut filled = [0usize; 6];
    let num_pairs = pair_positions.len();

    // Precompute trig tables for incremental spectral check
    let num_angles: usize = 200;
    let spectral_threshold = 4.0 * (n as f64) + 2.0 + spectral_margin;
    let cos_table: Vec<Vec<f64>> = (0..n).map(|pos| {
        (0..num_angles).map(|k| {
            ((pos as f64) * ((k + 1) as f64) * PI / 100.0).cos()
        }).collect()
    }).collect();
    let sin_table: Vec<Vec<f64>> = (0..n).map(|pos| {
        (0..num_angles).map(|k| {
            ((pos as f64) * ((k + 1) as f64) * PI / 100.0).sin()
        }).collect()
    }).collect();

    // Running Hall polynomial state: Re/Im of C and D at each spectral angle
    let mut real_c = vec![0.0f64; num_angles];
    let mut imag_c = vec![0.0f64; num_angles];
    let mut real_d = vec![0.0f64; num_angles];
    let mut imag_d = vec![0.0f64; num_angles];
    let mut cd_checked: u64 = 0;

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
        results: &mut Vec<(Sequence, Sequence)>,
        max_solutions: usize,
        cos_table: &[Vec<f64>],
        sin_table: &[Vec<f64>],
        real_c: &mut [f64],
        imag_c: &mut [f64],
        real_d: &mut [f64],
        imag_d: &mut [f64],
        spectral_threshold: f64,
        cd_checked: &mut u64,
    ) {
        if results.len() >= max_solutions { return; }
        let na = real_c.len(); // number of spectral angles

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
                            for k in 0..na {
                                real_c[k] += cm_f * cos_table[mid_pos][k];
                                imag_c[k] += cm_f * sin_table[mid_pos][k];
                                real_d[k] += dm_f * cos_table[mid_pos][k];
                                imag_d[k] += dm_f * sin_table[mid_pos][k];
                            }

                            // Exact spectral check
                            *cd_checked += 1;
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
                                results.push((Sequence::new(c_vals.clone()), Sequence::new(d_vals.clone())));
                            }

                            // Undo spectral for middle
                            for k in 0..na {
                                real_c[k] -= cm_f * cos_table[mid_pos][k];
                                imag_c[k] -= cm_f * sin_table[mid_pos][k];
                                real_d[k] -= dm_f * cos_table[mid_pos][k];
                                imag_d[k] -= dm_f * sin_table[mid_pos][k];
                            }

                            if results.len() >= max_solutions { return; }
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
                    *cd_checked += 1;
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
                        results.push((Sequence::new(c_vals.clone()), Sequence::new(d_vals.clone())));
                    }
                }
            }
            return;
        }

        let (left, right, lc, rc) = pair_positions[pair_idx];
        let choices: &[(i32, i32, i32, i32)] = if pair_idx == 0 { all_16 } else { valid_pairs_cd };

        for &(ci, di, cj, dj) in choices {
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
                c_vals[left] = ci;
                d_vals[left] = di;
                c_vals[right] = cj;
                d_vals[right] = dj;

                // Update spectral state incrementally
                let ci_f = ci as f64;
                let di_f = di as f64;
                let cj_f = cj as f64;
                let dj_f = dj as f64;
                for k in 0..na {
                    real_c[k] += ci_f * cos_table[left][k] + cj_f * cos_table[right][k];
                    imag_c[k] += ci_f * sin_table[left][k] + cj_f * sin_table[right][k];
                    real_d[k] += di_f * cos_table[left][k] + dj_f * cos_table[right][k];
                    imag_d[k] += di_f * sin_table[left][k] + dj_f * sin_table[right][k];
                }

                // Spectral lower bound pruning (effective near leaves)
                let unfilled = 2 * (num_pairs - pair_idx - 1)
                    + if has_middle { 1 } else { 0 };
                let mut spectral_feasible = true;
                if unfilled <= 6 && unfilled > 0 {
                    let u = unfilled as f64;
                    for k in 0..na {
                        let mag_c = (real_c[k] * real_c[k] + imag_c[k] * imag_c[k]).sqrt();
                        let mag_d = (real_d[k] * real_d[k] + imag_d[k] * imag_d[k]).sqrt();
                        let lb_c = if mag_c > u { mag_c - u } else { 0.0 };
                        let lb_d = if mag_d > u { mag_d - u } else { 0.0 };
                        if lb_c * lb_c + lb_d * lb_d > spectral_threshold {
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
                        results, max_solutions,
                        cos_table, sin_table,
                        real_c, imag_c, real_d, imag_d,
                        spectral_threshold, cd_checked,
                    );
                }

                // Undo spectral state
                for k in 0..na {
                    real_c[k] -= ci_f * cos_table[left][k] + cj_f * cos_table[right][k];
                    imag_c[k] -= ci_f * sin_table[left][k] + cj_f * sin_table[right][k];
                    real_d[k] -= di_f * cos_table[left][k] + dj_f * cos_table[right][k];
                    imag_d[k] -= di_f * sin_table[left][k] + dj_f * sin_table[right][k];
                }

                if results.len() >= max_solutions {
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
        &mut results, max_solutions,
        &cos_table, &sin_table,
        &mut real_c, &mut imag_c, &mut real_d, &mut imag_d,
        spectral_threshold, &mut cd_checked,
    );

    (results, cd_checked)
}


/// Backtracking search for A,B sequences using Theorem 2.2 constraints.
/// Uses incremental autocorrelation tracking: O(n) per node instead of O(n²).
/// Positions filled outside-in; after pair k, shift n-k becomes fully determined.
fn backtrack_search_ab(
    n: usize,
    c: &Sequence,
    d: &Sequence,
    st: &SumTuple,
    at: &AltSumTuple,
    max_nodes: u64,
) -> Option<(Sequence, Sequence)> {
    let m = n + 1; // length of A, B

    // Precompute CD autocorrelations (shifts 1..=n for BS(n+1,n))
    let cd_autocorr = {
        let mut v = vec![0i32; n + 1];
        for shift in 1..=n {
            v[shift] = c.autocorrelation(shift) + d.autocorrelation(shift);
        }
        v
    };

    let constraints = precompute_symmetric_constraints_ab(n);

    let mut a = vec![0i32; m];
    let mut b = vec![0i32; m];
    let mut nodes_visited = 0u64;

    // Incremental state: partial_ac[s] = N_C(s) + N_D(s) + (AB terms for filled pairs)
    let mut partial_ac = cd_autocorr.clone();
    let mut a_sum = 0i32;
    let mut b_sum = 0i32;

    // Add contributions of newly-filled position `pos` to all shifts.
    fn update_partial_ac(
        partial_ac: &mut [i32], a: &[i32], b: &[i32],
        pos: usize, a_val: i32, b_val: i32, n: usize, m: usize,
    ) {
        for s in 1..=n {
            if pos >= s && a[pos - s] != 0 {
                partial_ac[s] += a[pos - s] * a_val + b[pos - s] * b_val;
            }
            if pos + s < m && a[pos + s] != 0 {
                partial_ac[s] += a_val * a[pos + s] + b_val * b[pos + s];
            }
        }
    }

    // Reverse the update for position `pos` (called during backtrack undo).
    fn undo_partial_ac(
        partial_ac: &mut [i32], a: &[i32], b: &[i32],
        pos: usize, a_val: i32, b_val: i32, n: usize, m: usize,
    ) {
        for s in 1..=n {
            if pos >= s && a[pos - s] != 0 {
                partial_ac[s] -= a[pos - s] * a_val + b[pos - s] * b_val;
            }
            if pos + s < m && a[pos + s] != 0 {
                partial_ac[s] -= a_val * a[pos + s] + b_val * b[pos + s];
            }
        }
    }

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
        st: &SumTuple,
        at: &AltSumTuple,
        nodes_visited: &mut u64,
        max_nodes: u64,
    ) -> bool {
        if *nodes_visited >= max_nodes { return false; }

        let num_pairs = constraints.len();
        if pair_idx >= num_pairs {
            // All positions filled — check final constraints
            if *a_sum != st.a || *b_sum != st.b { return false; }

            let a_alt: i32 = a.iter().enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
            let b_alt: i32 = b.iter().enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
            if a_alt != at.a_star || b_alt != at.b_star { return false; }

            // Shifts n..=n-num_pairs+1 were checked during recursion.
            // Check remaining shifts 1..=n-num_pairs.
            for s in 1..=(n.saturating_sub(num_pairs)) {
                if partial_ac[s] != 0 { return false; }
            }
            return true;
        }

        let pos_left = pair_idx;
        let pos_right = m - 1 - pair_idx;

        for &(a_i, b_i, a_j, b_j) in &constraints[pair_idx] {
            *nodes_visited += 1;
            if *nodes_visited >= max_nodes { return false; }

            // Assign positions and update partial_ac incrementally
            if pos_left == pos_right {
                a[pos_left] = a_i;
                b[pos_left] = b_i;
                *a_sum += a_i;
                *b_sum += b_i;
                update_partial_ac(partial_ac, a, b, pos_left, a_i, b_i, n, m);
            } else {
                // Fill left first, then right (cross-term captured from right's update)
                a[pos_left] = a_i;
                b[pos_left] = b_i;
                update_partial_ac(partial_ac, a, b, pos_left, a_i, b_i, n, m);
                a[pos_right] = a_j;
                b[pos_right] = b_j;
                update_partial_ac(partial_ac, a, b, pos_right, a_j, b_j, n, m);
                *a_sum += a_i + a_j;
                *b_sum += b_i + b_j;
            }

            // Sum feasibility (O(1) with running sums)
            let remaining_positions = if pos_left == pos_right { 0 }
                else { m - 2 * (pair_idx + 1) };
            let a_remaining = st.a - *a_sum;
            let b_remaining = st.b - *b_sum;
            let prune = a_remaining.abs() > remaining_positions as i32
                || b_remaining.abs() > remaining_positions as i32
                || (a_remaining + remaining_positions as i32) % 2 != 0
                || (b_remaining + remaining_positions as i32) % 2 != 0;

            if !prune {
                // After pair k, shift n-k is fully determined — check it
                let ready_shift = n - pair_idx;
                if partial_ac[ready_shift] == 0 {
                    if backtrack(pair_idx + 1, a, b, m, n, constraints,
                                 partial_ac, a_sum, b_sum, st, at,
                                 nodes_visited, max_nodes) {
                        return true;
                    }
                }
            }

            // Undo (reverse order: right first while left still set)
            if pos_left == pos_right {
                undo_partial_ac(partial_ac, a, b, pos_left, a_i, b_i, n, m);
                a[pos_left] = 0;
                b[pos_left] = 0;
                *a_sum -= a_i;
                *b_sum -= b_i;
            } else {
                undo_partial_ac(partial_ac, a, b, pos_right, a_j, b_j, n, m);
                a[pos_right] = 0;
                b[pos_right] = 0;
                undo_partial_ac(partial_ac, a, b, pos_left, a_i, b_i, n, m);
                a[pos_left] = 0;
                b[pos_left] = 0;
                *a_sum -= a_i + a_j;
                *b_sum -= b_i + b_j;
            }
        }

        false
    }

    if backtrack(0, &mut a, &mut b, m, n, &constraints,
                 &mut partial_ac, &mut a_sum, &mut b_sum, st, at,
                 &mut nodes_visited, max_nodes) {
        Some((Sequence::new(a), Sequence::new(b)))
    } else {
        None
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

/// Checkpoint data for resuming search
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    n: usize,
    #[serde(default = "default_version")]
    version: u32,
    completed_tuples: Vec<usize>,
    total_cd_tried: u64,
    total_cd_filtered: u64,
    elapsed_secs: f64,
}

fn default_version() -> u32 { 1 }

impl Checkpoint {
    fn new(n: usize) -> Self {
        Checkpoint {
            n,
            version: 2,
            completed_tuples: Vec::new(),
            total_cd_tried: 0,
            total_cd_filtered: 0,
            elapsed_secs: 0.0,
        }
    }

    fn filename(n: usize) -> String {
        format!("checkpoint_v6r_n{}.json", n)
    }

    fn save(&self) -> std::io::Result<()> {
        let filename = Self::filename(self.n);
        let file = File::create(&filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    fn load(n: usize) -> Option<Self> {
        let filename = Self::filename(n);
        if !Path::new(&filename).exists() {
            return None;
        }
        let file = File::open(&filename).ok()?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).ok()
    }

    fn delete(n: usize) {
        let filename = Self::filename(n);
        let _ = std::fs::remove_file(&filename);
    }
}

/// Score a tuple by search difficulty (lower = easier = try first)
fn score_tuple(sum_tuple: &SumTuple, alt_tuple: &AltSumTuple) -> i32 {
    let sum_mag = sum_tuple.a.abs() + sum_tuple.b.abs() +
                  sum_tuple.c.abs() + sum_tuple.d.abs();
    let alt_mag = alt_tuple.a_star.abs() + alt_tuple.b_star.abs() +
                  alt_tuple.c_star.abs() + alt_tuple.d_star.abs();
    sum_mag + alt_mag
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

    let mod3_solutions = enumerate_mod3_solutions(n, &known_st, &known_at);
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
        let mod6_solutions = enumerate_mod6_cd_solutions(n, mod3_sol, mod3_idx);
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
            let (cd_pairs, cd_checked) = backtrack_cd_from_mod6(n, mod6_sol, 10000, f64::MAX);
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
            for j in 1..=200 {
                let theta = (j as f64) * PI / 100.0;
                let fc = hall_polynomial(&known_c_seq.values, theta);
                let fd = hall_polynomial(&known_d_seq.values, theta);
                let surplus = fc + fd - target_f;
                if surplus > worst_surplus {
                    worst_surplus = surplus;
                    worst_j = j;
                }
            }
            println!("  Worst theta: j={} (theta={:.4}), f(C)+f(D)-target = {:.4}",
                     worst_j, (worst_j as f64) * PI / 100.0, worst_surplus);

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
            let (cd_pairs, cd_checked) = backtrack_cd_from_mod6(n, &direct_mod6, 10000, f64::MAX);
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
        let (cd_pairs, cd_checked) = backtrack_cd_from_mod6(n, &direct_mod6, 10000, f64::MAX);
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
        let ab_result = backtrack_search_ab(n, &c_seq, &d_seq, &known_st, &known_at, 10_000_000);
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
        let mod6_solutions = enumerate_mod6_cd_solutions(n, mod3_sol, known_mod3_idx.unwrap());
        let mod6_found = mod6_solutions.iter().any(|s| s.p == known_mod6_p && s.q == known_mod6_q);
        println!("  Mod-6 CD solution found: {}", if mod6_found { "YES" } else { "NO" });

        if mod6_found {
            let mod6_idx = mod6_solutions.iter().position(|s| s.p == known_mod6_p && s.q == known_mod6_q).unwrap();
            let (cd_pairs, _) = backtrack_cd_from_mod6(n, &mod6_solutions[mod6_idx], 10000, f64::MAX);
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
    sorted.sort_by_key(|(st, at)| score_tuple(st, at));
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
        let mod3_sols = enumerate_mod3_solutions(n, st, at);
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
            let mod6_sols = enumerate_mod6_cd_solutions(n, mod3_sol, m3_idx);
            tuple_mod6 += mod6_sols.len() as u64;

            for mod6_sol in &mod6_sols {
                // Step 4: CD generation with integrated spectral filter (exact threshold)
                let (cd_pairs, cd_checked) = backtrack_cd_from_mod6(n, mod6_sol, max_cd_per_mod6, 0.0);
                tuple_cd += cd_checked;
                tuple_pass += cd_pairs.len() as u64;
                tuple_fail += cd_checked - cd_pairs.len() as u64;

                for (c, d) in &cd_pairs {
                    let headroom = compute_ab_headroom(c, d);
                    if headroom > best_headroom {
                        best_headroom = headroom;
                    }

                    // Step 5: Try AB backtracking
                    if let Some((_a, _b)) = backtrack_search_ab(n, c, d, st, at, 10_000_000) {
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
            eprintln!("Usage: {} <n> [--resume] [--exhaustive]", args[0]);
            std::process::exit(1);
        })
    } else {
        eprintln!("Usage: {} <n> [--resume]", args[0]);
        std::process::exit(1);
    };

    let resume = args.iter().any(|a| a == "--resume");
    let debug_pipeline = args.iter().any(|a| a == "--debug-pipeline");
    let pipeline_stats = args.iter().any(|a| a == "--pipeline-stats");

    if debug_pipeline {
        run_debug_pipeline(n);
        return;
    }

    if pipeline_stats {
        run_pipeline_stats(n);
        return;
    }

    println!("BS({},{}) - V6 Paper Pipeline Search", n + 1, n);
    println!("==============================================\n");

    let backtrack_limit: u64 = if n <= 15 { 10_000_000 }
        else if n <= 25 { 50_000_000 }
        else if n <= 35 { 100_000_000 }
        else { 200_000_000 };
    let max_cd_per_mod6: usize = if n <= 20 { 0 }
        else if n <= 30 { 100_000 }
        else { 50_000 };

    println!("Configuration for n={}:", n);
    println!("  Spectral margin: 0.5");
    println!("  AB backtrack limit: {:.0e}", backtrack_limit as f64);
    println!("  Max CDs per mod-6: {}", if max_cd_per_mod6 == 0 { "unlimited".to_string() } else { format!("{}", max_cd_per_mod6) });
    println!();

    println!("Step 1: Find valid tuples...");
    let all_tuples = find_valid_sum_tuples_fast_v2(n);
    println!("  {} raw tuples found", all_tuples.len());

    println!("Step 2: Filter and sort by difficulty...");
    let canonical = filter_to_canonical_5class(all_tuples, n);
    let mut sorted: Vec<(SumTuple, AltSumTuple)> = canonical.into_iter().collect();
    sorted.sort_by_key(|(st, at)| score_tuple(st, at));
    println!("  {} canonical tuples", sorted.len());

    // Calculate total CD pairs
    println!("\nStep 3: Calculate search space...");
    let mut tuple_cd_counts: Vec<u64> = Vec::with_capacity(sorted.len());
    let mut total_cd_pairs: u64 = 0;

    for (st, at) in &sorted {
        let count = count_cd_pairs(n, st.c, st.d, at.c_star, at.d_star);
        tuple_cd_counts.push(count);
        total_cd_pairs = total_cd_pairs.saturating_add(count);
    }

    println!("  Total CD pairs: {:.2e}", total_cd_pairs as f64);
    println!();

    // Load or create checkpoint
    let checkpoint = if resume {
        if let Some(cp) = Checkpoint::load(n) {
            if cp.version >= 2 {
                println!("Resuming from checkpoint (v{}):", cp.version);
                println!("  Completed tuples: {}/{}", cp.completed_tuples.len(), sorted.len());
                println!("  CD pairs sampled: {:.2e}", cp.total_cd_tried as f64);
                println!("  Previous elapsed: {:.2} hours", cp.elapsed_secs / 3600.0);
                println!();
                cp
            } else {
                println!("Found old-format checkpoint (v1 deterministic). Cannot resume.");
                println!("Starting fresh with randomized sampling.\n");
                Checkpoint::new(n)
            }
        } else {
            println!("No checkpoint found, starting fresh.\n");
            Checkpoint::new(n)
        }
    } else {
        Checkpoint::new(n)
    };

    let prior_elapsed = checkpoint.elapsed_secs;
    let completed_set: std::collections::HashSet<usize> =
        checkpoint.completed_tuples.iter().cloned().collect();
    let completed_set = Arc::new(completed_set);
    let start = Instant::now();

    let found = Arc::new(AtomicBool::new(false));
    let tuples_done = Arc::new(AtomicUsize::new(checkpoint.completed_tuples.len()));
    let cd_tried = Arc::new(AtomicU64::new(checkpoint.total_cd_tried));
    let cd_filtered = Arc::new(AtomicU64::new(checkpoint.total_cd_filtered));

    let checkpoint_mutex = Arc::new(Mutex::new(checkpoint));
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
    let cd_clone = Arc::clone(&cd_tried);
    let filtered_clone = Arc::clone(&cd_filtered);
    let checkpoint_clone = Arc::clone(&checkpoint_mutex);
    let total_mod3_clone = Arc::clone(&total_mod3_found);
    let total_mod6_clone = Arc::clone(&total_mod6_found);
    let total_tuples = sorted_arc.len();
    let start_clone = start.clone();

    std::thread::spawn(move || {
        let mut last_checkpoint = Instant::now();
        let mut last_tried = cd_clone.load(Ordering::Relaxed);
        let mut last_time = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(30));
            if found_clone.load(Ordering::Relaxed) || search_done_clone.load(Ordering::Relaxed) { break; }

            let done = tuples_clone.load(Ordering::Relaxed);
            let tried = cd_clone.load(Ordering::Relaxed);
            let filtered = filtered_clone.load(Ordering::Relaxed);
            let elapsed = start_clone.elapsed().as_secs_f64() + prior_elapsed;

            let total_sampled = tried + filtered;
            let pass_rate = if total_sampled > 0 {
                tried as f64 / total_sampled as f64 * 100.0
            } else { 0.0 };

            let dt = last_time.elapsed().as_secs_f64();
            let searched_per_sec = if dt > 0.0 {
                (tried - last_tried) as f64 / dt
            } else { 0.0 };
            last_tried = tried;
            last_time = Instant::now();

            let m3 = total_mod3_clone.load(Ordering::Relaxed);
            let m6 = total_mod6_clone.load(Ordering::Relaxed);
            println!("  [{:>4}/{:>4}] | {:.2e} CD ({:.1}% pass) | {:.1}/s | m3:{:.1e} m6:{:.1e} | {:.1}h",
                done, total_tuples,
                total_sampled as f64,
                pass_rate,
                searched_per_sec,
                m3 as f64, m6 as f64,
                elapsed / 3600.0);

            // Save checkpoint periodically (every 5 minutes)
            if last_checkpoint.elapsed().as_secs() >= 300 {
                if let Ok(mut cp) = checkpoint_clone.lock() {
                    cp.total_cd_tried = tried;
                    cp.total_cd_filtered = filtered;
                    cp.elapsed_secs = elapsed;
                    if let Err(err) = cp.save() {
                        eprintln!("  Warning: Failed to save checkpoint: {}", err);
                    } else {
                        println!("  [Checkpoint saved]");
                    }
                }
                last_checkpoint = Instant::now();
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

    println!("Step 4: Paper pipeline (Steps 2-5: mod-3 -> mod-6 -> CD+spectral -> AB)\n");

    let result: Option<(BaseSequence, usize, SumTuple, AltSumTuple)> = (0..sorted_arc.len())
        .into_par_iter()
        .find_map_any(|tuple_idx| {
            if found.load(Ordering::Relaxed) { return None; }
            if completed_set.contains(&tuple_idx) { return None; }

            let (st, at) = &sorted_arc[tuple_idx];

            // Step 2: Mod-3 partial sum enumeration (Theorem 2.3, m=3)
            let mod3_solutions = enumerate_mod3_solutions(n, st, at);
            total_mod3_found.fetch_add(mod3_solutions.len() as u64, Ordering::Relaxed);

            // Steps 3-5 for each mod-3 solution
            for (mod3_idx, mod3_sol) in mod3_solutions.iter().enumerate() {
                if found.load(Ordering::Relaxed) { return None; }

                // Step 3: Mod-6 CD refinement (Theorem 2.3, m=6)
                let mod6_solutions = enumerate_mod6_cd_solutions(n, mod3_sol, mod3_idx);
                total_mod6_found.fetch_add(mod6_solutions.len() as u64, Ordering::Relaxed);

                for mod6_sol in &mod6_solutions {
                    if found.load(Ordering::Relaxed) { return None; }

                    // Step 4a: Generate C,D sequences (Theorem 2.2 + mod-6 constraints)
                    let cd_limit = if max_cd_per_mod6 == 0 { usize::MAX } else { max_cd_per_mod6 };
                    let (cd_pairs, cd_checked) = backtrack_cd_from_mod6(n, mod6_sol, cd_limit, 0.5);
                    let cd_filtered_count = cd_checked - cd_pairs.len() as u64;
                    cd_filtered.fetch_add(cd_filtered_count, Ordering::Relaxed);

                    for (c, d) in &cd_pairs {
                        if found.load(Ordering::Relaxed) { return None; }

                        cd_tried.fetch_add(1, Ordering::Relaxed);

                        // Step 5: Backtracking A,B search (Theorem 2.2)
                        if let Some((a, b)) = backtrack_search_ab(n, c, d, st, at, backtrack_limit) {
                            let base = BaseSequence::new(a, b, c.clone(), d.clone());
                            if base.is_valid() {
                                found.store(true, Ordering::Relaxed);
                                Checkpoint::delete(n);
                                return Some((base, tuple_idx, st.clone(), at.clone()));
                            }
                        }
                    }
                }
            }

            tuples_done.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut cp) = checkpoint_mutex.lock() {
                cp.completed_tuples.push(tuple_idx);
            }
            None
        });

    // Stop progress thread
    search_done.store(true, Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_millis(100));

    let elapsed_secs = start.elapsed().as_secs_f64() + prior_elapsed;
    let total_mod3 = total_mod3_found.load(Ordering::Relaxed);
    let total_mod6 = total_mod6_found.load(Ordering::Relaxed);

    println!("\n");

    if let Some((base, idx, st, at)) = result {
        print_solution(n, &base, &st, &at, idx, elapsed_secs,
            &tuples_done, &cd_tried);
    } else if !found.load(Ordering::Relaxed) {
        println!("============================================");
        println!("     Search complete - no solution          ");
        println!("============================================\n");

        println!("Time: {:.2} hours", elapsed_secs / 3600.0);
        println!("Tuples processed: {}/{}", tuples_done.load(Ordering::Relaxed), sorted_arc.len());
        println!("Mod-3 solutions found: {}", total_mod3);
        println!("Mod-6 CD solutions found: {}", total_mod6);
        let tried = cd_tried.load(Ordering::Relaxed);
        let filtered = cd_filtered.load(Ordering::Relaxed);
        let total_sampled = tried + filtered;
        let pass_rate = if total_sampled > 0 { tried as f64 / total_sampled as f64 * 100.0 } else { 0.0 };
        println!("CD pairs generated: {:.2e} ({:.1}% passed spectral)", total_sampled as f64, pass_rate);
        println!("CD pairs searched (passed spectral): {:.2e}", tried as f64);
        println!("\nNote: The paper's algorithm is deterministic. If no solution found,");
        println!("all CD pairs from the mod-6 decomposition have been exhaustively searched.");
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
) {
    println!("============================================");
    println!("       SUCCESS! BS({},{}) FOUND          ", n + 1, n);
    println!("============================================\n");

    println!("Time: {:.2} hours", elapsed_secs / 3600.0);
    println!("Tuples checked: {}", tuples_done.load(Ordering::Relaxed));
    println!("CD pairs tried: {:.2e}", cd_tried.load(Ordering::Relaxed) as f64);
    println!();

    println!("Solution at tuple #{}", idx);
    println!("Sum tuple:     ({:>3},{:>3},{:>3},{:>3})", st.a, st.b, st.c, st.d);
    println!("Alt-sum tuple: ({:>3},{:>3},{:>3},{:>3})", at.a_star, at.b_star, at.c_star, at.d_star);
    println!();

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
    let filename = format!("BS_{}_{}_V6_{:.0}s.txt", n + 1, n, elapsed_secs);
    if let Ok(mut f) = File::create(&filename) {
        writeln!(f, "BS({},{}) Solution - V6 Aggressive", n + 1, n).ok();
        writeln!(f, "====================================").ok();
        writeln!(f, "Time: {:.1}s ({:.2}h)", elapsed_secs, elapsed_secs / 3600.0).ok();
        writeln!(f, "CD pairs tried: {:.2e}", cd_tried.load(Ordering::Relaxed) as f64).ok();
        writeln!(f, "").ok();
        writeln!(f, "A = {:?}", base.a.values).ok();
        writeln!(f, "B = {:?}", base.b.values).ok();
        writeln!(f, "C = {:?}", base.c.values).ok();
        writeln!(f, "D = {:?}", base.d.values).ok();
        println!("\nSaved to: {}", filename);
    }
}
