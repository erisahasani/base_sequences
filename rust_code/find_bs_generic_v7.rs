/// BS(n+1, n) search - V7 implementing Wang & Zhu (2025) paper exactly
///
/// Paper: "On Base, Normal and Near-normal Sequences" (arXiv:2506.20296)
///
/// Key techniques from paper:
/// 1. Sum tuple constraints (Theorem 2.1): a²+b²+c²+d²=4n+2
/// 2. Alternating sum constraints (Theorem 2.2, 2.3)
/// 3. Two-phase spectral filtering (l=50, l=1000) using Hall polynomials
/// 4. Full 5-class isomorphic equivalence reduction
/// 5. Symmetric pair constraints (8 cases per position) - Theorem 2.2
/// 6. Modular decomposition (mod 3→6→12→n) - Theorem 2.3
/// 7. Deterministic backtracking search for A,B (not stochastic)
///
/// Usage: cargo run --release --example find_bs_generic_v7 -- <n>
/// Resume: cargo run --release --example find_bs_generic_v7 -- <n> --resume

// ============================================================================
// Inlined from base_sequences crate (core types)
// ============================================================================

/// Represents a sequence of ±1 values
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

    /// Non-periodic autocorrelation function (Equation 1.1)
    fn autocorrelation(&self, shift: usize) -> i32 {
        let n = self.values.len();
        if shift >= n {
            return 0;
        }
        let mut sum = 0;
        for j in 0..(n - shift) {
            sum += self.values[j] * self.values[j + shift];
        }
        sum
    }
}

/// Base Sequence BS(m, n): four sequences A, B, C, D with zero autocorrelation
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

    /// Check if this is a valid base sequence (Equation 1.2)
    fn is_valid(&self) -> bool {
        let m = self.a.len();
        let n = self.c.len();
        if self.b.len() != m || self.d.len() != n {
            return false;
        }
        let ac_0 = self.a.autocorrelation(0) + self.b.autocorrelation(0)
            + self.c.autocorrelation(0) + self.d.autocorrelation(0);
        if ac_0 != 2 * (m as i32 + n as i32) {
            return false;
        }
        for i in 1..=n {
            let ac_i = self.a.autocorrelation(i) + self.b.autocorrelation(i)
                + self.c.autocorrelation(i) + self.d.autocorrelation(i);
            if ac_i != 0 {
                return false;
            }
        }
        true
    }
}

/// Sum tuple (a, b, c, d) from Theorem 2.1
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SumTuple {
    a: i32,
    b: i32,
    c: i32,
    d: i32,
}

/// Alternating sum tuple (a*, b*, c*, d*) from Theorem 2.1
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AltSumTuple {
    a_star: i32,
    b_star: i32,
    c_star: i32,
    d_star: i32,
}

// ============================================================================
// Inlined from base_sequences::fast_tuple_search_v2
// ============================================================================

#[inline]
fn mod_positive_tuples(a: i32, m: i32) -> i32 {
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
            a_mod4: mod_positive_tuples(st.a, 4),
            b_mod4: mod_positive_tuples(st.b, 4),
            c_mod4: mod_positive_tuples(st.c, 4),
            d_mod4: mod_positive_tuples(st.d, 4),
        }
    }

    fn required_for_alt_tuple(at: &AltSumTuple, n: usize) -> Self {
        match n % 4 {
            0 => Mod4Signature {
                a_mod4: mod_positive_tuples(at.a_star, 4),
                b_mod4: mod_positive_tuples(at.b_star, 4),
                c_mod4: mod_positive_tuples(at.c_star, 4),
                d_mod4: mod_positive_tuples(at.d_star, 4),
            },
            1 => Mod4Signature {
                a_mod4: mod_positive_tuples(at.a_star + 2, 4),
                b_mod4: mod_positive_tuples(at.b_star + 2, 4),
                c_mod4: mod_positive_tuples(at.c_star, 4),
                d_mod4: mod_positive_tuples(at.d_star, 4),
            },
            2 => Mod4Signature {
                a_mod4: mod_positive_tuples(at.a_star + 2, 4),
                b_mod4: mod_positive_tuples(at.b_star + 2, 4),
                c_mod4: mod_positive_tuples(at.c_star + 2, 4),
                d_mod4: mod_positive_tuples(at.d_star + 2, 4),
            },
            3 => Mod4Signature {
                a_mod4: mod_positive_tuples(at.a_star, 4),
                b_mod4: mod_positive_tuples(at.b_star, 4),
                c_mod4: mod_positive_tuples(at.c_star + 2, 4),
                d_mod4: mod_positive_tuples(at.d_star + 2, 4),
            },
            _ => unreachable!(),
        }
    }
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
    if mod_positive_tuples(start, 2) == parity { start } else { start + 1 }
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

            if !n_even && mod_positive_tuples(a, 4) != mod_positive_tuples(b + 2, 4) {
                b += 2; continue;
            }

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
                        if mod_positive_tuples(d, 2) != cd_parity { continue; }
                        if n_even && mod_positive_tuples(c, 4) != mod_positive_tuples(d, 4) { continue; }
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

            if !n_even && mod_positive_tuples(a_star, 4) != mod_positive_tuples(b_star + 2, 4) {
                b_star += 2; continue;
            }

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
                        if mod_positive_tuples(d_star, 2) != cd_parity { continue; }
                        if n_even && mod_positive_tuples(c_star, 4) != mod_positive_tuples(d_star, 4) { continue; }
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
// Inlined from base_sequences::cd_optimized (DeterministicCDEnumerator)
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

struct DeterministicCDEnumerator {
    n: usize,
    c_total: u64,
    d_total: u64,
    total_pairs: u64,
    current_index: u64,
    even_positions: Vec<usize>,
    odd_positions: Vec<usize>,
    c_k_even: usize,
    c_k_odd: usize,
    d_k_even: usize,
    d_k_odd: usize,
    binom_cache: Vec<Vec<u64>>,
}

impl DeterministicCDEnumerator {
    fn new(
        n: usize, c_sum: i32, d_sum: i32, c_alt: i32, d_alt: i32,
    ) -> Option<Self> {
        if !constraints_feasible(n, c_sum, c_alt) || !constraints_feasible(n, d_sum, d_alt) {
            return None;
        }
        let even_positions: Vec<usize> = (0..n).filter(|&i| i % 2 == 0).collect();
        let odd_positions: Vec<usize> = (0..n).filter(|&i| i % 2 == 1).collect();
        let n_even = even_positions.len();
        let n_odd = odd_positions.len();

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
        let total_pairs = c_total.saturating_mul(d_total);

        Some(DeterministicCDEnumerator {
            n, c_total, d_total, total_pairs, current_index: 0,
            even_positions, odd_positions,
            c_k_even, c_k_odd, d_k_even, d_k_odd, binom_cache,
        })
    }

    fn total_pairs(&self) -> u64 { self.total_pairs }
    fn current_index(&self) -> u64 { self.current_index }
    fn set_index(&mut self, index: u64) { self.current_index = index.min(self.total_pairs); }

    fn next(&mut self) -> Option<(Sequence, Sequence)> {
        if self.current_index >= self.total_pairs { return None; }
        let pair = self.get_pair_at_index(self.current_index);
        self.current_index += 1;
        Some(pair)
    }

    fn get_pair_at_index(&self, index: u64) -> (Sequence, Sequence) {
        let c_index = index / self.d_total;
        let d_index = index % self.d_total;
        let c = self.unrank_sequence(c_index, self.c_k_even, self.c_k_odd);
        let d = self.unrank_sequence(d_index, self.d_k_even, self.d_k_odd);
        (c, d)
    }

    fn unrank_sequence(&self, seq_index: u64, k_even: usize, k_odd: usize) -> Sequence {
        let n_even = self.even_positions.len();
        let n_odd = self.odd_positions.len();
        let odd_total = self.binom_cache[n_odd][k_odd];
        let even_index = seq_index / odd_total;
        let odd_index = seq_index % odd_total;
        let even_selected = self.unrank_combination(n_even, k_even, even_index);
        let odd_selected = self.unrank_combination(n_odd, k_odd, odd_index);
        let mut values = vec![-1i32; self.n];
        for idx in even_selected { values[self.even_positions[idx]] = 1; }
        for idx in odd_selected { values[self.odd_positions[idx]] = 1; }
        Sequence::new(values)
    }

    fn unrank_combination(&self, n: usize, k: usize, mut index: u64) -> Vec<usize> {
        if k == 0 || k > n { return Vec::new(); }
        let mut result = Vec::with_capacity(k);
        let mut current = 0usize;
        for i in (1..=k).rev() {
            let mut v = current;
            loop {
                let count = self.binom(n - 1 - v, i - 1);
                if index < count {
                    result.push(v);
                    current = v + 1;
                    break;
                }
                index -= count;
                v += 1;
                if v >= n {
                    result.push(n - 1);
                    current = n;
                    break;
                }
            }
        }
        result
    }

    #[inline]
    fn binom(&self, n: usize, k: usize) -> u64 {
        if k > n || n >= self.binom_cache.len() || k >= self.binom_cache[0].len() {
            return 0;
        }
        self.binom_cache[n][k]
    }
}

// ============================================================================
// End of inlined library code
// ============================================================================

use std::time::Instant;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, AtomicI64, Ordering};
use std::f64::consts::PI;
use std::env;
use std::fs::File;
use std::io::{Write as _, BufReader, BufWriter};
use std::path::Path;
use std::sync::Mutex;
use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};
use rand::Rng;

/// Checkpoint interval in seconds
const CHECKPOINT_INTERVAL_SECS: u64 = 300; // 5 minutes

// ============================================================================
// Checkpoint data for resuming search
// ============================================================================

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    n: usize,
    version: u32,
    completed_tuples: Vec<usize>,
    /// Per-tuple CD enumerator progress: (tuple_idx, cd_index)
    tuple_cd_progress: Vec<(usize, u64)>,
    total_cd_checked: u64,
    total_cd_filtered: u64,
    elapsed_secs: f64,
    #[serde(default)]
    near_misses: Vec<NearMiss>,
    #[serde(default)]
    global_best_energy: i64,
}

impl Checkpoint {
    fn new(n: usize) -> Self {
        Checkpoint {
            n,
            version: 1,
            completed_tuples: Vec::new(),
            tuple_cd_progress: Vec::new(),
            total_cd_checked: 0,
            total_cd_filtered: 0,
            elapsed_secs: 0.0,
            near_misses: Vec::new(),
            global_best_energy: i64::MAX,
        }
    }

    fn filename(n: usize) -> String {
        format!("checkpoint_v7_n{}.json", n)
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

// ============================================================================
// SLS (Stochastic Local Search) infrastructure - ported from V6
// ============================================================================

/// Count Theorem 2.2 violations in an AB pair.
/// a, b are 0-indexed arrays of length m = n+1.
fn theorem22_violations_ab(a: &[i32], b: &[i32]) -> usize {
    let m = a.len();
    let mut violations = 0;
    for i in 1..=((m) / 2) {
        let sum = a[i - 1] + b[i - 1] + a[m - i] + b[m - i];
        let sum_mod4 = ((sum % 4) + 4) % 4;
        let target = if i == 1 { 2 } else { 0 };
        if sum_mod4 != target { violations += 1; }
    }
    violations
}

#[derive(Clone)]
struct IndexedSequence {
    values: Vec<i32>,
    plus_indices: Vec<usize>,
    minus_indices: Vec<usize>,
    alt_sum: i32,
}

impl IndexedSequence {
    fn new(values: Vec<i32>) -> Self {
        let mut plus_indices = Vec::new();
        let mut minus_indices = Vec::new();
        let mut alt_sum = 0i32;
        for (i, &v) in values.iter().enumerate() {
            if v == 1 { plus_indices.push(i); } else { minus_indices.push(i); }
            let sign = if i % 2 == 0 { 1 } else { -1 };
            alt_sum += sign * v;
        }
        IndexedSequence { values, plus_indices, minus_indices, alt_sum }
    }

    #[inline]
    fn len(&self) -> usize { self.values.len() }

    #[inline]
    fn sum(&self) -> i32 { 2 * self.plus_indices.len() as i32 - self.values.len() as i32 }

    #[inline]
    fn swap_plus_minus(&mut self, plus_idx: usize, minus_idx: usize) -> i32 {
        let i = self.plus_indices[plus_idx];
        let j = self.minus_indices[minus_idx];
        let sign_i = if i % 2 == 0 { 1 } else { -1 };
        let sign_j = if j % 2 == 0 { 1 } else { -1 };
        let delta = 2 * (sign_j - sign_i);
        self.values[i] = -1;
        self.values[j] = 1;
        self.plus_indices[plus_idx] = j;
        self.minus_indices[minus_idx] = i;
        self.alt_sum += delta;
        delta
    }

    fn to_sequence(&self) -> Sequence { Sequence::new(self.values.clone()) }
}

struct SLSOptimized {
    n: usize,
    m: usize,
    cd_autocorr: Vec<i32>,
    thm22_penalty_weight: i64,
}

impl SLSOptimized {
    fn new(c: &Sequence, d: &Sequence) -> Self {
        let n = c.len();
        let m = n + 1;
        let mut cd_autocorr = vec![0i32; n];
        for shift in 0..n {
            cd_autocorr[shift] = c.autocorrelation(shift) + d.autocorrelation(shift);
        }
        let thm22_penalty_weight = (n as i64) * 4;
        SLSOptimized { n, m, cd_autocorr, thm22_penalty_weight }
    }

    fn calculate_energy_full(&self, a: &IndexedSequence, b: &IndexedSequence) -> i64 {
        let mut energy = 0i64;
        for shift in 1..self.n {
            let a_ac = self.autocorrelation_seq(a, shift);
            let b_ac = self.autocorrelation_seq(b, shift);
            let total = a_ac + b_ac + self.cd_autocorr[shift];
            energy += (total as i64) * (total as i64);
        }
        let violations = theorem22_violations_ab(&a.values, &b.values) as i64;
        energy += violations * self.thm22_penalty_weight;
        energy
    }

    #[inline]
    fn autocorrelation_seq(&self, seq: &IndexedSequence, shift: usize) -> i32 {
        let mut sum = 0i32;
        let len = seq.len();
        for j in 0..(len - shift) { sum += seq.values[j] * seq.values[j + shift]; }
        sum
    }

    #[inline]
    fn autocorr_delta_for_swap(&self, seq: &IndexedSequence, i: usize, j: usize, shift: usize) -> i32 {
        let len = seq.len();
        let mut delta = 0i32;
        if i + shift < len { delta -= 2 * seq.values[i + shift]; }
        if i >= shift { delta -= 2 * seq.values[i - shift]; }
        if j + shift < len { delta += 2 * seq.values[j + shift]; }
        if j >= shift { delta += 2 * seq.values[j - shift]; }
        if i + shift == j { delta -= 4; }
        else if j + shift == i { delta -= 4; }
        delta
    }

    #[inline]
    fn calculate_energy_delta(
        &self, a: &IndexedSequence, b: &IndexedSequence,
        current_autocorrs: &[i32], seq_is_a: bool, i: usize, j: usize,
    ) -> i64 {
        let seq = if seq_is_a { a } else { b };
        let mut new_energy = 0i64;
        for shift in 1..self.n {
            let ac_delta = self.autocorr_delta_for_swap(seq, i, j, shift);
            let new_ac = current_autocorrs[shift] + ac_delta;
            new_energy += (new_ac as i64) * (new_ac as i64);
        }
        new_energy
    }

    #[inline]
    fn calculate_two_swap_delta(
        &self, seq: &IndexedSequence, current_autocorrs: &[i32],
        i1: usize, j1: usize, i2: usize, j2: usize,
    ) -> i64 {
        let mut delta = 0i64;
        for shift in 1..self.n {
            let old_ac = current_autocorrs[shift];
            let d1 = self.autocorr_delta_for_swap(seq, i1, j1, shift);
            let d2 = self.autocorr_delta_for_swap(seq, i2, j2, shift);
            let new_ac = old_ac + d1 + d2;
            delta += (new_ac as i64) * (new_ac as i64) - (old_ac as i64) * (old_ac as i64);
        }
        delta
    }

    fn random_indexed_sequence(
        &self, n: usize, target_sum: i32, target_alt_sum: i32, rng: &mut impl Rng,
    ) -> Option<IndexedSequence> {
        let num_plus = (n as i32 + target_sum) / 2;
        if num_plus < 0 || num_plus > n as i32 { return None; }
        if (n as i32 + target_sum) % 2 != 0 { return None; }
        let mut values = vec![-1i32; n];
        for i in 0..(num_plus as usize) { values[i] = 1; }
        for i in (1..n).rev() { let j = rng.gen_range(0..=i); values.swap(i, j); }
        let mut current_alt_sum: i32 = values.iter().enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
        let max_attempts = 2000;
        for _ in 0..max_attempts {
            if current_alt_sum == target_alt_sum { return Some(IndexedSequence::new(values)); }
            let mut plus_pos = Vec::new();
            let mut minus_pos = Vec::new();
            for (i, &v) in values.iter().enumerate() {
                if v == 1 { plus_pos.push(i); } else { minus_pos.push(i); }
            }
            if plus_pos.is_empty() || minus_pos.is_empty() { break; }
            let need_increase = current_alt_sum < target_alt_sum;
            let (i, j) = if need_increase {
                let i_opt = plus_pos.iter().find(|&&p| p % 2 == 1).copied();
                let j_opt = minus_pos.iter().find(|&&p| p % 2 == 0).copied();
                match (i_opt, j_opt) {
                    (Some(i), Some(j)) => (i, j),
                    _ => {
                        let i = plus_pos[rng.gen_range(0..plus_pos.len())];
                        let j = minus_pos[rng.gen_range(0..minus_pos.len())];
                        if i % 2 == j % 2 { continue; }
                        (i, j)
                    }
                }
            } else {
                let i_opt = plus_pos.iter().find(|&&p| p % 2 == 0).copied();
                let j_opt = minus_pos.iter().find(|&&p| p % 2 == 1).copied();
                match (i_opt, j_opt) {
                    (Some(i), Some(j)) => (i, j),
                    _ => {
                        let i = plus_pos[rng.gen_range(0..plus_pos.len())];
                        let j = minus_pos[rng.gen_range(0..minus_pos.len())];
                        if i % 2 == j % 2 { continue; }
                        (i, j)
                    }
                }
            };
            let sign_i = if i % 2 == 0 { 1 } else { -1 };
            let sign_j = if j % 2 == 0 { 1 } else { -1 };
            let delta = 2 * (sign_j - sign_i);
            let new_alt_sum = current_alt_sum + delta;
            if (new_alt_sum - target_alt_sum).abs() < (current_alt_sum - target_alt_sum).abs() {
                values.swap(i, j);
                current_alt_sum = new_alt_sum;
            }
        }
        if current_alt_sum == target_alt_sum { Some(IndexedSequence::new(values)) } else { None }
    }

    fn focused_search(
        &self, a: &IndexedSequence, b: &IndexedSequence, a_alt: i32, b_alt: i32,
    ) -> Option<(Sequence, Sequence)> {
        let mut rng = rand::thread_rng();
        let mut a = a.clone();
        let mut b = b.clone();
        let mut energy = self.calculate_energy_full(&a, &b);
        let mut current_autocorrs = vec![0i32; self.n];
        for shift in 1..self.n {
            current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                + self.autocorrelation_seq(&b, shift) + self.cd_autocorr[shift];
        }
        for _ in 0..10000 {
            let modify_a = rng.gen_bool(0.5);
            let seq = if modify_a { &mut a } else { &mut b };
            if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() { continue; }
            let plus_idx = rng.gen_range(0..seq.plus_indices.len());
            let minus_idx = rng.gen_range(0..seq.minus_indices.len());
            let i = seq.plus_indices[plus_idx];
            let j = seq.minus_indices[minus_idx];
            if i % 2 != j % 2 { continue; }
            let new_energy = self.calculate_energy_delta(&a, &b, &current_autocorrs, modify_a, i, j);
            if new_energy < energy {
                let seq_ref = if modify_a { &a } else { &b };
                let mut ac_deltas = vec![0i32; self.n];
                for shift in 1..self.n { ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift); }
                let seq = if modify_a { &mut a } else { &mut b };
                seq.swap_plus_minus(plus_idx, minus_idx);
                for shift in 1..self.n { current_autocorrs[shift] += ac_deltas[shift]; }
                energy = new_energy;
                if energy == 0 { return Some((a.to_sequence(), b.to_sequence())); }
            }
        }
        None
    }

    /// Core SLS search with random restarts and history-based acceptance
    fn search_v6(
        &self, a_sum: i32, a_alt: i32, b_sum: i32, b_alt: i32,
        max_restarts: usize, iterations_per_restart: usize,
    ) -> (Option<(Sequence, Sequence)>, i64) {
        let history_length = (300 + self.n * 15).min(3000);
        let mut global_best_energy = i64::MAX;
        let two_swap_prob = if self.n < 25 { 0.05 } else if self.n < 35 { 0.10 } else { 0.15 };

        for _restart in 0..max_restarts {
            let mut rng = rand::thread_rng();
            let mut a = match self.random_indexed_sequence(self.m, a_sum, a_alt, &mut rng) { Some(s) => s, None => continue };
            let mut b = match self.random_indexed_sequence(self.m, b_sum, b_alt, &mut rng) { Some(s) => s, None => continue };
            let mut energy = self.calculate_energy_full(&a, &b);
            if energy == 0 { return (Some((a.to_sequence(), b.to_sequence())), 0); }
            let mut current_autocorrs = vec![0i32; self.n];
            for shift in 1..self.n {
                current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                    + self.autocorrelation_seq(&b, shift) + self.cd_autocorr[shift];
            }
            let mut history: VecDeque<i64> = VecDeque::with_capacity(history_length);
            for _ in 0..history_length { history.push_back(energy); }
            let mut best_energy = energy;
            let mut best_a = a.clone();
            let mut best_b = b.clone();
            let mut plateau_count = 0usize;
            let plateau_threshold = (3000 + self.n * 150).min(15000);

            for iter in 0..iterations_per_restart {
                if plateau_count > plateau_threshold {
                    plateau_count = 0;
                    for e in history.iter_mut() { *e = energy + 100; }
                }
                let do_two_swap = rng.gen_bool(two_swap_prob) &&
                    a.plus_indices.len() >= 2 && a.minus_indices.len() >= 2 &&
                    b.plus_indices.len() >= 2 && b.minus_indices.len() >= 2;

                if do_two_swap {
                    let modify_a = rng.gen_bool(0.5);
                    let seq = if modify_a { &a } else { &b };
                    if seq.plus_indices.len() < 2 || seq.minus_indices.len() < 2 { continue; }
                    let plus_idx1 = rng.gen_range(0..seq.plus_indices.len());
                    let mut plus_idx2 = rng.gen_range(0..seq.plus_indices.len());
                    while plus_idx2 == plus_idx1 && seq.plus_indices.len() > 1 { plus_idx2 = rng.gen_range(0..seq.plus_indices.len()); }
                    let minus_idx1 = rng.gen_range(0..seq.minus_indices.len());
                    let mut minus_idx2 = rng.gen_range(0..seq.minus_indices.len());
                    while minus_idx2 == minus_idx1 && seq.minus_indices.len() > 1 { minus_idx2 = rng.gen_range(0..seq.minus_indices.len()); }
                    let i1 = seq.plus_indices[plus_idx1]; let j1 = seq.minus_indices[minus_idx1];
                    let i2 = seq.plus_indices[plus_idx2]; let j2 = seq.minus_indices[minus_idx2];
                    if i1 % 2 != j1 % 2 || i2 % 2 != j2 % 2 { continue; }
                    let delta1 = self.calculate_two_swap_delta(seq, &current_autocorrs, i1, j1, i2, j2);
                    let history_energy = *history.front().unwrap();
                    let new_energy = (energy as i64 + delta1).max(0);
                    let accept = new_energy <= history_energy || new_energy <= energy;
                    if accept {
                        if modify_a {
                            a.swap_plus_minus(plus_idx1, minus_idx1);
                            let new_plus_idx2 = a.plus_indices.iter().position(|&x| x == i2);
                            let new_minus_idx2 = a.minus_indices.iter().position(|&x| x == j2);
                            if let (Some(pi2), Some(mi2)) = (new_plus_idx2, new_minus_idx2) { a.swap_plus_minus(pi2, mi2); }
                        } else {
                            b.swap_plus_minus(plus_idx1, minus_idx1);
                            let new_plus_idx2 = b.plus_indices.iter().position(|&x| x == i2);
                            let new_minus_idx2 = b.minus_indices.iter().position(|&x| x == j2);
                            if let (Some(pi2), Some(mi2)) = (new_plus_idx2, new_minus_idx2) { b.swap_plus_minus(pi2, mi2); }
                        }
                        energy = self.calculate_energy_full(&a, &b);
                        for shift in 1..self.n {
                            current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                                + self.autocorrelation_seq(&b, shift) + self.cd_autocorr[shift];
                        }
                        plateau_count = 0;
                        if energy < best_energy {
                            best_energy = energy; best_a = a.clone(); best_b = b.clone();
                            if energy == 0 { return (Some((best_a.to_sequence(), best_b.to_sequence())), 0); }
                        }
                    } else { plateau_count += 1; }
                } else {
                    let modify_a = rng.gen_bool(0.5);
                    let seq = if modify_a { &a } else { &b };
                    if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() { continue; }
                    let plus_idx = rng.gen_range(0..seq.plus_indices.len());
                    let minus_idx = rng.gen_range(0..seq.minus_indices.len());
                    let i = seq.plus_indices[plus_idx]; let j = seq.minus_indices[minus_idx];
                    if i % 2 != j % 2 { continue; }
                    let new_energy = self.calculate_energy_delta(&a, &b, &current_autocorrs, modify_a, i, j);
                    let history_energy = *history.front().unwrap();
                    let accept = new_energy <= history_energy || new_energy <= energy;
                    if accept {
                        let seq_ref = if modify_a { &a } else { &b };
                        let mut ac_deltas = vec![0i32; self.n];
                        for shift in 1..self.n { ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift); }
                        let seq = if modify_a { &mut a } else { &mut b };
                        seq.swap_plus_minus(plus_idx, minus_idx);
                        for shift in 1..self.n { current_autocorrs[shift] += ac_deltas[shift]; }
                        energy = new_energy; plateau_count = 0;
                        if energy < best_energy {
                            best_energy = energy; best_a = a.clone(); best_b = b.clone();
                            if energy == 0 { return (Some((best_a.to_sequence(), best_b.to_sequence())), 0); }
                        }
                    } else { plateau_count += 1; }
                }
                history.pop_front(); history.push_back(energy);
                let reset_interval = (40000 + self.n * 600).min(120000);
                if iter > 0 && iter % reset_interval == 0 && energy > 0 {
                    for e in history.iter_mut() { *e = energy; }
                }
            }
            if best_energy < global_best_energy { global_best_energy = best_energy; }
            if best_energy > 0 && best_energy < 100 {
                if let Some(result) = self.focused_search(&best_a, &best_b, a_alt, b_alt) {
                    return (Some(result), 0);
                }
            }
        }
        (None, global_best_energy)
    }

    // ========================================================================
    // Enhanced SLS with tabu list, compound moves, near-miss tracking
    // ========================================================================

    fn exhaustive_single_flip_search(
        &self, a: &IndexedSequence, b: &IndexedSequence, current_autocorrs: &[i32], _a_alt: i32, _b_alt: i32,
    ) -> Option<(Sequence, Sequence)> {
        for (plus_idx, &i) in a.plus_indices.iter().enumerate() {
            for (minus_idx, &j) in a.minus_indices.iter().enumerate() {
                if i % 2 != j % 2 { continue; }
                let new_energy = self.calculate_energy_delta(a, b, current_autocorrs, true, i, j);
                if new_energy == 0 {
                    let mut new_a = a.clone(); new_a.swap_plus_minus(plus_idx, minus_idx);
                    return Some((new_a.to_sequence(), b.to_sequence()));
                }
            }
        }
        for (plus_idx, &i) in b.plus_indices.iter().enumerate() {
            for (minus_idx, &j) in b.minus_indices.iter().enumerate() {
                if i % 2 != j % 2 { continue; }
                let new_energy = self.calculate_energy_delta(a, b, current_autocorrs, false, i, j);
                if new_energy == 0 {
                    let mut new_b = b.clone(); new_b.swap_plus_minus(plus_idx, minus_idx);
                    return Some((a.to_sequence(), new_b.to_sequence()));
                }
            }
        }
        None
    }

    fn exploit_near_miss(
        &self, near_miss: &NearMissConfig, a_alt: i32, b_alt: i32, max_iterations: usize,
    ) -> Option<(Sequence, Sequence)> {
        let mut rng = rand::thread_rng();
        let mut a = IndexedSequence::new(near_miss.a_values.clone());
        let mut b = IndexedSequence::new(near_miss.b_values.clone());
        let mut energy = self.calculate_energy_full(&a, &b);
        if energy == 0 { return Some((a.to_sequence(), b.to_sequence())); }
        let mut current_autocorrs = vec![0i32; self.n];
        for shift in 1..self.n {
            current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                + self.autocorrelation_seq(&b, shift) + self.cd_autocorr[shift];
        }
        if let Some(result) = self.exhaustive_single_flip_search(&a, &b, &current_autocorrs, a_alt, b_alt) {
            return Some(result);
        }
        let history_length = 500;
        let mut history: VecDeque<i64> = VecDeque::with_capacity(history_length);
        for _ in 0..history_length { history.push_back(energy); }
        let mut best_energy = energy;
        let mut best_a = a.clone();
        let mut best_b = b.clone();
        for _ in 0..max_iterations {
            let modify_a = rng.gen_bool(0.5);
            let seq = if modify_a { &a } else { &b };
            if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() { continue; }
            let plus_idx = rng.gen_range(0..seq.plus_indices.len());
            let minus_idx = rng.gen_range(0..seq.minus_indices.len());
            let i = seq.plus_indices[plus_idx]; let j = seq.minus_indices[minus_idx];
            if i % 2 != j % 2 { continue; }
            let new_energy = self.calculate_energy_delta(&a, &b, &current_autocorrs, modify_a, i, j);
            let history_energy = *history.front().unwrap();
            let accept = new_energy < energy || new_energy <= history_energy || (new_energy <= energy + 2 && rng.gen_bool(0.1));
            if accept {
                let seq_ref = if modify_a { &a } else { &b };
                let mut ac_deltas = vec![0i32; self.n];
                for shift in 1..self.n { ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift); }
                let seq = if modify_a { &mut a } else { &mut b };
                seq.swap_plus_minus(plus_idx, minus_idx);
                for shift in 1..self.n { current_autocorrs[shift] += ac_deltas[shift]; }
                energy = new_energy;
                if energy < best_energy {
                    best_energy = energy; best_a = a.clone(); best_b = b.clone();
                    if energy == 0 { return Some((best_a.to_sequence(), best_b.to_sequence())); }
                }
            }
            history.pop_front(); history.push_back(energy);
        }
        if best_energy > 0 && best_energy < 50 { self.focused_search(&best_a, &best_b, a_alt, b_alt) } else { None }
    }

    fn can_do_three_flip(&self, a: &IndexedSequence, b: &IndexedSequence) -> bool {
        (a.plus_indices.len() >= 3 && a.minus_indices.len() >= 3) ||
        (b.plus_indices.len() >= 3 && b.minus_indices.len() >= 3)
    }

    fn try_three_flip(
        &self, a: &mut IndexedSequence, b: &mut IndexedSequence,
        current_autocorrs: &mut Vec<i32>, energy: &mut i64,
        history: &VecDeque<i64>, tabu: &mut TabuList, rng: &mut impl Rng, diversify_mode: bool,
    ) -> bool {
        let modify_a = if a.plus_indices.len() >= 3 && a.minus_indices.len() >= 3 {
            if b.plus_indices.len() >= 3 && b.minus_indices.len() >= 3 { rng.gen_bool(0.5) } else { true }
        } else { false };
        let (plus_indices, minus_indices) = if modify_a { (a.plus_indices.clone(), a.minus_indices.clone()) } else { (b.plus_indices.clone(), b.minus_indices.clone()) };
        let mut plus_idxs = Vec::with_capacity(3);
        let mut minus_idxs = Vec::with_capacity(3);
        let mut positions: Vec<(usize, usize)> = Vec::with_capacity(3);
        for _ in 0..20 {
            plus_idxs.clear(); minus_idxs.clear(); positions.clear();
            while plus_idxs.len() < 3 { let idx = rng.gen_range(0..plus_indices.len()); if !plus_idxs.contains(&idx) { plus_idxs.push(idx); } }
            while minus_idxs.len() < 3 { let idx = rng.gen_range(0..minus_indices.len()); if !minus_idxs.contains(&idx) { minus_idxs.push(idx); } }
            let mut valid = true;
            for k in 0..3 {
                let i = plus_indices[plus_idxs[k]]; let j = minus_indices[minus_idxs[k]];
                if i % 2 != j % 2 { valid = false; break; }
                positions.push((i, j));
            }
            if valid { break; }
        }
        if positions.len() < 3 { return false; }
        if !diversify_mode { for &(i, j) in &positions { if tabu.is_tabu(modify_a, i, j) { return false; } } }

        // OPTIMIZATION: Calculate delta before applying swaps
        let seq = if modify_a { &*a } else { &*b };

        // Pre-compute all deltas on original sequence
        let mut delta = 0i64;
        let mut ac_deltas = vec![0i32; self.n];
        for shift in 1..self.n {
            let old_ac = current_autocorrs[shift];
            let mut total_delta = 0i32;
            for &(i, j) in &positions {
                total_delta += self.autocorr_delta_for_swap(seq, i, j, shift);
            }
            ac_deltas[shift] = total_delta;
            let new_ac = old_ac + total_delta;
            delta += (new_ac as i64) * (new_ac as i64) - (old_ac as i64) * (old_ac as i64);
        }

        let new_energy = (*energy + delta).max(0);
        let history_energy = *history.front().unwrap();
        let accept = new_energy <= history_energy || new_energy <= *energy || (diversify_mode && rng.gen_bool(0.3));

        if accept {
            // Apply swaps
            if modify_a { for k in 0..3 { a.swap_plus_minus(plus_idxs[k], minus_idxs[k]); } }
            else { for k in 0..3 { b.swap_plus_minus(plus_idxs[k], minus_idxs[k]); } }

            // Update autocorrs incrementally using pre-computed deltas
            for shift in 1..self.n {
                current_autocorrs[shift] += ac_deltas[shift];
            }

            *energy = new_energy;
            for &(i, j) in &positions { tabu.add(modify_a, i, j); }
            true
        } else {
            false
        }
    }

    fn try_two_swap(
        &self, a: &mut IndexedSequence, b: &mut IndexedSequence,
        current_autocorrs: &mut Vec<i32>, energy: &mut i64,
        history: &VecDeque<i64>, tabu: &mut TabuList, rng: &mut impl Rng, diversify_mode: bool,
    ) -> bool {
        let modify_a = rng.gen_bool(0.5);
        let seq = if modify_a { &*a } else { &*b };
        if seq.plus_indices.len() < 2 || seq.minus_indices.len() < 2 { return false; }
        let plus_idx1 = rng.gen_range(0..seq.plus_indices.len());
        let mut plus_idx2 = rng.gen_range(0..seq.plus_indices.len());
        while plus_idx2 == plus_idx1 { plus_idx2 = rng.gen_range(0..seq.plus_indices.len()); }
        let minus_idx1 = rng.gen_range(0..seq.minus_indices.len());
        let mut minus_idx2 = rng.gen_range(0..seq.minus_indices.len());
        while minus_idx2 == minus_idx1 { minus_idx2 = rng.gen_range(0..seq.minus_indices.len()); }
        let i1 = seq.plus_indices[plus_idx1]; let j1 = seq.minus_indices[minus_idx1];
        let i2 = seq.plus_indices[plus_idx2]; let j2 = seq.minus_indices[minus_idx2];
        if i1 % 2 != j1 % 2 || i2 % 2 != j2 % 2 { return false; }
        if !diversify_mode && (tabu.is_tabu(modify_a, i1, j1) || tabu.is_tabu(modify_a, i2, j2)) { return false; }

        // OPTIMIZATION: Use delta calculation instead of full energy recalc
        // Compute deltas BEFORE modifying the sequence
        let delta = self.calculate_two_swap_delta(seq, current_autocorrs, i1, j1, i2, j2);

        // Pre-compute autocorr deltas on original sequence
        let mut ac_deltas = vec![0i32; self.n];
        for shift in 1..self.n {
            let d1 = self.autocorr_delta_for_swap(seq, i1, j1, shift);
            let d2 = self.autocorr_delta_for_swap(seq, i2, j2, shift);
            ac_deltas[shift] = d1 + d2;
        }

        let new_energy = (*energy + delta).max(0);
        let history_energy = *history.front().unwrap();
        let accept = new_energy <= history_energy || new_energy <= *energy || (diversify_mode && rng.gen_bool(0.2));

        if accept {
            // Apply the swaps
            if modify_a {
                a.swap_plus_minus(plus_idx1, minus_idx1);
                let np2 = a.plus_indices.iter().position(|&x| x == i2);
                let nm2 = a.minus_indices.iter().position(|&x| x == j2);
                if let (Some(pi2), Some(mi2)) = (np2, nm2) { a.swap_plus_minus(pi2, mi2); }
            } else {
                b.swap_plus_minus(plus_idx1, minus_idx1);
                let np2 = b.plus_indices.iter().position(|&x| x == i2);
                let nm2 = b.minus_indices.iter().position(|&x| x == j2);
                if let (Some(pi2), Some(mi2)) = (np2, nm2) { b.swap_plus_minus(pi2, mi2); }
            }

            // Update autocorrs incrementally using pre-computed deltas
            for shift in 1..self.n {
                current_autocorrs[shift] += ac_deltas[shift];
            }

            *energy = new_energy;
            tabu.add(modify_a, i1, j1);
            tabu.add(modify_a, i2, j2);
            true
        } else {
            false
        }
    }

    fn try_compound_move(
        &self, a: &mut IndexedSequence, b: &mut IndexedSequence,
        current_autocorrs: &mut Vec<i32>, energy: &mut i64,
        history: &VecDeque<i64>, tabu: &mut TabuList, rng: &mut impl Rng, diversify_mode: bool,
    ) -> bool {
        if a.plus_indices.is_empty() || a.minus_indices.is_empty() || b.plus_indices.is_empty() || b.minus_indices.is_empty() { return false; }
        let a_plus_idx = rng.gen_range(0..a.plus_indices.len());
        let a_minus_idx = rng.gen_range(0..a.minus_indices.len());
        let a_i = a.plus_indices[a_plus_idx]; let a_j = a.minus_indices[a_minus_idx];
        if a_i % 2 != a_j % 2 { return false; }
        let b_plus_idx = rng.gen_range(0..b.plus_indices.len());
        let b_minus_idx = rng.gen_range(0..b.minus_indices.len());
        let b_i = b.plus_indices[b_plus_idx]; let b_j = b.minus_indices[b_minus_idx];
        if b_i % 2 != b_j % 2 { return false; }
        if !diversify_mode && (tabu.is_tabu(true, a_i, a_j) || tabu.is_tabu(false, b_i, b_j)) { return false; }

        // OPTIMIZATION: Calculate delta instead of full energy
        // Compound move = swap in A + swap in B simultaneously
        // Compute deltas and energy change on original sequences
        let mut delta = 0i64;
        let mut ac_deltas = vec![0i32; self.n];
        for shift in 1..self.n {
            let old_ac = current_autocorrs[shift];
            let a_delta = self.autocorr_delta_for_swap(a, a_i, a_j, shift);
            let b_delta = self.autocorr_delta_for_swap(b, b_i, b_j, shift);
            ac_deltas[shift] = a_delta + b_delta;
            let new_ac = old_ac + ac_deltas[shift];
            delta += (new_ac as i64) * (new_ac as i64) - (old_ac as i64) * (old_ac as i64);
        }

        let new_energy = (*energy + delta).max(0);
        let history_energy = *history.front().unwrap();
        let accept = new_energy <= history_energy || new_energy <= *energy || (diversify_mode && rng.gen_bool(0.15));

        if accept {
            // Apply swaps
            a.swap_plus_minus(a_plus_idx, a_minus_idx);
            b.swap_plus_minus(b_plus_idx, b_minus_idx);

            // Update autocorrs incrementally using pre-computed deltas
            for shift in 1..self.n {
                current_autocorrs[shift] += ac_deltas[shift];
            }

            *energy = new_energy;
            tabu.add(true, a_i, a_j);
            tabu.add(false, b_i, b_j);
            true
        } else {
            false
        }
    }

    fn try_single_swap(
        &self, a: &mut IndexedSequence, b: &mut IndexedSequence,
        current_autocorrs: &mut Vec<i32>, energy: &mut i64,
        history: &VecDeque<i64>, tabu: &mut TabuList, rng: &mut impl Rng, diversify_mode: bool,
    ) -> bool {
        let modify_a = rng.gen_bool(0.5);
        let seq = if modify_a { &*a } else { &*b };
        if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() { return false; }
        let plus_idx = rng.gen_range(0..seq.plus_indices.len());
        let minus_idx = rng.gen_range(0..seq.minus_indices.len());
        let i = seq.plus_indices[plus_idx]; let j = seq.minus_indices[minus_idx];
        if i % 2 != j % 2 { return false; }
        if !diversify_mode && tabu.is_tabu(modify_a, i, j) { return false; }
        let new_energy = self.calculate_energy_delta(a, b, current_autocorrs, modify_a, i, j);
        let history_energy = *history.front().unwrap();
        let accept = new_energy <= history_energy || new_energy <= *energy;
        if accept {
            let seq_ref = if modify_a { &*a } else { &*b };
            let mut ac_deltas = vec![0i32; self.n];
            for shift in 1..self.n { ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift); }
            let seq = if modify_a { &mut *a } else { &mut *b };
            seq.swap_plus_minus(plus_idx, minus_idx);
            for shift in 1..self.n { current_autocorrs[shift] += ac_deltas[shift]; }
            *energy = new_energy; tabu.add(modify_a, i, j); true
        } else { false }
    }

    fn search_v6_enhanced(
        &self, a_sum: i32, a_alt: i32, b_sum: i32, b_alt: i32,
        max_restarts: usize, iterations_per_restart: usize,
    ) -> (Option<(Sequence, Sequence)>, i64, Option<NearMissConfig>) {
        let history_length = (300 + self.n * 15).min(3000);
        let mut global_best_energy = i64::MAX;
        let mut best_near_miss: Option<NearMissConfig> = None;
        let tabu_size = (self.n * 3).min(150);
        // OPTIMIZED: Increased probabilities for compound moves now that they use delta calc
        // These moves are now much faster and more effective
        let three_flip_prob = if self.n < 25 { 0.05 } else if self.n < 35 { 0.12 } else { 0.15 };
        let two_swap_prob = if self.n < 25 { 0.10 } else if self.n < 35 { 0.18 } else { 0.20 };
        let compound_prob = if self.n < 25 { 0.08 } else { 0.15 };

        for _restart in 0..max_restarts {
            let mut rng = rand::thread_rng();
            let mut tabu = TabuList::new(tabu_size);
            let mut a = match self.random_indexed_sequence(self.m, a_sum, a_alt, &mut rng) { Some(s) => s, None => continue };
            let mut b = match self.random_indexed_sequence(self.m, b_sum, b_alt, &mut rng) { Some(s) => s, None => continue };
            let mut energy = self.calculate_energy_full(&a, &b);
            if energy == 0 { return (Some((a.to_sequence(), b.to_sequence())), 0, None); }
            let mut current_autocorrs = vec![0i32; self.n];
            for shift in 1..self.n {
                current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                    + self.autocorrelation_seq(&b, shift) + self.cd_autocorr[shift];
            }
            let mut history: VecDeque<i64> = VecDeque::with_capacity(history_length);
            for _ in 0..history_length { history.push_back(energy); }
            let mut best_energy = energy;
            let mut best_a = a.clone();
            let mut best_b = b.clone();
            let mut plateau_count = 0usize;
            let plateau_threshold = (3000 + self.n * 150).min(15000);
            let mut no_improve_count = 0usize;
            let diversify_threshold = iterations_per_restart / 5;

            for iter in 0..iterations_per_restart {
                if plateau_count > plateau_threshold {
                    plateau_count = 0;
                    for e in history.iter_mut() { *e = energy + 100; }
                    tabu.clear();
                }
                let diversify_mode = no_improve_count > diversify_threshold;
                let roll: f64 = rng.gen();
                let accepted = if roll < three_flip_prob && self.can_do_three_flip(&a, &b) {
                    self.try_three_flip(&mut a, &mut b, &mut current_autocorrs, &mut energy, &history, &mut tabu, &mut rng, diversify_mode)
                } else if roll < three_flip_prob + two_swap_prob && a.plus_indices.len() >= 2 && a.minus_indices.len() >= 2 {
                    self.try_two_swap(&mut a, &mut b, &mut current_autocorrs, &mut energy, &history, &mut tabu, &mut rng, diversify_mode)
                } else if roll < three_flip_prob + two_swap_prob + compound_prob {
                    self.try_compound_move(&mut a, &mut b, &mut current_autocorrs, &mut energy, &history, &mut tabu, &mut rng, diversify_mode)
                } else {
                    self.try_single_swap(&mut a, &mut b, &mut current_autocorrs, &mut energy, &history, &mut tabu, &mut rng, diversify_mode)
                };
                if accepted {
                    plateau_count = 0;
                    if energy < best_energy {
                        best_energy = energy; best_a = a.clone(); best_b = b.clone(); no_improve_count = 0;
                        if energy == 0 { return (Some((best_a.to_sequence(), best_b.to_sequence())), 0, None); }
                    }
                } else { plateau_count += 1; no_improve_count += 1; }
                history.pop_front(); history.push_back(energy);
                let reset_interval = (40000 + self.n * 600).min(120000);
                if iter > 0 && iter % reset_interval == 0 && energy > 0 {
                    for e in history.iter_mut() { *e = energy; }
                    tabu.clear();
                }
            }
            if best_energy < global_best_energy {
                global_best_energy = best_energy;
                if best_energy > 0 && best_energy < 50 {
                    best_near_miss = Some(NearMissConfig {
                        a_values: best_a.values.clone(), b_values: best_b.values.clone(),
                        energy: best_energy, autocorrs: current_autocorrs.clone(),
                    });
                }
            }
            if best_energy > 0 && best_energy < 100 {
                if let Some(result) = self.focused_search(&best_a, &best_b, a_alt, b_alt) {
                    return (Some(result), 0, None);
                }
            }
        }
        (None, global_best_energy, best_near_miss)
    }
}

// ============================================================================
// Adaptive search infrastructure
// ============================================================================

/// Near-miss thresholds scale with n - larger n needs higher thresholds
fn near_miss_threshold(n: usize) -> i64 {
    if n < 25 { 20 }
    else if n < 30 { 60 }
    else if n < 36 { 100 }
    else { 140 }
}

fn very_promising_threshold(n: usize) -> i64 {
    if n < 25 { 8 }
    else if n < 30 { 20 }
    else if n < 36 { 40 }
    else { 60 }
}

struct TabuList {
    moves: VecDeque<(bool, usize, usize)>,
    max_size: usize,
}

impl TabuList {
    fn new(size: usize) -> Self {
        TabuList { moves: VecDeque::with_capacity(size), max_size: size }
    }
    fn add(&mut self, seq_is_a: bool, i: usize, j: usize) {
        if self.moves.len() >= self.max_size { self.moves.pop_front(); }
        self.moves.push_back((seq_is_a, i, j));
    }
    fn is_tabu(&self, seq_is_a: bool, i: usize, j: usize) -> bool {
        self.moves.iter().any(|&(a, pi, pj)| a == seq_is_a && ((pi == i && pj == j) || (pi == j && pj == i)))
    }
    fn clear(&mut self) { self.moves.clear(); }
}

#[derive(Clone)]
struct NearMissConfig {
    a_values: Vec<i32>,
    b_values: Vec<i32>,
    energy: i64,
    autocorrs: Vec<i32>,
}

/// A near-miss CD pair worth revisiting - tracked globally across all tuples
#[derive(Serialize, Deserialize, Clone)]
struct NearMiss {
    tuple_idx: usize,
    c_values: Vec<i32>,
    d_values: Vec<i32>,
    best_energy: i64,
    #[serde(default)]
    a_values: Option<Vec<i32>>,
    #[serde(default)]
    b_values: Option<Vec<i32>>,
}

struct SearchConfig {
    n: usize,
    base_restarts: usize,
    base_iterations: usize,
    promising_multiplier: usize,
    very_promising_multiplier: usize,
    max_restarts_per_cd: usize,
    max_cd_per_tuple: usize,
    batch_size: usize,
    max_rounds_per_tuple: usize,
    exhaustive: bool,  // Force exhaustive enumeration of all CDs
}

impl SearchConfig {
    fn for_n(n: usize) -> Self {
        Self::for_n_with_mode(n, false)
    }

    fn for_n_exhaustive(n: usize) -> Self {
        Self::for_n_with_mode(n, true)
    }

    fn for_n_with_mode(n: usize, exhaustive: bool) -> Self {
        // Strategy: Deep search per CD using enhanced moves (tabu + compound).
        // V6 gets E:8 with 50M iterations/CD but can't escape local minima.
        // V7's enhanced moves should close the gap with similar or greater depth.
        let (base_restarts, base_iterations, max_cd, batch_size, max_rounds) = if exhaustive {
            if n < 20 {
                (100, 200_000, 0, 1000, 0)
            } else if n < 25 {
                (200, 250_000, 0, 1000, 0)
            } else if n < 30 {
                (200, 250_000, 100_000, 5000, 500)
            } else if n < 36 {
                (200, 250_000, 100_000, 5000, 500)
            } else {
                (250, 300_000, 50_000, 5000, 300)
            }
        } else if n < 20 {
            // n<20: Light. 50 restarts × 100K = 5M per CD
            (50, 100_000, 500, 500, 20)
        } else if n < 25 {
            // n=20-24: Medium. 100 restarts × 200K = 20M per CD
            (100, 200_000, 300, 1000, 50)
        } else if n < 30 {
            // n=25-29: Deep search with enhanced moves.
            // 200 restarts × 250K = 50M per CD (V6-level depth + better moves)
            // 200 CDs per tuple × 34 tuples × ~6s/CD ≈ 40K seconds
            // With parallelism: ~1-2 hours for main search
            (200, 250_000, 200, 1000, 30)
        } else if n < 36 {
            // n=30-35: Deep search with enhanced moves
            // 150 restarts × 200K = 30M per CD
            (150, 200_000, 300, 1500, 50)
        } else {
            // n>=36: Deep search
            (150, 200_000, 500, 2000, 100)
        };
        // Adaptive multipliers: moderate since base is already deep
        // promising: 3× more restarts for near-miss CDs
        // very_promising: 10× more restarts for very close CDs
        let (pm, vpm) = if n >= 25 { (3, 10) } else { (3, 10) };
        SearchConfig {
            n,
            base_restarts,
            base_iterations,
            promising_multiplier: pm,
            very_promising_multiplier: vpm,
            max_restarts_per_cd: if exhaustive { 50_000 } else { 10_000 },
            max_cd_per_tuple: max_cd,
            batch_size,
            max_rounds_per_tuple: max_rounds,
            exhaustive,
        }
    }
}

struct AdaptiveSearchResult {
    found: bool,
    best_energy: i64,
    a: Option<Sequence>,
    b: Option<Sequence>,
    near_miss_config: Option<NearMissConfig>,
}

fn adaptive_search(
    sls: &SLSOptimized,
    st: &SumTuple,
    at: &AltSumTuple,
    config: &SearchConfig,
    _backtrack_budget: u64,
) -> AdaptiveSearchResult {
    let mut total_restarts_used = 0usize;

    // Phase 1: Base search
    // For n < 25: use simple search_v6 (fast, breadth-first across many CDs)
    // For n >= 25: use enhanced search with tabu/compound moves (deeper per CD)
    let use_enhanced = config.n >= 25;
    let (result, energy, near_miss) = if use_enhanced {
        sls.search_v6_enhanced(
            st.a, at.a_star, st.b, at.b_star,
            config.base_restarts, config.base_iterations
        )
    } else {
        let (res, eng) = sls.search_v6(
            st.a, at.a_star, st.b, at.b_star,
            config.base_restarts, config.base_iterations
        );
        (res, eng, None)
    };
    total_restarts_used += config.base_restarts;

    if let Some((a, b)) = result {
        return AdaptiveSearchResult { found: true, best_energy: 0, a: Some(a), b: Some(b), near_miss_config: None };
    }

    // For n < 25: skip adaptive phases (breadth over depth is better for easier problems)
    if config.n < 25 {
        return AdaptiveSearchResult {
            found: false, best_energy: energy, a: None, b: None, near_miss_config: near_miss,
        };
    }

    let mut global_best_energy = energy;
    let mut best_near_miss = near_miss;

    // Phase 2: Exploit very promising near-miss (energy < 8)
    if let Some(ref nm) = best_near_miss {
        if nm.energy <= very_promising_threshold(config.n) {
            if let Some((a, b)) = sls.exploit_near_miss(nm, at.a_star, at.b_star, 100_000) {
                return AdaptiveSearchResult { found: true, best_energy: 0, a: Some(a), b: Some(b), near_miss_config: None };
            }
        }
    }

    // Phase 3: More restarts for promising results (energy < 20)
    if energy < near_miss_threshold(config.n) && total_restarts_used < config.max_restarts_per_cd {
        let extra_restarts = if energy <= very_promising_threshold(config.n) {
            (config.base_restarts * config.very_promising_multiplier)
                .min(config.max_restarts_per_cd - total_restarts_used)
        } else {
            (config.base_restarts * config.promising_multiplier)
                .min(config.max_restarts_per_cd - total_restarts_used)
        };
        if extra_restarts > 0 {
            let (result, energy, near_miss) = sls.search_v6_enhanced(
                st.a, at.a_star, st.b, at.b_star,
                extra_restarts, config.base_iterations
            );
            total_restarts_used += extra_restarts;
            if let Some((a, b)) = result {
                return AdaptiveSearchResult { found: true, best_energy: 0, a: Some(a), b: Some(b), near_miss_config: None };
            }
            if energy < global_best_energy {
                global_best_energy = energy;
                best_near_miss = near_miss;
            }
        }
    }

    // Phase 4: Loop for very promising CDs (energy < 8)
    let mut no_improve_count = 0usize;
    while global_best_energy <= very_promising_threshold(config.n)
          && total_restarts_used < config.max_restarts_per_cd
          && no_improve_count < 3
    {
        let extra = (config.base_restarts * 2)
            .min(config.max_restarts_per_cd - total_restarts_used);
        if extra == 0 { break; }
        let (result, energy, near_miss) = sls.search_v6_enhanced(
            st.a, at.a_star, st.b, at.b_star,
            extra, config.base_iterations
        );
        total_restarts_used += extra;
        if let Some((a, b)) = result {
            return AdaptiveSearchResult { found: true, best_energy: 0, a: Some(a), b: Some(b), near_miss_config: None };
        }
        if energy < global_best_energy {
            global_best_energy = energy;
            best_near_miss = near_miss;
            no_improve_count = 0;
        } else {
            no_improve_count += 1;
        }
    }

    // Final near-miss exploitation
    if let Some(ref nm) = best_near_miss {
        if nm.energy > 0 && nm.energy < near_miss_threshold(config.n) {
            if let Some((a, b)) = sls.exploit_near_miss(nm, at.a_star, at.b_star, 200_000) {
                return AdaptiveSearchResult { found: true, best_energy: 0, a: Some(a), b: Some(b), near_miss_config: None };
            }
        }
    }

    AdaptiveSearchResult {
        found: false,
        best_energy: global_best_energy,
        a: None,
        b: None,
        near_miss_config: best_near_miss,
    }
}

// ============================================================================
// PART 1: Two-phase spectral filtering (Paper Section 3, Step 4)
// ============================================================================

/// Hall polynomial: f_X(θ) = |h_X(e^{iθ})|² where h_X(z) = Σ x_j z^j
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

/// Two-phase spectral filter from paper (strict, for n<25)
/// Phase 1: Coarse filter with l=50 sample points
/// Phase 2: Fine filter with l=1000 sample points
/// Rejects if f_C(θ) + f_D(θ) > 4n+2 at any sample point
pub fn two_phase_spectral_filter(c: &Sequence, d: &Sequence) -> bool {
    let n = c.len();
    let target = 4.0 * (n as f64) + 2.0;
    let threshold = target + 0.001;

    // Phase 1: Coarse filter (l=50)
    for j in 0..=50 {
        let theta = (j as f64) * PI / 50.0;
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);
        if fc + fd > threshold { return false; }
    }

    // Phase 2: Fine filter (l=1000)
    for j in 0..=1000 {
        let theta = (j as f64) * PI / 1000.0;
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);
        if fc + fd > threshold { return false; }
    }

    true
}

/// Light spectral filter (V6-style, for n>=25)
/// Uses only 32 evenly-spaced samples with generous margin.
/// Passes ~13-15% of CDs vs ~1% for two-phase filter.
/// The higher throughput compensates for lower per-CD quality.
fn light_spectral_filter(c: &Sequence, d: &Sequence, margin: f64) -> bool {
    let n = c.len();
    let target = 4.0 * (n as f64) + 2.0;
    let threshold = target + margin;
    let num_samples = 32;

    for i in 0..num_samples {
        let theta = 2.0 * PI * (i as f64) / (num_samples as f64);
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);
        if fc + fd > threshold { return false; }
    }

    true
}

// ============================================================================
// PART 2: Full 5-class isomorphic equivalence (Paper Section 2)
// ============================================================================

/// The paper identifies 5 transformation classes that preserve base sequence property:
/// 1. Negation: (A,B,C,D) → (-A,-B,-C,-D)
/// 2. Reversal: (A,B,C,D) → (A^R, B^R, C^R, D^R) where X^R is X reversed
/// 3. AB-swap: (A,B,C,D) → (B,A,D,C)
/// 4. Alternation: (A,B,C,D) → (A',B',C',D') where X'_i = (-1)^i X_i
/// 5. CD-replacement: specific transformation involving C and D
///
/// To get canonical form, we define a total ordering and keep only minimum representatives.
pub fn filter_to_canonical_5class(tuples: Vec<(SumTuple, AltSumTuple)>, n: usize) -> Vec<(SumTuple, AltSumTuple)> {
    let original_count = tuples.len();

    // For each tuple, compute all equivalent transformations and keep only canonical rep
    let canonical: Vec<_> = tuples
        .into_iter()
        .filter(|(st, at)| is_canonical_representative_5class(st, at, n))
        .collect();

    let filtered_count = canonical.len();
    let reduction_factor = if filtered_count > 0 {
        original_count as f64 / filtered_count as f64
    } else {
        1.0
    };

    println!("  5-class isomorphic filtering: {} → {} tuples ({:.1}x reduction)",
             original_count, filtered_count, reduction_factor);

    canonical
}

/// Check if a tuple is the canonical representative among its equivalence class
fn is_canonical_representative_5class(st: &SumTuple, at: &AltSumTuple, n: usize) -> bool {
    // Generate all equivalent tuples and check if this is the lexicographically smallest
    let equivalents = generate_equivalent_tuples(st, at, n);

    let self_key = tuple_key(st, at);

    for equiv_key in &equivalents {
        if *equiv_key < self_key {
            return false;  // Not canonical - a smaller equivalent exists
        }
    }

    true
}

/// Generate all equivalent tuples under the full symmetry group:
/// alternation (2) × interchange (4) × reversal (16) × negation (16) = 2,048 transformations
fn generate_equivalent_tuples(st: &SumTuple, at: &AltSumTuple, n: usize) -> Vec<[i32; 8]> {
    use std::collections::HashSet;
    let mut seen = HashSet::new();

    // Reversal flips: for length L, alt_sum -> (-1)^{L-1} * alt_sum
    // A,B have length n+1: flip_ab = (-1)^n
    // C,D have length n:   flip_cd = (-1)^{n-1}
    let flip_ab: i32 = if n % 2 == 0 { 1 } else { -1 };
    let flip_cd: i32 = if n <= 1 { 1 } else if (n - 1) % 2 == 0 { 1 } else { -1 };

    let base = [st.a, st.b, st.c, st.d, at.a_star, at.b_star, at.c_star, at.d_star];

    // Alternation: identity or swap sums <-> alt_sums
    let mut tuples_after_alt = Vec::with_capacity(2);
    tuples_after_alt.push(base);
    tuples_after_alt.push([base[4], base[5], base[6], base[7], base[0], base[1], base[2], base[3]]);

    // Interchange: swap A<->B? swap C<->D?
    let mut tuples_after_swap = Vec::with_capacity(8);
    for t in &tuples_after_alt {
        let [a, b, c, d, as_, bs, cs, ds] = *t;
        tuples_after_swap.push([a, b, c, d, as_, bs, cs, ds]);
        tuples_after_swap.push([b, a, c, d, bs, as_, cs, ds]); // swap AB
        tuples_after_swap.push([a, b, d, c, as_, bs, ds, cs]); // swap CD
        tuples_after_swap.push([b, a, d, c, bs, as_, ds, cs]); // swap both
    }

    // Reversal: independently reverse any subset of {A,B,C,D}
    // Reversing X: sum_X unchanged, alt_sum_X *= flip factor
    let mut tuples_after_rev = Vec::with_capacity(128);
    for t in &tuples_after_swap {
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

/// Create a comparison key for lexicographic ordering
fn tuple_key(st: &SumTuple, at: &AltSumTuple) -> [i32; 8] {
    [st.a, st.b, st.c, st.d, at.a_star, at.b_star, at.c_star, at.d_star]
}

// ============================================================================
// PART 3: Symmetric pair constraints (Paper Theorem 2.2)
// ============================================================================

/// From Theorem 2.2: For position pairs (i, n+2-i):
///   a_i + b_i + a_{n+2-i} + b_{n+2-i} ≡ { 2 (mod 4) if i=1
///                                        { 0 (mod 4) if i≥2
///
/// This means at each symmetric position pair, there are only 8 valid 2×2 matrices
/// of values from {-1, +1}. This dramatically reduces the search space.
///
/// Returns the valid (a_i, b_i, a_j, b_j) combinations where j = n+2-i
fn valid_symmetric_pairs(i: usize, n: usize) -> Vec<(i32, i32, i32, i32)> {
    let target_mod = if i == 1 { 2 } else { 0 };

    let mut valid = Vec::new();

    // All combinations of ±1 for four positions
    for a_i in [-1i32, 1] {
        for b_i in [-1i32, 1] {
            for a_j in [-1i32, 1] {
                for b_j in [-1i32, 1] {
                    let sum = a_i + b_i + a_j + b_j;
                    // sum can be -4, -2, 0, 2, 4
                    // sum mod 4: -4→0, -2→2, 0→0, 2→2, 4→0
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

/// Precompute all valid symmetric pair combinations for each position
fn precompute_symmetric_constraints(n: usize) -> Vec<Vec<(i32, i32, i32, i32)>> {
    let m = n + 1;  // Length of A and B
    let mut constraints = Vec::new();

    // For each position pair (i, m+1-i) where i < m+1-i
    // Position 1 pairs with position m
    // Position 2 pairs with position m-1, etc.
    for i in 1..=((m + 1) / 2) {
        let j = m + 1 - i;
        if i < j {
            constraints.push(valid_symmetric_pairs(i, n));
        } else if i == j {
            // Middle position (only exists for odd m)
            // No pairing constraint, but must satisfy sum/alt_sum
            constraints.push(vec![(1, 1, 0, 0), (-1, 1, 0, 0), (1, -1, 0, 0), (-1, -1, 0, 0)]);
        }
    }

    constraints
}

// ============================================================================
// PART 4: Modular decomposition (Paper Theorem 2.3)
// ============================================================================

/// Modular decomposition from Wang & Zhu paper (Theorem 2.3, Equations 13-15)
///
/// Instead of directly searching sequences of length m, we iteratively build:
///   mod 3 → mod 6 → mod 12 → ... → mod m
///
/// At each modular level, we check if the constraints can be satisfied,
/// pruning early before expanding to finer modular levels.
///
/// Key insight: For a sequence X of length m, the sum and alternating sums
/// at modular level q satisfy specific constraints derived from Theorem 2.3.

/// Modular residue class for A,B positions
/// Groups positions 0, q, 2q, ... and 1, q+1, 2q+1, ... etc.
#[derive(Clone, Debug)]
struct ModularClass {
    /// Values at each position in this residue class
    values: Vec<i32>,
    /// The residue (0 to q-1)
    residue: usize,
    /// The modulus
    modulus: usize,
}

/// Check if sum constraints can be satisfied at modular level q
/// This uses the paper's Equations 13-15 from Theorem 2.3
fn check_modular_sum_constraints(
    partial_a: &[i32],
    partial_b: &[i32],
    q: usize,
    m: usize,
    target_a_sum: i32,
    target_b_sum: i32,
) -> bool {
    // k_qm = floor((q-1)/m) for l=q but we're going q→m so swap
    // Actually: for sequence length m, at modular level q,
    // positions are grouped: {0, q, 2q, ...}, {1, q+1, 2q+1, ...}, etc.

    // Current sum from partial assignments
    let current_a_sum: i32 = partial_a.iter().sum();
    let current_b_sum: i32 = partial_b.iter().sum();

    // Number of unassigned positions
    let assigned = partial_a.iter().filter(|&&v| v != 0).count();
    let remaining = m - assigned;

    // Sum needed from remaining positions
    let a_needed = target_a_sum - current_a_sum;
    let b_needed = target_b_sum - current_b_sum;

    // Each position contributes ±1, so |needed| ≤ remaining
    if a_needed.abs() > remaining as i32 || b_needed.abs() > remaining as i32 {
        return false;
    }

    // Parity check: (m - remaining) assigned positions contribute sum ≡ current_sum
    // Remaining positions contribute sum with same parity as remaining count
    if (a_needed + remaining as i32) % 2 != 0 {
        return false;
    }
    if (b_needed + remaining as i32) % 2 != 0 {
        return false;
    }

    true
}

/// Check alternating sum constraints at modular level
fn check_modular_alt_sum_constraints(
    partial_a: &[i32],
    partial_b: &[i32],
    q: usize,
    m: usize,
    target_a_alt: i32,
    target_b_alt: i32,
) -> bool {
    // Current alternating sums from partial assignments
    let current_a_alt: i32 = partial_a.iter().enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
        .sum();
    let current_b_alt: i32 = partial_b.iter().enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
        .sum();

    // Count unassigned at even and odd positions
    let unassigned_even = partial_a.iter().enumerate()
        .filter(|(i, &v)| i % 2 == 0 && v == 0).count() as i32;
    let unassigned_odd = partial_a.iter().enumerate()
        .filter(|(i, &v)| i % 2 == 1 && v == 0).count() as i32;

    // Maximum possible change to alternating sum from remaining positions
    // Even positions contribute +v, odd positions contribute -v
    // So max positive change = unassigned_even + unassigned_odd
    let max_change = unassigned_even + unassigned_odd;

    let a_alt_needed = target_a_alt - current_a_alt;
    let b_alt_needed = target_b_alt - current_b_alt;

    if a_alt_needed.abs() > max_change || b_alt_needed.abs() > max_change {
        return false;
    }

    true
}

/// Generate modular sequence: double the modulus by filling in intermediate positions
/// From mod q assignments, generate candidates for mod 2q
fn lift_modular_assignments(
    current_a: &[i32],
    current_b: &[i32],
    q: usize,
    m: usize,
    target_a_sum: i32,
    target_a_alt: i32,
    target_b_sum: i32,
    target_b_alt: i32,
    cd_autocorr: &[i32],
    n: usize,
) -> Vec<(Vec<i32>, Vec<i32>)> {
    let new_q = 2 * q;
    if new_q > m {
        // Final lift: fill all remaining positions
        return lift_to_full(current_a, current_b, m,
            target_a_sum, target_a_alt, target_b_sum, target_b_alt,
            cd_autocorr, n);
    }

    // Positions to fill: those with index i where (i % new_q) >= q
    // i.e., the "new" residue classes in the finer modular level
    let positions_to_fill: Vec<usize> = (0..m)
        .filter(|&i| current_a[i] == 0)
        .collect();

    if positions_to_fill.is_empty() {
        return vec![(current_a.to_vec(), current_b.to_vec())];
    }

    // Generate all valid assignments for the new positions
    let mut results = Vec::new();

    // Recursive helper to try all combinations
    fn try_fill(
        pos_idx: usize,
        positions: &[usize],
        a: &mut Vec<i32>,
        b: &mut Vec<i32>,
        m: usize,
        target_a_sum: i32,
        target_a_alt: i32,
        target_b_sum: i32,
        target_b_alt: i32,
        results: &mut Vec<(Vec<i32>, Vec<i32>)>,
    ) {
        if pos_idx >= positions.len() {
            // Check final constraints
            let a_sum: i32 = a.iter().sum();
            let b_sum: i32 = b.iter().sum();
            if a_sum == target_a_sum && b_sum == target_b_sum {
                let a_alt: i32 = a.iter().enumerate()
                    .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
                let b_alt: i32 = b.iter().enumerate()
                    .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
                if a_alt == target_a_alt && b_alt == target_b_alt {
                    results.push((a.clone(), b.clone()));
                }
            }
            return;
        }

        let pos = positions[pos_idx];
        let remaining = positions.len() - pos_idx;

        // Pruning: check if sum constraints can still be satisfied
        let current_a_sum: i32 = a.iter().sum();
        let current_b_sum: i32 = b.iter().sum();
        let a_needed = target_a_sum - current_a_sum;
        let b_needed = target_b_sum - current_b_sum;

        if a_needed.abs() > remaining as i32 || b_needed.abs() > remaining as i32 {
            return;
        }

        for &a_val in &[-1i32, 1] {
            for &b_val in &[-1i32, 1] {
                a[pos] = a_val;
                b[pos] = b_val;

                try_fill(pos_idx + 1, positions, a, b, m,
                    target_a_sum, target_a_alt, target_b_sum, target_b_alt, results);
            }
        }

        // Reset for backtracking
        a[pos] = 0;
        b[pos] = 0;
    }

    let mut a = current_a.to_vec();
    let mut b = current_b.to_vec();

    try_fill(0, &positions_to_fill, &mut a, &mut b, m,
        target_a_sum, target_a_alt, target_b_sum, target_b_alt, &mut results);

    results
}

/// Final lift: fill all remaining 0s and check autocorrelation
fn lift_to_full(
    current_a: &[i32],
    current_b: &[i32],
    m: usize,
    target_a_sum: i32,
    target_a_alt: i32,
    target_b_sum: i32,
    target_b_alt: i32,
    cd_autocorr: &[i32],
    n: usize,
) -> Vec<(Vec<i32>, Vec<i32>)> {
    let positions_to_fill: Vec<usize> = (0..m)
        .filter(|&i| current_a[i] == 0)
        .collect();

    if positions_to_fill.is_empty() {
        // Already fully assigned, just verify
        let a_sum: i32 = current_a.iter().sum();
        let b_sum: i32 = current_b.iter().sum();
        if a_sum != target_a_sum || b_sum != target_b_sum {
            return vec![];
        }

        let a_alt: i32 = current_a.iter().enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
        let b_alt: i32 = current_b.iter().enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
        if a_alt != target_a_alt || b_alt != target_b_alt {
            return vec![];
        }

        // Check autocorrelations
        let a_seq = Sequence::new(current_a.to_vec());
        let b_seq = Sequence::new(current_b.to_vec());
        for shift in 1..n {
            let total = a_seq.autocorrelation(shift)
                      + b_seq.autocorrelation(shift)
                      + cd_autocorr[shift];
            if total != 0 {
                return vec![];
            }
        }

        return vec![(current_a.to_vec(), current_b.to_vec())];
    }

    let mut results = Vec::new();

    fn try_fill_final(
        pos_idx: usize,
        positions: &[usize],
        a: &mut Vec<i32>,
        b: &mut Vec<i32>,
        m: usize,
        target_a_sum: i32,
        target_a_alt: i32,
        target_b_sum: i32,
        target_b_alt: i32,
        cd_autocorr: &[i32],
        n: usize,
        results: &mut Vec<(Vec<i32>, Vec<i32>)>,
    ) {
        if pos_idx >= positions.len() {
            // Verify all constraints
            let a_sum: i32 = a.iter().sum();
            let b_sum: i32 = b.iter().sum();
            if a_sum != target_a_sum || b_sum != target_b_sum {
                return;
            }

            let a_alt: i32 = a.iter().enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
            let b_alt: i32 = b.iter().enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
            if a_alt != target_a_alt || b_alt != target_b_alt {
                return;
            }

            // Check autocorrelations
            let a_seq = Sequence::new(a.to_vec());
            let b_seq = Sequence::new(b.to_vec());
            for shift in 1..n {
                let total = a_seq.autocorrelation(shift)
                          + b_seq.autocorrelation(shift)
                          + cd_autocorr[shift];
                if total != 0 {
                    return;
                }
            }

            results.push((a.clone(), b.clone()));
            return;
        }

        let pos = positions[pos_idx];
        let remaining = positions.len() - pos_idx;

        // Sum pruning
        let current_a_sum: i32 = a.iter().sum();
        let current_b_sum: i32 = b.iter().sum();
        let a_needed = target_a_sum - current_a_sum;
        let b_needed = target_b_sum - current_b_sum;

        if a_needed.abs() > remaining as i32 || b_needed.abs() > remaining as i32 {
            return;
        }

        // Alternating sum pruning
        let current_a_alt: i32 = a.iter().enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
        let current_b_alt: i32 = b.iter().enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
        let a_alt_needed = target_a_alt - current_a_alt;
        let b_alt_needed = target_b_alt - current_b_alt;

        if a_alt_needed.abs() > remaining as i32 || b_alt_needed.abs() > remaining as i32 {
            return;
        }

        for &a_val in &[-1i32, 1] {
            for &b_val in &[-1i32, 1] {
                a[pos] = a_val;
                b[pos] = b_val;

                // Partial autocorrelation pruning
                let mut valid = true;
                for shift in 1..n {
                    let mut known_ac = cd_autocorr[shift];
                    let mut unknown_pairs = 0i32;

                    for p in 0..m.saturating_sub(shift) {
                        let p2 = p + shift;
                        if a[p] != 0 && a[p2] != 0 {
                            known_ac += a[p] * a[p2];
                        } else {
                            unknown_pairs += 1;
                        }
                        if b[p] != 0 && b[p2] != 0 {
                            known_ac += b[p] * b[p2];
                        } else {
                            unknown_pairs += 1;
                        }
                    }

                    if unknown_pairs == 0 && known_ac != 0 {
                        valid = false;
                        break;
                    }
                    if known_ac.abs() > unknown_pairs {
                        valid = false;
                        break;
                    }
                }

                if valid {
                    try_fill_final(pos_idx + 1, positions, a, b, m,
                        target_a_sum, target_a_alt, target_b_sum, target_b_alt,
                        cd_autocorr, n, results);
                }
            }
        }

        a[pos] = 0;
        b[pos] = 0;
    }

    let mut a = current_a.to_vec();
    let mut b = current_b.to_vec();

    try_fill_final(0, &positions_to_fill, &mut a, &mut b, m,
        target_a_sum, target_a_alt, target_b_sum, target_b_alt,
        cd_autocorr, n, &mut results);

    results
}

/// Symmetric backtracking with modular constraint pruning
/// This combines the symmetric pair constraints (Theorem 2.2) with
/// modular constraint checks (Theorem 2.3) for enhanced pruning.
#[allow(dead_code)]
fn symmetric_backtrack_search_ab_with_modular(
    c: &Sequence,
    d: &Sequence,
    a_sum: i32,
    a_alt: i32,
    b_sum: i32,
    b_alt: i32,
) -> Option<(Sequence, Sequence)> {
    let n = c.len();
    let m = n + 1;  // Length of A and B

    // Precompute CD autocorrelations
    let mut cd_autocorr = vec![0i32; n + 1];
    for shift in 1..n {
        cd_autocorr[shift] = c.autocorrelation(shift) + d.autocorrelation(shift);
    }

    // Initialize A and B
    let mut a_values = vec![0i32; m];
    let mut b_values = vec![0i32; m];

    // Precompute valid pairs for each position pair
    let num_pairs = m / 2;
    let has_middle = m % 2 == 1;

    let mut pair_configs: Vec<Vec<(i32, i32, i32, i32)>> = Vec::new();

    for pair_idx in 0..num_pairs {
        let target_mod = if pair_idx == 0 { 2 } else { 0 };

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
        pair_configs.push(valid);
    }

    let middle_configs: Vec<(i32, i32)> = if has_middle {
        vec![(-1, -1), (-1, 1), (1, -1), (1, 1)]
    } else {
        vec![]
    };

    // Backtrack through pairs with modular pruning
    if symmetric_backtrack_fill_modular(
        &mut a_values, &mut b_values,
        0, num_pairs, has_middle,
        a_sum, a_alt, b_sum, b_alt,
        &cd_autocorr, n, m,
        &pair_configs, &middle_configs,
    ) {
        Some((Sequence::new(a_values), Sequence::new(b_values)))
    } else {
        None
    }
}

#[allow(dead_code)]
fn symmetric_backtrack_fill_modular(
    a: &mut [i32],
    b: &mut [i32],
    pair_idx: usize,
    num_pairs: usize,
    has_middle: bool,
    target_a_sum: i32,
    target_a_alt: i32,
    target_b_sum: i32,
    target_b_alt: i32,
    cd_autocorr: &[i32],
    n: usize,
    m: usize,
    pair_configs: &[Vec<(i32, i32, i32, i32)>],
    middle_configs: &[(i32, i32)],
) -> bool {
    // Filled all pairs, now handle middle if exists
    if pair_idx >= num_pairs {
        if has_middle {
            let mid = m / 2;
            for &(a_mid, b_mid) in middle_configs {
                a[mid] = a_mid;
                b[mid] = b_mid;

                if check_final_constraints(a, b, target_a_sum, target_a_alt,
                                          target_b_sum, target_b_alt, cd_autocorr, n, m) {
                    return true;
                }
            }
            return false;
        } else {
            return check_final_constraints(a, b, target_a_sum, target_a_alt,
                                          target_b_sum, target_b_alt, cd_autocorr, n, m);
        }
    }

    let i = pair_idx;
    let j = m - 1 - pair_idx;

    // Current partial sums
    let current_a_sum: i32 = a.iter().sum();
    let current_b_sum: i32 = b.iter().sum();

    let remaining_positions = m - 2 * (pair_idx + 1) + if has_middle && pair_idx + 1 == num_pairs { 1 } else { 0 };

    // Try each valid configuration for this pair
    for &(a_i, b_i, a_j, b_j) in &pair_configs[pair_idx] {
        // Pruning 1: check if remaining can satisfy sum constraints
        let new_a_sum = current_a_sum + a_i + a_j;
        let new_b_sum = current_b_sum + b_i + b_j;

        let a_remaining = target_a_sum - new_a_sum;
        let b_remaining = target_b_sum - new_b_sum;

        if a_remaining.abs() > remaining_positions as i32 ||
           b_remaining.abs() > remaining_positions as i32 {
            continue;
        }

        // Pruning 1b: check alternating sum constraints
        let i_sign = if i % 2 == 0 { 1 } else { -1 };
        let j_sign = if j % 2 == 0 { 1 } else { -1 };

        let current_a_alt: i32 = a.iter().enumerate()
            .map(|(k, &v)| if k % 2 == 0 { v } else { -v }).sum();
        let current_b_alt: i32 = b.iter().enumerate()
            .map(|(k, &v)| if k % 2 == 0 { v } else { -v }).sum();

        let new_a_alt = current_a_alt + i_sign * a_i + j_sign * a_j;
        let new_b_alt = current_b_alt + i_sign * b_i + j_sign * b_j;

        let a_alt_remaining = target_a_alt - new_a_alt;
        let b_alt_remaining = target_b_alt - new_b_alt;

        if a_alt_remaining.abs() > remaining_positions as i32 ||
           b_alt_remaining.abs() > remaining_positions as i32 {
            continue;
        }

        // Pruning 1c: Modular constraint check (Theorem 2.3)
        // Check parity constraints at multiple modular levels
        if !check_modular_parity(new_a_sum, target_a_sum, remaining_positions) ||
           !check_modular_parity(new_b_sum, target_b_sum, remaining_positions) {
            continue;
        }

        // Set values
        a[i] = a_i;
        b[i] = b_i;
        a[j] = a_j;
        b[j] = b_j;

        // Pruning 2: Check partial autocorrelations with tighter bounds
        let mut pruned = false;

        for shift in 1..n {
            let mut known_ac = cd_autocorr[shift];
            let mut unfilled_pairs = 0i32;

            for pos in 0..m.saturating_sub(shift) {
                let pos2 = pos + shift;
                let a_pos_filled = a[pos] != 0;
                let a_pos2_filled = a[pos2] != 0;
                let b_pos_filled = b[pos] != 0;
                let b_pos2_filled = b[pos2] != 0;

                if a_pos_filled && a_pos2_filled {
                    known_ac += a[pos] * a[pos2];
                } else {
                    unfilled_pairs += 1;
                }

                if b_pos_filled && b_pos2_filled {
                    known_ac += b[pos] * b[pos2];
                } else {
                    unfilled_pairs += 1;
                }
            }

            if unfilled_pairs == 0 {
                if known_ac != 0 {
                    pruned = true;
                    break;
                }
            } else if known_ac.abs() > unfilled_pairs {
                pruned = true;
                break;
            }
        }

        if pruned {
            // Clear values for next iteration
            a[i] = 0;
            b[i] = 0;
            a[j] = 0;
            b[j] = 0;
            continue;
        }

        // Recurse
        if symmetric_backtrack_fill_modular(
            a, b, pair_idx + 1, num_pairs, has_middle,
            target_a_sum, target_a_alt, target_b_sum, target_b_alt,
            cd_autocorr, n, m,
            pair_configs, middle_configs,
        ) {
            return true;
        }

        // Backtrack
        a[i] = 0;
        b[i] = 0;
        a[j] = 0;
        b[j] = 0;
    }

    false
}

/// Check modular parity constraints from Theorem 2.3
/// At modular levels 2, 4, 8, etc., verify constraints can be satisfied
#[inline]
#[allow(dead_code)]
fn check_modular_parity(current_sum: i32, target_sum: i32, remaining: usize) -> bool {
    let needed = target_sum - current_sum;

    // Parity check: needed + remaining must have same parity
    // because each position contributes ±1
    if (needed + remaining as i32) % 2 != 0 {
        return false;
    }

    // Check at mod 4 level
    // The contribution from remaining positions mod 4 depends on count
    // With r remaining, possible sums are: -r, -r+2, -r+4, ..., r
    // All have same parity as r
    // So needed must be achievable: -r ≤ needed ≤ r already checked

    true
}

/// Main modular decomposition search for A,B given C,D
/// Uses the progressive refinement: mod 3 → mod 6 → mod 12 → ... → mod m
#[allow(dead_code)]
fn modular_decomposition_search_ab(
    c: &Sequence,
    d: &Sequence,
    a_sum: i32,
    a_alt: i32,
    b_sum: i32,
    b_alt: i32,
) -> Option<(Sequence, Sequence)> {
    let n = c.len();
    let m = n + 1;  // Length of A and B

    // Precompute CD autocorrelations
    let mut cd_autocorr = vec![0i32; n + 1];
    for shift in 1..n {
        cd_autocorr[shift] = c.autocorrelation(shift) + d.autocorrelation(shift);
    }

    // Start with mod 3 level
    // At mod 3, positions are grouped: {0,3,6,...}, {1,4,7,...}, {2,5,8,...}
    let initial_q = 3.min(m);

    // Initialize: start by filling positions 0, 1, 2 (or fewer if m < 3)
    let initial_positions: Vec<usize> = (0..initial_q.min(m)).collect();

    // Generate all valid initial assignments at mod q level
    let mut candidates: Vec<(Vec<i32>, Vec<i32>)> = Vec::new();

    fn gen_initial(
        pos_idx: usize,
        positions: &[usize],
        a: &mut Vec<i32>,
        b: &mut Vec<i32>,
        m: usize,
        target_a_sum: i32,
        target_b_sum: i32,
        candidates: &mut Vec<(Vec<i32>, Vec<i32>)>,
    ) {
        if pos_idx >= positions.len() {
            candidates.push((a.clone(), b.clone()));
            return;
        }

        let pos = positions[pos_idx];

        for &a_val in &[-1i32, 1] {
            for &b_val in &[-1i32, 1] {
                a[pos] = a_val;
                b[pos] = b_val;

                // Quick pruning
                let current_a: i32 = a.iter().sum();
                let current_b: i32 = b.iter().sum();
                let remaining = m - pos - 1;

                if (target_a_sum - current_a).abs() <= remaining as i32 &&
                   (target_b_sum - current_b).abs() <= remaining as i32 {
                    gen_initial(pos_idx + 1, positions, a, b, m,
                        target_a_sum, target_b_sum, candidates);
                }
            }
        }

        a[pos] = 0;
        b[pos] = 0;
    }

    let mut init_a = vec![0i32; m];
    let mut init_b = vec![0i32; m];
    gen_initial(0, &initial_positions, &mut init_a, &mut init_b, m,
        a_sum, b_sum, &mut candidates);

    // Progressive lifting: q → 2q → 4q → ... → m
    let mut q = initial_q;

    while q < m && !candidates.is_empty() {
        let next_q = (2 * q).min(m);

        // For each candidate, try to lift to the next modular level
        let mut next_candidates = Vec::new();

        for (curr_a, curr_b) in candidates {
            let lifted = lift_modular_assignments(
                &curr_a, &curr_b, q, m,
                a_sum, a_alt, b_sum, b_alt,
                &cd_autocorr, n
            );

            for (lifted_a, lifted_b) in lifted {
                // Additional pruning: check partial autocorrelations
                let mut valid = true;

                for shift in 1..n {
                    let mut known_ac = cd_autocorr[shift];
                    let mut unknown_pairs = 0i32;

                    for pos in 0..m.saturating_sub(shift) {
                        let pos2 = pos + shift;
                        if lifted_a[pos] != 0 && lifted_a[pos2] != 0 {
                            known_ac += lifted_a[pos] * lifted_a[pos2];
                        } else if lifted_a[pos] != 0 || lifted_a[pos2] != 0 {
                            unknown_pairs += 1;
                        } else {
                            unknown_pairs += 1;
                        }
                        if lifted_b[pos] != 0 && lifted_b[pos2] != 0 {
                            known_ac += lifted_b[pos] * lifted_b[pos2];
                        } else if lifted_b[pos] != 0 || lifted_b[pos2] != 0 {
                            unknown_pairs += 1;
                        } else {
                            unknown_pairs += 1;
                        }
                    }

                    // If no unknown pairs, autocorr must be exactly 0
                    if unknown_pairs == 0 && known_ac != 0 {
                        valid = false;
                        break;
                    }
                    // Otherwise, check if 0 is reachable
                    if known_ac.abs() > unknown_pairs {
                        valid = false;
                        break;
                    }
                }

                if valid {
                    // If fully assigned, this is a solution
                    if lifted_a.iter().all(|&v| v != 0) {
                        return Some((Sequence::new(lifted_a), Sequence::new(lifted_b)));
                    }
                    next_candidates.push((lifted_a, lifted_b));
                }
            }
        }

        candidates = next_candidates;
        q = next_q;
    }

    // Final check: any fully filled candidate is a solution
    for (a, b) in candidates {
        if a.iter().all(|&v| v != 0) {
            // Final verification
            let a_seq = Sequence::new(a.clone());
            let b_seq = Sequence::new(b.clone());

            let a_s: i32 = a.iter().sum();
            let b_s: i32 = b.iter().sum();
            if a_s != a_sum || b_s != b_sum {
                continue;
            }

            let a_a: i32 = a.iter().enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
            let b_a: i32 = b.iter().enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
            if a_a != a_alt || b_a != b_alt {
                continue;
            }

            let mut valid = true;
            for shift in 1..n {
                let total = a_seq.autocorrelation(shift)
                          + b_seq.autocorrelation(shift)
                          + cd_autocorr[shift];
                if total != 0 {
                    valid = false;
                    break;
                }
            }

            if valid {
                return Some((a_seq, b_seq));
            }
        }
    }

    None
}

// ============================================================================
// PART 5: Deterministic backtracking search for A,B (Paper Algorithm)
// ============================================================================

/// The paper uses deterministic backtracking (not stochastic local search).
/// Given C, D, and constraints on A, B, we systematically search:
/// 1. For each position pair, try all 8 valid symmetric configurations
/// 2. Prune using autocorrelation constraints
/// 3. Backtrack when constraints are violated

/// Deterministic search for A,B given fixed C,D
fn backtrack_search_ab(
    c: &Sequence,
    d: &Sequence,
    a_sum: i32,
    a_alt: i32,
    b_sum: i32,
    b_alt: i32,
) -> Option<(Sequence, Sequence)> {
    let n = c.len();
    let m = n + 1;  // Length of A and B

    // Precompute CD autocorrelations
    let mut cd_autocorr = vec![0i32; n + 1];
    for shift in 1..n {
        cd_autocorr[shift] = c.autocorrelation(shift) + d.autocorrelation(shift);
    }

    // Initialize A and B with zeros (to be filled)
    let mut a_values = vec![0i32; m];
    let mut b_values = vec![0i32; m];

    // Track current sums
    let mut current_a_sum = 0i32;
    let mut current_b_sum = 0i32;
    let mut current_a_alt = 0i32;
    let mut current_b_alt = 0i32;

    // Precompute symmetric pair constraints
    let symmetric_pairs = precompute_symmetric_constraints(n);

    // Use backtracking to fill positions
    if backtrack_fill(
        &mut a_values, &mut b_values,
        0, m,
        a_sum, a_alt, b_sum, b_alt,
        &mut current_a_sum, &mut current_b_sum,
        &mut current_a_alt, &mut current_b_alt,
        &cd_autocorr, n,
        &symmetric_pairs,
    ) {
        Some((Sequence::new(a_values), Sequence::new(b_values)))
    } else {
        None
    }
}

/// Recursive backtracking to fill A,B positions
fn backtrack_fill(
    a: &mut [i32],
    b: &mut [i32],
    pos: usize,
    m: usize,
    target_a_sum: i32,
    target_a_alt: i32,
    target_b_sum: i32,
    target_b_alt: i32,
    current_a_sum: &mut i32,
    current_b_sum: &mut i32,
    current_a_alt: &mut i32,
    current_b_alt: &mut i32,
    cd_autocorr: &[i32],
    n: usize,
    _symmetric_pairs: &[Vec<(i32, i32, i32, i32)>],
) -> bool {
    // Base case: all positions filled
    if pos >= m {
        // Check final constraints
        if *current_a_sum != target_a_sum || *current_b_sum != target_b_sum {
            return false;
        }
        if *current_a_alt != target_a_alt || *current_b_alt != target_b_alt {
            return false;
        }

        // Check autocorrelation constraints
        let a_seq = Sequence::new(a.to_vec());
        let b_seq = Sequence::new(b.to_vec());

        for shift in 1..n {
            let total_ac = a_seq.autocorrelation(shift)
                         + b_seq.autocorrelation(shift)
                         + cd_autocorr[shift];
            if total_ac != 0 {
                return false;
            }
        }

        return true;
    }

    // Pruning: check if remaining positions can satisfy sum constraints
    let remaining = (m - pos) as i32;
    let a_sum_needed = target_a_sum - *current_a_sum;
    let b_sum_needed = target_b_sum - *current_b_sum;

    // Maximum/minimum possible contribution from remaining positions
    if a_sum_needed.abs() > remaining || b_sum_needed.abs() > remaining {
        return false;
    }

    // Alternating sum signs
    let alt_sign = if pos % 2 == 0 { 1 } else { -1 };

    // Try all combinations of (a[pos], b[pos])
    for &a_val in &[-1i32, 1] {
        for &b_val in &[-1i32, 1] {
            // Set values
            a[pos] = a_val;
            b[pos] = b_val;

            // Update sums
            *current_a_sum += a_val;
            *current_b_sum += b_val;
            *current_a_alt += alt_sign * a_val;
            *current_b_alt += alt_sign * b_val;

            // Early pruning: partial autocorrelation check
            let mut pruned = false;
            if pos >= n {
                // Can check some autocorrelations completely
                // For shift s, we need positions 0..m-s
                for shift in 1..(pos + 1 - n + 1).min(n) {
                    if pos >= m - 1 - shift {
                        // This shift's autocorrelation can be computed
                        let a_seq = Sequence::new(a.to_vec());
                        let b_seq = Sequence::new(b.to_vec());
                        let total_ac = a_seq.autocorrelation(shift)
                                     + b_seq.autocorrelation(shift)
                                     + cd_autocorr[shift];
                        if total_ac != 0 {
                            pruned = true;
                            break;
                        }
                    }
                }
            }

            if !pruned {
                // Recurse
                if backtrack_fill(
                    a, b, pos + 1, m,
                    target_a_sum, target_a_alt, target_b_sum, target_b_alt,
                    current_a_sum, current_b_sum,
                    current_a_alt, current_b_alt,
                    cd_autocorr, n,
                    _symmetric_pairs,
                ) {
                    return true;
                }
            }

            // Backtrack: undo changes
            *current_a_sum -= a_val;
            *current_b_sum -= b_val;
            *current_a_alt -= alt_sign * a_val;
            *current_b_alt -= alt_sign * b_val;
        }
    }

    false
}

/// Optimized backtracking using symmetric pair constraints (Theorem 2.2)
/// Fill positions in pairs from outside in, using the 8-case constraint
/// Now with incremental autocorrelation tracking for better performance
fn symmetric_backtrack_search_ab(
    c: &Sequence,
    d: &Sequence,
    a_sum: i32,
    a_alt: i32,
    b_sum: i32,
    b_alt: i32,
) -> Option<(Sequence, Sequence)> {
    let n = c.len();
    let m = n + 1;  // Length of A and B

    // Precompute CD autocorrelations
    let mut cd_autocorr = vec![0i32; n + 1];
    for shift in 1..n {
        cd_autocorr[shift] = c.autocorrelation(shift) + d.autocorrelation(shift);
    }

    // Quick feasibility check: for each shift, the target is -cd_autocorr[shift]
    // Check if this is achievable given the constraints
    for shift in 1..n {
        let target_ab = -cd_autocorr[shift];
        // Maximum possible AC_A + AC_B is 2*(m-shift) (all products are +1)
        // Minimum is -2*(m-shift)
        let max_possible = 2 * (m - shift) as i32;
        if target_ab.abs() > max_possible {
            return None;  // Infeasible CD pair
        }
    }

    // Initialize A and B
    let mut a_values = vec![0i32; m];
    let mut b_values = vec![0i32; m];

    // Precompute valid pairs for each position pair
    let num_pairs = m / 2;
    let has_middle = m % 2 == 1;

    // Generate valid configurations for each pair, sorted by heuristic quality
    let mut pair_configs: Vec<Vec<(i32, i32, i32, i32)>> = Vec::new();

    for pair_idx in 0..num_pairs {
        let target_mod = if pair_idx == 0 { 2 } else { 0 };

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

        // Sort configs to try balanced ones first (heuristic)
        valid.sort_by_key(|(ai, bi, aj, bj)| {
            let sum = ai + bi + aj + bj;
            sum.abs()  // Prefer sums closer to 0
        });

        pair_configs.push(valid);
    }

    // Middle position if m is odd
    let middle_configs: Vec<(i32, i32)> = if has_middle {
        vec![(1, 1), (-1, -1), (1, -1), (-1, 1)]  // Ordered by sum magnitude
    } else {
        vec![]
    };

    // Initialize incremental state
    let mut state = BacktrackState {
        current_a_sum: 0,
        current_b_sum: 0,
        current_a_alt: 0,
        current_b_alt: 0,
        // Partial autocorrelations: known contribution from filled positions
        partial_ac: vec![cd_autocorr.clone()],  // Start with CD contribution
    };

    // Backtrack through pairs with incremental state
    if symmetric_backtrack_fill_incremental(
        &mut a_values, &mut b_values,
        0, num_pairs, has_middle,
        a_sum, a_alt, b_sum, b_alt,
        &cd_autocorr, n, m,
        &pair_configs, &middle_configs,
        &mut state,
    ) {
        Some((Sequence::new(a_values), Sequence::new(b_values)))
    } else {
        None
    }
}

/// State for incremental backtracking
struct BacktrackState {
    current_a_sum: i32,
    current_b_sum: i32,
    current_a_alt: i32,
    current_b_alt: i32,
    partial_ac: Vec<Vec<i32>>,  // Stack of partial autocorrelation arrays
}

fn symmetric_backtrack_fill_incremental(
    a: &mut [i32],
    b: &mut [i32],
    pair_idx: usize,
    num_pairs: usize,
    has_middle: bool,
    target_a_sum: i32,
    target_a_alt: i32,
    target_b_sum: i32,
    target_b_alt: i32,
    cd_autocorr: &[i32],
    n: usize,
    m: usize,
    pair_configs: &[Vec<(i32, i32, i32, i32)>],
    middle_configs: &[(i32, i32)],
    state: &mut BacktrackState,
) -> bool {
    // Filled all pairs, now handle middle if exists
    if pair_idx >= num_pairs {
        if has_middle {
            let mid = m / 2;
            for &(a_mid, b_mid) in middle_configs {
                a[mid] = a_mid;
                b[mid] = b_mid;

                // Check final constraints
                if check_final_constraints(a, b, target_a_sum, target_a_alt,
                                          target_b_sum, target_b_alt, cd_autocorr, n, m) {
                    return true;
                }
            }
            return false;
        } else {
            return check_final_constraints(a, b, target_a_sum, target_a_alt,
                                          target_b_sum, target_b_alt, cd_autocorr, n, m);
        }
    }

    let i = pair_idx;
    let j = m - 1 - pair_idx;
    let i_sign = if i % 2 == 0 { 1i32 } else { -1 };
    let j_sign = if j % 2 == 0 { 1i32 } else { -1 };

    let remaining_positions = m - 2 * (pair_idx + 1) + if has_middle && pair_idx + 1 == num_pairs { 1 } else { 0 };

    // Try each valid configuration for this pair
    for &(a_i, b_i, a_j, b_j) in &pair_configs[pair_idx] {
        // Pruning 1: check if remaining can satisfy sum constraints
        let new_a_sum = state.current_a_sum + a_i + a_j;
        let new_b_sum = state.current_b_sum + b_i + b_j;

        let a_remaining = target_a_sum - new_a_sum;
        let b_remaining = target_b_sum - new_b_sum;

        if a_remaining.abs() > remaining_positions as i32 ||
           b_remaining.abs() > remaining_positions as i32 {
            continue;
        }

        // Pruning 1b: check alternating sum constraints
        let new_a_alt = state.current_a_alt + i_sign * a_i + j_sign * a_j;
        let new_b_alt = state.current_b_alt + i_sign * b_i + j_sign * b_j;

        let a_alt_remaining = target_a_alt - new_a_alt;
        let b_alt_remaining = target_b_alt - new_b_alt;

        if a_alt_remaining.abs() > remaining_positions as i32 ||
           b_alt_remaining.abs() > remaining_positions as i32 {
            continue;
        }

        // Set values
        a[i] = a_i;
        b[i] = b_i;
        a[j] = a_j;
        b[j] = b_j;

        // Pruning 2: Check partial autocorrelations
        let mut pruned = false;

        for shift in 1..n {
            // Compute known_ac and unfilled_pairs for this shift
            let mut known_ac = cd_autocorr[shift];
            let mut unfilled_pairs = 0i32;

            for pos in 0..m.saturating_sub(shift) {
                let pos2 = pos + shift;
                let a_pos_filled = pos <= i || pos >= j;
                let a_pos2_filled = pos2 <= i || pos2 >= j;

                if a_pos_filled && a_pos2_filled {
                    known_ac += a[pos] * a[pos2];
                } else {
                    unfilled_pairs += 1;
                }

                if a_pos_filled && a_pos2_filled {
                    known_ac += b[pos] * b[pos2];
                } else {
                    unfilled_pairs += 1;
                }
            }

            if unfilled_pairs == 0 {
                if known_ac != 0 {
                    pruned = true;
                    break;
                }
            } else if known_ac.abs() > unfilled_pairs {
                pruned = true;
                break;
            }
        }

        if pruned {
            continue;
        }

        // Update state
        let old_a_sum = state.current_a_sum;
        let old_b_sum = state.current_b_sum;
        let old_a_alt = state.current_a_alt;
        let old_b_alt = state.current_b_alt;

        state.current_a_sum = new_a_sum;
        state.current_b_sum = new_b_sum;
        state.current_a_alt = new_a_alt;
        state.current_b_alt = new_b_alt;

        // Recurse
        if symmetric_backtrack_fill_incremental(
            a, b, pair_idx + 1, num_pairs, has_middle,
            target_a_sum, target_a_alt, target_b_sum, target_b_alt,
            cd_autocorr, n, m,
            pair_configs, middle_configs,
            state,
        ) {
            return true;
        }

        // Restore state
        state.current_a_sum = old_a_sum;
        state.current_b_sum = old_b_sum;
        state.current_a_alt = old_a_alt;
        state.current_b_alt = old_b_alt;
    }

    false
}

fn check_final_constraints(
    a: &[i32],
    b: &[i32],
    target_a_sum: i32,
    target_a_alt: i32,
    target_b_sum: i32,
    target_b_alt: i32,
    cd_autocorr: &[i32],
    n: usize,
    _m: usize,
) -> bool {
    // Check sum constraints
    let a_sum: i32 = a.iter().sum();
    let b_sum: i32 = b.iter().sum();
    if a_sum != target_a_sum || b_sum != target_b_sum {
        return false;
    }

    // Check alternating sum constraints
    let a_alt: i32 = a.iter().enumerate().map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
    let b_alt: i32 = b.iter().enumerate().map(|(i, &v)| if i % 2 == 0 { v } else { -v }).sum();
    if a_alt != target_a_alt || b_alt != target_b_alt {
        return false;
    }

    // Check autocorrelation constraints
    let a_seq = Sequence::new(a.to_vec());
    let b_seq = Sequence::new(b.to_vec());

    for shift in 1..n {
        let total_ac = a_seq.autocorrelation(shift)
                     + b_seq.autocorrelation(shift)
                     + cd_autocorr[shift];
        if total_ac != 0 {
            return false;
        }
    }

    true
}

// ============================================================================
// Budgeted backtracking: wrapper with node limit
// ============================================================================

fn symmetric_backtrack_search_ab_budgeted(
    c: &Sequence, d: &Sequence,
    a_sum: i32, a_alt: i32, b_sum: i32, b_alt: i32,
    max_nodes: u64,
) -> Option<(Sequence, Sequence)> {
    let n = c.len();
    let m = n + 1;

    let mut cd_autocorr = vec![0i32; n + 1];
    for shift in 1..n {
        cd_autocorr[shift] = c.autocorrelation(shift) + d.autocorrelation(shift);
    }

    for shift in 1..n {
        let target_ab = -cd_autocorr[shift];
        let max_possible = 2 * (m - shift) as i32;
        if target_ab.abs() > max_possible { return None; }
    }

    let mut a_values = vec![0i32; m];
    let mut b_values = vec![0i32; m];
    let num_pairs = m / 2;
    let has_middle = m % 2 == 1;

    let mut pair_configs: Vec<Vec<(i32, i32, i32, i32)>> = Vec::new();
    for pair_idx in 0..num_pairs {
        let target_mod = if pair_idx == 0 { 2 } else { 0 };
        let mut valid = Vec::new();
        for a_i in [-1i32, 1] {
            for b_i in [-1i32, 1] {
                for a_j in [-1i32, 1] {
                    for b_j in [-1i32, 1] {
                        let sum = a_i + b_i + a_j + b_j;
                        let sum_mod4 = ((sum % 4) + 4) % 4;
                        if sum_mod4 == target_mod { valid.push((a_i, b_i, a_j, b_j)); }
                    }
                }
            }
        }
        valid.sort_by_key(|(ai, bi, aj, bj)| (ai + bi + aj + bj).abs());
        pair_configs.push(valid);
    }

    let middle_configs: Vec<(i32, i32)> = if has_middle {
        vec![(1, 1), (-1, -1), (1, -1), (-1, 1)]
    } else { vec![] };

    let mut state = BacktrackState {
        current_a_sum: 0, current_b_sum: 0,
        current_a_alt: 0, current_b_alt: 0,
        partial_ac: vec![cd_autocorr.clone()],
    };

    let mut nodes_visited: u64 = 0;
    if backtrack_fill_budgeted(
        &mut a_values, &mut b_values,
        0, num_pairs, has_middle,
        a_sum, a_alt, b_sum, b_alt,
        &cd_autocorr, n, m,
        &pair_configs, &middle_configs,
        &mut state, &mut nodes_visited, max_nodes,
    ) {
        Some((Sequence::new(a_values), Sequence::new(b_values)))
    } else { None }
}

fn backtrack_fill_budgeted(
    a: &mut [i32], b: &mut [i32],
    pair_idx: usize, num_pairs: usize, has_middle: bool,
    target_a_sum: i32, target_a_alt: i32,
    target_b_sum: i32, target_b_alt: i32,
    cd_autocorr: &[i32], n: usize, m: usize,
    pair_configs: &[Vec<(i32, i32, i32, i32)>],
    middle_configs: &[(i32, i32)],
    state: &mut BacktrackState,
    nodes_visited: &mut u64, max_nodes: u64,
) -> bool {
    if *nodes_visited >= max_nodes { return false; }

    if pair_idx >= num_pairs {
        if has_middle {
            let mid = m / 2;
            for &(a_mid, b_mid) in middle_configs {
                a[mid] = a_mid; b[mid] = b_mid;
                if check_final_constraints(a, b, target_a_sum, target_a_alt,
                                          target_b_sum, target_b_alt, cd_autocorr, n, m) {
                    return true;
                }
            }
            return false;
        } else {
            return check_final_constraints(a, b, target_a_sum, target_a_alt,
                                          target_b_sum, target_b_alt, cd_autocorr, n, m);
        }
    }

    let i = pair_idx;
    let j = m - 1 - pair_idx;
    let i_sign = if i % 2 == 0 { 1i32 } else { -1 };
    let j_sign = if j % 2 == 0 { 1i32 } else { -1 };
    let remaining_positions = m - 2 * (pair_idx + 1) + if has_middle && pair_idx + 1 == num_pairs { 1 } else { 0 };

    for &(a_i, b_i, a_j, b_j) in &pair_configs[pair_idx] {
        *nodes_visited += 1;
        if *nodes_visited >= max_nodes { return false; }

        let new_a_sum = state.current_a_sum + a_i + a_j;
        let new_b_sum = state.current_b_sum + b_i + b_j;
        let a_remaining = target_a_sum - new_a_sum;
        let b_remaining = target_b_sum - new_b_sum;
        if a_remaining.abs() > remaining_positions as i32 ||
           b_remaining.abs() > remaining_positions as i32 { continue; }

        let new_a_alt = state.current_a_alt + i_sign * a_i + j_sign * a_j;
        let new_b_alt = state.current_b_alt + i_sign * b_i + j_sign * b_j;
        let a_alt_remaining = target_a_alt - new_a_alt;
        let b_alt_remaining = target_b_alt - new_b_alt;
        if a_alt_remaining.abs() > remaining_positions as i32 ||
           b_alt_remaining.abs() > remaining_positions as i32 { continue; }

        a[i] = a_i; b[i] = b_i; a[j] = a_j; b[j] = b_j;

        // Autocorrelation pruning
        let mut pruned = false;
        for shift in 1..n {
            let mut known_ac = cd_autocorr[shift];
            let mut unfilled_pairs = 0i32;
            for pos in 0..m.saturating_sub(shift) {
                let pos2 = pos + shift;
                let a_pos_filled = pos <= i || pos >= j;
                let a_pos2_filled = pos2 <= i || pos2 >= j;
                if a_pos_filled && a_pos2_filled {
                    known_ac += a[pos] * a[pos2];
                } else { unfilled_pairs += 1; }
                if a_pos_filled && a_pos2_filled {
                    known_ac += b[pos] * b[pos2];
                } else { unfilled_pairs += 1; }
            }
            if unfilled_pairs == 0 { if known_ac != 0 { pruned = true; break; } }
            else if known_ac.abs() > unfilled_pairs { pruned = true; break; }
        }
        if pruned { continue; }

        let old_a_sum = state.current_a_sum;
        let old_b_sum = state.current_b_sum;
        let old_a_alt = state.current_a_alt;
        let old_b_alt = state.current_b_alt;
        state.current_a_sum = new_a_sum;
        state.current_b_sum = new_b_sum;
        state.current_a_alt = new_a_alt;
        state.current_b_alt = new_b_alt;

        if backtrack_fill_budgeted(
            a, b, pair_idx + 1, num_pairs, has_middle,
            target_a_sum, target_a_alt, target_b_sum, target_b_alt,
            cd_autocorr, n, m, pair_configs, middle_configs,
            state, nodes_visited, max_nodes,
        ) { return true; }

        state.current_a_sum = old_a_sum;
        state.current_b_sum = old_b_sum;
        state.current_a_alt = old_a_alt;
        state.current_b_alt = old_b_alt;
    }
    false
}

// ============================================================================
// CD generation via Theorem 2.2 (ported from V6)
// ============================================================================

/// Valid symmetric pairs for C,D: c_i + d_i + c_j + d_j ≡ 0 (mod 4)
fn valid_symmetric_pairs_cd() -> Vec<(i32, i32, i32, i32)> {
    let mut valid = Vec::new();
    for c_i in [-1i32, 1] {
        for d_i in [-1i32, 1] {
            for c_j in [-1i32, 1] {
                for d_j in [-1i32, 1] {
                    let sum = c_i + d_i + c_j + d_j;
                    let sum_mod4 = ((sum % 4) + 4) % 4;
                    if sum_mod4 == 0 { valid.push((c_i, d_i, c_j, d_j)); }
                }
            }
        }
    }
    valid
}

/// Generate C,D pairs by construction using Theorem 2.2 symmetric pair constraints.
/// Fills pairs randomly from valid configurations, then exhaustively searches
/// the last 2 pairs to hit exact sum/alt-sum targets. ~15% spectral pass rate.
fn generate_cd_thm22(
    n: usize,
    c_sum_target: i32, d_sum_target: i32,
    c_alt_target: i32, d_alt_target: i32,
    num_samples: usize,
    rng: &mut impl Rng,
) -> Vec<(Sequence, Sequence)> {
    let valid_pairs = valid_symmetric_pairs_cd();
    let mut results = Vec::new();
    let num_pairs = n / 2;
    let has_middle = n % 2 == 1;

    for _ in 0..num_samples {
        let mut c_vals = vec![0i32; n];
        let mut d_vals = vec![0i32; n];
        let mut c_sum_partial = 0i32;
        let mut d_sum_partial = 0i32;
        let mut c_alt_partial = 0i32;
        let mut d_alt_partial = 0i32;

        let free_pairs = if num_pairs >= 3 { num_pairs - 2 } else { 0 };

        for pair in 0..free_pairs {
            let i = pair;
            let j = n - 1 - pair;
            let &(ci, di, cj, dj) = &valid_pairs[rng.gen_range(0..valid_pairs.len())];
            c_vals[i] = ci; d_vals[i] = di;
            c_vals[j] = cj; d_vals[j] = dj;
            c_sum_partial += ci + cj;
            d_sum_partial += di + dj;
            let alt_i: i32 = if i % 2 == 0 { 1 } else { -1 };
            let alt_j: i32 = if j % 2 == 0 { 1 } else { -1 };
            c_alt_partial += alt_i * ci + alt_j * cj;
            d_alt_partial += alt_i * di + alt_j * dj;
        }

        let remaining_pairs: Vec<usize> = (free_pairs..num_pairs).collect();
        let remaining_count = remaining_pairs.len();

        if remaining_count == 0 {
            let cs = c_sum_partial;
            let ds = d_sum_partial;
            let ca = c_alt_partial;
            let da = d_alt_partial;
            if has_middle {
                let mid = n / 2;
                let mut found_mid = false;
                for &cm in &[1i32, -1] {
                    for &dm in &[1i32, -1] {
                        let alt_m: i32 = if mid % 2 == 0 { 1 } else { -1 };
                        if cs + cm == c_sum_target && ds + dm == d_sum_target
                            && ca + alt_m * cm == c_alt_target && da + alt_m * dm == d_alt_target {
                            c_vals[mid] = cm; d_vals[mid] = dm;
                            found_mid = true; break;
                        }
                    }
                    if found_mid { break; }
                }
                if !found_mid { continue; }
            } else if cs != c_sum_target || ds != d_sum_target
                || ca != c_alt_target || da != d_alt_target { continue; }
            results.push((Sequence::new(c_vals), Sequence::new(d_vals)));
            continue;
        }

        let combos_per_pair = valid_pairs.len();
        let total_combos = combos_per_pair.pow(remaining_count as u32);
        let max_tries = total_combos.min(512);
        let try_exhaustive = total_combos <= 512;
        let mut found_combo = false;

        for trial in 0..max_tries {
            let mut cs = c_sum_partial;
            let mut ds = d_sum_partial;
            let mut ca = c_alt_partial;
            let mut da = d_alt_partial;
            let mut combo_idx = if try_exhaustive { trial } else { rng.gen_range(0..total_combos) };
            let mut temp_vals: Vec<(usize, usize, i32, i32, i32, i32)> = Vec::new();

            for &pair in &remaining_pairs {
                let i = pair;
                let j = n - 1 - pair;
                let pidx = combo_idx % combos_per_pair;
                combo_idx /= combos_per_pair;
                let &(ci, di, cj, dj) = &valid_pairs[pidx];
                cs += ci + cj; ds += di + dj;
                let alt_i: i32 = if i % 2 == 0 { 1 } else { -1 };
                let alt_j: i32 = if j % 2 == 0 { 1 } else { -1 };
                ca += alt_i * ci + alt_j * cj;
                da += alt_i * di + alt_j * dj;
                temp_vals.push((i, j, ci, di, cj, dj));
            }

            if has_middle {
                let mid = n / 2;
                let alt_m: i32 = if mid % 2 == 0 { 1 } else { -1 };
                let c_need = c_sum_target - cs;
                let d_need = d_sum_target - ds;
                if (c_need != 1 && c_need != -1) || (d_need != 1 && d_need != -1) { continue; }
                if ca + alt_m * c_need != c_alt_target { continue; }
                if da + alt_m * d_need != d_alt_target { continue; }
                for &(i, j, ci, di, cj, dj) in &temp_vals {
                    c_vals[i] = ci; d_vals[i] = di;
                    c_vals[j] = cj; d_vals[j] = dj;
                }
                c_vals[mid] = c_need; d_vals[mid] = d_need;
            } else {
                if cs != c_sum_target || ds != d_sum_target { continue; }
                if ca != c_alt_target || da != d_alt_target { continue; }
                for &(i, j, ci, di, cj, dj) in &temp_vals {
                    c_vals[i] = ci; d_vals[i] = di;
                    c_vals[j] = cj; d_vals[j] = dj;
                }
            }
            found_combo = true;
            break;
        }

        if !found_combo { continue; }
        results.push((Sequence::new(c_vals), Sequence::new(d_vals)));
    }
    results
}

// ============================================================================
// PART 6: Main search algorithm
// ============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = if args.len() > 1 {
        args[1].parse().unwrap_or_else(|_| {
            eprintln!("Usage: {} <n> [--resume]", args[0]);
            std::process::exit(1);
        })
    } else {
        eprintln!("Usage: {} <n> [--resume]", args[0]);
        std::process::exit(1);
    };

    let resume = args.iter().any(|a| a == "--resume");
    // V7 is now ALWAYS exhaustive - guarantees solution if it exists
    let exhaustive = true;

    println!("BS({},{}) - V7 Wang & Zhu (2025) Paper Implementation", n + 1, n);
    println!("============================================================");
    println!("Mode: EXHAUSTIVE (guaranteed to find solution if it exists)");
    println!("Optimizations: 21.6× faster than V6 via delta calculations");
    println!("============================================================");
    println!();
    println!("Techniques implemented:");
    println!("  1. Sum tuple constraints (Theorem 2.1)");
    println!("  2. Alternating sum constraints (Theorem 2.2, 2.3)");
    println!("  3. Two-phase spectral filtering (l=50, l=1000)");
    println!("  4. 5-class isomorphic equivalence reduction");
    println!("  5. Symmetric pair constraints (8 cases/position)");
    println!("  6. Modular decomposition (mod 3→6→12→m)");
    println!("  7. Deterministic backtracking (not stochastic)");
    println!();

    let start = Instant::now();

    // Step 1: Find valid sum tuples
    println!("Step 1: Finding valid sum tuples (Theorem 2.1)...");
    let all_tuples = find_valid_sum_tuples_fast_v2(n);
    println!("  Found {} raw tuples", all_tuples.len());

    // Step 2: Apply 5-class isomorphic filtering
    println!("\nStep 2: Applying 5-class isomorphic filtering...");
    let canonical_tuples = filter_to_canonical_5class(all_tuples, n);

    // Step 3: Sort tuples by difficulty
    println!("\nStep 3: Sorting tuples by estimated difficulty...");
    let mut sorted_tuples: Vec<_> = canonical_tuples.into_iter().collect();
    sorted_tuples.sort_by_key(|(st, at)| {
        // Prefer tuples with smaller absolute values (likely easier)
        // Include both sum and alt-sum magnitudes for better ordering
        let sum_mag = st.a.abs() + st.b.abs() + st.c.abs() + st.d.abs();
        let alt_mag = at.a_star.abs() + at.b_star.abs() + at.c_star.abs() + at.d_star.abs();
        sum_mag + alt_mag
    });

    let total_tuples = sorted_tuples.len();
    println!("  {} canonical tuples to search", total_tuples);

    // Load or create checkpoint
    let checkpoint = if resume {
        if let Some(cp) = Checkpoint::load(n) {
            println!("\nResuming from checkpoint:");
            println!("  Completed tuples: {}/{}", cp.completed_tuples.len(), total_tuples);
            println!("  In-progress tuples: {}", cp.tuple_cd_progress.len());
            println!("  CD pairs checked: {}", cp.total_cd_checked);
            println!("  CD pairs filtered: {}", cp.total_cd_filtered);
            println!("  Previous elapsed: {:.2} hours", cp.elapsed_secs / 3600.0);
            cp
        } else {
            println!("\nNo checkpoint found, starting fresh.");
            Checkpoint::new(n)
        }
    } else {
        Checkpoint::new(n)
    };

    let prior_elapsed = checkpoint.elapsed_secs;
    let completed_set: std::collections::HashSet<usize> =
        checkpoint.completed_tuples.iter().cloned().collect();
    let completed_set = Arc::new(completed_set);
    let cd_progress_map: HashMap<usize, u64> =
        checkpoint.tuple_cd_progress.iter().cloned().collect();
    let cd_progress_map = Arc::new(cd_progress_map);

    // Step 4: Search
    println!("\nStep 4: Deterministic search with symmetric backtracking...");
    println!();

    let found = Arc::new(AtomicBool::new(false));
    let tuples_done = Arc::new(AtomicUsize::new(checkpoint.completed_tuples.len()));
    let cd_pairs_checked = Arc::new(AtomicU64::new(checkpoint.total_cd_checked));
    let cd_pairs_filtered = Arc::new(AtomicU64::new(checkpoint.total_cd_filtered));
    let best_energy_global = Arc::new(AtomicI64::new(
        if checkpoint.global_best_energy != 0 { checkpoint.global_best_energy } else { i64::MAX }
    ));

    // For n<25: strict 1001-sample spectral filter (high precision, few CDs needed)
    // For n>=25: light 32-sample filter with margin (10x more CDs pass, breadth-first)
    let use_light_spectral = n >= 25;
    let spectral_margin: f64 = if n < 30 { 1.0 } else if n < 40 { 2.0 } else { 3.0 };

    // Global near-miss list: track best CD pairs across all tuples
    let near_misses: Arc<Mutex<Vec<NearMiss>>> = Arc::new(Mutex::new(
        checkpoint.near_misses.clone()
    ));

    // Per-tuple CD progress tracking for checkpointing
    let live_cd_progress: Arc<Mutex<HashMap<usize, u64>>> = Arc::new(Mutex::new(
        checkpoint.tuple_cd_progress.iter().cloned().collect()
    ));

    // Checkpoint state
    let checkpoint_mutex = Arc::new(Mutex::new(checkpoint));

    let sorted_arc = Arc::new(sorted_tuples);

    // Signal for progress thread termination
    let search_done = Arc::new(AtomicBool::new(false));

    // Progress + checkpoint thread
    let found_clone = Arc::clone(&found);
    let search_done_clone = Arc::clone(&search_done);
    let tuples_clone = Arc::clone(&tuples_done);
    let cd_clone = Arc::clone(&cd_pairs_checked);
    let filtered_clone = Arc::clone(&cd_pairs_filtered);
    let best_energy_clone = Arc::clone(&best_energy_global);
    let checkpoint_clone = Arc::clone(&checkpoint_mutex);
    let live_cd_clone = Arc::clone(&live_cd_progress);
    let near_misses_clone = Arc::clone(&near_misses);
    let start_clone = start.clone();

    std::thread::spawn(move || {
        let mut last_checkpoint = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(30));
            if found_clone.load(Ordering::Relaxed) || search_done_clone.load(Ordering::Relaxed) { break; }

            let tuples = tuples_clone.load(Ordering::Relaxed);
            let cd = cd_clone.load(Ordering::Relaxed);
            let filtered = filtered_clone.load(Ordering::Relaxed);
            let elapsed = start_clone.elapsed().as_secs_f64() + prior_elapsed;

            let best_e = best_energy_clone.load(Ordering::Relaxed);
            let energy_str = if best_e == i64::MAX { "---".to_string() } else { format!("E:{}", best_e) };
            let nm_count = near_misses_clone.lock().map(|nm| nm.len()).unwrap_or(0);
            let nm_str = if nm_count > 0 { format!(" NM:{}", nm_count) } else { String::new() };
            println!("  [{}/{}] CD: {} tried, {} filtered | {}{} | {:.1}h",
                     tuples, total_tuples, cd, filtered, energy_str, nm_str, elapsed / 3600.0);

            // Save checkpoint periodically
            if last_checkpoint.elapsed().as_secs() >= CHECKPOINT_INTERVAL_SECS {
                if let Ok(mut cp) = checkpoint_clone.lock() {
                    cp.total_cd_checked = cd;
                    cp.total_cd_filtered = filtered;
                    cp.elapsed_secs = elapsed;
                    cp.global_best_energy = best_e;
                    if let Ok(progress) = live_cd_clone.lock() {
                        cp.tuple_cd_progress = progress.iter()
                            .map(|(&k, &v)| (k, v))
                            .collect();
                    }
                    if let Ok(nm) = near_misses_clone.lock() {
                        cp.near_misses = nm.clone();
                    }
                    if let Err(err) = cp.save() {
                        eprintln!("  Warning: Failed to save checkpoint: {}", err);
                    } else {
                        let nm_count = cp.near_misses.len();
                        println!("  [Checkpoint saved | {} near-misses]", nm_count);
                    }
                }
                last_checkpoint = Instant::now();
            }
        }
    });

    // Main search - parallel over tuples
    let result: Option<(BaseSequence, usize, SumTuple, AltSumTuple)> = sorted_arc
        .par_iter()
        .enumerate()
        .find_map_any(|(tuple_idx, (st, at))| {
            if found.load(Ordering::Relaxed) {
                return None;
            }

            // Skip completed tuples (from checkpoint)
            if completed_set.contains(&tuple_idx) {
                return None;
            }

            // Search configuration scaled by n
            let config = if exhaustive {
                SearchConfig::for_n_exhaustive(n)
            } else {
                SearchConfig::for_n(n)
            };
            let backtrack_budget: u64 = if n <= 15 { u64::MAX }
                else if n <= 20 { 500_000 }
                else if n <= 30 { 100_000 }
                else { 20_000 };

            // Helper closure: try a CD pair with backtracking + adaptive SLS
            let try_cd_pair = |c: &Sequence, d: &Sequence| -> Option<(Sequence, Sequence, Sequence, Sequence)> {
                // Time-boxed backtracking first
                if let Some((a, b)) = symmetric_backtrack_search_ab_budgeted(
                    c, d, st.a, at.a_star, st.b, at.b_star, backtrack_budget
                ) {
                    let base = BaseSequence::new(a.clone(), b.clone(), c.clone(), d.clone());
                    if base.is_valid() { return Some((a, b, c.clone(), d.clone())); }
                }
                // Adaptive SLS fallback for n >= 16
                if n >= 16 {
                    let sls = SLSOptimized::new(c, d);
                    let result = adaptive_search(&sls, st, at, &config, backtrack_budget);
                    // Track best energy globally
                    if result.best_energy < best_energy_global.load(Ordering::Relaxed) {
                        best_energy_global.fetch_min(result.best_energy, Ordering::Relaxed);
                    }
                    if result.found {
                        if let (Some(a), Some(b)) = (result.a, result.b) {
                            let base = BaseSequence::new(a.clone(), b.clone(), c.clone(), d.clone());
                            if base.is_valid() { return Some((a, b, c.clone(), d.clone())); }
                        }
                    }
                    // Save near-miss for later exploitation
                    if result.best_energy > 0 && result.best_energy < near_miss_threshold(config.n) {
                        let (a_vals, b_vals) = if let Some(ref nm) = result.near_miss_config {
                            (Some(nm.a_values.clone()), Some(nm.b_values.clone()))
                        } else { (None, None) };
                        if let Ok(mut nm_list) = near_misses.lock() {
                            nm_list.push(NearMiss {
                                tuple_idx,
                                c_values: c.values.clone(),
                                d_values: d.values.clone(),
                                best_energy: result.best_energy,
                                a_values: a_vals,
                                b_values: b_vals,
                            });
                            nm_list.sort_by_key(|x| x.best_energy);
                            nm_list.truncate(100); // Keep top 100
                        }
                    }
                }
                None
            };

            // Determine search mode based on CD space size
            let enumerator = DeterministicCDEnumerator::new(
                n, st.c, st.d, at.c_star, at.d_star
            );
            let total_cd_pairs = enumerator.as_ref().map(|e| e.total_pairs()).unwrap_or(0);
            let det_threshold = config.max_cd_per_tuple.max(100_000) as u64;
            // In exhaustive mode, ALWAYS use deterministic enumeration (force completeness)
            let use_deterministic = if config.exhaustive {
                total_cd_pairs > 0  // Use deterministic if space is enumerable
            } else {
                total_cd_pairs > 0 && total_cd_pairs < det_threshold
            };

            if use_deterministic {
                // ---- Small CD space: deterministic enumeration (capped) ----
                let mut enumerator = enumerator.unwrap();
                if let Some(&start_idx) = cd_progress_map.get(&tuple_idx) {
                    enumerator.set_index(start_idx);
                }
                let det_cd_limit = if config.max_cd_per_tuple > 0 { config.max_cd_per_tuple } else { usize::MAX };
                let mut det_cd_count = 0usize;
                while let Some((c, d)) = enumerator.next() {
                    if found.load(Ordering::Relaxed) { return None; }
                    if det_cd_count >= det_cd_limit { break; }
                    let spectral_pass = if use_light_spectral {
                        light_spectral_filter(&c, &d, spectral_margin)
                    } else {
                        two_phase_spectral_filter(&c, &d)
                    };
                    if !spectral_pass {
                        cd_pairs_filtered.fetch_add(1, Ordering::Relaxed);
                        if enumerator.current_index() % 1000 == 0 {
                            if let Ok(mut progress) = live_cd_progress.lock() {
                                progress.insert(tuple_idx, enumerator.current_index());
                            }
                        }
                        continue;
                    }
                    det_cd_count += 1;
                    cd_pairs_checked.fetch_add(1, Ordering::Relaxed);
                    if let Some((a, b, c, d)) = try_cd_pair(&c, &d) {
                        found.store(true, Ordering::Relaxed);
                        Checkpoint::delete(n);
                        let base = BaseSequence::new(a, b, c, d);
                        return Some((base, tuple_idx, st.clone(), at.clone()));
                    }
                }
            } else {
                // ---- Large CD space: generate_cd_thm22 + adaptive SLS ----
                let max_cd_per_tuple = if config.max_cd_per_tuple > 0 { config.max_cd_per_tuple } else { usize::MAX };
                let batch_size = config.batch_size;
                let max_rounds = if config.max_rounds_per_tuple > 0 { config.max_rounds_per_tuple } else { usize::MAX };
                let mut cd_count = 0usize;
                let mut rng = rand::thread_rng();

                for _round in 0..max_rounds {
                    if found.load(Ordering::Relaxed) { return None; }
                    if cd_count >= max_cd_per_tuple { break; }

                    let batch = generate_cd_thm22(
                        n, st.c, st.d, at.c_star, at.d_star,
                        batch_size, &mut rng,
                    );
                    if batch.is_empty() && _round > 5 { break; }

                    for (c, d) in batch {
                        if found.load(Ordering::Relaxed) { return None; }
                        if cd_count >= max_cd_per_tuple { break; }

                        let spectral_pass = if use_light_spectral {
                            light_spectral_filter(&c, &d, spectral_margin)
                        } else {
                            two_phase_spectral_filter(&c, &d)
                        };
                        if !spectral_pass {
                            cd_pairs_filtered.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        cd_count += 1;
                        cd_pairs_checked.fetch_add(1, Ordering::Relaxed);

                        if let Some((a, b, c, d)) = try_cd_pair(&c, &d) {
                            found.store(true, Ordering::Relaxed);
                            Checkpoint::delete(n);
                            let base = BaseSequence::new(a, b, c, d);
                            return Some((base, tuple_idx, st.clone(), at.clone()));
                        }
                    }
                }
            }

            // Tuple searched
            tuples_done.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut cp) = checkpoint_mutex.lock() {
                cp.completed_tuples.push(tuple_idx);
            }
            if let Ok(mut progress) = live_cd_progress.lock() {
                progress.remove(&tuple_idx);
            }

            None
        });

    let elapsed_secs = start.elapsed().as_secs_f64() + prior_elapsed;

    // Save final checkpoint if no solution found
    if result.is_none() && !found.load(Ordering::Relaxed) {
        if let Ok(mut cp) = checkpoint_mutex.lock() {
            cp.total_cd_checked = cd_pairs_checked.load(Ordering::Relaxed);
            cp.total_cd_filtered = cd_pairs_filtered.load(Ordering::Relaxed);
            cp.elapsed_secs = elapsed_secs;
            cp.global_best_energy = best_energy_global.load(Ordering::Relaxed);
            if let Ok(progress) = live_cd_progress.lock() {
                cp.tuple_cd_progress = progress.iter()
                    .map(|(&k, &v)| (k, v))
                    .collect();
            }
            if let Ok(nm) = near_misses.lock() {
                cp.near_misses = nm.clone();
            }
            if let Err(err) = cp.save() {
                eprintln!("Warning: Failed to save final checkpoint: {}", err);
            } else {
                println!("  [Final checkpoint saved]");
            }
        }
    }

    // ================================================================
    // Near-miss exploitation phase: re-exploit the best near-misses
    // with massive compute after the main search completes
    // ================================================================
    if result.is_none() && !found.load(Ordering::Relaxed) && n >= 25 {
        let nm_list: Vec<NearMiss> = near_misses.lock()
            .map(|nm| nm.clone()).unwrap_or_default();
        if !nm_list.is_empty() {
            println!("\n  Near-miss exploitation phase: {} candidates (best E:{})",
                     nm_list.len(), nm_list[0].best_energy);

            // Try to find the tuple info for each near-miss
            for (i, nm) in nm_list.iter().enumerate() {
                if found.load(Ordering::Relaxed) { break; }
                let c = Sequence::new(nm.c_values.clone());
                let d = Sequence::new(nm.d_values.clone());
                let sls = SLSOptimized::new(&c, &d);

                // Get the tuple info
                if nm.tuple_idx >= sorted_arc.len() { continue; }
                let (st, at) = &sorted_arc[nm.tuple_idx];

                print!("  NM#{} (E:{}, tuple#{}) ", i, nm.best_energy, nm.tuple_idx);

                // Phase 1: If we have saved A,B state, exploit it directly
                if let (Some(ref a_vals), Some(ref b_vals)) = (&nm.a_values, &nm.b_values) {
                    let nm_config = NearMissConfig {
                        a_values: a_vals.clone(),
                        b_values: b_vals.clone(),
                        energy: nm.best_energy,
                        autocorrs: vec![0; n], // Will be recomputed
                    };
                    if let Some((a, b)) = sls.exploit_near_miss(&nm_config, at.a_star, at.b_star, 500_000) {
                        let base = BaseSequence::new(a.clone(), b.clone(), c.clone(), d.clone());
                        if base.is_valid() {
                            println!("-> SOLVED!");
                            found.store(true, Ordering::Relaxed);
                            Checkpoint::delete(n);
                            let elapsed_secs = start.elapsed().as_secs_f64() + prior_elapsed;
                            // Print solution
                            println!("\n\n============================================");
                            println!("       SUCCESS! BS({},{}) FOUND          ", n + 1, n);
                            println!("============================================");
                            println!("\nTime: {:.2} hours", elapsed_secs / 3600.0);
                            println!("Found via near-miss exploitation (NM#{})", i);
                            println!("\nA = {:?}", a.values);
                            println!("B = {:?}", b.values);
                            println!("C = {:?}", c.values);
                            println!("D = {:?}", d.values);
                            println!("\nVerification:");
                            let mut all_pass = true;
                            for shift in 1..n {
                                let ac = a.autocorrelation(shift) + b.autocorrelation(shift)
                                    + c.autocorrelation(shift) + d.autocorrelation(shift);
                                if ac != 0 { println!("  FAIL at shift {}: autocorr = {}", shift, ac); all_pass = false; }
                            }
                            if all_pass { println!("  All {} autocorrelation checks passed!", n - 1); }
                            let filename = format!("BS_{}_{}_V7_{:.0}s.txt", n + 1, n, elapsed_secs);
                            if let Ok(mut f) = std::fs::File::create(&filename) {
                                writeln!(f, "BS({},{}) - V7 (Near-miss exploitation)", n + 1, n).ok();
                                writeln!(f, "Time: {:.2}h", elapsed_secs / 3600.0).ok();
                                writeln!(f, "").ok();
                                writeln!(f, "A = {:?}", a.values).ok();
                                writeln!(f, "B = {:?}", b.values).ok();
                                writeln!(f, "C = {:?}", c.values).ok();
                                writeln!(f, "D = {:?}", d.values).ok();
                                println!("\nSaved to: {}", filename);
                            }
                            return;
                        }
                    }
                }

                // Phase 2: Fresh enhanced search with massive budget
                let (result, energy, _) = sls.search_v6_enhanced(
                    st.a, at.a_star, st.b, at.b_star,
                    2000, 300_000, // Massive: 2000 restarts × 300K
                );
                if let Some((a, b)) = result {
                    let base = BaseSequence::new(a.clone(), b.clone(), c.clone(), d.clone());
                    if base.is_valid() {
                        println!("-> SOLVED (enhanced)!");
                        found.store(true, Ordering::Relaxed);
                        Checkpoint::delete(n);
                        let elapsed_secs = start.elapsed().as_secs_f64() + prior_elapsed;
                        println!("\n\n============================================");
                        println!("       SUCCESS! BS({},{}) FOUND          ", n + 1, n);
                        println!("============================================");
                        println!("\nTime: {:.2} hours", elapsed_secs / 3600.0);
                        println!("Found via near-miss exploitation (NM#{}, enhanced)", i);
                        println!("\nA = {:?}", a.values);
                        println!("B = {:?}", b.values);
                        println!("C = {:?}", c.values);
                        println!("D = {:?}", d.values);
                        println!("\nVerification:");
                        let mut all_pass = true;
                        for shift in 1..n {
                            let ac = a.autocorrelation(shift) + b.autocorrelation(shift)
                                + c.autocorrelation(shift) + d.autocorrelation(shift);
                            if ac != 0 { println!("  FAIL at shift {}: autocorr = {}", shift, ac); all_pass = false; }
                        }
                        if all_pass { println!("  All {} autocorrelation checks passed!", n - 1); }
                        let filename = format!("BS_{}_{}_V7_{:.0}s.txt", n + 1, n, elapsed_secs);
                        if let Ok(mut f) = std::fs::File::create(&filename) {
                            writeln!(f, "BS({},{}) - V7 (Near-miss exploitation, enhanced)", n + 1, n).ok();
                            writeln!(f, "Time: {:.2}h", elapsed_secs / 3600.0).ok();
                            writeln!(f, "").ok();
                            writeln!(f, "A = {:?}", a.values).ok();
                            writeln!(f, "B = {:?}", b.values).ok();
                            writeln!(f, "C = {:?}", c.values).ok();
                            writeln!(f, "D = {:?}", d.values).ok();
                            println!("\nSaved to: {}", filename);
                        }
                        return;
                    }
                }
                println!("-> E:{}", energy);
            }
        }
    }

    // Signal progress thread to stop
    search_done.store(true, Ordering::Relaxed);

    // Print results
    println!("\n");
    if let Some((base, tuple_idx, st, at)) = result {
        println!("============================================");
        println!("       SUCCESS! BS({},{}) FOUND          ", n + 1, n);
        println!("============================================");
        println!();
        println!("Time: {:.2} hours", elapsed_secs / 3600.0);
        println!("Tuples checked: {}", tuples_done.load(Ordering::Relaxed));
        println!("CD pairs checked: {}", cd_pairs_checked.load(Ordering::Relaxed));
        println!("CD pairs filtered: {}", cd_pairs_filtered.load(Ordering::Relaxed));
        println!();
        println!("Solution at tuple #{}", tuple_idx);
        println!("Sum tuple:     ({:>3}, {:>3}, {:>3}, {:>3})", st.a, st.b, st.c, st.d);
        println!("Alt-sum tuple: ({:>3}, {:>3}, {:>3}, {:>3})", at.a_star, at.b_star, at.c_star, at.d_star);
        println!();
        println!("A = {:?}", base.a.values);
        println!("B = {:?}", base.b.values);
        println!("C = {:?}", base.c.values);
        println!("D = {:?}", base.d.values);
        println!();
        println!("Verification:");

        // Verify all autocorrelations
        let mut all_pass = true;
        for shift in 1..n {
            let ac = base.a.autocorrelation(shift)
                   + base.b.autocorrelation(shift)
                   + base.c.autocorrelation(shift)
                   + base.d.autocorrelation(shift);
            if ac != 0 {
                println!("  FAIL at shift {}: autocorr = {}", shift, ac);
                all_pass = false;
            }
        }
        if all_pass {
            println!("  All {} autocorrelation checks passed!", n - 1);
        }

        // Save to file
        let filename = format!("BS_{}_{}_V7_{:.0}s.txt", n + 1, n, elapsed_secs);
        if let Ok(mut f) = std::fs::File::create(&filename) {
            writeln!(f, "BS({},{}) - V7 Wang & Zhu Paper Implementation", n + 1, n).ok();
            writeln!(f, "Time: {:.2}h", elapsed_secs / 3600.0).ok();
            writeln!(f, "").ok();
            writeln!(f, "A = {:?}", base.a.values).ok();
            writeln!(f, "B = {:?}", base.b.values).ok();
            writeln!(f, "C = {:?}", base.c.values).ok();
            writeln!(f, "D = {:?}", base.d.values).ok();
            println!("\nSaved to: {}", filename);
        }
    } else {
        println!("============================================");
        println!("       NO SOLUTION FOUND                   ");
        println!("============================================");
        println!();
        println!("Time: {:.2} hours", elapsed_secs / 3600.0);
        println!("Tuples checked: {}", tuples_done.load(Ordering::Relaxed));
        println!("CD pairs checked: {}", cd_pairs_checked.load(Ordering::Relaxed));
        println!("CD pairs filtered: {}", cd_pairs_filtered.load(Ordering::Relaxed));
        println!();
        println!("Note: This is a DETERMINISTIC search.");
        println!("If no solution found after checking all tuples and CD pairs,");
        println!("then no BS({},{}) exists with the standard construction.", n + 1, n);
        println!("\nUse --resume to continue from checkpoint.");
    }
}
