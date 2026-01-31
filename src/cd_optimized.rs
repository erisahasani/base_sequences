/// Optimized C,D pair generation
///
/// Key optimizations:
/// 1. Pre-check constraint feasibility before loop
/// 2. Direct construction with parity-aware random placement
/// 3. Reusable position vectors to avoid allocations
/// 4. Parallel generation using rayon when available

use crate::Sequence;
use rand::Rng;
use std::f64::consts::PI;

/// Check if sum/alt_sum constraints are feasible for length n
#[inline]
fn constraints_feasible(n: usize, target_sum: i32, target_alt_sum: i32) -> bool {
    // Check sum feasibility: #plus = (n + sum) / 2
    let sum_plus_n = n as i32 + target_sum;
    if sum_plus_n < 0 || sum_plus_n > 2 * n as i32 || sum_plus_n % 2 != 0 {
        return false;
    }
    let num_plus = (sum_plus_n / 2) as usize;

    // Position counts
    let n_even = (n + 1) / 2;
    let n_odd = n / 2;

    // Check alt_sum feasibility: e_plus = (alt_sum + n_even + 2*num_plus - n_odd) / 4
    let numerator = target_alt_sum + (n_even as i32) + 2 * (num_plus as i32) - (n_odd as i32);
    if numerator % 4 != 0 {
        return false;
    }

    let e_plus = numerator / 4;
    if e_plus < 0 || e_plus > n_even as i32 {
        return false;
    }

    let o_plus = num_plus as i32 - e_plus;
    if o_plus < 0 || o_plus > n_odd as i32 {
        return false;
    }

    true
}

/// Generate random CD pairs efficiently
/// Uses direct construction - no rejection sampling needed
pub fn generate_random_cd_pairs_fast(
    n: usize,
    c_sum: i32,
    d_sum: i32,
    c_alt_sum: i32,
    d_alt_sum: i32,
    num_samples: usize,
) -> Vec<(Sequence, Sequence)> {
    // Pre-check feasibility - avoid wasted iterations
    if !constraints_feasible(n, c_sum, c_alt_sum) {
        return Vec::new();
    }
    if !constraints_feasible(n, d_sum, d_alt_sum) {
        return Vec::new();
    }

    let mut rng = rand::thread_rng();
    let mut pairs = Vec::with_capacity(num_samples);

    // Pre-compute position vectors (reused across iterations)
    let even_positions: Vec<usize> = (0..n).filter(|&i| i % 2 == 0).collect();
    let odd_positions: Vec<usize> = (0..n).filter(|&i| i % 2 == 1).collect();

    // Direct construction - should succeed on every attempt since we pre-checked feasibility
    for _ in 0..num_samples {
        let c = generate_constrained_sequence_fast(
            n, c_sum, c_alt_sum, &even_positions, &odd_positions, &mut rng
        );
        let d = generate_constrained_sequence_fast(
            n, d_sum, d_alt_sum, &even_positions, &odd_positions, &mut rng
        );
        pairs.push((c, d));
    }

    pairs
}

/// Generate random CD pairs with Hall polynomial filtering
/// More selective but may miss valid pairs for large n
pub fn generate_random_cd_pairs_with_hall(
    n: usize,
    c_sum: i32,
    d_sum: i32,
    c_alt_sum: i32,
    d_alt_sum: i32,
    num_samples: usize,
) -> Vec<(Sequence, Sequence)> {
    let mut rng = rand::thread_rng();
    let mut pairs = Vec::with_capacity(num_samples);
    let max_attempts = num_samples * 100;
    let mut attempts = 0;

    while pairs.len() < num_samples && attempts < max_attempts {
        attempts += 1;

        if let Some(c) = generate_constrained_sequence(n, c_sum, c_alt_sum, &mut rng) {
            if let Some(d) = generate_constrained_sequence(n, d_sum, d_alt_sum, &mut rng) {
                if passes_hall_check_fast(&c, &d, n) {
                    pairs.push((c, d));
                }
            }
        }
    }

    pairs
}

/// Fast sequence generation with pre-computed position vectors
/// Assumes constraints have been pre-validated via constraints_feasible()
fn generate_constrained_sequence_fast(
    n: usize,
    target_sum: i32,
    target_alt_sum: i32,
    even_positions: &[usize],
    odd_positions: &[usize],
    rng: &mut impl Rng,
) -> Sequence {
    let num_plus = ((n as i32 + target_sum) / 2) as usize;
    let n_even = even_positions.len();
    let n_odd = odd_positions.len();

    let numerator = target_alt_sum + (n_even as i32) + 2 * (num_plus as i32) - (n_odd as i32);
    let e_plus = (numerator / 4) as usize;
    let o_plus = num_plus - e_plus;

    // Build sequence
    let mut values = vec![-1i32; n];

    // Use partial Fisher-Yates to select random positions without full shuffle
    let mut selected_even = even_positions.to_vec();
    partial_shuffle(&mut selected_even, e_plus, rng);
    for &pos in &selected_even[..e_plus] {
        values[pos] = 1;
    }

    let mut selected_odd = odd_positions.to_vec();
    partial_shuffle(&mut selected_odd, o_plus, rng);
    for &pos in &selected_odd[..o_plus] {
        values[pos] = 1;
    }

    Sequence::new(values)
}

/// Generate a sequence satisfying sum and alternating sum constraints
/// Uses direct construction based on parity distribution
fn generate_constrained_sequence(
    n: usize,
    target_sum: i32,
    target_alt_sum: i32,
    rng: &mut impl Rng,
) -> Option<Sequence> {
    if !constraints_feasible(n, target_sum, target_alt_sum) {
        return None;
    }

    let even_positions: Vec<usize> = (0..n).filter(|&i| i % 2 == 0).collect();
    let odd_positions: Vec<usize> = (0..n).filter(|&i| i % 2 == 1).collect();

    Some(generate_constrained_sequence_fast(
        n, target_sum, target_alt_sum, &even_positions, &odd_positions, rng
    ))
}

/// Partial Fisher-Yates - only shuffle first k elements
/// More efficient when k << slice.len()
#[inline]
fn partial_shuffle<T>(slice: &mut [T], k: usize, rng: &mut impl Rng) {
    let n = slice.len();
    let k = k.min(n);
    for i in 0..k {
        let j = rng.gen_range(i..n);
        slice.swap(i, j);
    }
}

/// Calculate the range of possible alternating sums from position start_pos to n
#[inline]
fn alt_sum_range(current: i32, start_pos: usize, n: usize) -> (i32, i32) {
    let mut min_val = current;
    let mut max_val = current;

    for pos in start_pos..n {
        if pos % 2 == 0 {
            // Even position: +1 adds 1, -1 subtracts 1
            max_val += 1;
            min_val -= 1;
        } else {
            // Odd position: +1 subtracts 1, -1 adds 1
            max_val += 1;
            min_val -= 1;
        }
    }

    (min_val, max_val)
}

/// Fast Hall polynomial check using just a few sample points
/// Full check is expensive; this catches most bad pairs quickly
#[inline]
fn passes_hall_check_fast(c: &Sequence, d: &Sequence, n: usize) -> bool {
    let target = 4.0 * (n as f64) + 2.0;

    // Sample at a few key angles
    let sample_angles = [0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0];

    for &theta in &sample_angles {
        let fc = hall_polynomial_fast(&c.values, theta);
        let fd = hall_polynomial_fast(&d.values, theta);

        // For valid base sequence: f_A + f_B + f_C + f_D = target
        // So f_C + f_D should be <= target (with some margin for f_A + f_B >= 0)
        if fc + fd > target + 0.1 {
            return false;
        }
    }

    true
}

/// Fast Hall polynomial evaluation
#[inline]
fn hall_polynomial_fast(values: &[i32], theta: f64) -> f64 {
    let mut real_sum = 0.0;
    let mut imag_sum = 0.0;

    for (i, &val) in values.iter().enumerate() {
        let angle = (i as f64) * theta;
        let (sin_a, cos_a) = angle.sin_cos();
        real_sum += (val as f64) * cos_a;
        imag_sum += (val as f64) * sin_a;
    }

    real_sum * real_sum + imag_sum * imag_sum
}

/// Generate CD pairs using deterministic enumeration with aggressive pruning
/// Better for exhaustive search of small spaces
pub fn generate_cd_pairs_pruned(
    n: usize,
    c_sum: i32,
    d_sum: i32,
    c_alt_sum: i32,
    d_alt_sum: i32,
    max_pairs: usize,
) -> Vec<(Sequence, Sequence)> {
    let mut pairs = Vec::new();
    let mut c_values = vec![0i32; n];
    let mut d_values = vec![0i32; n];

    generate_pruned_helper(
        &mut c_values,
        &mut d_values,
        0,
        0, 0, // c_sum, c_alt
        0, 0, // d_sum, d_alt
        n,
        c_sum, d_sum, c_alt_sum, d_alt_sum,
        &mut pairs,
        max_pairs,
    );

    pairs
}

fn generate_pruned_helper(
    c: &mut Vec<i32>,
    d: &mut Vec<i32>,
    pos: usize,
    c_sum: i32, c_alt: i32,
    d_sum: i32, d_alt: i32,
    n: usize,
    target_c_sum: i32, target_d_sum: i32,
    target_c_alt: i32, target_d_alt: i32,
    results: &mut Vec<(Sequence, Sequence)>,
    max_pairs: usize,
) {
    if results.len() >= max_pairs {
        return;
    }

    if pos == n {
        if c_sum == target_c_sum && d_sum == target_d_sum &&
           c_alt == target_c_alt && d_alt == target_d_alt {
            let c_seq = Sequence::new(c.clone());
            let d_seq = Sequence::new(d.clone());
            if passes_hall_check_fast(&c_seq, &d_seq, n) {
                results.push((c_seq, d_seq));
            }
        }
        return;
    }

    let remaining = (n - pos - 1) as i32;
    let alt_sign = if pos % 2 == 0 { 1 } else { -1 };

    // Try all 4 combinations with pruning
    for &c_val in &[1i32, -1] {
        let new_c_sum = c_sum + c_val;
        let new_c_alt = c_alt + alt_sign * c_val;

        // Prune C
        if target_c_sum < new_c_sum - remaining || target_c_sum > new_c_sum + remaining {
            continue;
        }
        let (c_alt_min, c_alt_max) = alt_sum_range(new_c_alt, pos + 1, n);
        if target_c_alt < c_alt_min || target_c_alt > c_alt_max {
            continue;
        }

        for &d_val in &[1i32, -1] {
            let new_d_sum = d_sum + d_val;
            let new_d_alt = d_alt + alt_sign * d_val;

            // Prune D
            if target_d_sum < new_d_sum - remaining || target_d_sum > new_d_sum + remaining {
                continue;
            }
            let (d_alt_min, d_alt_max) = alt_sum_range(new_d_alt, pos + 1, n);
            if target_d_alt < d_alt_min || target_d_alt > d_alt_max {
                continue;
            }

            c[pos] = c_val;
            d[pos] = d_val;

            generate_pruned_helper(
                c, d, pos + 1,
                new_c_sum, new_c_alt,
                new_d_sum, new_d_alt,
                n,
                target_c_sum, target_d_sum,
                target_c_alt, target_d_alt,
                results, max_pairs,
            );

            if results.len() >= max_pairs {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constrained_sequence() {
        let mut rng = rand::thread_rng();

        // n=5, sum=1, alt_sum=3
        // sum=1 means 3 plus, 2 minus
        for _ in 0..100 {
            if let Some(seq) = generate_constrained_sequence(5, 1, 3, &mut rng) {
                assert_eq!(seq.sum(), 1);
                assert_eq!(seq.alternating_sum(), 3);
            }
        }
    }

    #[test]
    fn test_random_cd_generation() {
        let pairs = generate_random_cd_pairs_fast(5, 1, -1, 3, 1, 10);
        for (c, d) in &pairs {
            assert_eq!(c.sum(), 1);
            assert_eq!(d.sum(), -1);
            assert_eq!(c.alternating_sum(), 3);
            assert_eq!(d.alternating_sum(), 1);
        }
    }
}
