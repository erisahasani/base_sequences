/// Spectral filtering for CD pairs using Theorem 2.4 bounds
///
/// For valid base sequences, the Hall polynomial must satisfy:
///   f_A(θ) + f_B(θ) + f_C(θ) + f_D(θ) = 4n + 2 for all θ
///
/// Since f_A, f_B ≥ 0 (squared magnitudes), this means:
///   f_C(θ) + f_D(θ) ≤ 4n + 2 for all θ
///
/// We can aggressively filter CD pairs that violate this bound.

use crate::Sequence;
use std::f64::consts::PI;

/// Spectral quality score for a CD pair
/// Lower score = better chance of finding valid A,B
#[derive(Clone, Debug)]
pub struct SpectralScore {
    pub max_excess: f64,      // Maximum excess over bound
    pub mean_excess: f64,     // Average excess over bound
    pub violation_count: u32, // Number of sample points violating bound
    pub quality: f64,         // Overall quality score (lower = better)
}

/// Compute spectral quality score for a CD pair
pub fn spectral_score(c: &Sequence, d: &Sequence) -> SpectralScore {
    let n = c.len();
    let target = 4.0 * (n as f64) + 2.0;

    // Sample more angles for more accurate filtering
    let num_samples = 64;
    let mut max_excess = 0.0f64;
    let mut total_excess = 0.0f64;
    let mut violations = 0u32;

    for i in 0..num_samples {
        let theta = 2.0 * PI * (i as f64) / (num_samples as f64);
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);
        let total = fc + fd;

        if total > target {
            let excess = total - target;
            max_excess = max_excess.max(excess);
            total_excess += excess;
            violations += 1;
        }
    }

    let mean_excess = if violations > 0 {
        total_excess / violations as f64
    } else {
        0.0
    };

    // Quality score combines multiple factors
    // - Direct violations are very bad
    // - Large max_excess is bad
    // - Mean excess indicates overall difficulty
    let quality = max_excess * 10.0 + mean_excess * 5.0 + (violations as f64) * 0.5;

    SpectralScore {
        max_excess,
        mean_excess,
        violation_count: violations,
        quality,
    }
}

/// Check if CD pair passes strict spectral bound
pub fn passes_spectral_bound(c: &Sequence, d: &Sequence, margin: f64) -> bool {
    let n = c.len();
    let target = 4.0 * (n as f64) + 2.0;
    let threshold = target + margin;

    // Sample at many angles
    let num_samples = 32;
    for i in 0..num_samples {
        let theta = 2.0 * PI * (i as f64) / (num_samples as f64);
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);

        if fc + fd > threshold {
            return false;
        }
    }

    true
}

/// Fast Hall polynomial evaluation: |sum(a_j * e^{i*j*theta})|^2
#[inline]
pub fn hall_polynomial(values: &[i32], theta: f64) -> f64 {
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

/// Estimate how much "room" is left for A,B given C,D
/// This is the minimum value of (4n+2 - f_C - f_D) over sample angles
pub fn compute_ab_headroom(c: &Sequence, d: &Sequence) -> f64 {
    let n = c.len();
    let target = 4.0 * (n as f64) + 2.0;

    let num_samples = 32;
    let mut min_headroom = f64::MAX;

    for i in 0..num_samples {
        let theta = 2.0 * PI * (i as f64) / (num_samples as f64);
        let fc = hall_polynomial(&c.values, theta);
        let fd = hall_polynomial(&d.values, theta);
        let headroom = target - fc - fd;
        min_headroom = min_headroom.min(headroom);
    }

    min_headroom
}

/// Filter and sort CD pairs by spectral quality
/// Returns pairs sorted by quality (best first)
pub fn filter_and_sort_cd_pairs(
    pairs: Vec<(Sequence, Sequence)>,
    margin: f64,
) -> Vec<(Sequence, Sequence, SpectralScore)> {
    let mut scored: Vec<_> = pairs
        .into_iter()
        .filter_map(|(c, d)| {
            // Quick rejection check
            if !passes_spectral_bound(&c, &d, margin) {
                return None;
            }
            let score = spectral_score(&c, &d);
            Some((c, d, score))
        })
        .collect();

    // Sort by quality (lower = better)
    scored.sort_by(|a, b| a.2.quality.partial_cmp(&b.2.quality).unwrap());

    scored
}

/// Estimate difficulty of finding A,B for given sum constraints and CD pair
/// Combines spectral bounds with constraint analysis
pub fn estimate_search_difficulty(
    c: &Sequence,
    d: &Sequence,
    a_sum: i32,
    a_alt: i32,
    b_sum: i32,
    b_alt: i32,
) -> f64 {
    let m = c.len() + 1;

    // Spectral headroom
    let headroom = compute_ab_headroom(c, d);

    // Constraint tightness
    // More extreme sum/alt_sum values are harder
    let a_constraint_tightness = (a_sum.abs() + a_alt.abs()) as f64 / (2.0 * m as f64);
    let b_constraint_tightness = (b_sum.abs() + b_alt.abs()) as f64 / (2.0 * m as f64);

    // Combine factors (lower = easier)
    let difficulty = 100.0 / headroom.max(0.1)
        + a_constraint_tightness * 20.0
        + b_constraint_tightness * 20.0;

    difficulty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hall_polynomial() {
        // All +1 sequence
        let seq = vec![1i32; 5];
        // At theta=0, Hall polynomial = n^2
        let result = hall_polynomial(&seq, 0.0);
        assert!((result - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_spectral_score() {
        let c = Sequence::new(vec![1, -1, 1, -1, 1]);
        let d = Sequence::new(vec![-1, 1, -1, 1, -1]);
        let score = spectral_score(&c, &d);
        // Just check it runs
        assert!(score.quality >= 0.0);
    }
}
