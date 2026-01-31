/// Ultra-fast sum tuple finding using HashMap-based matching
///
/// Improvements over v1:
/// 1. HashMap-based matching (O(1) lookup instead of O(n²))
/// 2. Better memory layout
/// 3. Perfect square lookup table (avoids isqrt in tight loops)
/// 4. Direct parity iteration (step by 2)

use crate::{SumTuple, AltSumTuple};
use std::collections::{HashMap, HashSet};

#[inline]
fn mod_positive(a: i32, m: i32) -> i32 {
    ((a % m) + m) % m
}

/// Signature for matching sum tuples with alt tuples based on Equation 2.4
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

    /// Compute the signature that an alt_tuple needs to match a sum_tuple
    /// Based on Equation 2.4's constraints for different n mod 4
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

/// Find valid sum tuples using HashMap-based matching
/// Much faster than the nested loop approach for the matching phase
pub fn find_valid_sum_tuples_fast_v2(n: usize) -> Vec<(SumTuple, AltSumTuple)> {
    let target = (4 * n + 2) as i32;
    let max_sum_m = (n + 1) as i32;
    let max_sum_n = n as i32;

    println!("  Phase 1: Finding valid (a,b,c,d) tuples...");

    // Phase 1: Find all valid (a,b,c,d) sum tuples
    let sum_tuples = find_sum_tuples(n, target, max_sum_m, max_sum_n);
    println!("  Found {} valid (a,b,c,d) tuples", sum_tuples.len());

    println!("  Phase 2: Finding valid (a*,b*,c*,d*) tuples...");

    // Phase 2: Find all valid alternating sum tuples
    let alt_tuples = find_alt_tuples(n, target, max_sum_m, max_sum_n);
    println!("  Found {} valid (a*,b*,c*,d*) tuples", alt_tuples.len());

    println!("  Phase 3: HashMap-based matching (Equation 2.4)...");

    // Phase 3: Build HashMap from alt_tuples and match
    let mut alt_by_signature: HashMap<Mod4Signature, Vec<AltSumTuple>> = HashMap::new();

    for at in alt_tuples {
        let sig = Mod4Signature::required_for_alt_tuple(&at, n);
        alt_by_signature.entry(sig).or_default().push(at);
    }

    // Match sum_tuples with alt_tuples via O(1) HashMap lookup
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

/// Precompute perfect squares and their roots for fast lookup
fn build_perfect_square_map(max_val: i32) -> HashMap<i32, i32> {
    let mut map = HashMap::new();
    for i in 0..=max_val {
        map.insert(i * i, i);
    }
    map
}

/// Get the first value in range [-max_val, max_val] with given parity
#[inline]
fn first_with_parity(max_val: i32, parity: i32) -> i32 {
    let start = -max_val;
    if mod_positive(start, 2) == parity { start } else { start + 1 }
}

fn find_sum_tuples(n: usize, target: i32, max_sum_m: i32, max_sum_n: i32) -> Vec<SumTuple> {
    let mut sum_tuples = Vec::new();

    // Precompute perfect squares for O(1) lookup
    let perfect_squares = build_perfect_square_map(max_sum_n);

    // Calculate parities
    let ab_parity = ((n + 1) % 2) as i32;
    let cd_parity = (n % 2) as i32;
    let n_even = n % 2 == 0;

    let mut a = first_with_parity(max_sum_m, ab_parity);
    while a <= max_sum_m {
        let a_sq = a * a;
        if a_sq > target {
            a += 2;
            continue;
        }

        let mut b = first_with_parity(max_sum_m, ab_parity);
        while b <= max_sum_m {
            let ab_sum_sq = a_sq + b * b;
            if ab_sum_sq > target {
                b += 2;
                continue;
            }

            // For odd n, check equation 2.5 early (depends only on a,b)
            if !n_even && mod_positive(a, 4) != mod_positive(b + 2, 4) {
                b += 2;
                continue;
            }

            let remaining = target - ab_sum_sq;

            let mut c = first_with_parity(max_sum_n, cd_parity);
            while c <= max_sum_n {
                let c_sq = c * c;
                let d_sq_needed = remaining - c_sq;

                if d_sq_needed < 0 {
                    // c² too large - if c >= 0, it only gets worse
                    if c >= 0 {
                        break;
                    }
                    c += 2;
                    continue;
                }

                // Use HashMap lookup instead of isqrt
                if let Some(&d_abs) = perfect_squares.get(&d_sq_needed) {
                    // Try both ±d
                    for &d in &[d_abs, -d_abs] {
                        if d < -max_sum_n || d > max_sum_n {
                            continue;
                        }

                        // Check d has correct parity
                        if mod_positive(d, 2) != cd_parity {
                            continue;
                        }

                        // Check Equation 2.5 for even n (depends on c,d)
                        if n_even && mod_positive(c, 4) != mod_positive(d, 4) {
                            continue;
                        }

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

    // Precompute perfect squares for O(1) lookup
    let perfect_squares = build_perfect_square_map(max_sum_n);

    let ab_parity = ((n + 1) % 2) as i32;
    let cd_parity = (n % 2) as i32;
    let n_even = n % 2 == 0;

    let mut a_star = first_with_parity(max_sum_m, ab_parity);
    while a_star <= max_sum_m {
        let a_sq = a_star * a_star;
        if a_sq > target {
            a_star += 2;
            continue;
        }

        let mut b_star = first_with_parity(max_sum_m, ab_parity);
        while b_star <= max_sum_m {
            let ab_sum_sq = a_sq + b_star * b_star;
            if ab_sum_sq > target {
                b_star += 2;
                continue;
            }

            // For odd n, check equation 2.5 early
            if !n_even && mod_positive(a_star, 4) != mod_positive(b_star + 2, 4) {
                b_star += 2;
                continue;
            }

            let remaining = target - ab_sum_sq;

            let mut c_star = first_with_parity(max_sum_n, cd_parity);
            while c_star <= max_sum_n {
                let c_sq = c_star * c_star;
                let d_sq_needed = remaining - c_sq;

                if d_sq_needed < 0 {
                    if c_star >= 0 {
                        break;
                    }
                    c_star += 2;
                    continue;
                }

                if let Some(&d_abs) = perfect_squares.get(&d_sq_needed) {
                    for &d_star in &[d_abs, -d_abs] {
                        if d_star < -max_sum_n || d_star > max_sum_n {
                            continue;
                        }

                        if mod_positive(d_star, 2) != cd_parity {
                            continue;
                        }

                        if n_even && mod_positive(c_star, 4) != mod_positive(d_star, 4) {
                            continue;
                        }

                        alt_tuples.push(AltSumTuple {
                            a_star,
                            b_star,
                            c_star,
                            d_star,
                        });
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

/// Integer square root (avoids floating point issues)
/// Used in tests, replaced by HashMap lookup in hot path
#[inline]
#[allow(dead_code)]
fn isqrt(n: i32) -> i32 {
    if n < 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_tuple_search_v2() {
        let tuples = find_valid_sum_tuples_fast_v2(3);
        assert!(!tuples.is_empty());

        // Verify all tuples satisfy constraints
        for (st, at) in &tuples {
            let sum_sq = st.a * st.a + st.b * st.b + st.c * st.c + st.d * st.d;
            assert_eq!(sum_sq, 14); // 4*3 + 2 = 14

            let alt_sum_sq = at.a_star * at.a_star + at.b_star * at.b_star +
                             at.c_star * at.c_star + at.d_star * at.d_star;
            assert_eq!(alt_sum_sq, 14);
        }
    }

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(10), 3);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(78), 8); // sqrt(78) ≈ 8.83
    }

    #[test]
    fn test_v2_matches_v1() {
        // Compare results with original implementation
        use crate::fast_tuple_search::find_valid_sum_tuples_fast;

        for n in 1..=5 {
            let v1 = find_valid_sum_tuples_fast(n);
            let v2 = find_valid_sum_tuples_fast_v2(n);
            assert_eq!(v1.len(), v2.len(), "Mismatch for n={}", n);
        }
    }
}
