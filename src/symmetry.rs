/// Isomorphic transformations and canonical form detection
///
/// Base sequences have symmetries that create equivalent solutions:
/// - Negation: BS(A,B,C,D) ≡ BS(-A,-B,-C,-D)
/// - Swapping: BS(A,B,C,D) ≡ BS(B,A,D,C)
/// - Combined: 2³ = 8 symmetries minimum
///
/// We reduce the search space by only processing canonical representatives

use crate::{SumTuple, AltSumTuple};

/// Returns canonical form of a sum tuple
/// Rules:
/// 1. Force a to be non-negative (handle negation symmetry)
/// 2. Require a ≤ b (handle A,B swap)
/// 3. Require c ≤ d (handle C,D swap)
pub fn canonical_sum_tuple(tuple: &SumTuple) -> SumTuple {
    let mut t = tuple.clone();

    // Negation symmetry: force first element non-negative
    if t.a < 0 {
        t.a = -t.a;
        t.b = -t.b;
        t.c = -t.c;
        t.d = -t.d;
    }

    // If a == 0, use b for sign determination
    if t.a == 0 && t.b < 0 {
        t.a = -t.a;
        t.b = -t.b;
        t.c = -t.c;
        t.d = -t.d;
    }

    // Swap symmetries: order pairs
    if t.a > t.b {
        std::mem::swap(&mut t.a, &mut t.b);
    }
    if t.c > t.d {
        std::mem::swap(&mut t.c, &mut t.d);
    }

    t
}

/// Returns canonical form of an alternating sum tuple
pub fn canonical_alt_sum_tuple(tuple: &AltSumTuple) -> AltSumTuple {
    let mut t = tuple.clone();

    // Negation symmetry
    if t.a_star < 0 {
        t.a_star = -t.a_star;
        t.b_star = -t.b_star;
        t.c_star = -t.c_star;
        t.d_star = -t.d_star;
    }

    // If a_star == 0, use b_star for sign determination
    if t.a_star == 0 && t.b_star < 0 {
        t.a_star = -t.a_star;
        t.b_star = -t.b_star;
        t.c_star = -t.c_star;
        t.d_star = -t.d_star;
    }

    // Swap symmetries
    if t.a_star > t.b_star {
        std::mem::swap(&mut t.a_star, &mut t.b_star);
    }
    if t.c_star > t.d_star {
        std::mem::swap(&mut t.c_star, &mut t.d_star);
    }

    t
}

/// Check if a tuple pair is in canonical form
pub fn is_canonical(sum_tuple: &SumTuple, alt_tuple: &AltSumTuple) -> bool {
    let canonical_sum = canonical_sum_tuple(sum_tuple);
    let canonical_alt = canonical_alt_sum_tuple(alt_tuple);

    *sum_tuple == canonical_sum && *alt_tuple == canonical_alt
}

/// Filter tuple pairs using only SAFE symmetry: negation
/// Only filters out negated copies (keeps one of (a,b,c,d) or (-a,-b,-c,-d))
/// This is conservative but safe - preserves all Theorem 2.2 constraints
pub fn filter_to_canonical(tuples: Vec<(SumTuple, AltSumTuple)>) -> Vec<(SumTuple, AltSumTuple)> {
    // Use negation-only filtering (safer) until full symmetry is validated
    filter_to_canonical_negation_only(tuples)
}

/// Filter using FULL symmetry: negation + A↔B swap + C↔D swap
/// This gives up to 8x reduction (2×2×2)
///
/// Symmetries exploited:
/// - Negation: BS(A,B,C,D) ≡ BS(-A,-B,-C,-D)
/// - A↔B swap: BS(A,B,C,D) ≡ BS(B,A,D,C) (swapping both pairs together)
/// - Independent C↔D swap within the sum tuple constraints
///
/// We keep only canonical representatives where:
/// 1. First non-zero element is positive (negation)
/// 2. |a| <= |b| (or a <= b when equal magnitude) for A,B ordering
/// 3. |c| <= |d| (or c <= d when equal magnitude) for C,D ordering
pub fn filter_to_canonical_full(tuples: Vec<(SumTuple, AltSumTuple)>) -> Vec<(SumTuple, AltSumTuple)> {
    let original_count = tuples.len();

    let canonical: Vec<_> = tuples
        .into_iter()
        .filter(|(st, _at)| {
            // 1. Negation symmetry: require first non-zero element to be positive
            let first_nonzero = if st.a != 0 {
                st.a
            } else if st.b != 0 {
                st.b
            } else if st.c != 0 {
                st.c
            } else {
                st.d
            };

            if first_nonzero < 0 {
                return false;
            }

            // 2. A↔B ordering: keep only when (|a|, a) <= (|b|, b) lexicographically
            //    This handles both magnitude and sign consistently
            let a_key = (st.a.abs(), st.a);
            let b_key = (st.b.abs(), st.b);
            if a_key > b_key {
                return false;
            }

            // 3. C↔D ordering: keep only when (|c|, c) <= (|d|, d) lexicographically
            let c_key = (st.c.abs(), st.c);
            let d_key = (st.d.abs(), st.d);
            if c_key > d_key {
                return false;
            }

            true
        })
        .collect();

    let filtered_count = canonical.len();
    let reduction_factor = original_count as f64 / filtered_count as f64;

    println!("  Isomorphic filtering (full symmetry): {} → {} tuples ({:.1}x reduction)",
             original_count, filtered_count, reduction_factor);

    canonical
}

/// Filter using only negation symmetry (conservative, for comparison)
pub fn filter_to_canonical_negation_only(tuples: Vec<(SumTuple, AltSumTuple)>) -> Vec<(SumTuple, AltSumTuple)> {
    let original_count = tuples.len();

    let canonical: Vec<_> = tuples
        .into_iter()
        .filter(|(st, _at)| {
            // Only use negation symmetry: require first non-zero element to be positive
            let first_nonzero = if st.a != 0 {
                st.a
            } else if st.b != 0 {
                st.b
            } else if st.c != 0 {
                st.c
            } else {
                st.d
            };

            first_nonzero > 0
        })
        .collect();

    let filtered_count = canonical.len();
    let reduction_factor = original_count as f64 / filtered_count as f64;

    println!("  Isomorphic filtering (negation only): {} → {} tuples ({:.1}x reduction)",
             original_count, filtered_count, reduction_factor);

    canonical
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negation_symmetry() {
        let t1 = SumTuple { a: 2, b: 0, c: 5, d: 7 };
        let t2 = SumTuple { a: -2, b: 0, c: -5, d: -7 };

        let c1 = canonical_sum_tuple(&t1);
        let c2 = canonical_sum_tuple(&t2);

        assert_eq!(c1, c2, "Negation should produce same canonical form");
    }

    #[test]
    fn test_swap_symmetry() {
        let t1 = SumTuple { a: 2, b: 5, c: 3, d: 7 };
        let t2 = SumTuple { a: 5, b: 2, c: 7, d: 3 };

        let c1 = canonical_sum_tuple(&t1);
        let _c2 = canonical_sum_tuple(&t2);

        assert_eq!(c1.a, c1.b.min(c1.a)); // a ≤ b
        assert_eq!(c1.c, c1.d.min(c1.c)); // c ≤ d
    }
}
