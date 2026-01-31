/// Ultra-optimized Stochastic Local Search for finding A,B given C,D
///
/// Key optimizations:
/// 1. Precomputed C,D autocorrelations (computed once)
/// 2. Incremental energy updates (O(n) instead of O(n²) per move)
/// 3. Precomputed swap indices (O(1) valid swap selection)
/// 4. O(1) alternating sum preservation check
/// 5. Late Acceptance Hill Climbing (LAHC) - often better than SA
/// 6. Parallel restarts with early termination

use crate::Sequence;
use rand::Rng;
use std::collections::VecDeque;

/// Precomputed data for a sequence that supports fast swaps
#[derive(Clone)]
pub struct IndexedSequence {
    pub values: Vec<i32>,
    pub plus_indices: Vec<usize>,   // indices where value = +1
    pub minus_indices: Vec<usize>,  // indices where value = -1
    pub alt_sum: i32,               // cached alternating sum
}

impl IndexedSequence {
    pub fn new(values: Vec<i32>) -> Self {
        let mut plus_indices = Vec::new();
        let mut minus_indices = Vec::new();
        let mut alt_sum = 0i32;

        for (i, &v) in values.iter().enumerate() {
            if v == 1 {
                plus_indices.push(i);
            } else {
                minus_indices.push(i);
            }
            let sign = if i % 2 == 0 { 1 } else { -1 };
            alt_sum += sign * v;
        }

        IndexedSequence {
            values,
            plus_indices,
            minus_indices,
            alt_sum,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn sum(&self) -> i32 {
        // sum = #plus - #minus = 2*#plus - n
        2 * self.plus_indices.len() as i32 - self.values.len() as i32
    }

    /// Swap a +1 at position i with a -1 at position j
    /// Returns the change in alternating sum (delta)
    #[inline]
    pub fn swap_plus_minus(&mut self, plus_idx: usize, minus_idx: usize) -> i32 {
        let i = self.plus_indices[plus_idx];
        let j = self.minus_indices[minus_idx];

        // Calculate alternating sum delta BEFORE swapping
        let sign_i = if i % 2 == 0 { 1 } else { -1 };
        let sign_j = if j % 2 == 0 { 1 } else { -1 };

        // Old contribution: sign_i * (+1) + sign_j * (-1) = sign_i - sign_j
        // New contribution: sign_i * (-1) + sign_j * (+1) = -sign_i + sign_j
        // Delta = new - old = -2*sign_i + 2*sign_j = 2*(sign_j - sign_i)
        let delta = 2 * (sign_j - sign_i);

        // Perform the swap
        self.values[i] = -1;
        self.values[j] = 1;
        self.plus_indices[plus_idx] = j;
        self.minus_indices[minus_idx] = i;
        self.alt_sum += delta;

        delta
    }

    /// Undo a swap (swap back)
    #[inline]
    pub fn undo_swap(&mut self, plus_idx: usize, minus_idx: usize, delta: i32) {
        let i = self.minus_indices[minus_idx]; // Now contains the old plus position
        let j = self.plus_indices[plus_idx];   // Now contains the old minus position

        self.values[i] = 1;
        self.values[j] = -1;
        self.plus_indices[plus_idx] = i;
        self.minus_indices[minus_idx] = j;
        self.alt_sum -= delta;
    }

    pub fn to_sequence(&self) -> Sequence {
        Sequence::new(self.values.clone())
    }
}

/// Ultra-optimized SLS search
pub struct SLSOptimized {
    n: usize,                    // length of C and D
    m: usize,                    // length of A and B (n+1)
    cd_autocorr: Vec<i32>,       // precomputed C+D autocorrelations for each shift
}

/// Result from adaptive restart tracking
#[derive(Clone, Debug)]
pub struct RestartStats {
    pub best_energy: i64,
    pub iterations_used: usize,
    pub improved: bool,
}

impl SLSOptimized {
    pub fn new(c: &Sequence, d: &Sequence) -> Self {
        let n = c.len();
        let m = n + 1;

        // Precompute C+D autocorrelations (this never changes during search)
        let mut cd_autocorr = vec![0i32; n];
        for shift in 0..n {
            cd_autocorr[shift] = c.autocorrelation(shift) + d.autocorrelation(shift);
        }

        SLSOptimized { n, m, cd_autocorr }
    }

    /// Calculate full energy from scratch (used for initial energy)
    fn calculate_energy_full(&self, a: &IndexedSequence, b: &IndexedSequence) -> i64 {
        let mut energy = 0i64;
        for shift in 1..self.n {
            let a_ac = self.autocorrelation_seq(a, shift);
            let b_ac = self.autocorrelation_seq(b, shift);
            let total = a_ac + b_ac + self.cd_autocorr[shift];
            energy += (total as i64) * (total as i64);
        }
        energy
    }

    /// Calculate autocorrelation for a single sequence at given shift
    #[inline]
    fn autocorrelation_seq(&self, seq: &IndexedSequence, shift: usize) -> i32 {
        let mut sum = 0i32;
        let len = seq.len();
        for j in 0..(len - shift) {
            sum += seq.values[j] * seq.values[j + shift];
        }
        sum
    }

    /// Calculate the CHANGE in autocorrelation when swapping positions i and j
    /// where values[i] = +1 and values[j] = -1 (before swap)
    #[inline]
    pub fn autocorr_delta_for_swap(&self, seq: &IndexedSequence, i: usize, j: usize, shift: usize) -> i32 {
        let len = seq.len();
        let mut delta = 0i32;

        // The swap changes values[i] from +1 to -1, and values[j] from -1 to +1
        // This affects pairs (k, k+shift) where k=i-shift, k=i, k=j-shift, or k=j

        // Contribution from position i (as the left element of a pair)
        if i + shift < len {
            let neighbor = seq.values[i + shift];
            // Old: (+1) * neighbor, New: (-1) * neighbor
            // Delta: -2 * neighbor
            delta -= 2 * neighbor;
        }

        // Contribution from position i (as the right element of a pair)
        if i >= shift {
            let neighbor = seq.values[i - shift];
            // Old: neighbor * (+1), New: neighbor * (-1)
            // Delta: -2 * neighbor
            delta -= 2 * neighbor;
        }

        // Contribution from position j (as the left element of a pair)
        if j + shift < len {
            let neighbor = seq.values[j + shift];
            // Old: (-1) * neighbor, New: (+1) * neighbor
            // Delta: +2 * neighbor
            delta += 2 * neighbor;
        }

        // Contribution from position j (as the right element of a pair)
        if j >= shift {
            let neighbor = seq.values[j - shift];
            // Old: neighbor * (-1), New: neighbor * (+1)
            // Delta: +2 * neighbor
            delta += 2 * neighbor;
        }

        // Special case: if |i - j| == shift, we've double-counted
        // The pair (min(i,j), max(i,j)) involves both swapped positions
        if i + shift == j {
            // Pair (i, j): old = (+1)*(-1) = -1, new = (-1)*(+1) = -1
            // No change from this pair itself, but we counted it twice above
            // We added: -2*(-1) from i's right neighbor being j = +2
            //           +2*(+1) from j's left neighbor being i = +2
            // But actual change is 0, so subtract the extra +4
            delta -= 4;
        } else if j + shift == i {
            // Pair (j, i): old = (-1)*(+1) = -1, new = (+1)*(-1) = -1
            // Similar correction
            delta -= 4;
        }

        delta
    }

    /// Find the shift with highest absolute autocorrelation (the "worst" shift)
    #[inline]
    fn find_worst_shift(&self, current_autocorrs: &[i32]) -> usize {
        let mut worst_shift = 1;
        let mut worst_val = 0i32;
        for shift in 1..self.n {
            let abs_val = current_autocorrs[shift].abs();
            if abs_val > worst_val {
                worst_val = abs_val;
                worst_shift = shift;
            }
        }
        worst_shift
    }

    /// Find a swap that targets the worst shift (lightweight version)
    /// Returns (plus_idx, minus_idx, i, j) where i is +1 position and j is -1 position
    fn find_targeted_swap(
        &self,
        seq: &IndexedSequence,
        current_autocorrs: &[i32],
        _seq_is_a: bool,
        _a: &IndexedSequence,
        _b: &IndexedSequence,
        rng: &mut impl Rng,
    ) -> Option<(usize, usize, usize, usize)> {
        if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() {
            return None;
        }

        let worst_shift = self.find_worst_shift(current_autocorrs);
        let target_ac = current_autocorrs[worst_shift];

        // Try a few candidate swaps and pick one that reduces worst shift (fast O(1) check)
        let num_candidates = 3.min(seq.plus_indices.len()).min(seq.minus_indices.len());

        let mut best_swap: Option<(usize, usize, usize, usize, i32)> = None;

        for _ in 0..num_candidates {
            let plus_idx = rng.gen_range(0..seq.plus_indices.len());
            let minus_idx = rng.gen_range(0..seq.minus_indices.len());
            let i = seq.plus_indices[plus_idx];
            let j = seq.minus_indices[minus_idx];

            // Must preserve alternating sum
            if i % 2 != j % 2 {
                continue;
            }

            // Only calculate delta for worst shift (O(1) instead of O(n))
            let delta = self.autocorr_delta_for_swap(seq, i, j, worst_shift);
            let new_ac = target_ac + delta;

            // Improvement = reduction in |autocorr| for worst shift
            let improvement = target_ac.abs() - new_ac.abs();

            match &best_swap {
                None => best_swap = Some((plus_idx, minus_idx, i, j, improvement)),
                Some((_, _, _, _, best_imp)) if improvement > *best_imp => {
                    best_swap = Some((plus_idx, minus_idx, i, j, improvement));
                }
                _ => {}
            }
        }

        best_swap.map(|(pi, mi, i, j, _)| (pi, mi, i, j))
    }

    /// Calculate the CHANGE in energy when swapping positions i,j in sequence seq_idx
    /// Returns (new_energy, can be computed incrementally)
    #[inline]
    fn calculate_energy_delta(
        &self,
        a: &IndexedSequence,
        b: &IndexedSequence,
        current_autocorrs: &[i32],  // current A+B+C+D autocorrelations
        seq_is_a: bool,
        i: usize,  // position of +1
        j: usize,  // position of -1
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

    /// Main search using Late Acceptance Hill Climbing (LAHC)
    /// LAHC often outperforms simulated annealing for constraint satisfaction
    pub fn search_lahc(
        &self,
        a_sum: i32,
        a_alt: i32,
        b_sum: i32,
        b_alt: i32,
        max_restarts: usize,
        iterations_per_restart: usize,
    ) -> Option<(Sequence, Sequence)> {
        // Scale LAHC history length with n for better escape from local minima at larger n
        let history_length = (200 + self.n * 10).min(2000);

        for _ in 0..max_restarts {
            let mut rng = rand::thread_rng();

            // Generate initial random sequences
            let mut a = match self.random_indexed_sequence(self.m, a_sum, a_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };
            let mut b = match self.random_indexed_sequence(self.m, b_sum, b_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };

            // Calculate initial energy and autocorrelations
            let mut energy = self.calculate_energy_full(&a, &b);

            if energy == 0 {
                return Some((a.to_sequence(), b.to_sequence()));
            }

            // Current autocorrelations (A+B+C+D for each shift)
            let mut current_autocorrs = vec![0i32; self.n];
            for shift in 1..self.n {
                current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                    + self.autocorrelation_seq(&b, shift)
                    + self.cd_autocorr[shift];
            }

            // LAHC history (circular buffer of past energies)
            let mut history: VecDeque<i64> = VecDeque::with_capacity(history_length);
            for _ in 0..history_length {
                history.push_back(energy);
            }

            let mut best_energy = energy;
            let mut best_a = a.clone();
            let mut best_b = b.clone();

            for iter in 0..iterations_per_restart {
                // Choose which sequence to modify
                let modify_a = rng.gen_bool(0.5);
                let seq = if modify_a { &mut a } else { &mut b };
                let _target_alt = if modify_a { a_alt } else { b_alt };

                // Skip if no valid swaps possible
                if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() {
                    continue;
                }

                // Select random +1 and -1 positions
                let plus_idx = rng.gen_range(0..seq.plus_indices.len());
                let minus_idx = rng.gen_range(0..seq.minus_indices.len());
                let i = seq.plus_indices[plus_idx];
                let j = seq.minus_indices[minus_idx];

                // O(1) check: does this swap preserve alternating sum?
                // Swap preserves alt_sum iff both positions have same parity
                if i % 2 != j % 2 {
                    continue;  // Would change alternating sum
                }

                // Calculate new energy incrementally
                let new_energy = self.calculate_energy_delta(
                    &a, &b, &current_autocorrs, modify_a, i, j
                );

                // LAHC acceptance: accept if new_energy <= energy from L iterations ago
                let history_energy = *history.front().unwrap();
                let accept = new_energy <= history_energy || new_energy <= energy;

                if accept {
                    // Calculate autocorrelation deltas BEFORE the swap
                    // (autocorr_delta_for_swap assumes values[i]=+1, values[j]=-1)
                    let seq_ref = if modify_a { &a } else { &b };
                    let mut ac_deltas = vec![0i32; self.n];
                    for shift in 1..self.n {
                        ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift);
                    }

                    // Now perform the swap
                    let seq = if modify_a { &mut a } else { &mut b };
                    let _delta = seq.swap_plus_minus(plus_idx, minus_idx);

                    // Update autocorrelations with pre-computed deltas
                    for shift in 1..self.n {
                        current_autocorrs[shift] += ac_deltas[shift];
                    }

                    energy = new_energy;

                    if energy < best_energy {
                        best_energy = energy;
                        best_a = a.clone();
                        best_b = b.clone();

                        if energy == 0 {
                            return Some((best_a.to_sequence(), best_b.to_sequence()));
                        }
                    }
                }

                // Update LAHC history
                history.pop_front();
                history.push_back(energy);

                // Periodic restart of history if stuck (scale interval with n)
                let reset_interval = (30000 + self.n * 500).min(100000);
                if iter > 0 && iter % reset_interval == 0 && energy > 0 {
                    // Reset history to current energy (allows escape from local minima)
                    for e in history.iter_mut() {
                        *e = energy;
                    }
                }
            }

            // If best is very close to 0, try a focused search
            if best_energy > 0 && best_energy < 50 {
                if let Some(result) = self.focused_search(&best_a, &best_b, a_alt, b_alt) {
                    return Some(result);
                }
            }
        }

        None
    }

    /// V5 Enhanced search with shift-targeted moves and adaptive restart support
    /// Returns (solution, restart_stats) for adaptive restart decisions
    pub fn search_v5(
        &self,
        a_sum: i32,
        a_alt: i32,
        b_sum: i32,
        b_alt: i32,
        max_restarts: usize,
        iterations_per_restart: usize,
    ) -> (Option<(Sequence, Sequence)>, Vec<RestartStats>) {
        let history_length = (200 + self.n * 10).min(2000);
        // Only track stats for first few restarts to reduce overhead
        let stats_limit = 5.min(max_restarts);
        let mut restart_stats = Vec::with_capacity(stats_limit);

        // Probability of using targeted vs random move (scale with n)
        // Lower for smaller n since random exploration works well there
        let targeted_prob = if self.n < 20 {
            0.15  // 15% for smaller problems
        } else {
            (0.2 + (self.n as f64 - 20.0) * 0.01).min(0.4)  // 20-40% for larger problems
        };

        for restart_idx in 0..max_restarts {
            let mut rng = rand::thread_rng();

            let mut a = match self.random_indexed_sequence(self.m, a_sum, a_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };
            let mut b = match self.random_indexed_sequence(self.m, b_sum, b_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };

            let mut energy = self.calculate_energy_full(&a, &b);

            if energy == 0 {
                if restart_stats.len() < stats_limit {
                    restart_stats.push(RestartStats { best_energy: 0, iterations_used: 0, improved: true });
                }
                return (Some((a.to_sequence(), b.to_sequence())), restart_stats);
            }

            let mut current_autocorrs = vec![0i32; self.n];
            for shift in 1..self.n {
                current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                    + self.autocorrelation_seq(&b, shift)
                    + self.cd_autocorr[shift];
            }

            let mut history: VecDeque<i64> = VecDeque::with_capacity(history_length);
            for _ in 0..history_length {
                history.push_back(energy);
            }

            let initial_energy = energy;
            let mut best_energy = energy;
            let mut best_a = a.clone();
            let mut best_b = b.clone();
            let mut actual_iters = 0usize;

            for iter in 0..iterations_per_restart {
                actual_iters = iter;
                let modify_a = rng.gen_bool(0.5);
                let seq = if modify_a { &a } else { &b };

                if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() {
                    continue;
                }

                // Choose between targeted and random move
                let (plus_idx, minus_idx, i, j) = if rng.gen_bool(targeted_prob) {
                    // Targeted move
                    match self.find_targeted_swap(seq, &current_autocorrs, modify_a, &a, &b, &mut rng) {
                        Some(swap) => swap,
                        None => continue,
                    }
                } else {
                    // Random move
                    let plus_idx = rng.gen_range(0..seq.plus_indices.len());
                    let minus_idx = rng.gen_range(0..seq.minus_indices.len());
                    let i = seq.plus_indices[plus_idx];
                    let j = seq.minus_indices[minus_idx];

                    if i % 2 != j % 2 {
                        continue;
                    }
                    (plus_idx, minus_idx, i, j)
                };

                let new_energy = self.calculate_energy_delta(&a, &b, &current_autocorrs, modify_a, i, j);

                // LAHC acceptance
                let history_energy = *history.front().unwrap();
                let accept = new_energy <= history_energy || new_energy <= energy;

                if accept {
                    let seq_ref = if modify_a { &a } else { &b };
                    let mut ac_deltas = vec![0i32; self.n];
                    for shift in 1..self.n {
                        ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift);
                    }

                    let seq = if modify_a { &mut a } else { &mut b };
                    seq.swap_plus_minus(plus_idx, minus_idx);

                    for shift in 1..self.n {
                        current_autocorrs[shift] += ac_deltas[shift];
                    }

                    energy = new_energy;

                    if energy < best_energy {
                        best_energy = energy;
                        best_a = a.clone();
                        best_b = b.clone();

                        if energy == 0 {
                            if restart_stats.len() < stats_limit {
                                restart_stats.push(RestartStats {
                                    best_energy: 0,
                                    iterations_used: iter,
                                    improved: true,
                                });
                            }
                            return (Some((best_a.to_sequence(), best_b.to_sequence())), restart_stats);
                        }
                    }
                }

                // Update LAHC history
                history.pop_front();
                history.push_back(energy);

                // Periodic reset
                let reset_interval = (30000 + self.n * 500).min(100000);
                if iter > 0 && iter % reset_interval == 0 && energy > 0 {
                    for e in history.iter_mut() {
                        *e = energy;
                    }
                }
            }

            // Record restart stats (only for first few to reduce overhead)
            if restart_stats.len() < stats_limit {
                restart_stats.push(RestartStats {
                    best_energy,
                    iterations_used: actual_iters,
                    improved: best_energy < initial_energy,
                });
            }

            // Focused search if close
            if best_energy > 0 && best_energy < 50 {
                if let Some(result) = self.focused_search(&best_a, &best_b, a_alt, b_alt) {
                    return (Some(result), restart_stats);
                }
            }
        }

        (None, restart_stats)
    }

    /// Hybrid search: LAHC + Simulated Annealing
    pub fn search_hybrid(
        &self,
        a_sum: i32,
        a_alt: i32,
        b_sum: i32,
        b_alt: i32,
        max_restarts: usize,
        iterations_per_restart: usize,
    ) -> Option<(Sequence, Sequence)> {
        // Try LAHC first (often faster)
        if let Some(result) = self.search_lahc(a_sum, a_alt, b_sum, b_alt, max_restarts / 2, iterations_per_restart) {
            return Some(result);
        }

        // Fall back to SA if LAHC didn't find solution
        self.search_sa(a_sum, a_alt, b_sum, b_alt, max_restarts / 2, iterations_per_restart)
    }

    /// Enhanced hybrid search with better escape mechanisms
    /// Returns (solution, best_energy_found) - best_energy helps with early termination decisions
    pub fn search_hybrid_v2(
        &self,
        a_sum: i32,
        a_alt: i32,
        b_sum: i32,
        b_alt: i32,
        max_restarts: usize,
        iterations_per_restart: usize,
    ) -> (Option<(Sequence, Sequence)>, i64) {
        let history_length = (200 + self.n * 10).min(2000);
        let plateau_threshold = (2000 + self.n * 100).min(10000);
        let mut global_best_energy = i64::MAX;

        for restart in 0..max_restarts {
            let mut rng = rand::thread_rng();

            let mut a = match self.random_indexed_sequence(self.m, a_sum, a_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };
            let mut b = match self.random_indexed_sequence(self.m, b_sum, b_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };

            let mut energy = self.calculate_energy_full(&a, &b);

            if energy == 0 {
                return (Some((a.to_sequence(), b.to_sequence())), 0);
            }

            let mut current_autocorrs = vec![0i32; self.n];
            for shift in 1..self.n {
                current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                    + self.autocorrelation_seq(&b, shift)
                    + self.cd_autocorr[shift];
            }

            let mut history: std::collections::VecDeque<i64> = std::collections::VecDeque::with_capacity(history_length);
            for _ in 0..history_length {
                history.push_back(energy);
            }

            let mut best_energy = energy;
            let mut best_a = a.clone();
            let mut best_b = b.clone();
            let mut plateau_count = 0usize;

            // Alternate between LAHC and SA phases
            let use_sa = restart % 3 == 2;  // Every 3rd restart use SA
            let initial_temp = 50.0;
            let mut temperature = initial_temp;
            let cooling_rate = 0.99995;

            for iter in 0..iterations_per_restart {
                let modify_a = rng.gen_bool(0.5);
                let seq = if modify_a { &a } else { &b };

                if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() {
                    continue;
                }

                // Plateau escape: force diversification
                if plateau_count > plateau_threshold {
                    plateau_count = 0;
                    // Reset history to allow worse moves
                    for e in history.iter_mut() {
                        *e = energy + 50;
                    }
                    temperature = initial_temp * 0.5;
                }

                let plus_idx = rng.gen_range(0..seq.plus_indices.len());
                let minus_idx = rng.gen_range(0..seq.minus_indices.len());
                let i = seq.plus_indices[plus_idx];
                let j = seq.minus_indices[minus_idx];

                if i % 2 != j % 2 {
                    continue;
                }

                let new_energy = self.calculate_energy_delta(
                    &a, &b, &current_autocorrs, modify_a, i, j
                );

                // Acceptance decision
                let accept = if use_sa {
                    // Simulated annealing
                    let delta = new_energy as f64 - energy as f64;
                    delta < 0.0 || rng.gen_bool((-delta / temperature).exp().min(1.0))
                } else {
                    // LAHC
                    let history_energy = *history.front().unwrap();
                    new_energy <= history_energy || new_energy <= energy
                };

                if accept {
                    let seq_ref = if modify_a { &a } else { &b };
                    let mut ac_deltas = vec![0i32; self.n];
                    for shift in 1..self.n {
                        ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift);
                    }

                    let seq = if modify_a { &mut a } else { &mut b };
                    seq.swap_plus_minus(plus_idx, minus_idx);

                    for shift in 1..self.n {
                        current_autocorrs[shift] += ac_deltas[shift];
                    }

                    energy = new_energy;
                    plateau_count = 0;

                    if energy < best_energy {
                        best_energy = energy;
                        best_a = a.clone();
                        best_b = b.clone();

                        if energy == 0 {
                            return (Some((best_a.to_sequence(), best_b.to_sequence())), 0);
                        }
                    }
                } else {
                    plateau_count += 1;
                }

                // Update history/temperature
                history.pop_front();
                history.push_back(energy);
                temperature *= cooling_rate;
                if temperature < 0.01 {
                    temperature = 0.01;
                }

                // Periodic reset
                let reset_interval = (30000 + self.n * 500).min(100000);
                if iter > 0 && iter % reset_interval == 0 && energy > 0 {
                    for e in history.iter_mut() {
                        *e = energy;
                    }
                    temperature = initial_temp * 0.3;
                }
            }

            if best_energy < global_best_energy {
                global_best_energy = best_energy;
            }

            // Focused search if close
            if best_energy > 0 && best_energy < 50 {
                if let Some(result) = self.focused_search(&best_a, &best_b, a_alt, b_alt) {
                    return (Some(result), 0);
                }
            }
        }

        (None, global_best_energy)
    }

    /// Simulated Annealing search (backup method)
    pub fn search_sa(
        &self,
        a_sum: i32,
        a_alt: i32,
        b_sum: i32,
        b_alt: i32,
        max_restarts: usize,
        iterations_per_restart: usize,
    ) -> Option<(Sequence, Sequence)> {
        for _ in 0..max_restarts {
            let mut rng = rand::thread_rng();

            let mut a = match self.random_indexed_sequence(self.m, a_sum, a_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };
            let mut b = match self.random_indexed_sequence(self.m, b_sum, b_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };

            let mut energy = self.calculate_energy_full(&a, &b);

            if energy == 0 {
                return Some((a.to_sequence(), b.to_sequence()));
            }

            let mut current_autocorrs = vec![0i32; self.n];
            for shift in 1..self.n {
                current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                    + self.autocorrelation_seq(&b, shift)
                    + self.cd_autocorr[shift];
            }

            let mut best_energy = energy;
            let mut best_a = a.clone();
            let mut best_b = b.clone();

            let initial_temp = 50.0;
            let mut temperature = initial_temp;
            let cooling_rate = 0.99995;

            for iter in 0..iterations_per_restart {
                let modify_a = rng.gen_bool(0.5);
                let seq = if modify_a { &mut a } else { &mut b };

                if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() {
                    continue;
                }

                let plus_idx = rng.gen_range(0..seq.plus_indices.len());
                let minus_idx = rng.gen_range(0..seq.minus_indices.len());
                let i = seq.plus_indices[plus_idx];
                let j = seq.minus_indices[minus_idx];

                // O(1) alternating sum preservation check
                if i % 2 != j % 2 {
                    continue;
                }

                let new_energy = self.calculate_energy_delta(
                    &a, &b, &current_autocorrs, modify_a, i, j
                );

                let delta = new_energy as f64 - energy as f64;
                let accept = delta < 0.0 || rng.gen_bool((-delta / temperature).exp().min(1.0));

                if accept {
                    // Calculate deltas BEFORE the swap
                    let seq_ref = if modify_a { &a } else { &b };
                    let mut ac_deltas = vec![0i32; self.n];
                    for shift in 1..self.n {
                        ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift);
                    }

                    let seq = if modify_a { &mut a } else { &mut b };
                    seq.swap_plus_minus(plus_idx, minus_idx);

                    for shift in 1..self.n {
                        current_autocorrs[shift] += ac_deltas[shift];
                    }

                    energy = new_energy;

                    if energy < best_energy {
                        best_energy = energy;
                        best_a = a.clone();
                        best_b = b.clone();

                        if energy == 0 {
                            return Some((best_a.to_sequence(), best_b.to_sequence()));
                        }
                    }
                }

                temperature *= cooling_rate;
                if temperature < 0.01 {
                    temperature = 0.01;
                }

                // Reheat periodically (scale interval with n)
                let reheat_interval = (20000 + self.n * 300).min(80000);
                if iter > 0 && iter % reheat_interval == 0 && energy > 0 {
                    temperature = initial_temp * 0.3;
                }
            }
        }

        None
    }

    /// Focused search when we're close to a solution
    fn focused_search(
        &self,
        a: &IndexedSequence,
        b: &IndexedSequence,
        a_alt: i32,
        b_alt: i32,
    ) -> Option<(Sequence, Sequence)> {
        let mut rng = rand::thread_rng();
        let mut a = a.clone();
        let mut b = b.clone();

        let mut energy = self.calculate_energy_full(&a, &b);

        let mut current_autocorrs = vec![0i32; self.n];
        for shift in 1..self.n {
            current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                + self.autocorrelation_seq(&b, shift)
                + self.cd_autocorr[shift];
        }

        // Greedy hill climbing with restarts
        for _ in 0..10000 {
            let modify_a = rng.gen_bool(0.5);
            let seq = if modify_a { &mut a } else { &mut b };
            let _target_alt = if modify_a { a_alt } else { b_alt };

            if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() {
                continue;
            }

            let plus_idx = rng.gen_range(0..seq.plus_indices.len());
            let minus_idx = rng.gen_range(0..seq.minus_indices.len());
            let i = seq.plus_indices[plus_idx];
            let j = seq.minus_indices[minus_idx];

            if i % 2 != j % 2 {
                continue;
            }

            let new_energy = self.calculate_energy_delta(
                &a, &b, &current_autocorrs, modify_a, i, j
            );

            // Strict hill climbing
            if new_energy < energy {
                // Calculate deltas BEFORE the swap
                let seq_ref = if modify_a { &a } else { &b };
                let mut ac_deltas = vec![0i32; self.n];
                for shift in 1..self.n {
                    ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift);
                }

                let seq = if modify_a { &mut a } else { &mut b };
                seq.swap_plus_minus(plus_idx, minus_idx);

                for shift in 1..self.n {
                    current_autocorrs[shift] += ac_deltas[shift];
                }

                energy = new_energy;

                if energy == 0 {
                    return Some((a.to_sequence(), b.to_sequence()));
                }
            }
        }

        None
    }

    /// Generate a random indexed sequence satisfying sum and alternating sum constraints
    fn random_indexed_sequence(
        &self,
        n: usize,
        target_sum: i32,
        target_alt_sum: i32,
        rng: &mut impl Rng,
    ) -> Option<IndexedSequence> {
        // Calculate required number of +1s
        let num_plus = (n as i32 + target_sum) / 2;
        if num_plus < 0 || num_plus > n as i32 {
            return None;
        }
        if (n as i32 + target_sum) % 2 != 0 {
            return None;
        }

        // Start with a sequence that has correct sum
        let mut values = vec![-1i32; n];
        for i in 0..(num_plus as usize) {
            values[i] = 1;
        }

        // Shuffle to randomize
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            values.swap(i, j);
        }

        // Now adjust to get correct alternating sum
        // Current alt_sum
        let mut current_alt_sum: i32 = values.iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
            .sum();

        // We need to swap +1 and -1 at positions with DIFFERENT parities
        // to change the alternating sum
        let max_attempts = 2000;
        for _ in 0..max_attempts {
            if current_alt_sum == target_alt_sum {
                return Some(IndexedSequence::new(values));
            }

            // Find positions to swap
            let mut plus_pos = Vec::new();
            let mut minus_pos = Vec::new();
            for (i, &v) in values.iter().enumerate() {
                if v == 1 {
                    plus_pos.push(i);
                } else {
                    minus_pos.push(i);
                }
            }

            if plus_pos.is_empty() || minus_pos.is_empty() {
                break;
            }

            // To increase alt_sum: swap +1 at odd pos with -1 at even pos
            // To decrease alt_sum: swap +1 at even pos with -1 at odd pos

            let need_increase = current_alt_sum < target_alt_sum;

            let (i, j) = if need_increase {
                // Find +1 at odd position, -1 at even position
                let i_opt = plus_pos.iter().find(|&&p| p % 2 == 1).copied();
                let j_opt = minus_pos.iter().find(|&&p| p % 2 == 0).copied();
                match (i_opt, j_opt) {
                    (Some(i), Some(j)) => (i, j),
                    _ => {
                        // Try random swap
                        let i = plus_pos[rng.gen_range(0..plus_pos.len())];
                        let j = minus_pos[rng.gen_range(0..minus_pos.len())];
                        if i % 2 == j % 2 {
                            continue; // This won't change alt_sum
                        }
                        (i, j)
                    }
                }
            } else {
                // Find +1 at even position, -1 at odd position
                let i_opt = plus_pos.iter().find(|&&p| p % 2 == 0).copied();
                let j_opt = minus_pos.iter().find(|&&p| p % 2 == 1).copied();
                match (i_opt, j_opt) {
                    (Some(i), Some(j)) => (i, j),
                    _ => {
                        let i = plus_pos[rng.gen_range(0..plus_pos.len())];
                        let j = minus_pos[rng.gen_range(0..minus_pos.len())];
                        if i % 2 == j % 2 {
                            continue;
                        }
                        (i, j)
                    }
                }
            };

            // Calculate delta
            let sign_i = if i % 2 == 0 { 1 } else { -1 };
            let sign_j = if j % 2 == 0 { 1 } else { -1 };
            let delta = 2 * (sign_j - sign_i);

            // Only swap if it moves us closer to target
            let new_alt_sum = current_alt_sum + delta;
            if (new_alt_sum - target_alt_sum).abs() < (current_alt_sum - target_alt_sum).abs() {
                values.swap(i, j);
                current_alt_sum = new_alt_sum;
            }
        }

        if current_alt_sum == target_alt_sum {
            Some(IndexedSequence::new(values))
        } else {
            None
        }
    }

    /// V6 search with 2-swap moves for escaping local minima
    /// Designed for larger n (30+) where single swaps may get stuck
    pub fn search_v6(
        &self,
        a_sum: i32,
        a_alt: i32,
        b_sum: i32,
        b_alt: i32,
        max_restarts: usize,
        iterations_per_restart: usize,
    ) -> (Option<(Sequence, Sequence)>, i64) {
        let history_length = (300 + self.n * 15).min(3000);
        let mut global_best_energy = i64::MAX;

        // 2-swap probability: higher for larger n
        let two_swap_prob = if self.n < 25 {
            0.05
        } else if self.n < 35 {
            0.10
        } else {
            0.15
        };

        for _restart in 0..max_restarts {
            let mut rng = rand::thread_rng();

            let mut a = match self.random_indexed_sequence(self.m, a_sum, a_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };
            let mut b = match self.random_indexed_sequence(self.m, b_sum, b_alt, &mut rng) {
                Some(s) => s,
                None => continue,
            };

            let mut energy = self.calculate_energy_full(&a, &b);

            if energy == 0 {
                return (Some((a.to_sequence(), b.to_sequence())), 0);
            }

            let mut current_autocorrs = vec![0i32; self.n];
            for shift in 1..self.n {
                current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                    + self.autocorrelation_seq(&b, shift)
                    + self.cd_autocorr[shift];
            }

            let mut history: std::collections::VecDeque<i64> = std::collections::VecDeque::with_capacity(history_length);
            for _ in 0..history_length {
                history.push_back(energy);
            }

            let mut best_energy = energy;
            let mut best_a = a.clone();
            let mut best_b = b.clone();
            let mut plateau_count = 0usize;
            let plateau_threshold = (3000 + self.n * 150).min(15000);

            for iter in 0..iterations_per_restart {
                // Plateau escape
                if plateau_count > plateau_threshold {
                    plateau_count = 0;
                    for e in history.iter_mut() {
                        *e = energy + 100;
                    }
                }

                // Decide: 1-swap or 2-swap
                let do_two_swap = rng.gen_bool(two_swap_prob) &&
                                  a.plus_indices.len() >= 2 && a.minus_indices.len() >= 2 &&
                                  b.plus_indices.len() >= 2 && b.minus_indices.len() >= 2;

                if do_two_swap {
                    // 2-SWAP MOVE: Two swaps in the same sequence
                    let modify_a = rng.gen_bool(0.5);
                    let seq = if modify_a { &a } else { &b };

                    if seq.plus_indices.len() < 2 || seq.minus_indices.len() < 2 {
                        continue;
                    }

                    // Pick two different swap pairs
                    let plus_idx1 = rng.gen_range(0..seq.plus_indices.len());
                    let mut plus_idx2 = rng.gen_range(0..seq.plus_indices.len());
                    while plus_idx2 == plus_idx1 && seq.plus_indices.len() > 1 {
                        plus_idx2 = rng.gen_range(0..seq.plus_indices.len());
                    }

                    let minus_idx1 = rng.gen_range(0..seq.minus_indices.len());
                    let mut minus_idx2 = rng.gen_range(0..seq.minus_indices.len());
                    while minus_idx2 == minus_idx1 && seq.minus_indices.len() > 1 {
                        minus_idx2 = rng.gen_range(0..seq.minus_indices.len());
                    }

                    let i1 = seq.plus_indices[plus_idx1];
                    let j1 = seq.minus_indices[minus_idx1];
                    let i2 = seq.plus_indices[plus_idx2];
                    let j2 = seq.minus_indices[minus_idx2];

                    // Both swaps must preserve alternating sum
                    if i1 % 2 != j1 % 2 || i2 % 2 != j2 % 2 {
                        continue;
                    }

                    // Calculate combined energy delta (approximate)
                    let delta1 = self.calculate_two_swap_delta(seq, &current_autocorrs, i1, j1, i2, j2);

                    let history_energy = *history.front().unwrap();
                    let new_energy = (energy as i64 + delta1).max(0);
                    let accept = new_energy <= history_energy || new_energy <= energy;

                    if accept {
                        // Apply both swaps - do them separately to avoid borrow issues
                        if modify_a {
                            a.swap_plus_minus(plus_idx1, minus_idx1);
                            // Find new position of i2 after first swap
                            let new_plus_idx2 = a.plus_indices.iter().position(|&x| x == i2);
                            let new_minus_idx2 = a.minus_indices.iter().position(|&x| x == j2);
                            if let (Some(pi2), Some(mi2)) = (new_plus_idx2, new_minus_idx2) {
                                a.swap_plus_minus(pi2, mi2);
                            }
                        } else {
                            b.swap_plus_minus(plus_idx1, minus_idx1);
                            let new_plus_idx2 = b.plus_indices.iter().position(|&x| x == i2);
                            let new_minus_idx2 = b.minus_indices.iter().position(|&x| x == j2);
                            if let (Some(pi2), Some(mi2)) = (new_plus_idx2, new_minus_idx2) {
                                b.swap_plus_minus(pi2, mi2);
                            }
                        }

                        // Recalculate energy and autocorrs (safer for 2-swap)
                        energy = self.calculate_energy_full(&a, &b);
                        for shift in 1..self.n {
                            current_autocorrs[shift] = self.autocorrelation_seq(&a, shift)
                                + self.autocorrelation_seq(&b, shift)
                                + self.cd_autocorr[shift];
                        }
                        plateau_count = 0;

                        if energy < best_energy {
                            best_energy = energy;
                            best_a = a.clone();
                            best_b = b.clone();

                            if energy == 0 {
                                return (Some((best_a.to_sequence(), best_b.to_sequence())), 0);
                            }
                        }
                    } else {
                        plateau_count += 1;
                    }
                } else {
                    // Standard 1-SWAP MOVE
                    let modify_a = rng.gen_bool(0.5);
                    let seq = if modify_a { &a } else { &b };

                    if seq.plus_indices.is_empty() || seq.minus_indices.is_empty() {
                        continue;
                    }

                    let plus_idx = rng.gen_range(0..seq.plus_indices.len());
                    let minus_idx = rng.gen_range(0..seq.minus_indices.len());
                    let i = seq.plus_indices[plus_idx];
                    let j = seq.minus_indices[minus_idx];

                    if i % 2 != j % 2 {
                        continue;
                    }

                    let new_energy = self.calculate_energy_delta(
                        &a, &b, &current_autocorrs, modify_a, i, j
                    );

                    let history_energy = *history.front().unwrap();
                    let accept = new_energy <= history_energy || new_energy <= energy;

                    if accept {
                        let seq_ref = if modify_a { &a } else { &b };
                        let mut ac_deltas = vec![0i32; self.n];
                        for shift in 1..self.n {
                            ac_deltas[shift] = self.autocorr_delta_for_swap(seq_ref, i, j, shift);
                        }

                        let seq = if modify_a { &mut a } else { &mut b };
                        seq.swap_plus_minus(plus_idx, minus_idx);

                        for shift in 1..self.n {
                            current_autocorrs[shift] += ac_deltas[shift];
                        }

                        energy = new_energy;
                        plateau_count = 0;

                        if energy < best_energy {
                            best_energy = energy;
                            best_a = a.clone();
                            best_b = b.clone();

                            if energy == 0 {
                                return (Some((best_a.to_sequence(), best_b.to_sequence())), 0);
                            }
                        }
                    } else {
                        plateau_count += 1;
                    }
                }

                // Update history
                history.pop_front();
                history.push_back(energy);

                // Periodic reset
                let reset_interval = (40000 + self.n * 600).min(120000);
                if iter > 0 && iter % reset_interval == 0 && energy > 0 {
                    for e in history.iter_mut() {
                        *e = energy;
                    }
                }
            }

            if best_energy < global_best_energy {
                global_best_energy = best_energy;
            }

            // Focused search if close
            if best_energy > 0 && best_energy < 100 {
                if let Some(result) = self.focused_search(&best_a, &best_b, a_alt, b_alt) {
                    return (Some(result), 0);
                }
            }
        }

        (None, global_best_energy)
    }

    /// Calculate approximate energy delta for a 2-swap move
    #[inline]
    fn calculate_two_swap_delta(
        &self,
        seq: &IndexedSequence,
        current_autocorrs: &[i32],
        i1: usize, j1: usize,
        i2: usize, j2: usize,
    ) -> i64 {
        // Approximate: sum of individual deltas (ignoring interaction)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_sequence() {
        let seq = IndexedSequence::new(vec![1, -1, 1, -1, 1]);
        assert_eq!(seq.sum(), 1);
        assert_eq!(seq.alt_sum, 5); // 1 - (-1) + 1 - (-1) + 1 = 5
        assert_eq!(seq.plus_indices, vec![0, 2, 4]);
        assert_eq!(seq.minus_indices, vec![1, 3]);
    }

    #[test]
    fn test_swap_preserves_sum() {
        let mut seq = IndexedSequence::new(vec![1, -1, 1, -1, 1]);
        let original_sum = seq.sum();
        seq.swap_plus_minus(0, 0); // Swap first +1 with first -1
        assert_eq!(seq.sum(), original_sum);
    }

    #[test]
    fn test_autocorr_delta() {
        let c = Sequence::new(vec![1, -1, 1]);
        let d = Sequence::new(vec![-1, 1, -1]);
        let sls = SLSOptimized::new(&c, &d);

        let seq = IndexedSequence::new(vec![1, 1, -1, -1]);

        // Calculate actual autocorrelation change for shift=1
        let old_ac = sls.autocorrelation_seq(&seq, 1);

        // Manually swap positions 0 (+1) and 2 (-1)
        let mut seq2 = seq.clone();
        seq2.values[0] = -1;
        seq2.values[2] = 1;
        let new_ac = sls.autocorrelation_seq(&seq2, 1);

        let expected_delta = new_ac - old_ac;
        let computed_delta = sls.autocorr_delta_for_swap(&seq, 0, 2, 1);

        assert_eq!(computed_delta, expected_delta);
    }
}
