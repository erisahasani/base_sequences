/// Generic BS(n+1, n) search - V5 with scalable parameters
///
/// Usage: cargo run --release --example find_bs_generic_v5 -- <n>
/// Example: cargo run --release --example find_bs_generic_v5 -- 30
///
/// V5 Scalability features:
/// 1. Parameters scale with n automatically
/// 2. Spectral filtering aggressiveness scales with n
/// 3. Batch CD pair generation with streaming (memory efficient)
/// 4. Shift-targeted moves (focus on worst autocorrelation)
/// 5. Adaptive restart strategy (reduce restarts for unpromising pairs)

use base_sequences::{BaseSequence, Sequence, SumTuple, AltSumTuple};
use base_sequences::fast_tuple_search_v2::find_valid_sum_tuples_fast_v2;
use base_sequences::sls_optimized::SLSOptimized;
use base_sequences::cd_optimized::generate_random_cd_pairs_fast;
use base_sequences::symmetry::filter_to_canonical_negation_only;
use base_sequences::spectral_filter::{passes_spectral_bound, compute_ab_headroom};
use std::time::Instant;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};
use std::fs::File;
use std::io::Write;
use std::env;

/// Batch size for CD pair streaming
const CD_BATCH_SIZE: usize = 50;

/// Scaling parameters based on n
struct ScalingConfig {
    n: usize,
    base_cd_pairs: usize,
    max_cd_pairs: usize,
    base_restarts: usize,
    max_restarts: usize,
    base_iterations: usize,
    max_iterations: usize,
    spectral_margin: f64,
    difficulty_easy: i32,
    difficulty_hard: i32,
}

impl ScalingConfig {
    fn for_n(n: usize) -> Self {
        // Problem difficulty scales roughly as O(n^2) to O(n^3)
        // Adjust parameters accordingly

        let scale_factor = (n as f64 / 20.0).powf(1.5);

        // CD pairs: more needed for larger n
        let base_cd_pairs = (300.0 * scale_factor) as usize;
        let max_cd_pairs = (1500.0 * scale_factor).min(5000.0) as usize;

        // Restarts: scale moderately
        let base_restarts = (100.0 * scale_factor.sqrt()) as usize;
        let max_restarts = (300.0 * scale_factor.sqrt()).min(500.0) as usize;

        // Iterations: scale more aggressively for larger n
        let base_iterations = (150_000.0 * scale_factor) as usize;
        let max_iterations = (500_000.0 * scale_factor).min(2_000_000.0) as usize;

        // Spectral margin: tighter for larger n (more filtering benefit)
        let spectral_margin = (4.0 - (n as f64 - 20.0) * 0.05).max(1.0);

        // Difficulty thresholds scale with sqrt(4n+2)
        let target = ((4 * n + 2) as f64).sqrt();
        let difficulty_easy = (target * 2.5) as i32;
        let difficulty_hard = (target * 4.5) as i32;

        ScalingConfig {
            n,
            base_cd_pairs: base_cd_pairs.max(200),
            max_cd_pairs,
            base_restarts: base_restarts.max(50),
            max_restarts,
            base_iterations: base_iterations.max(100_000),
            max_iterations,
            spectral_margin,
            difficulty_easy,
            difficulty_hard,
        }
    }

    /// Adaptive parameters based on tuple difficulty
    fn adaptive_params(&self, tuple_score: i32) -> (usize, usize, usize) {
        let difficulty_factor = 1.0 + (tuple_score as f64) / (4.0 * self.n as f64);

        let cd_pairs = ((self.base_cd_pairs as f64) * difficulty_factor.sqrt()) as usize;
        let restarts = ((self.base_restarts as f64) * difficulty_factor.sqrt()) as usize;
        let iterations = ((self.base_iterations as f64) * difficulty_factor) as usize;

        (
            cd_pairs.min(self.max_cd_pairs),
            restarts.min(self.max_restarts),
            iterations.min(self.max_iterations),
        )
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

/// Filter CD pairs by spectral quality and return sorted (best first)
fn filter_cd_pairs_spectral(
    pairs: Vec<(Sequence, Sequence)>,
    margin: f64,
) -> Vec<(Sequence, Sequence, f64)> {
    let mut scored: Vec<_> = pairs
        .into_iter()
        .filter_map(|(c, d)| {
            if !passes_spectral_bound(&c, &d, margin) {
                return None;
            }
            let headroom = compute_ab_headroom(&c, &d);
            if headroom < 0.0 {
                return None;
            }
            Some((c, d, headroom))
        })
        .collect();

    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    scored
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = if args.len() > 1 {
        args[1].parse().unwrap_or_else(|_| {
            eprintln!("Usage: {} <n>", args[0]);
            eprintln!("Example: {} 30", args[0]);
            std::process::exit(1);
        })
    } else {
        eprintln!("Usage: {} <n>", args[0]);
        eprintln!("Example: {} 30", args[0]);
        std::process::exit(1);
    };

    let config = ScalingConfig::for_n(n);

    println!("BS({},{}) - V5 Generic Scalable Search", n + 1, n);
    println!("========================================\n");

    println!("Scaling configuration for n={}:", n);
    println!("  CD pairs: {}-{}", config.base_cd_pairs, config.max_cd_pairs);
    println!("  Restarts: {}-{}", config.base_restarts, config.max_restarts);
    println!("  Iterations: {}k-{}k", config.base_iterations/1000, config.max_iterations/1000);
    println!("  Spectral margin: {:.1}", config.spectral_margin);
    println!("  Difficulty thresholds: easy<{}, hard>{}", config.difficulty_easy, config.difficulty_hard);
    println!();

    let start = Instant::now();

    println!("Step 1: Find valid tuples...");
    let all_tuples = find_valid_sum_tuples_fast_v2(n);
    println!("  {} raw tuples found", all_tuples.len());

    println!("Step 2: Filter by symmetry and sort by difficulty...");
    let canonical = filter_to_canonical_negation_only(all_tuples);
    let mut sorted: Vec<(SumTuple, AltSumTuple)> = canonical.into_iter().collect();
    sorted.sort_by_key(|(st, at)| score_tuple(st, at));
    println!("  {} canonical tuples", sorted.len());

    let easy = sorted.iter().filter(|(st, at)| score_tuple(st, at) < config.difficulty_easy).count();
    let medium = sorted.iter().filter(|(st, at)| {
        let s = score_tuple(st, at);
        s >= config.difficulty_easy && s < config.difficulty_hard
    }).count();
    let hard = sorted.iter().filter(|(st, at)| score_tuple(st, at) >= config.difficulty_hard).count();
    println!("  Difficulty: {} easy, {} medium, {} hard", easy, medium, hard);

    // Estimate total work
    let avg_cd = (config.base_cd_pairs + config.max_cd_pairs) / 2;
    let avg_restarts = (config.base_restarts + config.max_restarts) / 2;
    let avg_iters = (config.base_iterations + config.max_iterations) / 2;
    let est_total_iters = sorted.len() as u64 * avg_cd as u64 * avg_restarts as u64 * avg_iters as u64;
    println!("  Estimated max iterations: {:.1}T", est_total_iters as f64 / 1e12);
    println!();

    let found = Arc::new(AtomicBool::new(false));
    let tuples_checked = Arc::new(AtomicUsize::new(0));
    let cd_pairs_tried = Arc::new(AtomicUsize::new(0));
    let cd_pairs_filtered = Arc::new(AtomicUsize::new(0));
    let total_iterations = Arc::new(AtomicU64::new(0));

    println!("Step 3: Parallel search with adaptive parameters\n");

    // Progress monitor
    let found_clone = Arc::clone(&found);
    let tuples_clone = Arc::clone(&tuples_checked);
    let cd_clone = Arc::clone(&cd_pairs_tried);
    let filtered_clone = Arc::clone(&cd_pairs_filtered);
    let iters_clone = Arc::clone(&total_iterations);
    let total = sorted.len();
    let start_clone = start.clone();

    std::thread::spawn(move || {
        let mut last_tuples = 0;
        let mut last_time = 0.0;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(300)); // 5 minutes
            if found_clone.load(Ordering::Relaxed) { break; }

            let t = tuples_clone.load(Ordering::Relaxed);
            let cd = cd_clone.load(Ordering::Relaxed);
            let filtered = filtered_clone.load(Ordering::Relaxed);
            let iters = iters_clone.load(Ordering::Relaxed);
            let e = start_clone.elapsed().as_secs_f64();

            if t > 0 {
                let recent_rate = if e > last_time && t > last_tuples {
                    (t - last_tuples) as f64 / (e - last_time)
                } else {
                    t as f64 / e
                };

                let eta_min = (total - t) as f64 / recent_rate / 60.0;
                let filter_rate = if cd + filtered > 0 {
                    100.0 * filtered as f64 / (cd + filtered) as f64
                } else { 0.0 };

                println!("  [{:>5}/{}] {:>5.1}% | {} CD ({:.0}% filt) | {:.2}B iters | {:.2} t/s | ETA: {:.1}h",
                    t, total,
                    100.0 * t as f64 / total as f64,
                    cd, filter_rate,
                    iters as f64 / 1e9,
                    recent_rate,
                    eta_min / 60.0);

                last_tuples = t;
                last_time = e;
            }
        }
    });

    // Main search with batch streaming
    let config = Arc::new(config);
    let config_clone = Arc::clone(&config);

    let result: Option<(BaseSequence, usize, SumTuple, AltSumTuple)> = sorted
        .par_iter()
        .enumerate()
        .find_map_any(|(idx, (st, at))| {
            if found.load(Ordering::Relaxed) { return None; }

            let tuple_score = score_tuple(st, at);
            let (total_cd_pairs, restarts, iterations) = config_clone.adaptive_params(tuple_score);

            // BATCH STREAMING: Generate CD pairs in batches instead of all at once
            let mut total_generated = 0usize;
            let mut total_filtered_out = 0usize;
            let mut pairs_tried_this_tuple = 0usize;

            // Process CD pairs in batches
            while total_generated < total_cd_pairs {
                if found.load(Ordering::Relaxed) { return None; }

                // Generate a batch
                let batch_size = CD_BATCH_SIZE.min(total_cd_pairs - total_generated);
                let cd_batch = generate_random_cd_pairs_fast(
                    config_clone.n, st.c, st.d, at.c_star, at.d_star, batch_size
                );

                total_generated += batch_size;

                if cd_batch.is_empty() {
                    // If we can't generate any CD pairs, no point continuing
                    break;
                }

                // Filter the batch
                let filtered_batch = filter_cd_pairs_spectral(cd_batch, config_clone.spectral_margin);
                total_filtered_out += batch_size - filtered_batch.len();

                if filtered_batch.is_empty() {
                    // All pairs filtered out in this batch, try next batch
                    continue;
                }

                pairs_tried_this_tuple += filtered_batch.len();

                // Process filtered CD pairs immediately (streaming)
                for (c, d, _headroom) in &filtered_batch {
                    if found.load(Ordering::Relaxed) { return None; }
                    cd_pairs_tried.fetch_add(1, Ordering::Relaxed);

                    let sls = SLSOptimized::new(c, d);
                    let search_iters = restarts as u64 * iterations as u64;
                    total_iterations.fetch_add(search_iters, Ordering::Relaxed);

                    if let Some((a, b)) = sls.search_hybrid(
                        st.a, at.a_star, st.b, at.b_star, restarts, iterations
                    ) {
                        let base = BaseSequence::new(a, b, c.clone(), d.clone());
                        if base.is_valid() {
                            found.store(true, Ordering::Relaxed);
                            return Some((base, idx, st.clone(), at.clone()));
                        }
                    }
                }
            }

            cd_pairs_filtered.fetch_add(total_filtered_out, Ordering::Relaxed);
            tuples_checked.fetch_add(1, Ordering::Relaxed);
            None
        });

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    println!("\n");

    if let Some((base, idx, st, at)) = result {
        println!("============================================");
        println!("       SUCCESS! BS({},{}) FOUND          ", n + 1, n);
        println!("============================================\n");

        println!("Time: {:.2?} ({:.2} hours)", elapsed, elapsed_secs / 3600.0);
        println!("Tuples checked: {}", tuples_checked.load(Ordering::Relaxed));
        println!("CD pairs tried: {}", cd_pairs_tried.load(Ordering::Relaxed));
        println!("CD pairs filtered: {}", cd_pairs_filtered.load(Ordering::Relaxed));
        println!("Total iterations: {:.2}B", total_iterations.load(Ordering::Relaxed) as f64 / 1e9);
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
        let filename = format!("BS_{}_{}_V5_{:.0}s.txt", n + 1, n, elapsed_secs);
        if let Ok(mut f) = File::create(&filename) {
            writeln!(f, "BS({},{}) Solution - V5", n + 1, n).ok();
            writeln!(f, "==================").ok();
            writeln!(f, "Time: {:.1}s", elapsed_secs).ok();
            writeln!(f, "Total CD pairs: {}", cd_pairs_tried.load(Ordering::Relaxed)).ok();
            writeln!(f, "").ok();
            writeln!(f, "A = {:?}", base.a.values).ok();
            writeln!(f, "B = {:?}", base.b.values).ok();
            writeln!(f, "C = {:?}", base.c.values).ok();
            writeln!(f, "D = {:?}", base.d.values).ok();
            println!("\nSaved to: {}", filename);
        }
    } else {
        println!("============================================");
        println!("         No solution found                  ");
        println!("============================================\n");

        println!("Time: {:.2?} ({:.2} hours)", elapsed, elapsed_secs / 3600.0);
        println!("Tuples checked: {}", tuples_checked.load(Ordering::Relaxed));
        println!("CD pairs tried: {}", cd_pairs_tried.load(Ordering::Relaxed));
        println!("CD pairs filtered: {}", cd_pairs_filtered.load(Ordering::Relaxed));
        println!("Total iterations: {:.2}B", total_iterations.load(Ordering::Relaxed) as f64 / 1e9);
    }
}
