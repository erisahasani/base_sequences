/// Generic BS(n+1, n) search - V6 with advanced optimizations for large n
///
/// Usage: cargo run --release --example find_bs_generic_v6 -- <n>
/// Example: cargo run --release --example find_bs_generic_v6 -- 40
///
/// V6 Improvements over V5:
/// 1. Progressive deepening: Start shallow, go deeper if promising
/// 2. Global best energy tracking: Abandon unpromising tuples early
/// 3. Better work distribution: Process tuples in rounds for fairness
/// 4. Improved CD pair scoring: Prioritize by spectral quality
/// 5. Adaptive iteration budgets based on observed convergence
/// 6. Multi-phase search: Quick scan then deep dive

use base_sequences::{BaseSequence, Sequence, SumTuple, AltSumTuple};
use base_sequences::fast_tuple_search_v2::find_valid_sum_tuples_fast_v2;
use base_sequences::sls_optimized::SLSOptimized;
use base_sequences::cd_optimized::generate_random_cd_pairs_fast;
use base_sequences::symmetry::filter_to_canonical_negation_only;
use base_sequences::spectral_filter::{passes_spectral_bound, compute_ab_headroom};
use std::time::Instant;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, AtomicI64, Ordering};
use std::fs::File;
use std::io::Write;
use std::env;

/// Batch size for CD pair streaming
const CD_BATCH_SIZE: usize = 50;

/// V6 scaling parameters - more aggressive for large n
struct ScalingConfig {
    n: usize,
    // Phase 1: Quick scan
    phase1_cd_pairs: usize,
    phase1_restarts: usize,
    phase1_iterations: usize,
    // Phase 2: Deep dive (for promising tuples)
    phase2_cd_pairs: usize,
    phase2_restarts: usize,
    phase2_iterations: usize,
    // Filtering
    spectral_margin: f64,
    // Thresholds
    promise_threshold: i64,  // Energy below this = promising
    abandon_threshold: i64,  // Energy above this after phase1 = skip phase2
}

impl ScalingConfig {
    fn for_n(n: usize) -> Self {
        // For n=40, we need much more aggressive scaling
        let scale = (n as f64 / 20.0).powf(1.8);

        // Phase 1: Quick exploration with limited resources
        let phase1_cd_pairs = (100.0 * scale.sqrt()) as usize;
        let phase1_restarts = (30.0 * scale.sqrt()) as usize;
        let phase1_iterations = (50_000.0 * scale) as usize;

        // Phase 2: Deep search for promising tuples only
        let phase2_cd_pairs = (500.0 * scale) as usize;
        let phase2_restarts = (150.0 * scale.sqrt()) as usize;
        let phase2_iterations = (300_000.0 * scale) as usize;

        // Spectral margin tightens with n
        let spectral_margin = (3.5 - (n as f64 - 20.0) * 0.03).max(0.5);

        // Promise threshold: scale with n^2 (energy is sum of squares)
        let promise_threshold = (50.0 * (n as f64 / 20.0).powi(2)) as i64;
        let abandon_threshold = (500.0 * (n as f64 / 20.0).powi(2)) as i64;

        ScalingConfig {
            n,
            phase1_cd_pairs: phase1_cd_pairs.max(50),
            phase1_restarts: phase1_restarts.max(20),
            phase1_iterations: phase1_iterations.max(30_000),
            phase2_cd_pairs: phase2_cd_pairs.max(200).min(3000),
            phase2_restarts: phase2_restarts.max(50).min(400),
            phase2_iterations: phase2_iterations.max(100_000).min(1_500_000),
            spectral_margin,
            promise_threshold,
            abandon_threshold,
        }
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

/// Result from phase 1 search
#[derive(Clone)]
struct Phase1Result {
    tuple_idx: usize,
    best_energy: i64,
    promising_cd_pairs: Vec<(Sequence, Sequence, f64)>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = if args.len() > 1 {
        args[1].parse().unwrap_or_else(|_| {
            eprintln!("Usage: {} <n>", args[0]);
            std::process::exit(1);
        })
    } else {
        eprintln!("Usage: {} <n>", args[0]);
        std::process::exit(1);
    };

    let config = ScalingConfig::for_n(n);

    println!("BS({},{}) - V6 Advanced Scalable Search", n + 1, n);
    println!("==========================================\n");

    println!("V6 Configuration for n={}:", n);
    println!("  Phase 1 (quick scan): {} CD × {} restarts × {}k iters",
             config.phase1_cd_pairs, config.phase1_restarts, config.phase1_iterations/1000);
    println!("  Phase 2 (deep dive):  {} CD × {} restarts × {}k iters",
             config.phase2_cd_pairs, config.phase2_restarts, config.phase2_iterations/1000);
    println!("  Spectral margin: {:.2}", config.spectral_margin);
    println!("  Promise threshold: {} energy", config.promise_threshold);
    println!("  Abandon threshold: {} energy", config.abandon_threshold);
    println!();

    let start = Instant::now();

    println!("Step 1: Find valid tuples...");
    let all_tuples = find_valid_sum_tuples_fast_v2(n);
    println!("  {} raw tuples found", all_tuples.len());

    println!("Step 2: Filter and sort by difficulty...");
    let canonical = filter_to_canonical_negation_only(all_tuples);
    let mut sorted: Vec<(SumTuple, AltSumTuple)> = canonical.into_iter().collect();
    sorted.sort_by_key(|(st, at)| score_tuple(st, at));
    println!("  {} canonical tuples", sorted.len());
    println!();

    let found = Arc::new(AtomicBool::new(false));
    let tuples_phase1 = Arc::new(AtomicUsize::new(0));
    let tuples_phase2 = Arc::new(AtomicUsize::new(0));
    let cd_pairs_tried = Arc::new(AtomicUsize::new(0));
    let total_iterations = Arc::new(AtomicU64::new(0));
    let global_best_energy = Arc::new(AtomicI64::new(i64::MAX));

    // Progress monitor
    let found_clone = Arc::clone(&found);
    let phase1_clone = Arc::clone(&tuples_phase1);
    let phase2_clone = Arc::clone(&tuples_phase2);
    let cd_clone = Arc::clone(&cd_pairs_tried);
    let iters_clone = Arc::clone(&total_iterations);
    let best_clone = Arc::clone(&global_best_energy);
    let total = sorted.len();
    let start_clone = start.clone();

    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(300));
            if found_clone.load(Ordering::Relaxed) { break; }

            let p1 = phase1_clone.load(Ordering::Relaxed);
            let p2 = phase2_clone.load(Ordering::Relaxed);
            let cd = cd_clone.load(Ordering::Relaxed);
            let iters = iters_clone.load(Ordering::Relaxed);
            let best = best_clone.load(Ordering::Relaxed);
            let e = start_clone.elapsed().as_secs_f64();

            println!("  Phase1: {}/{} | Phase2: {} | {} CD | {:.2}B iters | best_E: {} | {:.1}min",
                p1, total, p2, cd, iters as f64 / 1e9,
                if best == i64::MAX { "∞".to_string() } else { best.to_string() },
                e / 60.0);
        }
    });

    println!("Step 3: Two-phase parallel search\n");
    println!("  PHASE 1: Quick scan of all tuples...\n");

    let config = Arc::new(config);
    let sorted = Arc::new(sorted);

    // PHASE 1: Quick scan all tuples, identify promising ones
    let phase1_results: Vec<Phase1Result> = sorted
        .par_iter()
        .enumerate()
        .filter_map(|(idx, (st, at))| {
            if found.load(Ordering::Relaxed) { return None; }

            let mut best_energy = i64::MAX;
            let mut promising_pairs = Vec::new();

            // Generate and filter CD pairs
            let cd_batch = generate_random_cd_pairs_fast(
                config.n, st.c, st.d, at.c_star, at.d_star, config.phase1_cd_pairs
            );

            if cd_batch.is_empty() {
                tuples_phase1.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            let filtered = filter_cd_pairs_spectral(cd_batch, config.spectral_margin);

            for (c, d, headroom) in filtered {
                if found.load(Ordering::Relaxed) { return None; }
                cd_pairs_tried.fetch_add(1, Ordering::Relaxed);

                let sls = SLSOptimized::new(&c, &d);
                let iters = config.phase1_restarts as u64 * config.phase1_iterations as u64;
                total_iterations.fetch_add(iters, Ordering::Relaxed);

                // Quick search using V6 with 2-swap capability
                let (result, search_best_energy) = sls.search_v6(
                    st.a, at.a_star, st.b, at.b_star,
                    config.phase1_restarts, config.phase1_iterations
                );

                if let Some((a, b)) = result {
                    let base = BaseSequence::new(a, b, c.clone(), d.clone());
                    if base.is_valid() {
                        found.store(true, Ordering::Relaxed);
                        // Return immediately - we found it!
                        println!("\n  FOUND IN PHASE 1!\n");
                        return Some(Phase1Result {
                            tuple_idx: idx,
                            best_energy: 0,
                            promising_cd_pairs: vec![(c, d, headroom)],
                        });
                    }
                }

                // Track best energy for this tuple using actual search result
                if headroom > 0.0 {
                    promising_pairs.push((c, d, headroom));
                    if search_best_energy < best_energy {
                        best_energy = search_best_energy;
                    }
                }
            }

            tuples_phase1.fetch_add(1, Ordering::Relaxed);

            // Update global best
            loop {
                let current_best = global_best_energy.load(Ordering::Relaxed);
                if best_energy >= current_best {
                    break;
                }
                if global_best_energy.compare_exchange(
                    current_best, best_energy, Ordering::Relaxed, Ordering::Relaxed
                ).is_ok() {
                    break;
                }
            }

            // Return promising tuples for phase 2
            if best_energy < config.abandon_threshold && !promising_pairs.is_empty() {
                Some(Phase1Result {
                    tuple_idx: idx,
                    best_energy,
                    promising_cd_pairs: promising_pairs,
                })
            } else {
                None
            }
        })
        .collect();

    if found.load(Ordering::Relaxed) {
        // Solution found in phase 1 - get and print it
        let result = phase1_results.into_iter().find(|r| r.best_energy == 0);
        if let Some(r) = result {
            let (st, at) = &sorted[r.tuple_idx];
            let (c, d, _) = &r.promising_cd_pairs[0];

            // Reconstruct the solution
            let sls = SLSOptimized::new(c, d);
            if let Some((a, b)) = sls.search_hybrid(
                st.a, at.a_star, st.b, at.b_star, 100, 100_000
            ) {
                let base = BaseSequence::new(a, b, c.clone(), d.clone());
                print_solution(n, &base, st, at, r.tuple_idx, &start,
                              tuples_phase1.load(Ordering::Relaxed),
                              cd_pairs_tried.load(Ordering::Relaxed),
                              total_iterations.load(Ordering::Relaxed));
            }
        }
        return;
    }

    println!("\n  Phase 1 complete: {} promising tuples found\n", phase1_results.len());

    if phase1_results.is_empty() {
        println!("No promising tuples found. Try adjusting parameters.");
        return;
    }

    println!("  PHASE 2: Deep search on {} promising tuples...\n", phase1_results.len());

    // PHASE 2: Deep dive on promising tuples
    // Sort by best_energy (lowest first)
    let mut promising: Vec<_> = phase1_results;
    promising.sort_by_key(|r| r.best_energy);

    let result = promising
        .par_iter()
        .find_map_any(|phase1| {
            if found.load(Ordering::Relaxed) { return None; }

            let (st, at) = &sorted[phase1.tuple_idx];
            tuples_phase2.fetch_add(1, Ordering::Relaxed);

            // Generate more CD pairs for deep search
            let mut all_pairs = phase1.promising_cd_pairs.clone();

            // Add fresh CD pairs
            let extra_batch = generate_random_cd_pairs_fast(
                config.n, st.c, st.d, at.c_star, at.d_star,
                config.phase2_cd_pairs.saturating_sub(all_pairs.len())
            );
            let extra_filtered = filter_cd_pairs_spectral(extra_batch, config.spectral_margin);
            all_pairs.extend(extra_filtered);

            // Sort by headroom (best first)
            all_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            for (c, d, _headroom) in &all_pairs {
                if found.load(Ordering::Relaxed) { return None; }
                cd_pairs_tried.fetch_add(1, Ordering::Relaxed);

                let sls = SLSOptimized::new(c, d);
                let iters = config.phase2_restarts as u64 * config.phase2_iterations as u64;
                total_iterations.fetch_add(iters, Ordering::Relaxed);

                let (result, _best_e) = sls.search_v6(
                    st.a, at.a_star, st.b, at.b_star,
                    config.phase2_restarts, config.phase2_iterations
                );

                if let Some((a, b)) = result {
                    let base = BaseSequence::new(a, b, c.clone(), d.clone());
                    if base.is_valid() {
                        found.store(true, Ordering::Relaxed);
                        return Some((base, phase1.tuple_idx, st.clone(), at.clone()));
                    }
                }
            }

            None
        });

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    println!("\n");

    if let Some((base, idx, st, at)) = result {
        print_solution(n, &base, &st, &at, idx, &start,
                      tuples_phase1.load(Ordering::Relaxed) + tuples_phase2.load(Ordering::Relaxed),
                      cd_pairs_tried.load(Ordering::Relaxed),
                      total_iterations.load(Ordering::Relaxed));
    } else {
        println!("============================================");
        println!("         No solution found                  ");
        println!("============================================\n");

        println!("Time: {:.2?} ({:.2} hours)", elapsed, elapsed_secs / 3600.0);
        println!("Phase 1 tuples: {}", tuples_phase1.load(Ordering::Relaxed));
        println!("Phase 2 tuples: {}", tuples_phase2.load(Ordering::Relaxed));
        println!("CD pairs tried: {}", cd_pairs_tried.load(Ordering::Relaxed));
        println!("Total iterations: {:.2}B", total_iterations.load(Ordering::Relaxed) as f64 / 1e9);
        println!("Best energy seen: {}", global_best_energy.load(Ordering::Relaxed));
    }
}

fn print_solution(
    n: usize,
    base: &BaseSequence,
    st: &SumTuple,
    at: &AltSumTuple,
    idx: usize,
    start: &Instant,
    tuples_checked: usize,
    cd_pairs: usize,
    iterations: u64,
) {
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    println!("============================================");
    println!("       SUCCESS! BS({},{}) FOUND          ", n + 1, n);
    println!("============================================\n");

    println!("Time: {:.2?} ({:.2} hours)", elapsed, elapsed_secs / 3600.0);
    println!("Tuples checked: {}", tuples_checked);
    println!("CD pairs tried: {}", cd_pairs);
    println!("Total iterations: {:.2}B", iterations as f64 / 1e9);
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
        writeln!(f, "BS({},{}) Solution - V6", n + 1, n).ok();
        writeln!(f, "==================").ok();
        writeln!(f, "Time: {:.1}s", elapsed_secs).ok();
        writeln!(f, "CD pairs: {}", cd_pairs).ok();
        writeln!(f, "").ok();
        writeln!(f, "A = {:?}", base.a.values).ok();
        writeln!(f, "B = {:?}", base.b.values).ok();
        writeln!(f, "C = {:?}", base.c.values).ok();
        writeln!(f, "D = {:?}", base.d.values).ok();
        println!("\nSaved to: {}", filename);
    }
}
