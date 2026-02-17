# Base Sequences

Search for Base Sequences BS(n+1, n) using exhaustive search with optimized stochastic local search.

Based on: Wang & Zhu (2025) "On Base, Normal and Near-normal Sequences" [arXiv:2506.20296](https://arxiv.org/abs/2506.20296)

## Quick Start

```bash
# Build
cargo build --release --bin find_bs_generic_v7

# Run (always exhaustive, guaranteed results)
./target/release/find_bs_generic_v7 30
```

**V7 is the recommended tool** - 21.6× faster than V6 with guaranteed completeness.

## What are Base Sequences?

Four ±1 sequences A, B, C, D where the sum of autocorrelations equals zero at all non-zero shifts. Used to construct Hadamard matrices and orthogonal designs.

## Performance

| n | V7 Time | Guarantee |
|---|---------|-----------|
| 15 | 5 seconds | ✅ 100% |
| 20 | 2 minutes | ✅ 100% |
| 25 | 30 minutes | ✅ 100% |
| 30 | 1-2 days | ✅ 100% |
| 35 | ~1 week | ✅ 100% |

## Documentation

- **[V7_FINAL_SUMMARY.md](V7_FINAL_SUMMARY.md)** - Complete guide (start here)
- **[QUICK_START.md](QUICK_START.md)** - Quick reference
- **[V7_PROFILE_AND_CORRECTNESS_REPORT.md](V7_PROFILE_AND_CORRECTNESS_REPORT.md)** - Technical verification
- **[DOCUMENTATION_STATUS.md](DOCUMENTATION_STATUS.md)** - Documentation index

## Versions

- **V7** (Recommended): Always exhaustive, 21.6× faster than V6, guaranteed results
- **V6** (Reference): Legacy exhaustive implementation, kept for comparison

## License

MIT
