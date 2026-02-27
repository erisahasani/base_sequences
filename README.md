# Base Sequences

Search for Base Sequences BS(n+1, n) implementing the 5-step algorithm from Wang & Zhu (2025) "On Base, Normal and Near-normal Sequences" [arXiv:2506.20296](https://arxiv.org/abs/2506.20296).

## Quick Start

```bash
# Build (native CPU optimizations + LTO)
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin find_bs_generic_v6

# Run
./target/release/find_bs_generic_v6 25

# Resume from checkpoint (e.g. after spot interruption)
./target/release/find_bs_generic_v6 30 --resume

# Control thread count (defaults to all cores)
RAYON_NUM_THREADS=32 ./target/release/find_bs_generic_v6 30
```

## Algorithm

The search follows the paper's 5-step pipeline:

1. **Tuple discovery** — find (a,b,c,d) and (a\*,b\*,c\*,d\*) satisfying sum-of-squares constraints (Theorem 2.1), paired via mod-4 signature matching (Equation 2.4), reduced by 5-class isomorphic filtering
2. **Mod-3 partial sums** — enumerate position-class constraints modulo 3 (Theorem 2.3, m=3)
3. **Mod-6 CD refinement** — refine to modulo 6 (Theorem 2.3, m=6)
4. **CD generation + spectral filter** — backtrack to build C,D sequences, filtered by power spectral density bound (Theorem 2.4)
5. **AB search** — backtrack to find A,B sequences satisfying the PAF constraint (Theorem 2.2)

## Performance (V6)

| n | Time | CDs tried |
|---|------|-----------|
| 10 | instant | 7 |
| 15 | instant | 35 |
| 20 | 37s | 3,750 |
| 25 | 29s | 134 |

## Versions

- **V6** — pure paper pipeline, self-contained (~2700 lines), parallel via rayon
- **V7** — experimental variant

## Reference

Wang & Zhu (2025) "On Base, Normal and Near-normal Sequences" [arXiv:2506.20296](https://arxiv.org/abs/2506.20296)
