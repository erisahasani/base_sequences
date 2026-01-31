# Base Sequences

Search for Base Sequences BS(n+1, n) using stochastic local search.

Based on: Wang & Zhu (2025) "On Base, Normal and Near-normal Sequences" [arXiv:2506.20296](https://arxiv.org/abs/2506.20296)

## Usage

```bash
cargo run --release --example find_bs_generic_v5 -- 30
```

## What are Base Sequences?

Four ±1 sequences A, B, C, D where the sum of autocorrelations equals zero at all non-zero shifts. Used to construct Hadamard matrices.

## License

MIT
