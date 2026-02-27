# Base Sequences

Search for Base Sequences BS(n+1, n) implementing the algorithm from Wang & Zhu (2025) [arXiv:2506.20296](https://arxiv.org/abs/2506.20296).

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin find_bs_v6_parallel
./target/release/find_bs_v6_parallel 30
./target/release/find_bs_v6_parallel 30 --resume  # resume from checkpoint
```
