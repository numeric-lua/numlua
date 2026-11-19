# numlua performance notes

`numlua` is a facade, not a tensor engine. Most rows below therefore compare a
facade call with the equivalent direct `luatorch`/ATen call; the expected result
is parity, and a ratio above `1.0` is a facade overhead, not extra kernel work.
The `.npy` rows use a deliberately stripped raw Lua/FFI read/write as a lower
bound. These numbers are a snapshot for reporting, **not a CI gate**.

## Conditions

| Item | Value |
| --- | --- |
| Date of run | 2026-11-19 20:04 -03:00 |
| Host | Linux `machinedr` 7.3.0-rc6-1-MANJARO, x86-64 |
| CPU | AMD Ryzen 3 7320U with Radeon Graphics, 4 cores / 8 logical CPUs, 4 MiB L3 |
| Memory / load | 7.1 GiB RAM, load average about 4.8 on 8 logical CPUs during the run |
| Lua | LuaJIT 2.1.1788856981 |
| luatorch | 0.10.0 (native shim built `Release`, `-O3 -DNDEBUG`) |
| libtorch | 2.5.1+cpu |
| Compiler | gcc (GCC) 16.2.1 20260810 |
| Threads | 1 (`torch.set_num_threads(1)` in the harness) |
| Source tree | numlua 0.2.0 branch `agent/20261119-230000-add-numpy-parity-fixture-and-benchmark-s`, based on `921d8dc` |

The host is a shared, laptop-class machine. Its CPU throttles by roughly 10-20%
and the run took place at a load average of about 4.8 on 8 logical CPUs, so
differences below about 10% should be treated as noise rather than a
performance change.

## Method

```sh
./scripts/bench.sh --samples 5 --warmup 2
```

For each row the harness:

1. loads `numlua` and the direct `torch` baseline in the same process;
2. runs two warmup calls of each side;
3. takes five wall-clock samples (`clock_gettime(CLOCK_MONOTONIC)`) of a fixed
   iteration batch, alternating which side runs first;
4. reports the median per-call time for each side and the ratio
   `numlua / baseline`.

Sizes are fixed in `bench/bench.lua` (for example 4M-element reductions,
2048x2048 layout views, and 1024x1024 float32 `.npy` files). The `.npy`
baseline uses a precomputed header for writing and a known payload offset for
reading, so it omits header construction and validation; it is a lower bound,
not an equal-feature implementation. No pass/fail threshold is applied.

## Results

Median milliseconds per call; `ratio = numlua / baseline`, lower is better.

| Group | Benchmark | numlua ms | baseline ms | ratio | Baseline |
| --- | --- | ---: | ---: | ---: | --- |
| construction | `array` nested 256x256 | 1.6396 | 1.6662 | 0.98x | `torch.tensor` (ATen) |
| construction | `zeros` 1024x1024 | 1.0722 | 1.0869 | 0.99x | `torch.zeros` (ATen) |
| construction | `arange` 1M float32 | 1.3798 | 1.4613 | 0.94x | `torch.arange` (ATen) |
| reductions | `sum` all 4M float32 | 1.1822 | 1.1605 | 1.02x | `Tensor:sum` (ATen) |
| reductions | `sum` axis=2 1024x1024 | 0.3167 | 0.3673 | 0.86x | `Tensor:sum` (ATen) |
| reductions | `mean` all 1M int64 | 3.1951 | 3.1298 | 1.02x | `to(float64):mean` (ATen) |
| reductions | `max` axis=2 1024x1024 | 6.4157 | 6.0854 | 1.05x | `Tensor:max` (ATen) |
| broadcasting | `broadcast_to` 1024x1024 | 0.0506 | 0.0498 | 1.02x | `Tensor:expand` (ATen) |
| broadcasting | `broadcast_arrays` row+column | 0.1056 | 0.1043 | 1.01x | manual expand pair (ATen) |
| broadcasting | `where` broadcast 1024x1024 | 6.8649 | 6.8863 | 1.00x | `torch.where` (ATen) |
| layout | `transpose` 2048x2048 | 0.3672 | 0.3644 | 1.01x | `Tensor:permute` (ATen) |
| layout | `reshape` 4M contiguous | 0.3207 | 0.3205 | 1.00x | `Tensor:reshape` (ATen) |
| layout | `ravel` 4M contiguous | 0.3933 | 0.5142 | 0.76x | `Tensor:reshape` (ATen) |
| layout | `ravel` 1024x1024 transposed | 3.2657 | 3.2754 | 1.00x | permute + reshape (ATen) |
| layout | `tile` 1024x1024 x2 rows | 4.1776 | 2.8716 | 1.45x | `torch.cat` (ATen) |
| layout | `flip` 2048x2048 | 5.9818 | 7.4940 | 0.80x | `Tensor:flip` (ATen) |
| .npy I/O | `save` 1024x1024 float32 | 11.2013 | 8.5139 | 1.32x | raw Lua/FFI write |
| .npy I/O | `load` 1024x1024 float32 | 6.2107 | 4.9539 | 1.25x | raw Lua/FFI read |

## Interpretation and explicit losses

- **Delegation parity.** Construction, reductions, broadcasting, and view-heavy
  layout calls land between 0.94x and 1.05x of the direct ATen call. That is the
  intended shape of this library: the facade does argument checking and then
  calls the same kernel/view constructor.
- **`tile` is a real loss.** `numlua.tile` materializes blocks with
  `index_select` and measured 1.45x slower than the equivalent
  `torch.cat({ matrix, matrix }, 1)` for a 1024x1024 source. It is correct and
  remains a copy, but it is a candidate for a `cat`/`expand` fast path if tiling
  becomes hot.
- **`.npy` I/O has unavoidable facade cost.** Saving measured 1.32x and loading
  1.25x slower than the raw Lua/FFI baseline. The baseline does no header
  construction or parsing, no dtype/shape validation, and writes a precomputed
  header; `numlua` also validates arguments and makes the tensor contiguous on
  save. For this 4 MiB file the absolute cost is roughly 3 ms per save and
  1.3 ms per load.
- **Apparent wins are noise.** The 0.76x `ravel`, 0.80x `flip`, and 0.86x
  axis-sum rows are faster than their ATen baseline in this run even though both
  sides execute the same view/kernel call. They are sampling and scheduling
  noise on this shared host, not optimizations; treat them as parity.
- **No correlation with correctness coverage.** The fixture/regression work is
  verified by `./scripts/test.sh`; timing is intentionally not part of it.

## Reproduce and caveats

```sh
./scripts/bench.sh                 # defaults: 5 samples, 2 warmups
./scripts/bench.sh --samples 7 --warmup 3
```

- Numbers are machine-specific. Do not compare absolute milliseconds with
  another host; compare ratios measured in the same run.
- The workload is CPU-only and single-threaded. Multi-threaded throughput and
  GPU behavior are out of scope.
- Use the median and repeat runs when a row matters. The harness prints raw
  medians only; it does not write a result file or fail on a threshold.
