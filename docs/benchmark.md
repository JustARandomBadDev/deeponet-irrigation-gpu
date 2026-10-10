# ONNX Runtime CPU/CUDA benchmark

## Objective

This experiment compares the frozen DeepONet under three ONNX Runtime 1.30.0
execution paths:

1. `cpu`: CPUExecutionProvider with ordinary host tensors.
2. `cuda`: CUDAExecutionProvider with ordinary host tensors and ORT-managed
   transfers.
3. `cuda_iobinding`: CUDAExecutionProvider with persistent device buffers,
   explicit host/device copies, and ONNX Runtime I/O Binding.

This is an end-to-end application-level benchmark, not a kernel profiler. It
does not use CUDA Graphs, I/O transfer tuning, FP16, TensorRT, or custom CUDA.

## Environment and configuration

- GPU: NVIDIA GeForce RTX 2060, driver 615.78.08
- CPU: Intel Core i5-9600K at 3.70 GHz
- CUDA runtime/driver API: 13.4
- ONNX Runtime: 1.30.0, official CUDA 13 Linux x64 archive
- Build: CMake `Release`, C++20
- CPU threading: six intra-op threads and one inter-op thread
- Batches: 1, 8, 32, 128
- Warmup: 100 iterations before each measured repetition
- Measurement: 1000 iterations per repetition
- Complete repetitions: 3

Inputs are deterministic repetitions of the ten existing real golden-vector
samples. No input construction, fixture parsing, session creation, or model
loading occurs inside a measured loop. For batch sizes above ten, samples are
repeated cyclically before timing.

Each latency sample uses `std::chrono::steady_clock` and ends only after a
host-visible prediction is available. Naive CUDA therefore includes
ORT-managed transfers. I/O Binding includes three explicit host-to-device
copies, bound GPU execution, output synchronization, and the device-to-host
copy. Its four CUDA buffers are allocated once for maximum batch 128 and reused
throughout timing. No `cudaMalloc`, `cudaFree`, logging, or session recreation
occurs inside measured loops.

The persistent allocations are 370,176 bytes for `branch_input` and 512 bytes
each for `horizon`, `current_moisture`, and `prediction`, for 371,712 bytes in
total.

Reported statistics are calculated from all 3000 raw latency samples per mode
and batch. Standard deviation is the population standard deviation. Throughput
is `batch_size / mean_latency_seconds`.

## Correctness requirement

All paths must pass the existing `1e-4` FP32 golden tolerance before timing.
I/O Binding's maximum error against the golden values is `3.8147e-6`; its
outputs are bit-identical to naive CUDA for the golden fixture.

## Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/deeponet_golden_test --cpu
./build/deeponet_golden_test --cuda
./build/deeponet_golden_test --cuda-iobinding

./build/deeponet_benchmark \
  --provider all \
  --warmup 100 \
  --iterations 1000 \
  --repetitions 3
```

Individual modes can be run with `--provider cpu`, `--provider cuda`, or
`--provider cuda-iobinding`.

## Aggregate results

| Provider         | Batch | Mean ms | P50 ms | P95 ms | Stddev ms | Samples/s |
| ---------------- | ----: | ------: | -----: | -----: | --------: | --------: |
| CPU              |     1 |  0.0202 | 0.0180 | 0.0250 |    0.0231 |    49,514 |
| CPU              |     8 |  0.0291 | 0.0246 | 0.0374 |    0.0370 |   274,889 |
| CPU              |    32 |  0.0636 | 0.0514 | 0.1096 |    0.0751 |   502,776 |
| CPU              |   128 |  0.2386 | 0.1440 | 0.6492 |    0.2814 |   536,476 |
| CUDA naive       |     1 |  0.1296 | 0.1160 | 0.1554 |    0.0695 |     7,713 |
| CUDA naive       |     8 |  0.1259 | 0.1151 | 0.1353 |    0.0578 |    63,530 |
| CUDA naive       |    32 |  0.1381 | 0.1265 | 0.1469 |    0.0597 |   231,790 |
| CUDA naive       |   128 |  0.1574 | 0.1427 | 0.2784 |    0.0744 |   813,317 |
| CUDA I/O Binding |     1 |  0.1432 | 0.1184 | 0.3001 |    0.0979 |     6,986 |
| CUDA I/O Binding |     8 |  0.1394 | 0.1189 | 0.3023 |    0.0808 |    57,387 |
| CUDA I/O Binding |    32 |  0.1525 | 0.1347 | 0.2360 |    0.0814 |   209,810 |
| CUDA I/O Binding |   128 |  0.1932 | 0.1706 | 0.3589 |    0.0902 |   662,665 |

| Batch | CUDA naive / CPU | I/O Binding / CPU | I/O Binding change vs naive |
| ----: | ---------------: | ----------------: | --------------------------: |
|     1 |           0.156x |            0.141x |               10.41% slower |
|     8 |           0.231x |            0.209x |               10.71% slower |
|    32 |           0.461x |            0.417x |               10.48% slower |
|   128 |           1.516x |            1.235x |               22.73% slower |

Generated data:

- `results/benchmark/onnx_runtime_benchmark.csv`: per-repetition and aggregate
  latency/throughput statistics.
- `results/benchmark/environment.json`: hardware, software, thread, batch, and
  iteration metadata.

## Interpretation

Measured facts:

- CPU has lower end-to-end latency at batches 1, 8, and 32.
- Naive CUDA overtakes CPU only at batch 128, where mean latency is 1.516x
  faster and throughput is about 813,317 versus 536,476 samples/s.
- CUDA mean latency stays nearly flat from batch 1 through batch 32, so batching
  substantially raises throughput before batch 128.
- This I/O Binding implementation did not improve end-to-end latency at any
  tested batch. It is about 10% slower through batch 32 and 22.73% slower at
  batch 128.
- CPU batch-128 repetition means ranged from 0.1745 to 0.2990 ms. Occasional
  OS/runtime outliers make aggregate means and standard deviations materially
  larger than p50 values, especially for this row. The raw per-repetition rows
  are retained in the CSV rather than hiding that variability.

These measurements do not identify why I/O Binding is slower, nor do they
isolate kernel time from copies or launch overhead. A later profiling phase
should determine operator placement and duration, transfer behavior, stream
synchronization costs, and whether CUDA Graphs or a different transfer strategy
is justified.

