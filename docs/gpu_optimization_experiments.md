# GPU optimization experiments

## Scope and method

These experiments keep the frozen DeepONet and the existing CPU, naive CUDA,
and host-to-host CUDA I/O Binding baselines unchanged. Release builds use the
same deterministic repeated golden inputs, nine batch sizes, 100 warmups,
1,000 measured iterations, and three complete repetitions. Model/session
creation, graph capture, allocation, initial upload, and correctness readback
are outside measured iterations.

New benchmark providers are:

- `cuda-graph`: CUDA EP with `enable_cuda_graph=1` and fixed-address I/O
  Binding. A separate session is used for each dynamic batch shape. The first
  correctness run captures the graph; warmup and measured calls replay it.
- `cuda-iobinding-device`: inputs are uploaded and buffers bound once. Timed
  calls execute only `Run` plus output synchronization; the output remains on
  the GPU and is copied once after each measured repetition for correctness.
  This is not host-to-host latency.
- `cuda-fp16`: a separate ONNX graph with FP32 external I/O and FP16 internal
  weights/operations. It is never selected implicitly and does not replace the
  canonical FP32 model.

The raw repetition and aggregate results are in
`results/gpu_optimization_benchmark.csv` and
`results/gpu_optimization_fp16_benchmark.csv`. The machine-readable summary is
`results/gpu_optimization_experiments.json`.

## Results

Aggregate mean latency in milliseconds:

| Batch |    CPU |   CUDA | Host I/O Binding | CUDA Graph | GPU-resident |   FP16 |
| ----: | -----: | -----: | ---------------: | ---------: | -----------: | -----: |
|     1 | 0.0181 | 0.1201 |           0.1221 |     0.0535 |       0.1007 | 0.1269 |
|     8 | 0.0259 | 0.1166 |           0.1221 |     0.0607 |       0.0981 | 0.1336 |
|    32 | 0.0586 | 0.1278 |           0.1377 |     0.0861 |       0.1035 | 0.1411 |
|   128 | 0.1552 | 0.1449 |           0.1749 |     0.1373 |       0.1106 | 0.1502 |
|   256 | 0.2689 | 0.2048 |           0.2434 |     0.2083 |       0.1380 | 0.1786 |
|   512 | 0.4859 | 0.3012 |           0.3350 |     0.3006 |       0.1546 | 0.2762 |
|  1024 | 0.9388 | 0.4848 |           0.5177 |     0.4790 |       0.2082 | 0.4372 |
|  2048 | 1.9312 | 0.8553 |           0.8886 |     0.8494 |       0.3240 | 0.7762 |
|  4096 | 4.1299 | 1.6125 |           1.6433 |     1.5939 |       0.5734 | 1.4400 |

Speedup relative to the normal FP32 CUDA path:

| Batch | CUDA Graph | GPU-resident |   FP16 |
| ----: | ---------: | -----------: | -----: |
|     1 |     2.244x |       1.193x | 0.947x |
|     8 |     1.921x |       1.189x | 0.872x |
|    32 |     1.483x |       1.234x | 0.905x |
|   128 |     1.056x |       1.311x | 0.965x |
|   256 |     0.983x |       1.484x | 1.147x |
|   512 |     1.002x |       1.949x | 1.090x |
|  1024 |     1.012x |       2.329x | 1.109x |
|  2048 |     1.007x |       2.639x | 1.102x |
|  4096 |     1.012x |       2.812x | 1.120x |

Speedup relative to the FP32 CPU path:

| Batch |   CUDA | CUDA Graph | GPU-resident | Host I/O Binding |
| ----: | -----: | ---------: | -----------: | ---------------: |
|     1 | 0.151x |     0.339x |       0.180x |           0.148x |
|     8 | 0.222x |     0.426x |       0.264x |           0.212x |
|    32 | 0.459x |     0.680x |       0.566x |           0.426x |
|   128 | 1.071x |     1.130x |       1.404x |           0.887x |
|   256 | 1.313x |     1.291x |       1.948x |           1.104x |
|   512 | 1.613x |     1.617x |       3.144x |           1.451x |
|  1024 | 1.936x |     1.960x |       4.510x |           1.814x |
|  2048 | 2.258x |     2.274x |       5.960x |           2.173x |
|  4096 | 2.561x |     2.591x |       7.202x |           2.513x |

CUDA Graphs materially reduce small-batch launch/orchestration overhead but are
neutral once GEMMs dominate. GPU-resident I/O Binding is increasingly useful
with batch size, reaching 7.14 million samples/s at batch 4096. That number is
only applicable when adjacent pipeline stages already produce and consume GPU
tensors; it must not be presented as host-to-host application latency. Explicit
host-to-host I/O Binding remains slower than naive CUDA at every batch.

## Correctness and FP16

All FP32 modes have maximum absolute error `3.8147e-6`, below the established
`1e-4` tolerance. CUDA Graph and GPU-resident outputs match the existing CUDA
path.

FP16 conversion is deterministic and requires no retraining:

```bash
uv run --project python python python/scripts/convert_onnx_fp16.py
```

The generated `models/deeponet_reference_fp16.onnx` retains FP32 inputs and
output but uses FP16 internally. Its maximum difference from the frozen FP32
reference is `0.0137291` physical moisture units; mean difference at batch 4096
is `0.00731959`. FP16 is slower through batch 128 and gives a repeatable
9–15% gain from batch 256 onward. Nsight identifies Turing FP16 tensor-op GEMM
kernels (`tensorop_f16_s1688`), confirming that Tensor Core–eligible kernels
are selected.

## Profiling evidence

Short Nsight traces were collected separately from benchmark timings at batch
1 and 4096. The ordinary CUDA trace contains about 21 kernel launches per
inference. CUDA Graph traces replace repeated per-kernel API launches with one
`cudaGraphLaunch` per replay after the initial capture. They still contain the
host/device copies, explaining why graph replay does not address large-batch
transfer cost.

GPU-resident traces contain only setup/final-readback memcpy events; measured
iterations contain no H2D or D2H copy. This directly supports the gap between
the host-to-host I/O Binding and GPU-resident results.

The FP32 ONNX graph contains six `Gemm`, four `Relu`, two `Mul`, two `Add`, one
`ReduceSum`, and one `Constant` node. Bias is already part of `Gemm`; Nsight
still shows small standalone ReLU/elementwise kernels. Replacing those inside
ORT would require a custom operator or a broader custom inference path, while
CUDA Graphs already remove most of their small-batch launch cost. No custom
CUDA fusion was implemented.

## Ranking

1. **Useful:** CUDA Graph for batches 1–128; GPU-resident I/O Binding when the
   surrounding pipeline is GPU-resident; FP16 for batches 256+ when its measured
   numerical difference is acceptable.
2. **Neutral:** CUDA Graph from batch 256 upward.
3. **Harmful:** host-to-host I/O Binding for this workload; FP16 at batches
   1–128.

CUDA Graph behavior follows the constraints documented by the
[ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview).
