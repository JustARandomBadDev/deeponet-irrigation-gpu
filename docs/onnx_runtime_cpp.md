# C++ ONNX Runtime inference

The repository-root C++ project runs the frozen DeepONet ONNX graph with ONNX
Runtime. This phase establishes correctness with both `CPUExecutionProvider`
and `CUDAExecutionProvider`; it does not make performance claims.

## Dependency and prerequisites

CMake downloads the official GPU-enabled ONNX Runtime 1.30.0 binary archive:

`onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz`

The URL and SHA-256 are pinned in `CMakeLists.txt`. The archive supplies the C++
headers, core runtime, and CUDA provider. It requires a compatible NVIDIA
driver and CUDA 13 runtime libraries. No system ONNX Runtime installation is
required. Build executables receive an RPATH to the fetched library directory,
so no manual `LD_LIBRARY_PATH` is needed.

## Model contract

All tensors are `float32`; only the batch dimension is dynamic:

| Tensor             | Direction | Shape          | Meaning                                                  |
| ------------------ | --------- | -------------- | -------------------------------------------------------- |
| `branch_input`     | input     | `[batch, 723]` | normalized flattened history and three normalized trends |
| `horizon`          | input     | `[batch, 1]`   | horizon hours divided by 24                              |
| `current_moisture` | input     | `[batch, 1]`   | physical soil moisture at the prediction origin          |
| `prediction`       | output    | `[batch, 1]`   | physical future soil moisture                            |

The C++ runtime validates these names, shapes, and element types when it loads
the model. Residual scaling and current-moisture reconstruction are already in
the ONNX graph and are not repeated in C++. CUDA sessions explicitly disable
CPU EP fallback, so a successful CUDA test cannot silently execute the graph on
the CPU provider.

## Build and run

From repository root, convert the existing NPZ golden fixture without
recomputing predictions, then configure and build:

```bash
uv run --project python python python/scripts/export_golden_cpp.py
cmake -S . -B build
cmake --build build -j
```

Run the one-sample demos:

```bash
./build/deeponet_ort --cpu
./build/deeponet_ort --cuda
```

Run all golden cases at dynamic batch sizes 1, 8, and 10:

```bash
./build/deeponet_golden_test --cpu
./build/deeponet_golden_test --cuda
./build/deeponet_golden_test --cuda-iobinding
```

The test compares physical predictions against
`models/deeponet_reference_golden.bin` and fails if any absolute error exceeds
`1e-4`. The binary starts with the eight-byte magic `DONGOLD1`, followed by
little-endian `uint32` case and branch-dimension fields. Each record then holds
723 branch floats, one normalized-horizon float, one physical-current-moisture
float, and one expected-prediction float.

## Verified status

The runtime was verified on an NVIDIA GeForce RTX 2060 with CUDA 13.4. ONNX
Runtime reports `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and
`CPUExecutionProvider`; this project requests only CPU or CUDA in this phase.
For batches 1, 8, and 10, CPU matched all ten golden values exactly at float32
precision. CUDA had maximum absolute error `3.8147e-6` and aggregate mean
absolute error `4.0155e-7`. The direct CPU/CUDA comparison produced the same
maximum and mean differences. All results are below the `1e-4` tolerance.

## Current scope

The baseline implementation uses normal host-side ONNX Runtime tensors. With
the naive CUDA path, ONNX Runtime performs host/device transfers internally.
The I/O Binding path uses persistent device buffers and explicit transfers;
its correctness and performance methodology are documented in
[`benchmark.md`](benchmark.md). CUDA Graphs, profiling, FP16, and other GPU
optimizations remain outside this phase.
