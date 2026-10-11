# Implementation and reproducibility

This document describes how the repository is organized and how to run its
frozen deployment experiment. The experimental reasoning and scientific results
belong in the [French](experiment_fr.md) and
[English](experiment_en.md) experiment notebooks.

## 1. Architecture overview

```text
Arnesano dataset
      ↓
Python preprocessing
      ↓
PyTorch training and validation-only selection
      ↓
frozen deeponet_reference.pt
      ↓
ONNX export and golden vectors
      ↓
ONNX Runtime C++
   ↙       ↘
 CPU       CUDA
```

The canonical deployment model is frozen. Model development, preprocessing, and
training live under `python/`; the repository root is the C++20 inference and
benchmark project. `data/`, `models/`, `results/`, and `docs/` are shared at the
repository root.

## 2. Python responsibilities

Python provides:

- deterministic Arnesano v2 acquisition and temporal preprocessing;
- persistence, MLP, and DeepONet training/evaluation tools;
- frozen PyTorch checkpoint reconstruction;
- ONNX export and structural/numerical validation;
- deterministic Python and C++ golden fixtures;
- optional conversion to an experimental FP16 ONNX model.

The normal deployment-artifact rebuild is:

```bash
uv sync --project python
uv run --project python setup-onnx
```

`setup-onnx` loads `models/deeponet_reference.pt`, validates its architecture
against the deployment metadata, exports the FP32 ONNX model, checks its tensor
contract, compares PyTorch and ONNX Runtime predictions, and regenerates the
golden NPZ/BIN/JSON fixtures. It uses tracked golden inputs and does **not**
download the dataset, train a model, or change the frozen checkpoint.

The optional FP16 conversion is separate:

```bash
uv run --project python python python/scripts/convert_onnx_fp16.py
```

It creates `models/deeponet_reference_fp16.onnx`; it does not replace the
canonical FP32 model.

## 3. C++ responsibilities

The root CMake project fetches the pinned official ONNX Runtime 1.30.0 CUDA 13
binary archive. Its build RPATH points to the fetched runtime libraries, so the
executables do not require a manually configured `LD_LIBRARY_PATH`.

The C++ code provides:

- a small ONNX Runtime session wrapper with CPU and CUDA providers;
- dynamic-batch input validation and inference;
- persistent CUDA buffers and I/O Binding;
- the ONNX Runtime CUDA Graph experiment;
- a GPU-resident I/O Binding experiment;
- golden-vector correctness tests;
- a Release-only latency/throughput benchmark.

The main executables are:

- `deeponet_ort`: one-sample CPU or CUDA demonstration;
- `deeponet_golden_test`: batch 1/8/10 correctness checks;
- `deeponet_benchmark`: correctness prechecks and reproducible measurements.

## 4. ONNX contract

The canonical FP32 graph exposes exactly:

| Direction | Name               | Shape          | Type      |
| --------- | ------------------ | -------------- | --------- |
| Input     | `branch_input`     | `[batch, 723]` | `float32` |
| Input     | `horizon`          | `[batch, 1]`   | `float32` |
| Input     | `current_moisture` | `[batch, 1]`   | `float32` |
| Output    | `prediction`       | `[batch, 1]`   | `float32` |

`batch` is dynamic. The branch tensor contains the flattened normalized
`[144, 5]` history followed by three normalized historical moisture trends.
`horizon` is `horizon_hours / 24`; `current_moisture` and `prediction` are in
physical dataset units. Residual scaling and reconstruction are part of the
graph, not reimplemented in C++.

## 5. Benchmark modes

`deeponet_benchmark --provider MODE` accepts:

- `cpu`: CPU Execution Provider with ordinary host tensors.
- `cuda`: CUDA Execution Provider with ordinary host tensors and ORT-managed
  transfers.
- `cuda-iobinding`: host-to-host inference using persistent device buffers,
  explicit transfers, and I/O Binding. Inputs start on CPU and predictions
  return to CPU inside every measured iteration.
- `cuda-graph`: CUDA EP graph replay with fixed-address I/O Binding. Capture is
  completed before measured iterations; a session is created per batch shape.
- `cuda-iobinding-device`: GPU-resident experiment. Inputs are uploaded and
  buffers bound before timing; measured iterations execute inference without
  H2D/D2H copies, and one output copy is performed afterward for correctness.
- `cuda-fp16`: CUDA execution of the separate experimental FP16 ONNX graph,
  with FP32 external inputs and output.
- `all`: runs `cpu`, `cuda`, `cuda-iobinding`, `cuda-graph`, and
  `cuda-iobinding-device`. It deliberately does not include `cuda-fp16`.

Optional arguments are:

```text
--batch N
--warmup N
--iterations N
--repetitions N
```

Without `--batch`, all configured batch sizes are tested. Any strictly positive
batch can be selected explicitly.

## 6. Benchmark methodology

The benchmark refuses non-Release builds. Its defaults are:

- batch sizes: 1, 8, 32, 128, 256, 512, 1024, 2048, and 4096;
- 100 warmup iterations before each repetition;
- 1000 measured iterations per repetition;
- three complete repetitions;
- deterministic cyclic repetition of the ten golden cases for large batches;
- CPU intra-op threads equal to `std::thread::hardware_concurrency()` and one
  inter-op thread.

Model loading, session creation, fixture parsing, graph capture, and initial
buffer allocation are outside the measured loops. Each reported end-to-end
sample ends with a host-visible result, except the explicitly labeled
`cuda-iobinding-device` mode. That mode measures GPU-resident execution and
synchronization only.

The benchmark records mean, median/p50, p95, minimum, maximum, population
standard deviation, samples per second, and microseconds per sample. It writes:

```text
results/benchmark/onnx_runtime_benchmark.csv
results/benchmark/environment.json
```

Every mode performs a prediction check before timing. FP32 uses an absolute
tolerance of `1e-4`; the isolated FP16 experiment uses `5e-2`.

## 7. Reproducibility

### Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The first configure requires network access to download the pinned ONNX Runtime
archive. CUDA Toolkit is a configure-time requirement.

### Validate the frozen deployment

```bash
./build/deeponet_golden_test --cpu
./build/deeponet_golden_test --cuda
./build/deeponet_golden_test --cuda-iobinding
```

The CUDA commands require a compatible NVIDIA GPU/driver. The tests consume the
tracked ONNX graph and binary fixture directly; Python, PyTorch, and the dataset
are not required.

### Run the canonical benchmark

```bash
./build/deeponet_benchmark \
  --provider all \
  --warmup 100 \
  --iterations 1000 \
  --repetitions 3
```

Run FP16 separately when required:

```bash
./build/deeponet_benchmark \
  --provider cuda-fp16 \
  --warmup 100 \
  --iterations 1000 \
  --repetitions 3
```

### Optionally regenerate deployment artifacts

```bash
uv sync --project python
uv run --project python setup-onnx
```

The full historical training and tuning sequence is intentionally separate. Its
scripts and curated selection logs are available, but it is not represented as
a one-command reproduction path and is unnecessary for deployment.

## 8. Important artifacts

| Path                                        | Purpose                                  |
| ------------------------------------------- | ---------------------------------------- |
| `models/deeponet_reference.pt`              | Frozen PyTorch source checkpoint         |
| `models/deeponet_reference.onnx`            | Canonical FP32 C++ deployment model      |
| `models/deeponet_reference.metadata.json`   | Input, normalization, and model contract |
| `models/deeponet_reference_golden.npz`      | Python golden inputs and predictions     |
| `models/deeponet_reference_golden.bin`      | C++-readable golden fixture              |
| `models/deeponet_reference_golden.json`     | Golden binary format metadata            |
| `models/deeponet_reference_fp16.onnx`       | Experimental converted FP16 graph        |
| `results/gpu_optimization_experiments.json` | Curated final GPU summary                |

Generated datasets, build products, raw benchmark CSV files, profiler traces,
and most intermediate checkpoints are not canonical deployment inputs.
