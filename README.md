# DeepONet Irrigation GPU

Soil-moisture forecasting for precision irrigation, from leakage-safe temporal
preprocessing to C++ GPU inference. The project compares persistence, a residual
MLP, and a residual DeepONet, then deploys the frozen DeepONet through ONNX
Runtime for reproducible CPU/CUDA experiments.

The repository root is the C++20 deployment project. The supporting dataset,
training, evaluation, and ONNX-export toolchain lives under `python/`.

## Main results

| Predictor                        |  Test RMSE |
| -------------------------------- | ---------: |
| Persistence                      |     1.0489 |
| Validation-selected residual MLP | **0.9478** |
| Frozen residual DeepONet         |     1.2112 |

- Normal CUDA is **2.561×** faster than CPU at batch 4096, but CPU remains
  faster at batches 1, 8, and 32.
- CUDA Graph replay is **2.244×** faster than normal CUDA at batch 1; CPU still
  has lower absolute single-sample latency.
- GPU-resident I/O Binding reaches **7.14 million samples/s** at batch 4096,
  **2.812×** normal CUDA. This mode excludes host/device transfers from its
  measured loop and is not ordinary host-to-host latency.
- Experimental FP16 is **1.120×** faster than FP32 CUDA at batch 4096, with a
  maximum observed prediction difference of 0.013729 on the golden inputs.

The central result is workload-dependent: the simpler MLP predicts better,
while GPU acceleration of the DeepONet becomes useful when batching or a
GPU-resident pipeline amortizes fixed overhead.

## Quick start

### Build C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run correctness tests

```bash
./build/deeponet_golden_test --cpu
./build/deeponet_golden_test --cuda
./build/deeponet_golden_test --cuda-iobinding
```

### Run the benchmark

```bash
./build/deeponet_benchmark \
  --provider all \
  --warmup 100 \
  --iterations 1000 \
  --repetitions 3
```

`--provider all` runs the FP32 CPU/CUDA modes, including CUDA Graph and the
GPU-resident experiment. Run experimental FP16 separately with
`--provider cuda-fp16`.

### Rebuild deployment artifacts

```bash
uv sync --project python
uv run --project python setup-onnx
```

`setup-onnx` starts from the tracked frozen PyTorch checkpoint. It validates and
rebuilds the ONNX and golden fixtures; it does **not** train a model or require
the dataset.

## Documentation

- [Carnet d'expérimentation — Français](docs/experiment_fr.md)
- [Experiment notebook — English](docs/experiment_en.md)
- [Implementation and reproducibility](docs/implementation.md)

## Repository structure

```text
python/      data preparation, training, evaluation, export
include/     C++ public headers
src/         ONNX Runtime inference and correctness tools
benchmark/   benchmark driver
models/      frozen deployment artifacts
results/     curated results and generated experiment outputs
docs/        experiment and implementation documentation
```

## Requirements

- CMake 3.24 or newer and a C++20 compiler;
- CUDA Toolkit and a compatible NVIDIA driver for CUDA modes;
- network access during the first CMake configure to fetch the pinned ONNX
  Runtime 1.30.0 CUDA 13 binary archive;
- Python 3.11+ and [uv](https://docs.astral.sh/uv/) only for the optional Python
  toolchain.

The tracked FP32 ONNX model and C++ golden fixture allow normal C++ deployment
without Python, PyTorch, the training dataset, or retraining.
