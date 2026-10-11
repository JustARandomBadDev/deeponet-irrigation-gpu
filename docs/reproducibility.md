# Reproducibility workflows

The repository distinguishes deployment reproducibility from reconstruction of
the historical research process. The frozen deployment artifacts are versioned;
bulk datasets, intermediate checkpoints, predictions, and generated plots are
not.

## A. C++ deployment and benchmark

This is the primary clean-checkout workflow. It requires CMake, a C++20
compiler, the CUDA toolkit, and a compatible NVIDIA driver for CUDA execution.
It does not require Python, uv, PyTorch, dataset download, or model training.

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

The runtime consumes the tracked `models/deeponet_reference.onnx` and
`models/deeponet_reference_golden.bin` directly.

## B. Python frozen-model verification

The normal deployment-artifact rebuild uses only tracked frozen artifacts. It
does not require raw or prepared data and does not retrain either neural
network:

```bash
uv sync --project python
uv run --project python setup-onnx
```

The command reconstructs the model from `deeponet_reference.pt`, validates it
against tracked deployment metadata, and uses the ten-case golden NPZ as its
deterministic input fixture. It exports and validates ONNX, refreshes golden
predictions, regenerates the C++ fixture, and reports hashes. The verified
re-export in the reference environment was byte-identical to the tracked ONNX
and golden NPZ files; PyTorch and ONNX Runtime were exactly equal on those ten
cases.

For the broader historical train/validation/test equivalence audit, first run
the dataset setup commands in section C, then use
`python/scripts/export_deeponet_onnx.py` and `python/scripts/infer_onnx.py`.

## C. Training and research

The Python package contains deterministic preprocessing, training, bounded
experiment, selection, evaluation, and plotting scripts. Historical model
selection was a sequence of explicitly chosen experiments rather than one
monolithic command. The small selection logs retained under `results/` record
the runs and validation-only decisions, while individual intermediate
checkpoints and prediction arrays remain intentionally untracked.

Dataset construction starts with:

```bash
uv sync --project python
uv run --project python python python/scripts/setup_data.py
uv run --project python python python/scripts/prepare_dataset.py
```

Individual research entry points include `train_mlp.py`, `improve_mlp.py`,
`finalize_mlp.py`, and `train_deeponet.py`. Replaying the complete historical
search requires invoking the recorded experiment IDs in their documented order;
it is not presented as a one-command pipeline. Final-test guards deliberately
prevent overwriting the published frozen evaluations.

MLP summary data is comparison metadata, not a mathematical input to DeepONet
training. If those JSON summaries are absent in a research checkout,
`train_deeponet.py` now continues without embedding the comparison metadata.

## Frozen deployment artifacts

| Path                                      |   Bytes | SHA-256                                                            |
| ----------------------------------------- | ------: | ------------------------------------------------------------------ |
| `models/deeponet_reference.pt`            | 950,092 | `10ab845e136d9011f9322848fe43ee1d6ae76b0cb655cd9440d1fd04c09c6fc1` |
| `models/deeponet_reference.onnx`          | 942,832 | `6eef7579599dff7fe038cc7580bd8772116ec8f52cbecbb84044ec88c4695964` |
| `models/deeponet_reference.metadata.json` |   4,397 | `65f6e248fa27c453ef3b7ce3bde46e00bacbd3344d9b72fba247cf03eb8fc06a` |
| `models/deeponet_reference_golden.npz`    |   5,154 | `42d928db3e81c9c79c7104090b1b2296a23fb0e42a749e941b1f08f1d7c2f648` |
| `models/deeponet_reference_golden.bin`    |  29,056 | `7829c3a6678008759c7b6b781b78d7f920e6f7705f03b6eae961759dfe7c188b` |
| `models/deeponet_reference_golden.json`   |     744 | `8b2a8456426cc4cee6e8821c3b3255762974790c4525c30c7f3ba83d9e7b8003` |

These files total less than 2 MB. Git LFS is not required.

## Curated scientific records

Only these small result files are versioned:

- `mlp_improvement_experiments.json`: bounded residual-MLP experiments;
- `mlp_final_feature_experiments.json`: final feature-model selection;
- `deeponet_experiments.json`: bounded DeepONet search and final selection;
- `final_frozen_mlp_metrics.json`: official frozen MLP evaluation;
- `deeponet_final_test_metrics.json`: official frozen DeepONet evaluation;
- `deeponet_onnx_equivalence.json`: PyTorch/ONNX numerical audit.

Bulk prediction arrays, plots, temporary reports, datasets, intermediate model
checkpoints, virtual environments, and build products remain ignored.
