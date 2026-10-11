# DeepONet ONNX inference contract

The deployment graph is `models/deeponet_reference.onnx`, exported from the
immutable `models/deeponet_reference.pt` checkpoint with ONNX opset 17. All
inputs and the output use `float32`; only the batch dimension is dynamic.

## Inputs

`branch_input` has shape `[batch, 723]`. Its layout is:

1. 720 history values: the prepared `[144, 5]` sequence flattened in
   time-major order, with the feature dimension varying fastest. Feature order
   is `soil_moisture`, `applied_water_liters`, `weather_rain`, `weather_temp`,
   `weather_humidity`. Each value is normalized as
   `(physical_value - training_mean) / training_std` using the values recorded
   in `deeponet_reference.metadata.json`.
2. Three moisture trends, in this exact order: current minus 1-hour-ago,
   current minus 3-hours-ago, and current minus 6-hours-ago. Each difference is
   first calculated in physical soil-moisture units, then normalized as
   `(physical_trend - trend_training_mean) / trend_training_std`. The ONNX
   input therefore receives **normalized trends, not raw trends**.

`horizon` has shape `[batch, 1]`. It receives `horizon_hours / 24`, not raw
hours. Project-supported physical horizons are 1, 3, 6, 12, and 24 hours.

`current_moisture` has shape `[batch, 1]`. It is the soil-moisture value at the
prediction origin in physical dataset units, with no normalization.

## Output

`prediction` has shape `[batch, 1]` and is future soil moisture in physical
dataset units. The graph computes the Branch and Trunk latent vectors, their
inner product plus operator bias, multiplies the normalized residual by the
training residual standard deviation, and adds `current_moisture`. A C++ or
TensorRT caller must not apply residual reconstruction a second time.

The machine-readable metadata contains the exact normalization values and
feature definitions. The small golden NPZ fixture contains the three input
arrays and expected physical predictions for ten deterministic validation
samples. The corresponding binary fixture is consumed by the C++ tests.

## One-command deployment setup

From repository root:

```bash
uv sync --project python
uv run --project python setup-onnx
```

This command does not download or prepare Arnesano data and does not invoke any
training function. It uses the frozen checkpoint as the source model, the
tracked metadata for the preprocessing/model contract, and the tracked golden
NPZ for deterministic deployment inputs. It then:

1. validates the checkpoint flags, architecture, and preprocessing contract;
2. exports `deeponet_reference.onnx` with dynamic batching and opset 17;
3. runs `onnx.checker` and verifies all float32 input/output names and shapes;
4. compares PyTorch and ONNX Runtime predictions with the established
   `atol=1e-4`, `rtol=1e-5` tolerances;
5. refreshes expected predictions in the ten-case golden NPZ;
6. regenerates the C++ golden BIN and JSON files;
7. prints checkpoint, pre/post ONNX, and golden NPZ SHA-256 hashes.

The comprehensive historical exporter
`python/scripts/export_deeponet_onnx.py` remains available when the prepared
dataset is present. It checks additional train/validation/test samples; it is
not needed for deployment setup.

## Frozen artifact integrity

The canonical deployment sources are versioned with the repository:

- checkpoint SHA-256:
  `10ab845e136d9011f9322848fe43ee1d6ae76b0cb655cd9440d1fd04c09c6fc1`;
- ONNX SHA-256:
  `6eef7579599dff7fe038cc7580bd8772116ec8f52cbecbb84044ec88c4695964`;
- golden NPZ SHA-256:
  `42d928db3e81c9c79c7104090b1b2296a23fb0e42a749e941b1f08f1d7c2f648`.

See [`reproducibility.md`](reproducibility.md) for the complete tracked artifact
inventory and clean-checkout workflows.
