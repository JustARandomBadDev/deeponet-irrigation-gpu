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
samples; it is intended for later TensorRT integration tests.
