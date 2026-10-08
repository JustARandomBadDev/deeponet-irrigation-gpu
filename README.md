# deeponet-irrigation-gpu

DeepONet-based soil-moisture prediction on the Arnesano precision-irrigation
dataset, with a reproducible Python pipeline and later GPU inference
optimization using C++, CUDA, and TensorRT.

## Dataset setup

The data tooling is pinned to version 2 of *Soil Moisture, Irrigation Actuator
and Weather Dataset from a Multi-Sector Precision-Irrigation*:
[DOI 10.17632/c837v6p8ph.2](https://doi.org/10.17632/c837v6p8ph.2).
It never requests the unversioned latest release.

```bash
uv sync
uv run python scripts/setup_data.py
uv run python scripts/inspect_dataset.py
```

`setup_data.py` downloads the official version-qualified Mendeley Data ZIP,
checks its server-published SHA-256, validates the documented directory layout,
and records provenance in `data/raw/arnesano-v2.metadata.json`. Re-running it
validates and reuses both the archive and extracted files.

`inspect_dataset.py` checks the supplied README and data dictionary, lists the
downloaded files, then streams every CSV in chunks to report shape, schema,
missing values, documented timestamp ranges, and documented sector identifiers.

Downloaded and generated artifacts under `data/`, `models/`, and `results/`
are intentionally excluded from Git.

## Temporal dataset

The leakage-safe temporal pipeline uses the author-provided **merged sector 4**
table. Unlike the preprocessed tables, merged data preserves the complete
10-minute grid; sector 4 provides the most valid 24-hour histories and future
targets among sectors 1, 2, and 4. Sector 3 is rejected because its probe has a
long trailing zero flatline, and sector 5 is excluded because of its documented
water-volume inconsistency.

The five input features, in fixed order, are `soil_moisture`,
`applied_water_liters`, `weather_rain`, `weather_temp`, and
`weather_humidity`. Applied water is lagged by one bin so it is strictly
historical. The target is soil moisture at 1, 3, 6, 12, or 24 hours after the
prediction origin. Each branch contains 144 consecutive 10-minute observations
(24 hours of bins).

Soil moisture is range-checked using the supplied data dictionary, clipped at
100, checked for documented spikes and trailing flatlines, and interpolated
only when an entire interior gap is at most 20 minutes. Longer gaps reject the
window. Splits are contiguous 70/15/15 time ranges; histories and targets that
cross a boundary are purged. Feature mean/std statistics use only unique rows
referenced by training histories.

```bash
uv run python scripts/prepare_dataset.py
uv run python scripts/inspect_samples.py
```

Compact indexed NumPy arrays and complete preprocessing provenance are written
under `data/processed/arnesano_v2/`. The normalization statistics and exact
split ranges are stored in `data/processed/arnesano_v2/metadata.json`.

## Baselines

The persistence baseline predicts the last soil-moisture value in the
historical window. It is unusually strong here because most targets are
unchanged from the prediction origin (about 95% at 1 hour and 66% at 24 hours
in the held-out test period).

The original learned baseline is a deliberately small direct-target MLP: it
flattens the normalized `[144, 5]` history, appends the horizon in hours divided
by 24, and applies `Linear(721, 128) -> ReLU -> Linear(128, 64) -> ReLU ->
Linear(64, 1)`. Its output is mapped back to physical soil-moisture units using
the training-only soil-moisture mean and standard deviation. The dataset loader
reconstructs windows from the prepared indices; it does not duplicate
preprocessing or flatten model inputs.

The learned baseline predicts the residual `future soil moisture - current soil
moisture`. Its selected target scaling divides by the training residual standard
deviation without subtracting the residual mean. The final layer starts at zero,
so an initial zero network output reconstructs exact persistence. This was more
stable across the large chronological moisture shift than mean-centered residual
scaling.

Chronological splits remain unchanged throughout: train is fitted, validation is
used for model selection and calibration, and test is evaluated only after the
configuration is frozen. Experimental change weighting is applied only to train
losses; it is never used to resample or alter validation/test, and it was rejected
because sector-4 train already contains substantially more change than later
periods.

The now-frozen final MLP adds three train-normalized, historical-only moisture
trends: current moisture minus moisture 1, 3, and 6 hours earlier. Its input is
therefore `724 -> 256 -> 128 -> 64 -> 1`, with ReLU, Adam, MSE, learning rate
`1e-3`, batch size 512, no change weighting, seed 42, and early stopping.
Validation selected no shrinkage (`alpha=1`) and no deadband. It reaches RMSE
`0.8975` on validation and `0.9478` on the last MLP test evaluation, compared
with `1.2174` and `1.0489` for persistence. The prior feature-free MLP scored
`0.9347` on validation and `0.9310` on test. The feature model was selected from
validation only; its slightly weaker test result did not trigger post-test
tuning. The MLP baseline is final and frozen.

```bash
uv run python scripts/train_mlp.py
uv run python scripts/diagnose_baselines.py
uv run python scripts/finalize_mlp.py --summary
uv run python scripts/plot_baselines.py
uv run python -m unittest discover -s tests
```

The final temporal-feature experiments are logged under
`results/mlp_final_feature_experiments.json`. The final evaluator refuses to
overwrite the last MLP test artifacts. Evaluation reports MAE, RMSE, and R²
overall and per horizon, plus bias and errors for nearly unchanged and changing
samples.

## DeepONet reference

The frozen PyTorch reference is a residual Deep Operator Network. Its Branch
network flattens the normalized 144-by-5 historical state and appends the same
three train-normalized historical moisture trends used by the frozen MLP. Its
Trunk network independently embeds the query coordinate `horizon_hours / 24`.
Both produce 64-dimensional vectors, combined only through
`sum(branch * trunk) + bias`. That normalized residual is multiplied by the
training residual standard deviation and added to current soil moisture.

The validation-selected architecture is Branch
`723 -> 256 -> 128 -> 64`, Trunk `1 -> 64 -> 64 -> 64`, with 234,945 trainable
parameters, MSE, Adam, learning rate `1e-3`, batch size 512, and seed 42.
Validation RMSE is `1.0292`, compared with `1.2174` for persistence and `0.8975`
for the frozen MLP. Its one-time final test RMSE is `1.2112`, compared with
`1.0489` and `0.9478`, respectively. DeepONet therefore remains a useful
operator-structured reference for later GPU work, but it is not the most
accurate forecasting model on this chronological split. No post-test tuning was
performed.

```bash
uv run python scripts/train_deeponet.py --run initial_s42
uv run python scripts/train_deeponet.py --summary
uv run python scripts/evaluate_deeponet.py --plots-only
```

The bounded experiment log is `results/deeponet_experiments.json`; the frozen
checkpoint is `models/deeponet_reference.pt`. The final evaluator is guarded
against repeat test inference, while `--plots-only` safely regenerates figures
from the immutable saved predictions.

Generated checkpoints are stored under `models/`; metrics, prediction arrays,
and plots are stored under `results/`. Both directories are excluded from Git.
