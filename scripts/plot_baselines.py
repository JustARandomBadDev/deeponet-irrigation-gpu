from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deeponet-irrigation-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
HORIZONS = (1, 3, 6, 12, 24)


def plot_representative_interval() -> Path:
    path = RESULTS_DIR / "test_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run evaluate_baselines.py first")
    with np.load(path) as values:
        horizons = values["horizon_hours"].reshape(-1)
        mask = horizons == 24
        target_time = values["target_timestamp"][mask].astype("datetime64[m]")
        order = np.argsort(target_time)
        selection = order[: min(7 * 24 * 6, len(order))]
        time = target_time[selection]
        target = values["target"].reshape(-1)[mask][selection]
        persistence = values["persistence"].reshape(-1)[mask][selection]
        mlp = values["mlp"].reshape(-1)[mask][selection]

    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(time, target, label="Ground truth", linewidth=2)
    axis.plot(time, persistence, label="Persistence", alpha=0.85)
    axis.plot(time, mlp, label="MLP", alpha=0.85)
    axis.set(title="Representative test interval (24-hour horizon)", ylabel="Soil moisture")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    output = RESULTS_DIR / "test_predictions_24h.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_mae_by_horizon() -> Path:
    persistence = json.loads(
        (RESULTS_DIR / "persistence_metrics.json").read_text(encoding="utf-8")
    )["metrics"]["test"]["by_horizon"]
    mlp = json.loads(
        (RESULTS_DIR / "mlp_metrics.json").read_text(encoding="utf-8")
    )["metrics"]["test"]["by_horizon"]
    persistence_mae = [persistence[f"{horizon}h"]["mae"] for horizon in HORIZONS]
    mlp_mae = [mlp[f"{horizon}h"]["mae"] for horizon in HORIZONS]

    positions = np.arange(len(HORIZONS))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(positions - width / 2, persistence_mae, width, label="Persistence")
    axis.bar(positions + width / 2, mlp_mae, width, label="MLP")
    axis.set(
        title="Test MAE by prediction horizon",
        xlabel="Prediction horizon",
        ylabel="MAE",
        xticks=positions,
        xticklabels=[f"{horizon}h" for horizon in HORIZONS],
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = RESULTS_DIR / "mae_by_horizon.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main() -> None:
    for output in (plot_representative_interval(), plot_mae_by_horizon()):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
