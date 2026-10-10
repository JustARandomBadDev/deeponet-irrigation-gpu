from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deeponet-irrigation-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deeponet_irrigation.project_paths import REPOSITORY_ROOT


ROOT = REPOSITORY_ROOT
RESULTS_DIR = ROOT / "results"
HORIZONS = (1, 3, 6, 12, 24)


def load_final_log() -> dict:
    return json.loads(
        (RESULTS_DIR / "mlp_final_feature_experiments.json").read_text(
            encoding="utf-8"
        )
    )


def plot_validation_rmse() -> Path:
    report = load_final_log()
    series = {
        "Persistence": json.loads(
            (RESULTS_DIR / "mlp_improvement_experiments.json").read_text(
                encoding="utf-8"
            )
        )["baselines"]["persistence"]["metrics"],
        "Previous best MLP": report["control"]["validation_metrics"],
        "Final trend MLP": report["selected"]["validation_metrics"],
    }
    positions = np.arange(len(HORIZONS))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for offset, (label, metrics) in zip((-1, 0, 1), series.items(), strict=True):
        values = [
            metrics["by_horizon"][f"{horizon}h"]["rmse"]
            for horizon in HORIZONS
        ]
        axis.bar(positions + offset * width, values, width, label=label)
    axis.set(
        title="Validation RMSE by prediction horizon",
        xlabel="Prediction horizon",
        ylabel="RMSE (soil-moisture units)",
        xticks=positions,
        xticklabels=[f"{horizon}h" for horizon in HORIZONS],
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = RESULTS_DIR / "validation_rmse_by_horizon.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_validation_change_status() -> Path:
    report = load_final_log()
    persistence = json.loads(
        (RESULTS_DIR / "mlp_improvement_experiments.json").read_text(
            encoding="utf-8"
        )
    )["baselines"]["persistence"]["change_status"]
    series = {
        "Persistence": persistence,
        "Previous best MLP": report["control"]["validation_change_status"],
        "Final trend MLP": report["selected"]["validation_change_status"],
    }
    labels = ("Nearly unchanged", "Changing")
    positions = np.arange(2)
    width = 0.25
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for offset, (label, metrics) in zip((-1, 0, 1), series.items(), strict=True):
        values = [
            metrics["nearly_unchanged"]["rmse"],
            metrics["changing"]["rmse"],
        ]
        axis.bar(positions + offset * width, values, width, label=label)
    axis.set(
        title="Validation RMSE by change status",
        ylabel="RMSE (soil-moisture units)",
        xticks=positions,
        xticklabels=labels,
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = RESULTS_DIR / "validation_change_status_rmse.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_final_test_24h() -> Path:
    with np.load(RESULTS_DIR / "final_frozen_mlp_predictions.npz") as values:
        horizons = values["horizon_hours"].reshape(-1)
        mask = horizons == 24
        time = values["target_timestamp"][mask].astype("datetime64[m]")
        order = np.argsort(time)
        time = time[order]
        target = values["target"].reshape(-1)[mask][order]
        persistence = values["persistence"].reshape(-1)[mask][order]
        original = values["previous_best_mlp"].reshape(-1)[mask][order]
        improved = values["final_mlp"].reshape(-1)[mask][order]

    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(time, target, label="Ground truth", linewidth=2)
    axis.plot(time, persistence, label="Persistence", alpha=0.75)
    axis.plot(time, original, label="Previous best MLP", alpha=0.75)
    axis.plot(time, improved, label="Final trend MLP", alpha=0.9)
    axis.set(
        title="Final test predictions at the 24-hour horizon",
        ylabel="Soil moisture",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    output = RESULTS_DIR / "final_frozen_mlp_predictions_24h.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main() -> None:
    for output in (
        plot_validation_rmse(),
        plot_validation_change_status(),
        plot_final_test_24h(),
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
