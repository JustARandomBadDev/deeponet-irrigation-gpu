import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from deeponet_irrigation.dataset import (
    PreparedTemporalDataset,
    fit_residual_normalization,
    residual_targets,
    summarize_change_weighting,
)
from deeponet_irrigation.evaluation import (
    calibrate_residual_prediction,
    load_mlp_checkpoint,
    persistence_prediction,
)
from deeponet_irrigation.metrics import metrics_by_horizon
from deeponet_irrigation.models import MLPBaseline
from deeponet_irrigation.training import change_weights, select_best_experiment
from deeponet_irrigation.temporal_features import (
    TREND_FEATURES,
    WATER_FEATURES,
    derive_temporal_features,
    temporal_feature_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"


class BaselineTests(unittest.TestCase):
    def test_prepared_branch_shape(self) -> None:
        dataset = PreparedTemporalDataset(DATA_DIR, "train")
        self.assertEqual(dataset[0]["branch_input"].shape, (144, 5))

    def test_mlp_output_shape(self) -> None:
        model = MLPBaseline(history_steps=144, feature_count=5)
        output = model(torch.zeros(7, 144, 5), torch.ones(7, 1))
        self.assertEqual(output.shape, (7, 1))

    def test_residual_target_is_future_minus_current(self) -> None:
        dataset = PreparedTemporalDataset(DATA_DIR, "train")
        sample = dataset[0]
        torch.testing.assert_close(
            sample["residual_target"],
            sample["target"] - sample["current_soil_moisture"],
        )
        reconstructed = sample["current_soil_moisture"] + sample["residual_target"]
        torch.testing.assert_close(reconstructed, sample["target"])

    def test_residual_model_reconstructs_physical_prediction(self) -> None:
        model = MLPBaseline(
            history_steps=4,
            feature_count=2,
            hidden_sizes=(3, 2),
            target_mean=0.5,
            target_std=2.0,
            target_mode="residual",
            soil_feature_index=0,
            soil_mean=10.0,
            soil_std=3.0,
        )
        for parameter in model.network.parameters():
            torch.nn.init.zeros_(parameter)
        branch = torch.zeros(2, 4, 2)
        branch[:, -1, 0] = torch.tensor([1.0, -1.0])
        output = model(branch, torch.ones(2, 1))
        torch.testing.assert_close(output, torch.tensor([[13.5], [7.5]]))

    def test_std_only_zero_output_is_exact_persistence(self) -> None:
        model = MLPBaseline(
            history_steps=4,
            feature_count=2,
            hidden_sizes=(3, 2),
            target_mean=0.0,
            target_std=4.0,
            target_mode="residual",
            soil_feature_index=0,
            soil_mean=10.0,
            soil_std=3.0,
            residual_scaling_mode="std_only",
            output_initialization="zero",
        )
        branch = torch.zeros(2, 4, 2)
        branch[:, -1, 0] = torch.tensor([1.0, -1.0])
        output = model(branch, torch.ones(2, 1))
        torch.testing.assert_close(output, torch.tensor([[13.0], [7.0]]))
        self.assertEqual(model.architecture()["output_scaling"]["mean"], 0.0)

    def test_residual_normalization_is_fit_on_train_only(self) -> None:
        train = PreparedTemporalDataset(DATA_DIR, "train")
        validation = PreparedTemporalDataset(DATA_DIR, "validation")
        fitted = fit_residual_normalization(train)
        train_residual = residual_targets(train)
        combined = np.concatenate((train_residual, residual_targets(validation)))
        self.assertEqual(fitted["fit_split"], "train")
        self.assertAlmostEqual(fitted["mean"], float(train_residual.mean()))
        self.assertAlmostEqual(fitted["std"], float(train_residual.std(ddof=0)))
        self.assertNotAlmostEqual(fitted["mean"], float(combined.mean()), places=5)

    def test_change_weighting_statistics_are_train_only(self) -> None:
        train = PreparedTemporalDataset(DATA_DIR, "train")
        configuration = {
            "change_threshold": 0.5,
            "large_change_threshold": 5.0,
            "change_weight": 1.5,
            "large_change_weight": 2.0,
        }
        summary = summarize_change_weighting(train, configuration)
        weights = change_weights(
            torch.tensor([[0.0], [1.0], [6.0]]), configuration
        )
        self.assertEqual(summary["fit_split"], "train")
        self.assertEqual(summary["sample_count"], len(train))
        torch.testing.assert_close(weights, torch.tensor([[1.0], [1.5], [2.0]]))

    def test_residual_calibration_alpha_and_deadband(self) -> None:
        current = np.array([10.0, 10.0, 10.0])
        prediction = np.array([10.05, 10.4, 9.0])
        calibrated = calibrate_residual_prediction(
            prediction, current, alpha=0.5, deadband=0.1
        )
        np.testing.assert_allclose(calibrated, np.array([10.0, 10.2, 9.5]))

    def test_final_checkpoint_reload_preserves_residual_configuration(self) -> None:
        dataset = PreparedTemporalDataset(DATA_DIR, "train")
        model = MLPBaseline(
            history_steps=dataset.history_steps,
            feature_count=dataset.feature_count,
            target_mean=0.0,
            target_std=4.0,
            target_mode="residual",
            soil_feature_index=dataset.soil_index,
            soil_mean=dataset.soil_mean,
            soil_std=dataset.soil_std,
            residual_scaling_mode="std_only",
            output_initialization="zero",
            calibration_alpha=0.5,
            calibration_deadband=0.1,
            temporal_feature_names=TREND_FEATURES,
            temporal_feature_means=(0.0, 0.0, 0.0),
            temporal_feature_stds=(1.0, 1.0, 1.0),
            input_feature_order=tuple(dataset.metadata["feature_order"]),
            input_feature_means=tuple(
                dataset.metadata["normalization"]["statistics"][name]["mean"]
                for name in dataset.metadata["feature_order"]
            ),
            input_feature_stds=tuple(
                dataset.metadata["normalization"]["statistics"][name]["std"]
                for name in dataset.metadata["feature_order"]
            ),
        )
        contract = {
            key: dataset.metadata[key]
            for key in (
                "dataset",
                "selected_data_source",
                "selected_sector",
                "feature_order",
                "target_column",
                "history_steps",
                "prediction_horizons_hours",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "architecture": model.architecture(),
                    "model_state_dict": model.state_dict(),
                    "preprocessing_contract": contract,
                },
                path,
            )
            loaded, _ = load_mlp_checkpoint(path, dataset.metadata, torch.device("cpu"))
        self.assertEqual(loaded.residual_scaling_mode, "std_only")
        self.assertEqual(loaded.calibration_alpha, 0.5)
        self.assertEqual(loaded.calibration_deadband, 0.1)
        self.assertEqual(loaded.temporal_feature_names, TREND_FEATURES)
        branch = torch.randn(3, dataset.history_steps, dataset.feature_count)
        horizon = torch.tensor([[1.0], [12.0], [24.0]])
        with torch.inference_mode():
            torch.testing.assert_close(model(branch, horizon), loaded(branch, horizon))

    def test_derived_features_use_historical_rows_only(self) -> None:
        branch = torch.zeros(1, 144, 5)
        branch[0, :, 0] = torch.arange(144)
        branch[0, :, 1] = 1.0
        branch[0, :, 2] = 2.0
        names = TREND_FEATURES + WATER_FEATURES
        derived = derive_temporal_features(
            branch,
            names,
            feature_order=(
                "soil_moisture",
                "applied_water_liters",
                "weather_rain",
                "weather_temp",
                "weather_humidity",
            ),
            feature_means=torch.zeros(5),
            feature_stds=torch.ones(5),
        )
        torch.testing.assert_close(
            derived,
            torch.tensor([[6.0, 18.0, 36.0, 6.0, 36.0, 144.0, 12.0, 72.0, 288.0]]),
        )
        for definition in temporal_feature_metadata(names):
            self.assertFalse(definition["uses_future"])
            self.assertLessEqual(definition["latest_offset_steps"], 0)
            self.assertLessEqual(definition["earliest_offset_steps"], 0)

    def test_temporal_feature_model_has_expected_input_dimension(self) -> None:
        names = TREND_FEATURES + WATER_FEATURES
        model = MLPBaseline(
            history_steps=144,
            feature_count=5,
            hidden_sizes=(256, 128, 64),
            target_mode="residual",
            temporal_feature_names=names,
            temporal_feature_means=(0.0,) * len(names),
            temporal_feature_stds=(1.0,) * len(names),
            input_feature_order=(
                "soil_moisture",
                "applied_water_liters",
                "weather_rain",
                "weather_temp",
                "weather_humidity",
            ),
            input_feature_means=(0.0,) * 5,
            input_feature_stds=(1.0,) * 5,
        )
        self.assertEqual(model.architecture()["input_size"], 730)
        output = model(torch.zeros(2, 144, 5), torch.ones(2, 1))
        self.assertEqual(output.shape, (2, 1))

    def test_persistence_uses_last_historical_soil_value(self) -> None:
        branch = torch.zeros(2, 4, 3)
        branch[0, -1, 1] = 2.0
        branch[1, -1, 1] = -1.0
        prediction = persistence_prediction(
            branch, soil_index=1, soil_mean=10.0, soil_std=3.0
        )
        torch.testing.assert_close(prediction, torch.tensor([[16.0], [7.0]]))

    def test_metrics_are_grouped_by_horizon(self) -> None:
        target = np.array([1.0, 3.0, 10.0, 14.0])
        prediction = np.array([2.0, 2.0, 12.0, 12.0])
        horizon = np.array([1.0, 1.0, 24.0, 24.0])
        metrics = metrics_by_horizon(target, prediction, horizon)
        self.assertEqual(set(metrics["by_horizon"]), {"1h", "24h"})
        self.assertAlmostEqual(metrics["by_horizon"]["1h"]["mae"], 1.0)
        self.assertAlmostEqual(metrics["by_horizon"]["24h"]["mae"], 2.0)

    def test_validation_and_test_metrics_are_not_mixed(self) -> None:
        validation = metrics_by_horizon(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0])
        )
        test = metrics_by_horizon(
            np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([1.0, 1.0])
        )
        self.assertEqual(validation["overall"]["mae"], 0.0)
        self.assertEqual(test["overall"]["mae"], 1.0)

        validation_data = PreparedTemporalDataset(DATA_DIR, "validation")
        test_data = PreparedTemporalDataset(DATA_DIR, "test")
        validation_range = validation_data.metadata["splits"]["validation"]
        test_range = test_data.metadata["splits"]["test"]
        self.assertLess(validation_data.target_index.max(), test_data.current_index.min())
        self.assertLessEqual(
            validation_data.target_index.max(), validation_range["end_index"]
        )
        self.assertGreaterEqual(test_data.history_start_index.min(), test_range["start_index"])

    def test_experiment_selection_never_uses_test_metrics(self) -> None:
        def experiment(name: str, validation_rmse: float, test_rmse: float) -> dict:
            return {
                "experiment_id": name,
                "validation_metrics": {
                    "overall": {"rmse": validation_rmse, "r2": 0.0, "mae": 1.0},
                    "by_horizon": {
                        "12h": {"rmse": validation_rmse},
                        "24h": {"rmse": validation_rmse},
                    },
                },
                "test_metrics": {"overall": {"rmse": test_rmse}},
            }

        selected = select_best_experiment(
            [
                experiment("better-validation", 1.0, 100.0),
                experiment("better-test", 2.0, 0.0),
            ]
        )
        self.assertEqual(selected["experiment_id"], "better-validation")


if __name__ == "__main__":
    unittest.main()
