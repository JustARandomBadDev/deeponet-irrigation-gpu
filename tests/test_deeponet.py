import tempfile
import unittest
from pathlib import Path

import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset
from deeponet_irrigation.evaluation import load_deeponet_checkpoint
from deeponet_irrigation.models import ResidualDeepONet
from deeponet_irrigation.temporal_features import TREND_FEATURES
from deeponet_irrigation.training import select_best_deeponet_experiment


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"


def small_model(**overrides: object) -> ResidualDeepONet:
    settings = {
        "history_steps": 4,
        "feature_count": 2,
        "branch_hidden_dimensions": (5,),
        "trunk_hidden_dimensions": (3,),
        "latent_dimension": 3,
        "target_std": 4.0,
        "soil_feature_index": 0,
        "soil_mean": 10.0,
        "soil_std": 2.0,
        "feature_order": ("soil_moisture", "weather_temp"),
    }
    settings.update(overrides)
    return ResidualDeepONet(**settings)


class DeepONetTests(unittest.TestCase):
    def test_project_branch_dimensions(self) -> None:
        raw = ResidualDeepONet(
            history_steps=144,
            feature_count=5,
            feature_order=("soil_moisture", "water", "rain", "temp", "humidity"),
        )
        trends = ResidualDeepONet(
            history_steps=144,
            feature_count=5,
            feature_order=(
                "soil_moisture",
                "applied_water_liters",
                "weather_rain",
                "weather_temp",
                "weather_humidity",
            ),
            temporal_feature_names=TREND_FEATURES,
            temporal_feature_means=(0.0, 0.0, 0.0),
            temporal_feature_stds=(1.0, 1.0, 1.0),
            input_feature_means=(0.0,) * 5,
            input_feature_stds=(1.0,) * 5,
        )
        self.assertEqual(raw.branch_input_dimension, 720)
        self.assertEqual(trends.branch_input_dimension, 723)
        self.assertEqual(raw.trunk_input_dimension, 1)

    def test_branch_trunk_and_output_shapes(self) -> None:
        model = small_model()
        branch = torch.zeros(7, 4, 2)
        horizon = torch.ones(7, 1)
        self.assertEqual(model.branch_input_dimension, 8)
        self.assertEqual(model.trunk_input_dimension, 1)
        self.assertEqual(model.encode_branch(branch).shape, (7, 3))
        self.assertEqual(model.encode_trunk(horizon).shape, (7, 3))
        self.assertEqual(model(branch, horizon).shape, (7, 1))

    def test_inner_product_is_explicit_and_keeps_column_shape(self) -> None:
        model = small_model()
        with torch.no_grad():
            model.operator_bias.fill_(0.5)
        branch_latent = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]])
        trunk_latent = torch.tensor([[4.0, 5.0, 6.0], [3.0, 2.0, 1.0]])
        expected = torch.tensor([[32.5], [5.5]])
        torch.testing.assert_close(
            model.combine_latents(branch_latent, trunk_latent), expected
        )

    def test_zero_normalized_residual_reconstructs_persistence(self) -> None:
        model = small_model()
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        branch = torch.zeros(2, 4, 2)
        branch[:, -1, 0] = torch.tensor([1.5, -2.0])
        prediction = model(branch, torch.ones(2, 1))
        torch.testing.assert_close(prediction, torch.tensor([[13.0], [6.0]]))

    def test_residual_scaling_is_applied_once(self) -> None:
        model = small_model()
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        with torch.no_grad():
            model.operator_bias.fill_(0.5)
        branch = torch.zeros(1, 4, 2)
        # current=10, normalized delta=0.5, std=4, so prediction=12
        torch.testing.assert_close(model(branch, torch.ones(1, 1)), torch.tensor([[12.0]]))

    def test_checkpoint_reload_reproduces_predictions(self) -> None:
        dataset = PreparedTemporalDataset(DATA_DIR, "train")
        model = ResidualDeepONet(
            history_steps=dataset.history_steps,
            feature_count=dataset.feature_count,
            branch_hidden_dimensions=(8, 4),
            trunk_hidden_dimensions=(4,),
            latent_dimension=3,
            target_std=4.435837,
            soil_feature_index=dataset.soil_index,
            soil_mean=dataset.soil_mean,
            soil_std=dataset.soil_std,
            feature_order=tuple(dataset.metadata["feature_order"]),
            sampling_interval_minutes=dataset.metadata["sampling_interval_minutes"],
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
        branch = torch.randn(3, dataset.history_steps, dataset.feature_count)
        horizon = torch.tensor([[1.0], [6.0], [24.0]])
        expected = model(branch, horizon)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deeponet.pt"
            torch.save(
                {
                    "architecture": model.architecture(),
                    "model_state_dict": model.state_dict(),
                    "preprocessing_contract": contract,
                },
                path,
            )
            loaded, _ = load_deeponet_checkpoint(
                path, dataset.metadata, torch.device("cpu")
            )
        torch.testing.assert_close(expected, loaded(branch, horizon))

    def test_selection_uses_validation_not_test_metrics(self) -> None:
        def experiment(name: str, validation_rmse: float, test_rmse: float) -> dict:
            return {
                "id": name,
                "validation_metrics": {
                    "overall": {"rmse": validation_rmse, "r2": 0.0, "mae": 1.0},
                    "by_horizon": {
                        "12h": {"rmse": validation_rmse},
                        "24h": {"rmse": validation_rmse},
                    },
                },
                "validation_change_status": {
                    "changing": {"rmse": validation_rmse}
                },
                "total_parameters": 10,
                "test_metrics": {"overall": {"rmse": test_rmse}},
            }

        selected = select_best_deeponet_experiment(
            [experiment("validation", 1.0, 100.0), experiment("test", 2.0, 0.0)]
        )
        self.assertEqual(selected["id"], "validation")


if __name__ == "__main__":
    unittest.main()
