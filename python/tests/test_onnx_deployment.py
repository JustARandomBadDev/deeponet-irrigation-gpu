import importlib.util
import json
import unittest
import numpy as np
import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset
from deeponet_irrigation.onnx_deployment import (
    ONNX_INPUT_NAMES,
    DeepONetONNXWrapper,
    load_frozen_deeponet,
    prepare_onnx_inputs,
    pytorch_deployment_prediction,
    sample_batch,
)
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.temporal_features import derive_temporal_features


ROOT = REPOSITORY_ROOT
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
CHECKPOINT = ROOT / "models" / "deeponet_reference.pt"
ONNX_PATH = ROOT / "models" / "deeponet_reference.onnx"
METADATA_PATH = ROOT / "models" / "deeponet_reference.metadata.json"
GOLDEN_PATH = ROOT / "models" / "deeponet_reference_golden.npz"
ONNX_AVAILABLE = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)
ONNX_ARTIFACT_AVAILABLE = ONNX_AVAILABLE and ONNX_PATH.exists()


class ONNXDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = PreparedTemporalDataset(DATA_DIR, "validation")
        cls.model, cls.checkpoint = load_frozen_deeponet(
            CHECKPOINT, cls.dataset.metadata, torch.device("cpu")
        )
        cls.wrapper = DeepONetONNXWrapper(cls.model).eval()

    def test_frozen_checkpoint_reconstructs(self) -> None:
        self.assertTrue(self.checkpoint["reference_model"])
        self.assertEqual(self.model.branch_input_dimension, 723)
        self.assertEqual(self.model.trunk_input_dimension, 1)

    def test_prepared_contract_matches_frozen_pytorch_model(self) -> None:
        history, horizons, _, _ = sample_batch(
            self.dataset, horizon_hours=12, batch_size=8
        )
        inputs = prepare_onnx_inputs(self.model, history, horizons)
        self.assertEqual(inputs["branch_input"].shape, (8, 723))
        self.assertEqual(inputs["horizon"].shape, (8, 1))
        self.assertEqual(inputs["current_moisture"].shape, (8, 1))
        self.assertEqual(inputs["branch_input"].dtype, np.float32)
        physical_trends = derive_temporal_features(
            history,
            self.model.temporal_feature_names,
            feature_order=self.model.feature_order,
            feature_means=self.model.input_feature_means,
            feature_stds=self.model.input_feature_stds,
        )
        normalized_trends = (
            physical_trends - self.model.temporal_feature_means
        ) / self.model.temporal_feature_stds
        np.testing.assert_allclose(
            inputs["branch_input"][:, -3:], normalized_trends.numpy()
        )
        with torch.inference_mode():
            original = self.model(history, horizons).numpy()
        deployed = pytorch_deployment_prediction(self.wrapper, inputs)
        np.testing.assert_allclose(deployed, original, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(ONNX_ARTIFACT_AVAILABLE, "ONNX packages/artifact unavailable")
    def test_onnx_checker_and_contract(self) -> None:
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(onnx.load(ONNX_PATH))
        session = ort.InferenceSession(
            str(ONNX_PATH), providers=["CPUExecutionProvider"]
        )
        self.assertEqual(
            {value.name: value.shape for value in session.get_inputs()},
            {
                "branch_input": ["batch", 723],
                "horizon": ["batch", 1],
                "current_moisture": ["batch", 1],
            },
        )
        self.assertEqual(len(session.get_outputs()), 1)
        self.assertEqual(session.get_outputs()[0].name, "prediction")
        self.assertEqual(session.get_outputs()[0].shape, ["batch", 1])

    @unittest.skipUnless(ONNX_ARTIFACT_AVAILABLE, "ONNX packages/artifact unavailable")
    def test_dynamic_batches_and_numerical_agreement(self) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(ONNX_PATH), providers=["CPUExecutionProvider"]
        )
        for batch_size in (1, 8):
            history, horizons, _, _ = sample_batch(
                self.dataset, horizon_hours=24, batch_size=batch_size
            )
            inputs = prepare_onnx_inputs(self.model, history, horizons)
            expected = pytorch_deployment_prediction(self.wrapper, inputs)
            actual = session.run(
                None, {name: inputs[name] for name in ONNX_INPUT_NAMES}
            )[0]
            self.assertEqual(actual.shape, (batch_size, 1))
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    @unittest.skipUnless(METADATA_PATH.exists(), "Deployment metadata unavailable")
    def test_deployment_metadata_matches_contract(self) -> None:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(metadata["branch_input_dimension"], 723)
        self.assertEqual(metadata["dtype"], "float32")
        self.assertEqual(metadata["onnx_opset"], 17)
        self.assertTrue(metadata["dynamic_batch"])
        self.assertEqual(
            [item["name"] for item in metadata["inputs"]], list(ONNX_INPUT_NAMES)
        )
        self.assertEqual(
            metadata["derived_trend_features"]["onnx_branch_values"], "normalized"
        )

    @unittest.skipUnless(GOLDEN_PATH.exists(), "Golden test vectors unavailable")
    def test_golden_vectors_match_pytorch(self) -> None:
        with np.load(GOLDEN_PATH) as values:
            inputs = {name: values[name] for name in ONNX_INPUT_NAMES}
            expected = values["expected_prediction"]
        actual = pytorch_deployment_prediction(self.wrapper, inputs)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
