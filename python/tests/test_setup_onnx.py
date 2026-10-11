import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from deeponet_irrigation.onnx_deployment import validate_onnx_model_contract
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.setup_onnx import CHECKPOINT, setup_deployment


MODELS = REPOSITORY_ROOT / "models"


class SetupONNXTests(unittest.TestCase):
    def test_missing_checkpoint_fails_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pt"
            with self.assertRaisesRegex(
                FileNotFoundError, "setup-onnx never trains a replacement"
            ):
                setup_deployment(checkpoint_path=missing, print_summary=False)

    def test_setup_uses_frozen_checkpoint_without_dataset_or_training(self) -> None:
        self.assertEqual(CHECKPOINT, MODELS / "deeponet_reference.pt")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            checkpoint = models / "deeponet_reference.pt"
            metadata = models / "deeponet_reference.metadata.json"
            golden_npz = models / "deeponet_reference_golden.npz"
            onnx_model = models / "deeponet_reference.onnx"
            golden_binary = models / "deeponet_reference_golden.bin"
            golden_json = models / "deeponet_reference_golden.json"
            shutil.copy2(MODELS / checkpoint.name, checkpoint)
            shutil.copy2(MODELS / metadata.name, metadata)
            shutil.copy2(MODELS / golden_npz.name, golden_npz)

            with mock.patch(
                "deeponet_irrigation.training.train_model",
                side_effect=AssertionError("training must not run"),
            ):
                result = setup_deployment(
                    checkpoint_path=checkpoint,
                    onnx_path=onnx_model,
                    metadata_path=metadata,
                    golden_npz_path=golden_npz,
                    golden_binary_path=golden_binary,
                    golden_metadata_path=golden_json,
                    repository_root=root,
                    print_summary=False,
                )

            self.assertIsNone(result["onnx_sha256_before"])
            self.assertEqual(result["golden_case_count"], 10)
            self.assertLessEqual(result["max_absolute_error"], 1e-4)
            for artifact in (
                onnx_model,
                metadata,
                golden_npz,
                golden_binary,
                golden_json,
            ):
                self.assertTrue(artifact.is_file(), artifact)
            self.assertFalse((root / "data").exists())
            _, session, _ = validate_onnx_model_contract(onnx_model, 723)
            self.assertEqual(len(session.get_inputs()), 3)
            fixture_metadata = json.loads(golden_json.read_text(encoding="utf-8"))
            self.assertEqual(fixture_metadata["header"]["case_count"], 10)


if __name__ == "__main__":
    unittest.main()
