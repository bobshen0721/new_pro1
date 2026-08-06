import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModelConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        cls.download_source = (ROOT / "download_models.ps1").read_text(encoding="utf-8")
        cls.release_manifest = json.loads(
            (ROOT / "model-release-manifest.json").read_text(encoding="utf-8")
        )

    def test_primary_model_is_tea_asr_mini(self):
        self.assertIn(
            'TEA_MODEL_ID = "JacobLinCool/TEA-ASR-1.1-mini"',
            self.app_source,
        )
        self.assertIn('"folder": "TEA-ASR-1.1-mini"', self.download_source)
        self.assertNotIn(
            'TEA_MODEL_ID = "JacobLinCool/TEA-ASR-1.1"',
            self.app_source,
        )

    def test_qwen_models_are_forced_to_local_files(self):
        self.assertGreaterEqual(self.app_source.count('"local_files_only": True'), 1)
        self.assertIn("local_files_only=True", self.app_source)
        self.assertIn("MODEL_REQUIRED_FILES", self.app_source)
        self.assertIn("[switch]$VerifyOnly", self.download_source)

    def test_click_to_seek_contract_is_preserved(self):
        self.assertIn("e.target.closest('.transcript-row')", self.app_source)
        self.assertIn("audio.currentTime=Number(row.dataset.start||0)", self.app_source)
        self.assertIn("audio.play().catch(()=>{})", self.app_source)

    def test_models_v2_release_contains_mini(self):
        self.assertEqual(self.release_manifest["release_tag"], "models-v2")
        folders = {model["folder"] for model in self.release_manifest["models"]}
        self.assertEqual(
            folders,
            {
                "faster-whisper-large-v2",
                "TEA-ASR-1.1-mini",
                "Qwen3-ForcedAligner-0.6B",
            },
        )


if __name__ == "__main__":
    unittest.main()
