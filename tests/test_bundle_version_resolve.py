"""Bare model ids resolve to an installed bundle when the default quant is absent."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.engine.contracts.pipeline_registry import local_bundle_root, resolve_bundle_version


def _entry() -> SimpleNamespace:
    return SimpleNamespace(
        raw={
            "versions": {
                "mlx-q4": {
                    "default": True,
                    "local_path": "models/Video/ltx-2.5-distilled-mlx-q4",
                    "quantization": {"bits": 4},
                },
                "mlx-q8": {
                    "default": False,
                    "local_path": "models/Video/ltx-2.5-distilled-mlx-q8",
                    "quantization": {"bits": 8},
                },
            }
        }
    )


class BundleVersionResolveTests(unittest.TestCase):
    def test_bare_id_uses_installed_non_default_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q8 = root / "models/Video/ltx-2.5-distilled-mlx-q8"
            q8.mkdir(parents=True)
            entry = _entry()

            version, path = resolve_bundle_version(root, entry, None)

            self.assertEqual(version, "mlx-q8")
            self.assertEqual(path, q8.resolve())
            self.assertIsNone(local_bundle_root(root, entry, None))

    def test_bare_id_keeps_default_when_that_bundle_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q4 = root / "models/Video/ltx-2.5-distilled-mlx-q4"
            q8 = root / "models/Video/ltx-2.5-distilled-mlx-q8"
            q4.mkdir(parents=True)
            q8.mkdir(parents=True)

            version, path = resolve_bundle_version(root, _entry(), None)

            self.assertEqual(version, "mlx-q4")
            self.assertEqual(path, q4.resolve())

    def test_explicit_missing_version_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models/Video/ltx-2.5-distilled-mlx-q8").mkdir(parents=True)

            version, path = resolve_bundle_version(root, _entry(), "mlx-q4")

            self.assertEqual(version, "mlx-q4")
            self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
