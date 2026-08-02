"""Disk-backed latent cache for LoRA training (encode once, stream samples)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

_SCHEMA_VERSION = 1


def _ensure_batch_latent(latent: mx.array) -> mx.array:
    """Training expects ``[B, C, H, W]``; cache stores per-sample ``z[0]`` as ``[C, H, W]``."""
    if latent.ndim == 3:
        return latent[None]
    if latent.ndim == 4:
        return latent
    raise RuntimeError(
        f"Latent cache sample has unexpected rank {latent.ndim} (expected 3=[C,H,W] or 4=[B,C,H,W])"
    )


def _fingerprint(
    *,
    dataset_id: str,
    n_pairs: int,
    num_augmentations: int,
    resolution: tuple[int, int],
    family: str,
    caption_mode: str = "",
) -> str:
    raw = f"{dataset_id}|{n_pairs}|{num_augmentations}|{resolution[0]}x{resolution[1]}|{family}|{caption_mode}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class LatentCache:
    def __init__(self, work_dir: Path) -> None:
        self.root = Path(work_dir) / "latent_cache"
        self.manifest_path = self.root / "manifest.json"

    def is_valid(
        self,
        *,
        dataset_id: str,
        n_pairs: int,
        num_augmentations: int,
        resolution: tuple[int, int],
        family: str,
        n_samples: int,
        caption_mode: str = "",
    ) -> bool:
        if not self.manifest_path.is_file():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if int(manifest.get("schema_version", 0)) != _SCHEMA_VERSION:
            return False
        fp = _fingerprint(
            dataset_id=dataset_id,
            n_pairs=n_pairs,
            num_augmentations=num_augmentations,
            resolution=resolution,
            family=family,
            caption_mode=caption_mode,
        )
        if manifest.get("fingerprint") != fp:
            return False
        if int(manifest.get("n_samples", 0)) != int(n_samples):
            return False
        for i in range(n_samples):
            if not (self.root / f"{i:05d}.safetensors").is_file():
                return False
        return True

    def begin(
        self,
        *,
        dataset_id: str,
        n_pairs: int,
        num_augmentations: int,
        resolution: tuple[int, int],
        family: str,
        tensor_keys: list[str],
        caption_mode: str = "",
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._tensor_keys = list(tensor_keys)
        self._manifest = {
            "schema_version": _SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "fingerprint": _fingerprint(
                dataset_id=dataset_id,
                n_pairs=n_pairs,
                num_augmentations=num_augmentations,
                resolution=resolution,
                family=family,
                caption_mode=caption_mode,
            ),
            "family": family,
            "n_pairs": n_pairs,
            "num_augmentations": num_augmentations,
            "resolution": list(resolution),
            "caption_mode": caption_mode,
            "tensor_keys": tensor_keys,
            "n_samples": 0,
        }

    def write_sample(self, idx: int, tensors: dict[str, mx.array]) -> None:
        if not hasattr(self, "_manifest"):
            raise RuntimeError("LatentCache.begin() must be called before write_sample()")
        missing = [k for k in self._tensor_keys if k not in tensors]
        if missing:
            raise RuntimeError(f"Latent cache sample {idx} missing tensors: {missing}")
        path = self.root / f"{idx:05d}.safetensors"
        mx.save_safetensors(str(path), {k: tensors[k] for k in self._tensor_keys})
        self._manifest["n_samples"] = max(int(self._manifest["n_samples"]), idx + 1)

    def finalize(self) -> int:
        if not hasattr(self, "_manifest"):
            raise RuntimeError("LatentCache.begin() must be called before finalize()")
        self.manifest_path.write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return int(self._manifest["n_samples"])

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise RuntimeError(f"Latent cache manifest not found: {self.manifest_path}")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def load_sample(self, idx: int) -> dict[str, mx.array]:
        path = self.root / f"{idx:05d}.safetensors"
        if not path.is_file():
            raise RuntimeError(f"Latent cache sample missing: {path}")
        return dict(mx.load(str(path)))

    def sample_z_image(self, idx: int) -> tuple[mx.array, mx.array]:
        data = self.load_sample(idx)
        return _ensure_batch_latent(data["latent"]), data["cap"]

    def sample_flux(self, idx: int, *, num_augmentations: int = 1) -> tuple[mx.array, mx.array, mx.array]:
        del num_augmentations
        data = self.load_sample(idx)
        return _ensure_batch_latent(data["latent"]), data["t5"], data["clip"]

    def sample_qwen(self, idx: int) -> tuple[mx.array, mx.array, mx.array]:
        data = self.load_sample(idx)
        return _ensure_batch_latent(data["latent"]), data["txt"], data["txt_mask"]

    def materialize_flux(self, n_samples: int) -> list[tuple[mx.array, mx.array, mx.array]]:
        return [self.sample_flux(i) for i in range(n_samples)]

    def materialize_z_image(self, n_samples: int) -> list[tuple[mx.array, mx.array]]:
        return [self.sample_z_image(i) for i in range(n_samples)]

    def materialize_qwen(self, n_samples: int) -> list[tuple[mx.array, mx.array, mx.array]]:
        return [self.sample_qwen(i) for i in range(n_samples)]

    def load_prior(self, name: str = "prior") -> dict[str, mx.array]:
        path = self.root / f"{name}.safetensors"
        if not path.is_file():
            raise RuntimeError(f"Prior cache not found: {path}")
        return dict(mx.load(str(path)))

    def write_prior(self, tensors: dict[str, mx.array], *, name: str = "prior") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(self.root / f"{name}.safetensors"), tensors)
