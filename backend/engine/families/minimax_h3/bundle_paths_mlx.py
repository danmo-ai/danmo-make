"""MiniMax-H3 bundle directory layout (PipeNetwork DiT + upstream ``FL2VA/`` aux)."""
from __future__ import annotations

from pathlib import Path


def minimax_h3_aux_root(bundle_root: Path) -> Path:
    """Return ``FL2VA/`` when present (MiniMaxAI layout), else bundle root."""
    root = Path(bundle_root)
    fl2va = root / "FL2VA"
    if fl2va.is_dir() and (
        (fl2va / "model_index.json").is_file()
        or (fl2va / "video_vae").is_dir()
        or (fl2va / "text_encoder").is_dir()
    ):
        return fl2va
    return root


def minimax_h3_tokenizer_root(bundle_root: Path) -> Path:
    aux = minimax_h3_aux_root(bundle_root)
    for candidate in (aux / "processor", aux / "text_encoder", aux):
        if (candidate / "tokenizer.json").is_file():
            return candidate
    return aux
