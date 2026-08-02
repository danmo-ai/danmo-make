"""LTX video model weight remapping — LTX 2.3 (48-layer A/V) + legacy 28-layer diffusers."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import mlx.core as mx


def load_split_safetensors(
    path: str | Path,
    prefix: str | None = None,
) -> dict[str, mx.array]:
    """Load safetensors via ``mx.load``, optionally stripping a key prefix."""
    path = Path(path)
    raw = mx.load(str(path))
    if not prefix:
        return raw
    weights: dict[str, mx.array] = {}
    for key, tensor in raw.items():
        if key.startswith(prefix):
            weights[key[len(prefix) :]] = tensor
    return weights


def normalize_ltx23_bundle_keys(weights: dict[str, Any]) -> dict[str, Any]:
    """Strip ``transformer.`` prefix and normalize AdaLN / FF key tails for LTX23Model."""
    out: dict[str, Any] = {}
    for key, tensor in weights.items():
        nk = key
        if nk.startswith("transformer."):
            nk = nk[len("transformer.") :]
        nk = nk.replace(".linear1.", ".linear_1.")
        nk = nk.replace(".linear2.", ".linear_2.")
        # mlx-forge / diffusers FF naming → LTX23 FeedForward (proj_in / proj_out)
        nk = nk.replace(".ff.net.2.", ".ff.proj_out.")
        nk = nk.replace(".ff.net.0.proj.", ".ff.proj_in.")
        nk = nk.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
        nk = nk.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        nk = re.sub(r"(\.attn[12]\.)to_out\.0\.(weight|bias)$", r"\1to_out.\2", nk)
        out[nk] = tensor
    return out


def remap_ltx23_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Map bundle keys to ``LTX23Model`` / ``LTX23Transformer._param_map`` names.

    After :func:`normalize_ltx23_bundle_keys`, keys are mostly identity
    (``transformer_blocks.N.*``, ``patchify_proj.*``, ``adaln_single.*``, …).
    """
    return normalize_ltx23_bundle_keys(weights)


def _looks_like_ltx23_checkpoint(weights: dict[str, Any]) -> bool:
    for k in weights:
        if "patchify_proj" in k or "audio_patchify_proj" in k:
            return True
        if ".audio_attn1." in k or ".audio_to_video_attn." in k:
            return True
        m = re.match(r"^(?:transformer\.)?transformer_blocks\.(\d+)\.", k)
        if m and int(m.group(1)) >= 27:
            return True
    return False


def remap_ltx_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Route LTX 2.3 bundles to :func:`remap_ltx23_weights`; else legacy 28-layer diffusers."""
    if _looks_like_ltx23_checkpoint(weights):
        return remap_ltx23_weights(weights)
    return _remap_ltx28_diffusers_weights(weights)


def restore_diffusers_names_from_mlx_forge_ltx(weights: dict[str, Any]) -> dict[str, Any]:
    """Undo ``mlx_forge.recipes.ltx_23.sanitize_transformer_key`` for legacy remap path."""
    out: dict[str, Any] = {}
    for k, v in weights.items():
        nk = k
        nk = nk.replace(".ff.proj_out.", ".ff.net.2.")
        nk = nk.replace(".ff.proj_in.", ".ff.net.0.proj.")
        nk = nk.replace(".audio_ff.proj_out.", ".audio_ff.net.2.")
        nk = nk.replace(".audio_ff.proj_in.", ".audio_ff.net.0.proj.")
        nk = nk.replace(".linear2.", ".linear_2.")
        nk = nk.replace(".linear1.", ".linear_1.")
        nk = re.sub(r"(\.attn[12]\.)to_out\.(weight|bias)$", r"\1to_out.0.\2", nk)
        out[nk] = v
    return out


def looks_like_mlx_forge_ltx_transformer_keys(weights: dict[str, Any]) -> bool:
    """Detect mlx-forge ``sanitize_transformer_key`` naming (needs restore before ``remap_ltx_weights``)."""
    for k in weights:
        if ".ff.proj_in." in k or ".ff.proj_out." in k:
            return True
        if ".audio_ff.proj_in." in k or ".audio_ff.proj_out." in k:
            return True
    return False


def max_remapped_ltx_block_index(remapped: dict[str, Any]) -> int:
    """Largest ``N`` in ``blocks.N.*`` keys, or ``-1`` if none."""
    mx = -1
    for k in remapped:
        m = re.match(r"^blocks\.(\d+)\.", k)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx


def prepare_ltx_video_transformer_weights(config: Any, weights: dict[str, Any]) -> dict[str, Any]:
    """Pre-load LTX DiT weight normalize + depth guard (registry-driven from ``VideoPipeline``)."""
    w = weights
    if bool(getattr(config, "uses_mlx_forge_weight_restore", False)) and looks_like_mlx_forge_ltx_transformer_keys(w):
        w = restore_diffusers_names_from_mlx_forge_ltx(w)
    if bool(getattr(config, "validate_ltx_block_depth", False)):
        mx_blk = max_remapped_ltx_block_index(w)
        if mx_blk >= 0:
            n_blocks = mx_blk + 1
            depth = int(getattr(config, "depth", 0))
            if n_blocks != depth:
                raise RuntimeError(
                    f"LTX weights map to {n_blocks} transformer blocks after remap, "
                    f"but LTXConfig.depth={depth} (diffusers LTXVideoTransformer3DModel). "
                    f"Public MLX-forge / dgrauet LTX-2.3 bundles use 48 layers and are not supported "
                    f"by this transformer implementation; use ``:original`` or a 28-block "
                    f"diffusers-compatible checkpoint, or extend the LTX family implementation."
                )
    return w


# ---------------------------------------------------------------------------
# Legacy diffusers LTXVideoTransformer3DModel (28 layers)
# ---------------------------------------------------------------------------

def _remap_ltx28_diffusers_weights(weights: dict[str, Any]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}

    _TOP_MAP: dict[str, str] = {
        "proj_in.bias": "patch_embed.proj.bias",
        "time_embed.emb.timestep_embedder.linear_1.weight": "time_embed.mlp_in.weight",
        "time_embed.emb.timestep_embedder.linear_1.bias": "time_embed.mlp_in.bias",
        "time_embed.emb.timestep_embedder.linear_2.weight": "time_embed.mlp_out.weight",
        "time_embed.emb.timestep_embedder.linear_2.bias": "time_embed.mlp_out.bias",
        "time_embed.linear.weight": "time_embed_out.weight",
        "time_embed.linear.bias": "time_embed_out.bias",
        "scale_shift_table": "output_modulation",
        "caption_projection.linear_1.weight": "caption_proj_in.weight",
        "caption_projection.linear_1.bias": "caption_proj_in.bias",
        "caption_projection.linear_2.weight": "caption_proj_out.weight",
        "caption_projection.linear_2.bias": "caption_proj_out.bias",
        "proj_out.weight": "proj_out.weight",
        "proj_out.bias": "proj_out.bias",
    }

    _BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.(.*)$")

    for key, tensor in weights.items():
        if key == "proj_in.weight":
            remapped["patch_embed.proj.weight"] = _reshape_proj_in_weight(tensor)
            continue
        if key in _TOP_MAP:
            remapped[_TOP_MAP[key]] = tensor
            continue
        m = _BLOCK_RE.match(key)
        if m:
            block_idx = m.group(1)
            tail = m.group(2)
            new_key = _remap_block_tail(block_idx, tail)
            if new_key is not None:
                remapped[new_key] = tensor
            continue
        remapped[key] = tensor

    return remapped


_BLOCK_TAIL_MAP: dict[str, str] = {
    "attn1.norm_q.weight": "self_attn.q_norm.weight",
    "attn1.norm_k.weight": "self_attn.k_norm.weight",
    "attn1.to_q.weight": "self_attn.to_q.weight",
    "attn1.to_q.bias": "self_attn.to_q.bias",
    "attn1.to_k.weight": "self_attn.to_k.weight",
    "attn1.to_k.bias": "self_attn.to_k.bias",
    "attn1.to_v.weight": "self_attn.to_v.weight",
    "attn1.to_v.bias": "self_attn.to_v.bias",
    "attn1.to_out.0.weight": "self_attn.to_out.weight",
    "attn1.to_out.0.bias": "self_attn.to_out.bias",
    "attn2.norm_q.weight": "cross_attn.q_norm.weight",
    "attn2.norm_k.weight": "cross_attn.k_norm.weight",
    "attn2.to_q.weight": "cross_attn.to_q.weight",
    "attn2.to_q.bias": "cross_attn.to_q.bias",
    "attn2.to_k.weight": "cross_attn.to_k.weight",
    "attn2.to_k.bias": "cross_attn.to_k.bias",
    "attn2.to_v.weight": "cross_attn.to_v.weight",
    "attn2.to_v.bias": "cross_attn.to_v.bias",
    "attn2.to_out.0.weight": "cross_attn.to_out.weight",
    "attn2.to_out.0.bias": "cross_attn.to_out.bias",
    "ff.net.0.proj.weight": "mlp_in.weight",
    "ff.net.0.proj.bias": "mlp_in.bias",
    "ff.net.2.weight": "mlp_out.weight",
    "ff.net.2.bias": "mlp_out.bias",
    "scale_shift_table": "scale_shift_table",
}


def _remap_block_tail(block_idx: str, tail: str) -> str | None:
    mapped = _BLOCK_TAIL_MAP.get(tail)
    if mapped is None:
        return None
    return f"blocks.{block_idx}.{mapped}"


def _reshape_proj_in_weight(tensor: Any) -> Any:
    out_ch, in_ch = tensor.shape
    return tensor.reshape(out_ch, 1, 1, 1, in_ch)  # type: ignore[attr-defined]
