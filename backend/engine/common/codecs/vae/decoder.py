"""
AutoencoderKL VAE Decoder — using RuntimeContext.

Standard SD VAE architecture. Shared by all Flux/Z-Image/Qwen/FIBO image models.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

from backend.engine.runtime._base import RuntimeContext


# NCHW ↔ NHWC 转换（MLX Conv2d/GroupNorm 工作在 NHWC；CUDA 使用标准 NCHW）
def _to_nhwc(ctx, x): return ctx.permute(x, (0, 2, 3, 1))
def _to_nchw(ctx, x): return ctx.permute(x, (0, 3, 1, 2))


def _vae_cuda_nchw(ctx: RuntimeContext) -> bool:
    return getattr(ctx, "backend", None) == "cuda"


class ResnetBlock:
    """VAE ResNet: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv → +shortcut。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ctx: RuntimeContext,
        use_shortcut: bool = False,
        *,
        cast_after_norm: bool = False,
        norm_input_fp32: bool = False,
    ):
        nn = ctx; self.ctx = ctx
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.use_shortcut = use_shortcut
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if use_shortcut else None
        self.cast_after_norm = bool(cast_after_norm)
        self.norm_input_fp32 = bool(norm_input_fp32)

    def forward(self, x):
        ctx = self.ctx
        if _vae_cuda_nchw(ctx):
            if self.norm_input_fp32:
                torch = importlib.import_module("torch")
                x_norm = x.to(dtype=torch.float32)
            else:
                x_norm = x
            h = self.norm1(x_norm)
            h = ctx.silu(h)
            h = self.conv1(h)
            if self.norm_input_fp32:
                torch = importlib.import_module("torch")
                h = h.to(dtype=torch.float32)
            h = self.norm2(h)
            h = ctx.silu(h)
            h = self.conv2(h)
            if self.conv_shortcut is not None:
                return self.conv_shortcut(x) + h
            return x + h
        h = _to_nhwc(ctx, x)
        if self.norm_input_fp32:
            h = h.astype(importlib.import_module("mlx.core").float32)
        h = self.norm1(h)
        if self.cast_after_norm:
            h = h.astype(importlib.import_module("mlx.core").bfloat16)
        h = ctx.silu(h)
        h = self.conv1(h)
        if self.norm_input_fp32:
            h = h.astype(importlib.import_module("mlx.core").float32)
        h = self.norm2(h)
        if self.cast_after_norm:
            h = h.astype(importlib.import_module("mlx.core").bfloat16)
        h = ctx.silu(h)
        h = self.conv2(h)
        h = _to_nchw(ctx, h)
        if self.conv_shortcut is not None:
            s = _to_nhwc(ctx, x); s = self.conv_shortcut(s); x = _to_nchw(ctx, s)
        return x + h


class SpatialAttention:
    """VAE 中间块的空间注意力 (1-head)。"""

    def __init__(
        self,
        dim: int = 512,
        ctx: RuntimeContext = None,
        *,
        cast_after_norm: bool = False,
        norm_input_fp32: bool = False,
    ):
        nn = ctx; self.ctx = ctx
        self.norm = nn.GroupNorm(32, dim, eps=1e-6, pytorch_compatible=True)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.cast_after_norm = bool(cast_after_norm)
        self.norm_input_fp32 = bool(norm_input_fp32)

    def forward(self, x):
        ctx = self.ctx
        if _vae_cuda_nchw(ctx):
            if self.norm_input_fp32:
                torch = importlib.import_module("torch")
                h = self.norm(x.to(dtype=torch.float32))
            else:
                h = self.norm(x)
            B, C, H, W = h.shape
            h_bhwc = ctx.permute(h, (0, 2, 3, 1))
            scale = C ** -0.5
            q = ctx.reshape(self.to_q(h_bhwc), (B, H * W, 1, C))
            k = ctx.reshape(self.to_k(h_bhwc), (B, H * W, 1, C))
            v = ctx.reshape(self.to_v(h_bhwc), (B, H * W, 1, C))
            q = ctx.permute(q, (0, 2, 1, 3))
            k = ctx.permute(k, (0, 2, 1, 3))
            v = ctx.permute(v, (0, 2, 1, 3))
            out = ctx.attention(q, k, v, scale=scale)
            out = ctx.permute(out, (0, 2, 1, 3))
            out = ctx.reshape(out, (B, H, W, C))
            out = self.to_out(out)
            out = ctx.permute(out, (0, 3, 1, 2))
            return x + out
        # x: NCHW → NHWC for norm and linear (MLX)
        h = _to_nhwc(ctx, x)
        B, H, W, C = h.shape
        scale = C ** -0.5

        if self.norm_input_fp32:
            h = h.astype(importlib.import_module("mlx.core").float32)
        h = self.norm(h)
        if self.cast_after_norm:
            h = h.astype(importlib.import_module("mlx.core").bfloat16)
        q = ctx.reshape(self.to_q(h), (B, H * W, 1, C))
        k = ctx.reshape(self.to_k(h), (B, H * W, 1, C))
        v = ctx.reshape(self.to_v(h), (B, H * W, 1, C))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))

        out = ctx.attention(q, k, v, scale=scale)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, H, W, C))
        out = self.to_out(out)
        out = _to_nchw(ctx, out)
        return x + out


class Upsample:
    """最近邻 2x 上采样 → Conv2d。"""

    def __init__(self, in_ch: int, out_ch: int, ctx: RuntimeContext):
        self.ctx = ctx; self.conv = ctx.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        ctx = self.ctx
        if _vae_cuda_nchw(ctx):
            F = importlib.import_module("torch.nn.functional")
            x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")
            return self.conv(x_up)
        B, C, H, W = x.shape
        h = _to_nhwc(ctx, x)  # [B, H, W, C]
        # 沿 H 重复: [B, H, W, C] → [B, H, 1, W, C] → concat → [B, H*2, W, C]
        h = ctx.reshape(h, (B, H, 1, W, C))
        h = ctx.concat([h, h], axis=2)
        h = ctx.reshape(h, (B, H * 2, W, C))
        # 沿 W 重复: [B, H*2, W, C] → [B, H*2, W, 1, C] → concat → [B, H*2, W*2, C]
        h = ctx.reshape(h, (B, H * 2, W, 1, C))
        h = ctx.concat([h, h], axis=3)
        h = ctx.reshape(h, (B, H * 2, W * 2, C))
        h = self.conv(h)
        return _to_nchw(ctx, h)


def _parse_vae_decoder_arch(vae_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = vae_cfg or {}
    block_out = cfg.get("block_out_channels") or (128, 256, 512, 512)
    return {
        "block_out_channels": tuple(int(x) for x in block_out),
        "layers_per_block": int(cfg.get("layers_per_block", 3)),
        "norm_num_groups": int(cfg.get("norm_num_groups", 32)),
        "mid_block_add_attention": bool(cfg.get("mid_block_add_attention", True)),
    }


class VAEDecoder:
    """AutoencoderKL 解码器 — 适配 RuntimeContext 的通用 VAE 解码路径。

    架构: ConvIn → Mid(resnet[+attn]+resnet) → Up blocks → NormOut → ConvOut
    """

    def __init__(
        self,
        latent_channels: int = 16,
        ctx: RuntimeContext = None,
        scaling_factor: float = 1.0,
        shift_factor: float = 0.0,
        *,
        vae_cfg: dict[str, Any] | None = None,
        block_out_channels: tuple[int, ...] | None = None,
        layers_per_block: int | None = None,
        norm_num_groups: int | None = None,
        mid_block_add_attention: bool | None = None,
    ):
        arch = _parse_vae_decoder_arch(vae_cfg)
        if block_out_channels is not None:
            arch["block_out_channels"] = tuple(int(x) for x in block_out_channels)
        if layers_per_block is not None:
            arch["layers_per_block"] = int(layers_per_block)
        if norm_num_groups is not None:
            arch["norm_num_groups"] = int(norm_num_groups)
        if mid_block_add_attention is not None:
            arch["mid_block_add_attention"] = bool(mid_block_add_attention)

        blocks = arch["block_out_channels"]
        if len(blocks) < 2:
            raise RuntimeError(f"VAEDecoder: block_out_channels must have >= 2 entries, got {blocks!r}")

        self.ctx = ctx
        nn = ctx
        C = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        top = int(blocks[-1])
        layers = int(arch["layers_per_block"])
        groups = int(arch["norm_num_groups"])

        self.conv_in = nn.Conv2d(C, top, 3, padding=1)
        self.mid_resnet1 = ResnetBlock(top, top, ctx)
        self.mid_attn = SpatialAttention(top, ctx) if arch["mid_block_add_attention"] else None
        self.mid_resnet2 = ResnetBlock(top, top, ctx)

        rev = list(reversed(blocks))
        self._up_stages: list[tuple[list[ResnetBlock], Upsample | None]] = []
        for i, out_ch in enumerate(rev):
            in_ch = int(rev[i - 1]) if i > 0 else int(rev[0])
            out_ch = int(out_ch)
            resnets: list[ResnetBlock] = []
            for j in range(layers):
                if j == 0 and in_ch != out_ch:
                    resnets.append(ResnetBlock(in_ch, out_ch, ctx, use_shortcut=True))
                else:
                    resnets.append(ResnetBlock(out_ch, out_ch, ctx))
            ups = Upsample(out_ch, out_ch, ctx) if i < len(rev) - 1 else None
            self._up_stages.append((resnets, ups))
            setattr(self, f"up{i + 1}_resnets", resnets)
            if ups is not None:
                setattr(self, f"up{i + 1}_up", ups)

        out_ch = int(rev[-1])
        self.norm_out = nn.GroupNorm(groups, out_ch, eps=1e-6, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(out_ch, 3, 3, padding=1)

        self._param_map: dict[str, Any] = {}
        self._built = False

    def _build_param_map(self):
        if self._built: return
        _collect_nn_params(self, "", self._param_map)
        self._built = True

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        return weights

    def load_weights(
        self,
        weights: list[tuple[str, Any]],
        strict: bool = False,
        ctx: RuntimeContext | None = None,
        *,
        bundle_affine_bits: int | None = None,
        inference_mode: Any | None = None,
    ):
        if (
            inference_mode is not None
            and getattr(inference_mode, "kind", None) == "quantized"
            and getattr(inference_mode, "bits", None) in (4, 8)
        ):
            from backend.engine.common.model.quantized_load_mlx import load_weights_quantized_inference

            load_ctx = ctx if ctx is not None else self.ctx
            return load_weights_quantized_inference(
                self,
                weights,
                strict=strict,
                ctx=load_ctx,
                bundle_affine_bits=bundle_affine_bits,
                bits=int(inference_mode.bits),
                group_size=int(getattr(inference_mode, "group_size", 64)),
            )

        self._build_param_map()
        loaded, skipped = [], []
        for key, tensor in weights:
            if key in self._param_map:
                p = self._param_map[key]
                if p.shape == tensor.shape:
                    p[:] = tensor; loaded.append(key)
                else:
                    skipped.append(f"{key} shape:{p.shape} vs {tensor.shape}")
            else:
                skipped.append(key)
        if strict and skipped: raise ValueError(f"Weight errors: {skipped}")
        return loaded, skipped

    def forward(self, latents):
        ctx = self.ctx
        # 处理 5D 输入 [B, C, F, H, W]：去掉帧维度
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        # Apply scaling (matching diffusers/reference VAE)
        if self.scaling_factor != 1.0 or self.shift_factor != 0.0:
            latents = (latents / self.scaling_factor) + self.shift_factor
        if _vae_cuda_nchw(ctx):
            x = self.conv_in(latents)
        else:
            x = _to_nhwc(ctx, latents)
            x = self.conv_in(x)
            x = _to_nchw(ctx, x)

        x = self.mid_resnet1.forward(x)
        if self.mid_attn is not None:
            x = self.mid_attn.forward(x)
        x = self.mid_resnet2.forward(x)

        for resnets, ups in self._up_stages:
            for r in resnets:
                x = r.forward(x)
            if ups is not None:
                x = ups.forward(x)

        if _vae_cuda_nchw(ctx):
            x = self.norm_out(x)
            x = ctx.silu(x)
            x = self.conv_out(x)
        else:
            x = _to_nhwc(ctx, x)
            x = self.norm_out(x)
            x = ctx.silu(x)
            x = self.conv_out(x)
            x = _to_nchw(ctx, x)
        return x


def vae_output_to_uint8_hwc(image: Any, ctx: Any | None = None) -> Any:
    """VAE tensor/array [-1, 1] → HWC ``uint8`` numpy.

    Runs ``ctx.eval`` before host read when ``ctx.is_tensor`` applies (MLX lazy graphs).
    Uses float32 + ``rint`` for 8-bit conversion to reduce FP16 pepper-noise in flat areas.
    """
    import numpy as np

    if ctx is not None and getattr(ctx, "is_tensor", lambda _x: False)(image):
        ctx.eval(image)
    if hasattr(image, "detach"):
        arr = image.detach().cpu().numpy()
    elif hasattr(image, "numpy"):
        arr = image.numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] <= 4:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.asarray(arr, dtype=np.float32)
    arr = (arr + 1.0) / 2.0 * 255.0
    np.clip(arr, 0.0, 255.0, out=arr)
    return np.rint(arr).astype(np.uint8)


def flux2_preprocess_latents_for_decode(
    ctx: RuntimeContext,
    latents: Any,
    vae_weights: dict,
    scaling_factor: float,
    shift_factor: float,
) -> Any:
    """Flux2-style BN + post_quant path before standard VAE decode."""
    bn_mean = vae_weights.get("bn.running_mean", ctx.zeros((128,))).reshape(1, -1, 1, 1)
    bn_var = vae_weights.get("bn.running_var", ctx.ones((128,))).reshape(1, -1, 1, 1)
    latents = latents * ctx.sqrt(bn_var + 1e-4) + bn_mean

    b, c_, h_, w_ = latents.shape
    latents = latents.reshape(b, c_ // 4, 2, 2, h_, w_)
    latents = ctx.permute(latents, (0, 1, 4, 2, 5, 3))
    latents = latents.reshape(b, c_ // 4, h_ * 2, w_ * 2)

    latents = (latents / scaling_factor) + shift_factor
    latents = ctx.permute(latents, (0, 2, 3, 1))

    pw = vae_weights.get("post_quant_conv.weight")
    pb = vae_weights.get("post_quant_conv.bias")
    if pw is not None and pb is not None:
        pw_shape = tuple(int(x) for x in pw.shape)
        if len(pw_shape) == 4 and pw_shape[1] == pw_shape[2] == 1:
            kernel = pw
        else:
            kernel = ctx.permute(pw, (0, 2, 3, 1))
        latents = ctx.conv2d(latents, kernel, stride=1, padding=0)
        latents = latents + pb.reshape(1, 1, 1, -1)

    return ctx.permute(latents, (0, 3, 1, 2))


def reshape_packed_latents_to_nchw(latents: Any) -> Any:
    """``[B, seq, C]`` packed tokens → ``[B, C, H, W]`` for standard VAE decode."""
    if getattr(latents, "ndim", None) != 3:
        return latents
    b, seq_len, channels = latents.shape
    latent_h = int(seq_len ** 0.5)
    latent_w = seq_len // latent_h
    return latents.reshape(b, latent_h, latent_w, channels).transpose(0, 3, 1, 2)


def _conv_in_in_channels(shape: tuple[int, ...] | Any) -> int | None:
    """Infer conv_in input channels from PyTorch (O,I,kH,kW) or MLX (O,kH,kW,I) layouts."""
    if len(shape) != 4:
        return None
    o, a, b, c = (int(shape[i]) for i in range(4))
    k_small = b <= 7 and c <= 7
    # MLX / diffusers NHWC: (O, kH, kW, I)
    if a <= 7 and k_small and c > 7:
        return c
    # PyTorch NCHW: (O, I, kH, kW) — e.g. Z-Image (512, 16, 3, 3)
    if a > 7 and k_small:
        return a
    if k_small and 1 < a < o:
        return a
    if a <= 7 and k_small:
        return c
    return None


def infer_latent_channels(vae_cfg: dict[str, Any], vae_weights: dict[str, Any], *, default: int = 16) -> int:
    lc = vae_cfg.get("latent_channels")
    if lc is not None:
        return int(lc)
    for wkey in ("decoder.conv_in.weight", "conv_in.weight"):
        if wkey in vae_weights:
            inferred = _conv_in_in_channels(getattr(vae_weights[wkey], "shape", ()))
            if inferred is not None:
                return inferred
    wkey = "encoder.conv_out.weight"
    if wkey in vae_weights:
        sh = getattr(vae_weights[wkey], "shape", ())
        if len(sh) >= 1:
            return int(sh[0]) // 2
    return default


def read_vae_dir_config(vae_dir: Path | None) -> tuple[dict[str, Any], float, float]:
    vae_cfg: dict[str, Any] = {}
    scaling_factor = 1.0
    shift_factor = 0.0
    if vae_dir and (vae_dir / "config.json").exists():
        import json

        with open(vae_dir / "config.json") as f:
            vae_cfg = json.load(f)
        scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
        shift_factor = float(vae_cfg.get("shift_factor", 0.0))
    return vae_cfg, scaling_factor, shift_factor


def infer_flux2_vae_config(vae_weights: dict[str, Any]) -> dict[str, Any] | None:
    """Infer ``AutoencoderKLFlux2`` metadata when mlx-community bundles omit ``vae/config.json``."""
    if not vae_weights:
        return None
    if "bn.running_mean" not in vae_weights and "post_quant_conv.weight" not in vae_weights:
        return None
    return {
        "_class_name": "AutoencoderKLFlux2",
        "batch_norm_eps": 1e-4,
        "use_quant_conv": True,
        "use_post_quant_conv": True,
        "latent_channels": 32,
    }


def resolve_vae_bundle_config(
    vae_dir: Path | None,
    *,
    vae_weights: dict[str, Any] | None = None,
    ctx: RuntimeContext | None = None,
) -> tuple[dict[str, Any], float, float]:
    """Merge on-disk ``config.json`` with weight-based inference (Flux2 mlx bundles)."""
    vae_cfg, scaling_factor, shift_factor = read_vae_dir_config(vae_dir)
    if str(vae_cfg.get("_class_name") or "").strip():
        return vae_cfg, scaling_factor, shift_factor
    weights = vae_weights
    if weights is None and vae_dir and vae_dir.exists() and ctx is not None:
        weights = load_vae_weight_dict(ctx, vae_dir)
    inferred = infer_flux2_vae_config(weights or {})
    if inferred is None:
        return vae_cfg, scaling_factor, shift_factor
    merged = dict(inferred)
    merged.update(vae_cfg)
    return merged, scaling_factor, shift_factor


def release_vae_decoder_memory(ctx: RuntimeContext, vae: Any | None) -> None:
    """Drop a short-lived VAE decoder and free MLX cache after decode."""
    if vae is None:
        return
    del vae
    clear_cache_fn = getattr(ctx, "clear_cache", None)
    if clear_cache_fn is not None:
        clear_cache_fn()
    elif getattr(ctx, "backend", None) == "mlx":
        importlib.import_module("mlx.core").clear_cache()


def load_vae_weight_dict(
    ctx: RuntimeContext,
    vae_dir: Path | None,
    *,
    fail_if_config_only: bool = False,
) -> dict[str, Any]:
    vae_weights: dict[str, Any] = {}
    if not vae_dir or not vae_dir.exists():
        return vae_weights
    saf_paths = sorted(vae_dir.glob("*.safetensors"))
    if saf_paths:
        for sf in saf_paths:
            vae_weights.update(ctx.load_weights(str(sf)))
    elif fail_if_config_only and (vae_dir / "config.json").is_file():
        raise RuntimeError(
            f"VAE directory has config but no *.safetensors under {vae_dir}; "
            "cannot decode (install model weights)."
        )
    return vae_weights


def flux2_quant_preprocess_gate(vae_cfg: dict[str, Any], vae_weights: dict[str, Any]) -> bool:
    has_flux2_quant_tensors = (
        "bn.running_mean" in vae_weights or "post_quant_conv.weight" in vae_weights
    )
    if not has_flux2_quant_tensors:
        return False
    use_quant = bool(vae_cfg.get("use_quant_conv", False))
    use_post = bool(vae_cfg.get("use_post_quant_conv", False))
    if use_quant or use_post:
        return True
    # mlx-community Flux2 bundles ship quant VAE tensors without config.json flags.
    return not vae_cfg.get("latent_channels") and not use_quant and not use_post


def apply_flux2_latent_preprocess_if_enabled(
    ctx: RuntimeContext,
    latents: Any,
    vae_cfg: dict[str, Any],
    vae_weights: dict[str, Any],
    scaling_factor: float,
    shift_factor: float,
) -> tuple[Any, float, float]:
    if not flux2_quant_preprocess_gate(vae_cfg, vae_weights):
        return latents, scaling_factor, shift_factor
    latents = flux2_preprocess_latents_for_decode(
        ctx, latents, vae_weights, scaling_factor, shift_factor
    )
    return latents, 1.0, 0.0


def build_standard_vae_preview_session(
    ctx: RuntimeContext,
    vae_dir: Path | None,
    *,
    on_log: Callable[[str, str], None] | None = None,
) -> dict[str, Any] | None:
    """Load a standard AutoencoderKL decoder once for step previews."""
    from backend.engine.common.codecs.vae.weight_remap import load_vae_decoder_from_weights

    vae_cfg, scaling_factor, shift_factor = read_vae_dir_config(vae_dir)
    vae_weights = load_vae_weight_dict(ctx, vae_dir)
    if not vae_weights:
        return None

    use_special_preprocess = flux2_quant_preprocess_gate(vae_cfg, vae_weights)
    if use_special_preprocess:
        scaling_factor, shift_factor = 1.0, 0.0

    vae = VAEDecoder(
        latent_channels=infer_latent_channels(vae_cfg, vae_weights),
        ctx=ctx,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
        vae_cfg=vae_cfg,
    )
    decoder_w, loaded, _skipped = load_vae_decoder_from_weights(vae, vae_weights)
    if not decoder_w:
        return None
    if on_log:
        on_log(
            "info",
            f"preview VAE session ready decoder_tensors={len(decoder_w)} "
            f"loaded_params={len(loaded)}",
        )
    return {
        "kind": "standard",
        "vae": vae,
        "vae_cfg": vae_cfg,
        "vae_weights": vae_weights,
        "use_special_preprocess": use_special_preprocess,
        "orig_scaling": float(vae_cfg.get("scaling_factor", 1.0)),
        "orig_shift": float(vae_cfg.get("shift_factor", 0.0)),
    }


def create_loaded_vae_decoder(
    ctx: RuntimeContext,
    latents: Any,
    vae_weights: dict[str, Any],
    scaling_factor: float,
    shift_factor: float,
    *,
    vae_cfg: dict[str, Any] | None = None,
    default_channels: int = 16,
    require_conv_in: bool = True,
    bundle_affine_bits: int | None = None,
    inference_mode: Any | None = None,
) -> tuple[VAEDecoder, dict[str, Any], list[Any], list[Any]]:
    from backend.engine.common.codecs.vae.weight_remap import load_vae_decoder_from_weights

    channels = infer_latent_channels(vae_cfg or {}, vae_weights, default=default_channels)
    vae = VAEDecoder(
        latent_channels=channels,
        ctx=ctx,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
        vae_cfg=vae_cfg,
    )
    decoder_w, loaded, skipped = load_vae_decoder_from_weights(
        vae,
        vae_weights,
        require_conv_in=require_conv_in,
        ctx=ctx,
        bundle_affine_bits=bundle_affine_bits,
        inference_mode=inference_mode,
    )
    return vae, decoder_w, loaded, skipped


def vae_forward_to_pil(ctx: RuntimeContext, vae: Any, latents: Any):
    from PIL import Image

    return Image.fromarray(vae_output_to_uint8_hwc(vae.forward(latents), ctx))


def _flatten_mlx_parameters_tree(node: Any, prefix: str, result: dict) -> None:
    """Flatten nested ``mlx.nn.Module.parameters()`` dict/list trees to dotted keys."""
    if isinstance(node, dict):
        for name, child in node.items():
            key = f"{prefix}.{name}" if prefix else name
            _flatten_mlx_parameters_tree(child, key, result)
        return
    if isinstance(node, (list, tuple)):
        for i, child in enumerate(node):
            key = f"{prefix}.{i}" if prefix else str(i)
            _flatten_mlx_parameters_tree(child, key, result)
        return
    if prefix:
        result[prefix] = node


def _collect_nn_params(obj: Any, prefix: str, result: dict):
    if hasattr(obj, 'parameters') and callable(obj.parameters):
        try:
            params = obj.parameters()
            if isinstance(params, dict):
                _flatten_mlx_parameters_tree(params, prefix, result)
                return
        except Exception: pass
    for attr_name in sorted(dir(obj)):
        if attr_name.startswith('_') or attr_name in ('ctx', '_param_map', '_built'):
            continue
        try: attr = getattr(obj, attr_name)
        except Exception: continue
        if attr is None or isinstance(attr, (int, float, str, bool, type)): continue
        new = f"{prefix}.{attr_name}" if prefix else attr_name
        if hasattr(attr, 'parameters') and callable(attr.parameters):
            _collect_nn_params(attr, new, result)
        elif isinstance(attr, (list, tuple)):
            for i, item in enumerate(attr):
                _collect_nn_params(item, f"{new}.{i}", result)
        elif hasattr(attr, '__dict__') and not isinstance(attr, type):
            _collect_nn_params(attr, new, result)
