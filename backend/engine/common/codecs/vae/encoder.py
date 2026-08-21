"""VAE Encoder — SD AutoencoderKL encoder, reference implementation for z_image / flux2 VAE."""
from __future__ import annotations

import importlib
from typing import Any
from backend.engine.runtime._base import RuntimeContext
from .decoder import ResnetBlock, SpatialAttention, _to_nchw, _to_nhwc
from .weight_remap import vae_conv_weight_for_runtime


class Downsample:
    """2× spatial downsample (diffusers ``DownSampler``).

    Reference pads NCHW ``H,W`` by ``(0,1)`` on the tail, then ``Conv2d k=3, stride=2, padding=0``
    in NHWC — **not** ``padding=1`` on the conv (that diverges from reference Z-Image VAE and breaks
    img2img encode vs reference benchmarks).
    """

    def __init__(self, in_ch: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.conv = ctx.Conv2d(in_ch, in_ch, 3, stride=2, padding=0)

    def forward(self, x):
        ctx = self.ctx
        mx = importlib.import_module("mlx.core")

        x = mx.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
        h = _to_nhwc(ctx, x)
        h = self.conv(h)
        return _to_nchw(ctx, h)


class VAEEncoder:
    """SD AutoencoderKL Encoder — matches reference z-image/flux2 VAE encoder."""

    def __init__(self, latent_channels: int = 16, ctx: RuntimeContext = None,
                 scaling_factor: float = 1.0, shift_factor: float = 0.0,
                 z_channels: int = 4):
        nn = ctx; C = latent_channels
        self.ctx = ctx
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.out_channels = C * 2  # mean + logvar

        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)

        # Down blocks: channels double at each level, 2 resnets per block
        self.down1 = _EncoderBlock(128, 128, ctx, stride=2)  # 256→128
        self.down2 = _EncoderBlock(128, 256, ctx, stride=2)  # 128→64
        self.down3 = _EncoderBlock(256, 512, ctx, stride=2)  # 64→32
        self.down4 = _EncoderBlock(512, 512, ctx, stride=1)  # no downsample

        # Mid block
        self.mid_resnet1 = ResnetBlock(512, 512, ctx, cast_after_norm=True, norm_input_fp32=True)
        self.mid_attn = SpatialAttention(512, ctx, cast_after_norm=True, norm_input_fp32=True)
        self.mid_resnet2 = ResnetBlock(512, 512, ctx, cast_after_norm=True, norm_input_fp32=True)

        # Output
        self.norm_out = nn.GroupNorm(32, 512, eps=1e-6, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(512, self.out_channels, 3, padding=1)

    def encode(self, image: Any) -> Any:
        """编码 RGB 图像 [0,255] 或 [0,1] → latent [B, C, T=1, H/8, W/8]。
        
        Returns: [B, C, H/8, W/8] — 去掉 5D 帧维度（由调用方决定是否保留）。
        """
        ctx = self.ctx
        # z-image/flux2 VAE 直接接受 [0,1] 输入（不归一化）
        h = _to_nhwc(ctx, image)
        h = self.conv_in(h)
        h = _to_nchw(ctx, h)
        h = self.down1.forward(h)
        h = self.down2.forward(h)
        h = self.down3.forward(h)
        h = self.down4.forward(h)

        h = self.mid_resnet1.forward(h)
        h = self.mid_attn.forward(h)
        h = self.mid_resnet2.forward(h)

        h = _to_nhwc(ctx, h)
        h = self.norm_out(h); h = ctx.silu(h)
        h = self.conv_out(h)
        h = _to_nchw(ctx, h)

        # Split mean from logvar (first C channels = mean; ignore logvar for inference encode)
        half = self.out_channels // 2
        mean = h[:, :half]
        latent = (mean - self.shift_factor) * self.scaling_factor
        return latent[:, :, None, :, :]  # [B, C, 1, H, W]

    def encode_conv_out_nchw(self, image: Any) -> Any:
        """Encoder 输出 ``conv_out``（mean+logvar 拼接），用于 Flux2 ``quant_conv`` 之后再取 mean。"""
        ctx = self.ctx
        h = _to_nhwc(ctx, image)
        h = self.conv_in(h)
        h = _to_nchw(ctx, h)
        h = self.down1.forward(h)
        h = self.down2.forward(h)
        h = self.down3.forward(h)
        h = self.down4.forward(h)
        h = self.mid_resnet1.forward(h)
        h = self.mid_attn.forward(h)
        h = self.mid_resnet2.forward(h)
        h = _to_nhwc(ctx, h)
        h = h.astype(importlib.import_module("mlx.core").float32)
        h = self.norm_out(h)
        h = ctx.silu(h)
        h = self.conv_out(h)
        return _to_nchw(ctx, h)

    def load_weights(self, weights: list[tuple[str, Any]], strict: bool = False):
        if not hasattr(self, '_param_map'):
            self._build_param_map()
        loaded, skipped = [], []
        down_map = {0: 'down1', 1: 'down2', 2: 'down3', 3: 'down4'}
        for key, tensor in weights:
            k = key
            # Strip encoder. prefix
            if k.startswith('encoder.'):
                k = k[8:]
            # Remap down_blocks → down*
            for bi, bn in down_map.items():
                k = k.replace(f'down_blocks.{bi}', bn)
            # Remap downsamplers → down.conv
            k = k.replace('downsamplers.0.conv', 'down.conv')
            # Remap resnets → resnets (keep as-is since our class uses resnets)
            # Remap mid_block.resnets.0 → mid_resnet1, .1 → mid_resnet2
            k = k.replace('mid_block.resnets.0', 'mid_resnet1')
            k = k.replace('mid_block.resnets.1', 'mid_resnet2')
            k = k.replace('mid_block.attentions.0', 'mid_attn')
            k = k.replace('conv_norm_out', 'norm_out')
            k = k.replace('group_norm', 'norm')
            k = k.replace('.to_out.0.', '.to_out.')
            # Conv2d weight: diffusers (O, I, kH, kW) → runtime NHWC (O, kH, kW, I)
            if ".weight" in k and tensor.ndim == 4:
                tensor = vae_conv_weight_for_runtime(self.ctx, tensor)
            if k in self._param_map:
                p = self._param_map[k]
                if p.shape == tensor.shape:
                    p[:] = tensor.astype(p.dtype)
                    loaded.append(k)
                else:
                    skipped.append(f'{k} shape {p.shape} vs {tensor.shape}')
            else:
                skipped.append(k)
        if strict and skipped:
            raise ValueError(f"Unloaded: {skipped[:10]}")
        return loaded, skipped

    def _build_param_map(self):
        self._param_map = {}
        _collect_vae_params(self, '', self._param_map)

    def parameters(self):
        if not hasattr(self, '_param_map'):
            self._build_param_map()
        return list(self._param_map.items())

    def cast_floating_params(self, dtype: Any) -> None:
        """Cast nested MLX submodules to target dtype (for reference parity-sensitive paths)."""
        from backend.engine.runtime.mlx_dtype import cast_module_parameters

        visited: set[int] = set()

        def _walk(obj: Any) -> None:
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)
            if hasattr(obj, "parameters") and hasattr(obj, "update") and obj is not self:
                mod = getattr(obj.__class__, "__module__", "")
                if mod.startswith("mlx."):
                    cast_module_parameters(obj, dtype, eval_fn=self.ctx.eval)
                    return
            if isinstance(obj, list):
                for it in obj:
                    _walk(it)
                return
            if isinstance(obj, dict):
                for it in obj.values():
                    _walk(it)
                return
            if hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    if isinstance(v, (int, float, str, bool, type, bytes, bytearray)):
                        continue
                    _walk(v)

        _walk(self)
        self._build_param_map()


def _collect_vae_params(obj, prefix, result):
    for name in sorted(dir(obj)):
        if name.startswith('_') or name in ('ctx',):
            continue
        try: attr = getattr(obj, name)
        except: continue
        if attr is None or isinstance(attr, (int, float, str, bool, type)):
            continue
        np = f'{prefix}.{name}' if prefix else name
        if hasattr(attr, 'weight') and hasattr(attr, 'bias'):
            if hasattr(attr, 'weight') and isinstance(attr.weight, object):
                result[f'{np}.weight'] = attr.weight
                if attr.bias is not None:
                    result[f'{np}.bias'] = attr.bias
        elif hasattr(attr, '__dict__') and not isinstance(attr, type):
            _collect_vae_params(attr, np, result)
        elif isinstance(attr, list):
            for i, item in enumerate(attr):
                _collect_vae_params(item, f'{np}.{i}', result)


class _EncoderBlock:
    """Encoder block: 2 Resnets + optional Downsample。"""
    def __init__(self, in_ch: int, out_ch: int, ctx: RuntimeContext, stride: int = 2):
        self.resnets = [
            ResnetBlock(
                in_ch,
                out_ch,
                ctx,
                use_shortcut=(in_ch != out_ch),
                cast_after_norm=True,
                norm_input_fp32=True,
            ),
            ResnetBlock(out_ch, out_ch, ctx, cast_after_norm=True, norm_input_fp32=True),
        ]
        self.down = Downsample(out_ch, ctx) if stride > 1 else None

    def forward(self, x):
        for r in self.resnets:
            x = r.forward(x)
        if self.down is not None:
            x = self.down.forward(x)
        return x
