from __future__ import annotations

"""SeedVR2 3D VAE — 单文件，避免 encoder/decoder 多级碎片目录。"""

from typing import Any

import mlx.core as mx
from mlx import nn

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.ops.vae_stream_cache_mlx import VAEStreamCacheSession
from .weights_mlx import ModelConfig


# ----- common/conv3d.py -----


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 1,
        causal_temporal: bool = True,
        use_padding_causal: bool = False,
    ):
        super().__init__()
        self.causal_temporal = causal_temporal
        self.use_padding_causal = use_padding_causal

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kt, kh, kw = kernel_size
        self.weight = mx.zeros((out_channels, kt, kh, kw, in_channels))
        self.bias = mx.zeros((out_channels,))
        self._runtime_layer_id: str | None = None
        self._runtime_stream: VAEStreamCacheSession | None = None
        self._runtime_conv3d_backend: str = "native"

    def bind_runtime(
        self,
        *,
        layer_id: str,
        stream: VAEStreamCacheSession | None,
        conv3d_backend: str,
    ) -> None:
        self._runtime_layer_id = str(layer_id)
        self._runtime_stream = stream
        self._runtime_conv3d_backend = str(conv3d_backend or "native")

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T, H, W = x.shape
        kt, kh, kw = self.kernel_size
        pt, ph, pw = self.padding
        st, sh, sw = self.stride
        stream = self._runtime_stream
        layer_id = self._runtime_layer_id or "cc3d"
        x_orig = x

        if self.causal_temporal and kt > 1:
            if stream is not None and stream.enabled:
                x = stream.prepend_causal_tail(x, layer_id, kernel_temporal=kt)
                temporal_padding = 0
            else:
                causal_pad = (2 * self.padding[0]) if self.use_padding_causal else kt - 1
                if causal_pad > 0:
                    first_frame = x[:, :, :1, :, :]
                    pad_frames = mx.repeat(first_frame, causal_pad, axis=2)
                    x = mx.concatenate([pad_frames, x], axis=2)
                temporal_padding = 0
        else:
            temporal_padding = pt

        from backend.engine.common.ops.mfa_seedvr2_mlx import causal_conv3d_forward

        out = causal_conv3d_forward(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=(st, sh, sw),
            padding=(temporal_padding, ph, pw),
            backend=self._runtime_conv3d_backend,
        )

        if stream is not None and stream.enabled and self.causal_temporal and kt > 1:
            stream.update_causal_tail(x_orig, layer_id, kernel_temporal=kt)
        return out


# ----- common/attention_3d.py -----

class Attention3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=32, dims=channels, eps=1e-6, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T, H, W = x.shape
        residual = x
        x = x.transpose(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H * W)
        x = x.transpose(0, 2, 1)

        x = self.group_norm(x.astype(mx.float32)).astype(ModelConfig.precision)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = q[:, None, :, :]
        k = k[:, None, :, :]
        v = v[:, None, :, :]

        x = scaled_dot_product_attention_bhsd_mx(
            mx,
            q,
            k,
            v,
            scale=C**-0.5,
        )

        x = x.squeeze(1)
        x = self.to_out[0](x)
        x = x.transpose(0, 2, 1)
        x = x.reshape(B, T, C, H, W)
        x = x.transpose(0, 2, 1, 3, 4)
        return x + residual


# ----- encoder/resnet_block_3d.py -----

class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, dims=in_channels, eps=1e-6, pytorch_compatible=True)
        self.norm2 = nn.GroupNorm(num_groups=32, dims=out_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = x.transpose(0, 2, 3, 4, 1)
        x = self.norm1(x.astype(mx.float32)).astype(ModelConfig.precision)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv1(x)

        x = x.transpose(0, 2, 3, 4, 1)
        x = self.norm2(x.astype(mx.float32)).astype(ModelConfig.precision)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        output = x + residual
        return output


# ----- encoder/downsample_3d.py -----

class Downsample3D(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_only: bool = False,
    ):
        super().__init__()
        kt, st, pt = (1, 1, 0) if spatial_only else (3, 2, 1)
        self.conv = CausalConv3d(
            channels,
            channels,
            kernel_size=(kt, 3, 3),
            stride=(st, 2, 2),
            padding=(pt, 0, 0),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1), (0, 1)])
        return self.conv(x)


# ----- encoder/down_block_3d.py -----

class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
        temporal_down: bool = False,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock3D(in_channels=in_ch, out_channels=out_channels))

        self.downsamplers = []
        if add_downsample:
            self.downsamplers.append(Downsample3D(channels=out_channels, spatial_only=not temporal_down))

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x


# ----- encoder/mid_block_3d.py -----

class MidBlock3D(nn.Module):
    def __init__(
        self,
        channels: int = 512,
    ):
        super().__init__()
        self.attentions = [Attention3D(channels=channels)]
        self.resnets = [
            ResnetBlock3D(in_channels=channels, out_channels=channels),
            ResnetBlock3D(in_channels=channels, out_channels=channels),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


# ----- encoder/encoder_3d.py -----

class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        block_out_channels: tuple = (128, 256, 512, 512),
        layers_per_block: int = 2,
        temporal_down_blocks: int = 2,
    ):
        super().__init__()
        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = []
        output_channel = block_out_channels[0]
        num_blocks = len(block_out_channels)

        for i, channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = channel
            is_final_block = i == num_blocks - 1
            temporal_down = (i >= num_blocks - temporal_down_blocks - 1) and not is_final_block

            self.down_blocks.append(
                DownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    add_downsample=not is_final_block,
                    temporal_down=temporal_down,
                )
            )

        self.mid_block = MidBlock3D(channels=block_out_channels[-1])

        self.conv_norm_out = nn.GroupNorm(
            num_groups=32,
            dims=block_out_channels[-1],
            eps=1e-6,
            pytorch_compatible=True,
        )
        self.conv_out = CausalConv3d(
            in_channels=block_out_channels[-1],
            out_channels=2 * out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.mid_block(x)
        x = x.transpose(0, 2, 3, 4, 1)
        x = self.conv_norm_out(x.astype(mx.float32)).astype(ModelConfig.precision)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x


# ----- decoder/decoder_resnet_block_3d.py -----

class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, dims=in_channels, eps=1e-6, pytorch_compatible=True)
        self.norm2 = nn.GroupNorm(num_groups=32, dims=out_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        x = x.transpose(0, 2, 3, 4, 1)
        x = self.norm1(x.astype(mx.float32)).astype(ModelConfig.precision)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv1(x)

        x = x.transpose(0, 2, 3, 4, 1)
        x = self.norm2(x.astype(mx.float32)).astype(ModelConfig.precision)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return x + residual


# ----- decoder/upsample_3d.py -----

class Upsample3D(nn.Module):
    def __init__(
        self,
        channels: int,
        temporal_up: bool = False,
    ):
        super().__init__()
        spatial_factor = 2
        temporal_factor = 2 if temporal_up else 1
        total_factor = (spatial_factor**2) * temporal_factor

        self.conv = CausalConv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_padding_causal=True,
        )
        self.upscale_conv = CausalConv3d(
            channels,
            channels * total_factor,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor

    def __call__(self, x: mx.array) -> mx.array:
        B, C, T, H, W = x.shape
        x = self.upscale_conv(x)
        sf = self.spatial_factor
        tf = self.temporal_factor
        x = x.reshape(B, sf, sf, tf, C, T, H, W)
        x = x.transpose(0, 4, 5, 3, 6, 1, 7, 2)
        x = x.reshape(B, C, T * tf, H * sf, W * sf)
        if T == 1 and tf > 1:
            x = x[:, :, :1, :, :]
        x = self.conv(x)
        return x


# ----- decoder/decoder_mid_block_3d.py -----

class MidBlock3D(nn.Module):
    def __init__(
        self,
        channels: int = 512,
    ):
        super().__init__()
        self.attentions = [Attention3D(channels=channels)]
        self.resnets = [
            ResnetBlock3D(
                in_channels=channels,
                out_channels=channels,
            ),
            ResnetBlock3D(
                in_channels=channels,
                out_channels=channels,
            ),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


# ----- decoder/up_block_3d.py -----

class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        add_upsample: bool = True,
        temporal_up: bool = False,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                )
            )

        self.upsamplers = []
        if add_upsample:
            self.upsamplers.append(Upsample3D(channels=out_channels, temporal_up=temporal_up))

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x


# ----- decoder/decoder_3d.py -----

class Decoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: tuple = (128, 256, 512, 512),
        layers_per_block: int = 3,
        temporal_up_blocks: int = 2,
    ):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))

        self.conv_in = CausalConv3d(
            in_channels=in_channels,
            out_channels=reversed_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = MidBlock3D(
            channels=reversed_channels[0],
        )

        self.up_blocks = []
        output_channel = reversed_channels[0]
        num_blocks = len(reversed_channels)

        for i, channel in enumerate(reversed_channels):
            input_channel = output_channel
            output_channel = channel
            is_final_block = i == num_blocks - 1
            temporal_up = i < temporal_up_blocks

            self.up_blocks.append(
                UpBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    add_upsample=not is_final_block,
                    temporal_up=temporal_up,
                )
            )

        self.conv_norm_out = nn.GroupNorm(
            num_groups=32,
            dims=reversed_channels[-1],
            eps=1e-6,
            pytorch_compatible=True,
        )
        self.conv_out = CausalConv3d(
            in_channels=reversed_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = x.transpose(0, 2, 3, 4, 1)
        x = self.conv_norm_out(x.astype(mx.float32)).astype(ModelConfig.precision)
        x = x.transpose(0, 4, 1, 2, 3)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x


# ----- vae.py -----

class SeedVR2VAE(nn.Module):
    scaling_factor: float = 0.9152
    spatial_scale = 8

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: tuple = (128, 256, 512, 512),
    ):
        super().__init__()
        self.latent_channels = latent_channels

        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=2,
            temporal_down_blocks=2,
        )

        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=3,
            temporal_up_blocks=2,
        )
        self._vae_stream: VAEStreamCacheSession | None = None

    def bind_vae_runtime(
        self,
        *,
        stream: VAEStreamCacheSession | None,
        conv3d_backend: str = "native",
    ) -> None:
        """Attach per-decode stream state + conv3d backend to all ``CausalConv3d`` layers."""
        self._vae_stream = stream
        counter = 0

        def _visit(obj: Any) -> None:
            nonlocal counter
            if isinstance(obj, CausalConv3d):
                obj.bind_runtime(
                    layer_id=f"cc3d_{counter}",
                    stream=stream,
                    conv3d_backend=conv3d_backend,
                )
                counter += 1
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _visit(item)
                return
            if isinstance(obj, nn.Module):
                for val in vars(obj).values():
                    if isinstance(val, (CausalConv3d, nn.Module, list, tuple)):
                        _visit(val)

        _visit(self.encoder)
        _visit(self.decoder)

    def reset_vae_stream(self) -> None:
        if self._vae_stream is not None:
            self._vae_stream.reset()

    def encode(self, x: mx.array) -> mx.array:
        x = x[:, :, None, :, :] if x.ndim == 4 else x
        h = self.encoder(x)
        mean, _ = mx.split(h, 2, axis=1)
        latent = mean
        latent_scaled = latent * self.scaling_factor
        return latent_scaled

    def decode(self, z: mx.array) -> mx.array:
        z = z[:, :, None, :, :] if z.ndim == 4 else z
        z = z / self.scaling_factor
        decoded = self.decoder(z)
        return decoded
