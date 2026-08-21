"""Wan video Transformer — public entry (MLX via ``DelegatingDiTStem``)."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem
from backend.engine.config.model_configs import WanConfig
from backend.engine.runtime._base import RuntimeContext

from .conditioning import (
    build_wan_i2v_side_channels,
    expand_wan_cond_latent,
    masks_like,
    prepare_ti2v_i2v_latents,
    wan_i2v_uses_channel_concat,
    wan_seq_len,
)
from .transformer_mlx import WanModelMLX


class WanTransformer(DelegatingDiTStem):
    """Wan TI2V / T2V / I2V DiT — hooks on stem, math on ``WanModelMLX``."""

    def __init__(self, config: WanConfig, ctx: RuntimeContext, num_frames: int = 81):
        super().__init__(
            config,
            ctx,
            mlx_cls=WanModelMLX,
            num_frames=num_frames,
        )
        self._seq_len: int | None = None

    def forward(self, latents: Any, timestep: Any, txt_embeds: Any | None = None, **kwargs: Any) -> Any:
        if kwargs.get("seq_len") is None and self._seq_len is None:
            _, _, t, h, w = latents.shape
            self._seq_len = wan_seq_len(int(t), int(h), int(w), self.config.patch_size)
        kwargs.setdefault("seq_len", self._seq_len)
        return self._inner.forward(latents, timestep, txt_embeds=txt_embeds, **kwargs)

    def prepare_conditioning(self, request: Any, bundle_root: str | None = None) -> dict[str, Any]:
        cond: dict[str, Any] = {}
        if getattr(request, "source_asset_id", None):
            cond["wan_i2v"] = True
            cond["wan_bundle_root"] = bundle_root
            cond["wan_size"] = getattr(request, "size", "480x720")
        return cond

    def before_denoise(self, latents: Any, timesteps: Any, sigmas: Any, **cond: Any) -> tuple[Any, dict[str, Any]]:
        from backend.engine.inference.optimization_plan import plan_from_extra_cond, pop_inference_plan

        cond = dict(cond)
        n_steps = len(timesteps) if timesteps is not None else 0
        plan = pop_inference_plan(cond)
        if plan is None:
            plan = plan_from_extra_cond(
                cond,
                family="wan",
                config=self.config,
                entry=None,
                ctx=self.ctx,
                num_steps=n_steps,
            )
        if hasattr(self._inner, "apply_runtime_plan"):
            self._inner.apply_runtime_plan(plan)
        ctx = self.ctx
        self._inner.invalidate_text_cache()
        _, _, t, h, w = latents.shape
        self._seq_len = wan_seq_len(int(t), int(h), int(w), self.config.patch_size)
        cond["wan_seq_len"] = self._seq_len

        if cond.get("wan_i2v") and cond.get("wan_cond_latent") is not None:
            z = cond["wan_cond_latent"]
            _, _, t, h, w = latents.shape
            z = expand_wan_cond_latent(ctx, z, int(t))
            mask2 = cond.get("wan_i2v_mask")
            if mask2 is None:
                _, mask2_list = masks_like(ctx, [ctx.squeeze(latents, 0)], zero=True)
                mask2 = mask2_list[0]
            if wan_i2v_uses_channel_concat(self.config):
                temporal_scale = int(getattr(self.config, "temporal_vae_scale", 4))
                side = build_wan_i2v_side_channels(
                    ctx, z, int(t), int(h), int(w), temporal_vae_scale=temporal_scale,
                )
                self._inner.set_i2v_state(None, mask2, side=side)
            else:
                self._inner.set_i2v_state(z, mask2)
                latents = prepare_ti2v_i2v_latents(ctx, latents, z, mask2)

        if getattr(self.config, "expand_timesteps", False) and cond.get("wan_i2v"):
            cond["wan_expand_timesteps"] = True
            if cond.get("wan_i2v_mask") is None:
                _, mask2_list = masks_like(ctx, [ctx.squeeze(latents, 0)], zero=True)
                cond["wan_i2v_mask"] = mask2_list[0]
        return latents, cond

    def patch_latent_volume(self, latent: Any, source_id: float = 0.0, **kwargs: Any) -> Any:
        return self._inner.patch_latent_volume(latent, source_id, **kwargs)

    def forward_token_sequence(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.forward_token_sequence(*args, **kwargs)

    def unpatchify_token_grid(self, token_out: Any, grid: tuple[int, int, int]) -> Any:
        return self._inner.unpatchify_token_grid(token_out, grid)

    def step_callback(self, step_idx: int, latents: Any, noise_pred: Any) -> None:
        del latents, noise_pred
        inner = self._inner
        step_cache = getattr(inner, "_step_cache", None)
        if step_cache is not None:
            step_cache.set_step_counter(int(step_idx) + 1)

    def build_timestep_per_token(self, scalar_t: Any, seq_len: int, mask2: Any | None = None) -> Any:
        """Per-token timesteps for ``expand_timesteps`` (first-frame tokens → 0 when masked)."""
        ctx = self.ctx
        cfg = self.config
        ph, pw = int(cfg.patch_size[1]), int(cfg.patch_size[2])

        if hasattr(scalar_t, "item"):
            t_scalar = float(scalar_t.item())
        else:
            try:
                t_scalar = float(scalar_t.reshape(-1)[0])
            except Exception:
                t_scalar = float(scalar_t)
        t_arr = ctx.array(t_scalar, dtype=ctx.float32())

        if mask2 is None or not getattr(cfg, "expand_timesteps", False):
            return ctx.broadcast_to(ctx.reshape(t_arr, (1, 1)), (1, seq_len))

        m = mask2
        if m.ndim == 5:
            m = ctx.squeeze(m, 0)
        if m.ndim == 4:
            m0 = m[0]
        else:
            m0 = m

        m_sub = m0[:, ::ph, ::pw] if ph > 1 or pw > 1 else m0
        temp_ts = (m_sub * t_arr).reshape(-1)
        n_tok = int(temp_ts.shape[0])
        flat = temp_ts.reshape(-1)
        if n_tok < seq_len:
            pad = ctx.ones((seq_len - n_tok,), dtype=ctx.float32()) * t_arr
            flat = ctx.concat([flat, pad], axis=0)
        elif n_tok > seq_len:
            flat = flat[:seq_len]
        return ctx.reshape(flat, (1, seq_len))
