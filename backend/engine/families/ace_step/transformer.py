"""
ACE-Step Transformer — public entry; MLX implementation in ``transformer_mlx``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.model.dit_stem import require_mlx_ctx


class AceStepTransformer(TransformerBase):
    """ACE-Step DiT decoder — MLX via RuntimeContext."""

    def __init__(self, ctx: Any, **config: Any):
        super().__init__()
        self._ctx = ctx
        require_mlx_ctx(ctx, feature="ACE-Step DiT")
        from .transformer_mlx import AceStepDiTMLX

        self._model = AceStepDiTMLX(**config)
        self._backend = "mlx"
        self._build_param_map()

    # ------------------------------------------------------------------
    # TransformerBase interface
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Any,
        timestep: Any,
        timestep_r: Optional[Any] = None,
        encoder_hidden_states: Optional[Any] = None,
        context_latents: Optional[Any] = None,
        cache: Any = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        if timestep_r is None:
            timestep_r = timestep
        return self._model(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            encoder_hidden_states=encoder_hidden_states,
            context_latents=context_latents,
            cache=cache,
            use_cache=use_cache,
        )

    def parameters(self):
        return self._model.parameters()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Strip ``decoder.`` prefix from checkpoint keys."""
        remapped: dict[str, Any] = {}
        for key, tensor in weights.items():
            nk = key[len("decoder."):] if key.startswith("decoder.") else key
            remapped[nk] = tensor
        return remapped

    def _build_param_map(self):
        from .transformer_mlx import collect_ace_step_dit_param_map

        self._param_map = collect_ace_step_dit_param_map(self._model)

    def load_weights(
        self,
        weights,
        strict: bool = False,
        ctx: Any = None,
        *,
        bundle_affine_bits: int | None = None,
        inference_mode=None,
    ):
        load_ctx = ctx if ctx is not None else self._ctx
        if (
            inference_mode is not None
            and getattr(inference_mode, "kind", "dense") == "quantized"
            and getattr(inference_mode, "bits", None) in (4, 8)
        ):
            from backend.engine.common.model.quantized_load_mlx import load_weights_quantized_inference

            return load_weights_quantized_inference(
                self,
                weights,
                strict=strict,
                ctx=load_ctx,
                bundle_affine_bits=bundle_affine_bits,
                bits=int(inference_mode.bits),
                group_size=int(getattr(inference_mode, "group_size", 64) or 64),
                module_root=self._model,
            )
        return super().load_weights(
            weights,
            strict=strict,
            ctx=load_ctx,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=inference_mode,
        )
