"""Family dispatch for ``ImageGenerationRequest.structural_guide`` hooks."""
from __future__ import annotations

from typing import Any

from backend.engine._transformer_registry import (
    get_structural_guide_attach,
    get_structural_guide_augment,
)


def augment_request_for_structural_guide(
    request: Any,
    ctx: Any,
    *,
    family: str | None = None,
) -> Any:
    guide = getattr(request, "structural_guide", None)
    if guide is None:
        return request
    if not family or not str(family).strip():
        model_id = (getattr(request, "model", None) or "").split(":", 1)[0].strip()
        raise RuntimeError(
            f"structural_guide augment requires family= (model={model_id!r}); "
            "pass require_entry_family(entry) from the resolved registry entry"
        )
    fn = get_structural_guide_augment(family)
    if fn is None:
        raise RuntimeError(
            f"structural_guide is not supported for family={family!r}; "
            "register an augment in _STRUCTURAL_GUIDE_AUGMENT "
            "(ModelConfig.supports_structural_guide=false or unknown family)"
        )
    return fn(request, ctx)


def attach_structural_conditioning(
    pipeline: Any,
    *,
    request: Any,
    family: str,
    model: Any,
    entry: Any,
    version_key: str | None,
    extra_cond: dict[str, Any],
    width: int,
    height: int,
    ctx_exec: Any,
    on_log: Any | None,
) -> tuple[dict[str, Any], Any | None]:
    attach = get_structural_guide_attach(family)
    if attach is None:
        raise RuntimeError(
            f"structural_guide is not supported for family={family!r} "
            f"(ModelConfig.supports_structural_guide=false or unknown family)"
        )
    return attach(
        pipeline,
        request=request,
        family=family,
        model=model,
        entry=entry,
        version_key=version_key,
        extra_cond=extra_cond,
        width=width,
        height=height,
        ctx_exec=ctx_exec,
        on_log=on_log,
    )
