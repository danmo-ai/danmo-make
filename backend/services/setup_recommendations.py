"""Hardware-aware model recommendations for Quick Setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from backend.core.dependency_specs import dependency_model_ids
from backend.core.model_registry import ModelEntry, ModelRegistry
from backend.engine.platform.info import PlatformInfo
from backend.utils.size_parse import parse_human_size_to_gb

SetupSlot = Literal["image", "video", "audio", "llm", "vlm"]
SetupSlotStatus = Literal["recommended", "unavailable", "warning"]

_MEMORY_TIER_LOW = 16.0
_MEMORY_TIER_MID = 32.0

_VERSION_PREF_MLX = (
    "mlx-q4",
    "int4",
    "mlx-q8",
    "int8",
    "fp16",
    "bf16",
    "xl-sft",
)
_VERSION_PREF_CUDA = ("int4", "int8", "fp16", "bf16", "fp8", "xl-sft")

_PREFERRED_MODEL_ORDER: dict[str, dict[str, tuple[str, ...]]] = {
    "image": {
        "low": ("flux2-klein-4b", "z-image-turbo"),
        "mid": ("z-image-turbo", "flux2-klein-4b"),
        "high": ("z-image-turbo", "flux2-klein-4b"),
    },
    "video": {
        "low": ("wan-2.2-ti2v-5b",),
        "mid": ("wan-2.2-ti2v-5b",),
        "high": ("wan-2.2-ti2v-5b",),
    },
    "audio": {
        "low": ("ace-step-xl-sft",),
        "mid": ("ace-step-xl-sft",),
        "high": ("ace-step-xl-sft",),
    },
    "llm": {
        "low": ("qwen3.5-4b",),
        "mid": ("qwen3.5-4b",),
        "high": ("qwen3.5-4b",),
    },
    "vlm": {
        "low": ("qwen3-vl-4b-instruct",),
        "mid": ("qwen3-vl-4b-instruct",),
        "high": ("qwen3-vl-4b-instruct",),
    },
}

_SLOT_SPECS: dict[SetupSlot, dict[str, str]] = {
    "image": {"media": "image", "category": "base_models", "action": "create"},
    "video": {"media": "video", "category": "video_models", "action": "create"},
    "audio": {"media": "audio", "category": "music_models", "action": "create"},
    "llm": {"media": "llm", "category": "llm_models", "action": "chat"},
    "vlm": {"media": "llm", "category": "vlm_models", "action": "describe"},
}


@dataclass
class SetupSlotRecommendation:
    slot: SetupSlot
    status: SetupSlotStatus
    model_id: Optional[str] = None
    version_key: Optional[str] = None
    estimated_gb: Optional[float] = None
    warning: Optional[str] = None
    reason: Optional[str] = None
    installed: bool = False
    name: dict[str, str] = field(default_factory=dict)
    version_name: dict[str, str] = field(default_factory=dict)
    size_human: str = ""


@dataclass
class SetupRecommendations:
    reference_memory_gb: float
    memory_tier: str
    available_backends: list[str]
    primary_backend: str
    slots: list[SetupSlotRecommendation]


def _memory_tier(ref_gb: float) -> str:
    if ref_gb <= _MEMORY_TIER_LOW:
        return "low"
    if ref_gb <= _MEMORY_TIER_MID:
        return "mid"
    return "high"


def _resolve_reference_memory_gb(
    *,
    memory_gb: float,
    mlx_memory_limit: int,
    available_backends: list[str],
) -> float:
    cap = float(mlx_memory_limit) if mlx_memory_limit > 0 else memory_gb
    if "cuda" in available_backends and "mlx" not in available_backends:
        from backend.engine.families.ace_step.quality.resource_policy_cuda import (
            detect_cuda_memory_gb,
        )

        cuda_gb = detect_cuda_memory_gb()
        if cuda_gb is not None and cuda_gb > 0:
            return cuda_gb
    if memory_gb > 0 and cap > 0:
        return min(memory_gb, cap)
    if memory_gb > 0:
        return memory_gb
    if cap > 0:
        return cap
    return 16.0


def _model_backends(raw: dict[str, Any]) -> tuple[str, ...]:
    backends = raw.get("backends")
    if isinstance(backends, list) and backends:
        return tuple(str(b) for b in backends)
    return ("mlx",)


def _is_backend_compatible(raw: dict[str, Any], available_backends: list[str]) -> bool:
    model_backends = _model_backends(raw)
    return bool(set(model_backends) & set(available_backends))


def _slot_candidates(
    registry: ModelRegistry,
    slot: SetupSlot,
    available_backends: list[str],
) -> list[ModelEntry]:
    spec = _SLOT_SPECS[slot]
    out: list[ModelEntry] = []
    for entry in registry.all().values():
        raw = entry.raw if isinstance(entry.raw, dict) else {}
        if raw.get("media") != spec["media"]:
            continue
        if raw.get("category") != spec["category"]:
            continue
        if raw.get("recommended") is not True:
            continue
        if raw.get("commercial_use_allowed") is not True:
            continue
        actions = raw.get("actions") if isinstance(raw.get("actions"), dict) else {}
        if spec["action"] not in actions:
            continue
        if not _is_backend_compatible(raw, available_backends):
            continue
        out.append(entry)
    return out


def _version_sort_key(version_key: str, pref: tuple[str, ...]) -> tuple[int, str]:
    try:
        return (pref.index(version_key), version_key)
    except ValueError:
        return (len(pref), version_key)


def _version_estimated_gb(version_cfg: dict[str, Any]) -> float:
    size_gb = parse_human_size_to_gb(version_cfg.get("size")) or 0.0
    min_gb = version_cfg.get("min_unified_memory_gb")
    if isinstance(min_gb, (int, float)) and min_gb > 0:
        return max(size_gb, float(min_gb))
    return size_gb


def _pick_version(
    versions: dict[str, Any],
    *,
    ref_gb: float,
    primary_backend: str,
) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[str]]:
    if not versions:
        return None, None, None

    pref = _VERSION_PREF_MLX if primary_backend == "mlx" else _VERSION_PREF_CUDA
    budget = ref_gb * 0.70
    ranked: list[tuple[str, dict[str, Any], float]] = []
    for key, cfg in versions.items():
        if not isinstance(cfg, dict):
            continue
        est = _version_estimated_gb(cfg)
        ranked.append((key, cfg, est))

    if not ranked:
        return None, None, None

    ranked.sort(key=lambda row: (_version_sort_key(row[0], pref), -row[2]))

    fitting = [row for row in ranked if row[2] <= budget or row[2] <= 0]
    if fitting:
        best = max(fitting, key=lambda row: row[2])
        return best[0], best[1], None

    smallest = min(ranked, key=lambda row: row[2] if row[2] > 0 else float("inf"))
    warning = "insufficient_memory"
    return smallest[0], smallest[1], warning


def _pick_model_for_slot(
    registry: ModelRegistry,
    slot: SetupSlot,
    *,
    ref_gb: float,
    primary_backend: str,
    available_backends: list[str],
) -> SetupSlotRecommendation:
    tier = _memory_tier(ref_gb)
    preferred_ids = _PREFERRED_MODEL_ORDER[slot][tier]
    candidates = _slot_candidates(registry, slot, available_backends)
    candidate_by_id = {entry.id: entry for entry in candidates}

    ordered: list[ModelEntry] = []
    for mid in preferred_ids:
        entry = candidate_by_id.get(mid)
        if entry is not None:
            ordered.append(entry)
    for entry in candidates:
        if entry.id not in preferred_ids:
            ordered.append(entry)

    if not ordered:
        if slot in ("llm", "vlm") and "mlx" not in available_backends:
            return SetupSlotRecommendation(
                slot=slot,
                status="unavailable",
                reason="mlx_required",
            )
        return SetupSlotRecommendation(
            slot=slot,
            status="unavailable",
            reason="no_compatible_models",
        )

    entry = ordered[0]
    raw = entry.raw if isinstance(entry.raw, dict) else {}
    versions = raw.get("versions") if isinstance(raw.get("versions"), dict) else {}
    version_key, version_cfg, warning = _pick_version(
        versions,
        ref_gb=ref_gb,
        primary_backend=primary_backend,
    )
    if version_key is None or version_cfg is None:
        return SetupSlotRecommendation(
            slot=slot,
            status="unavailable",
            reason="no_versions",
        )

    est = _version_estimated_gb(version_cfg)
    status: SetupSlotStatus = "warning" if warning else "recommended"
    name = raw.get("name") if isinstance(raw.get("name"), dict) else {}
    version_name = version_cfg.get("name") if isinstance(version_cfg.get("name"), dict) else {}
    return SetupSlotRecommendation(
        slot=slot,
        status=status,
        model_id=entry.id,
        version_key=version_key,
        estimated_gb=round(est, 2) if est > 0 else None,
        warning=warning,
        name=dict(name),
        version_name=dict(version_name),
        size_human=str(version_cfg.get("size") or ""),
    )


def build_setup_recommendations(
    registry: ModelRegistry,
    *,
    memory_gb: float,
    mlx_memory_limit: int,
    detailed_status: Optional[dict[str, dict[str, Any]]] = None,
) -> SetupRecommendations:
    available_backends = PlatformInfo.detect()
    primary_backend = available_backends[0] if available_backends else ""
    ref_gb = _resolve_reference_memory_gb(
        memory_gb=memory_gb,
        mlx_memory_limit=mlx_memory_limit,
        available_backends=available_backends,
    )
    tier = _memory_tier(ref_gb)

    slots: list[SetupSlotRecommendation] = []
    for slot_key in ("image", "video", "audio", "llm", "vlm"):
        rec = _pick_model_for_slot(
            registry,
            slot_key,  # type: ignore[arg-type]
            ref_gb=ref_gb,
            primary_backend=primary_backend,
            available_backends=available_backends,
        )
        if rec.model_id and detailed_status is not None:
            rec.installed = bool(detailed_status.get(rec.model_id, {}).get("ready"))
        slots.append(rec)

    return SetupRecommendations(
        reference_memory_gb=round(ref_gb, 2),
        memory_tier=tier,
        available_backends=available_backends,
        primary_backend=primary_backend,
        slots=slots,
    )


def topological_install_order(
    items: list[dict[str, str]],
    registry: ModelRegistry,
) -> list[dict[str, str]]:
    """Order install items so registry dependencies come first."""
    by_id: dict[str, dict[str, str]] = {}
    for item in items:
        model_id = item.get("model_id") or item.get("model_name")
        if not model_id:
            continue
        by_id[model_id] = {
            "model_id": model_id,
            "version_key": item.get("version_key") or item.get("version") or "",
        }

    deps_map: dict[str, list[str]] = {}
    for model_id in by_id:
        entry = registry.get(model_id)
        if entry is None:
            deps_map[model_id] = []
            continue
        raw = entry.raw if isinstance(entry.raw, dict) else {}
        deps = dependency_model_ids(raw.get("dependencies"))
        base = raw.get("base_model")
        if isinstance(base, str) and base.strip():
            deps = list(dict.fromkeys([*deps, base.strip()]))
        deps_map[model_id] = deps

    ordered_ids: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            return
        visiting.add(node)
        for dep in deps_map.get(node, []):
            if dep in by_id:
                visit(dep)
        visiting.remove(node)
        visited.add(node)
        ordered_ids.append(node)

    for model_id in by_id:
        visit(model_id)

    extra = [mid for mid in by_id if mid not in ordered_ids]
    ordered_ids.extend(extra)

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for model_id in ordered_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        result.append(by_id[model_id])
    return result
