"""ACE-Step memory tier policy — duration/LM limits (inspired by upstream gpu_config)."""
from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

LM_DIR_CANDIDATES = (
    "acestep-5Hz-lm-4B",
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-0.6B",
)

LM_SIZE_BY_DIR = {
    "acestep-5Hz-lm-0.6B": "0.6B",
    "acestep-5Hz-lm-1.7B": "1.7B",
    "acestep-5Hz-lm-4B": "4B",
}


@dataclass(frozen=True)
class AceStepResourcePolicy:
    """Runtime limits derived from available memory."""

    memory_gb: float
    tier: str
    max_duration_with_lm: int
    max_duration_without_lm: int
    available_lm_models: Tuple[str, ...]
    lm_quantize_bits: Optional[int] = None

    def max_duration(self, *, lm_enabled: bool) -> int:
        return self.max_duration_with_lm if lm_enabled else self.max_duration_without_lm


_TIER_TABLE: dict[str, dict] = {
    "minimal": {
        "max_gb": 8,
        "max_duration_with_lm": 60,
        "max_duration_without_lm": 120,
        "lm_models": (),
        "quantize": None,
    },
    "low": {
        "max_gb": 16,
        "max_duration_with_lm": 120,
        "max_duration_without_lm": 180,
        "lm_models": ("acestep-5Hz-lm-0.6B",),
        "quantize": 8,
    },
    "medium": {
        "max_gb": 24,
        "max_duration_with_lm": 240,
        "max_duration_without_lm": 360,
        "lm_models": ("acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B"),
        "quantize": None,
    },
    "high": {
        "max_gb": 48,
        "max_duration_with_lm": 480,
        "max_duration_without_lm": 600,
        "lm_models": LM_DIR_CANDIDATES,
        "quantize": None,
    },
    "unlimited": {
        "max_gb": float("inf"),
        "max_duration_with_lm": 600,
        "max_duration_without_lm": 600,
        "lm_models": LM_DIR_CANDIDATES,
        "quantize": None,
    },
}


def _tier_for_memory_gb(memory_gb: float) -> str:
    if memory_gb <= 0:
        return "minimal"
    for name in ("minimal", "low", "medium", "high"):
        if memory_gb < _TIER_TABLE[name]["max_gb"]:
            return name
    return "unlimited"


def detect_memory_gb() -> float:
    """Best-effort unified / GPU memory estimate."""
    debug = os.environ.get("DANQING_ACESTEP_DEBUG_MEMORY_GB")
    if debug:
        try:
            return float(debug)
        except ValueError:
            pass

    if platform.system() == "Darwin":
        try:
            import subprocess

            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            if out.returncode == 0:
                total = int(out.stdout.strip())
                return total / (1024**3) * 0.75
        except Exception:
            pass
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size) / (1024**3) * 0.75
        except (ValueError, OSError, AttributeError):
            pass
    return 16.0


def resolve_resource_policy() -> AceStepResourcePolicy:
    memory_gb = detect_memory_gb()
    tier = _tier_for_memory_gb(memory_gb)
    row = _TIER_TABLE[tier]
    return AceStepResourcePolicy(
        memory_gb=memory_gb,
        tier=tier,
        max_duration_with_lm=int(row["max_duration_with_lm"]),
        max_duration_without_lm=int(row["max_duration_without_lm"]),
        available_lm_models=tuple(row["lm_models"]),
        lm_quantize_bits=row["quantize"],
    )


def clamp_duration(
    duration: float,
    *,
    lm_enabled: bool,
    policy: AceStepResourcePolicy,
    registry_max: int = 600,
) -> Tuple[float, Optional[str]]:
    """Clamp duration to tier + registry caps; return (duration, warning_or_none)."""
    cap = min(registry_max, policy.max_duration(lm_enabled=lm_enabled))
    dur = float(duration)
    if dur <= cap:
        return dur, None
    return (
        float(cap),
        (
            f"请求时长 {dur:.0f}s 超过当前机器档位 ({policy.tier}, "
            f"约 {policy.memory_gb:.0f}GB 可用) 的上限 {cap}s；"
            f"已截断至 {cap}s。"
        ),
    )


def resolve_lm_dir_for_policy(
    bundle_root: Path,
    policy: AceStepResourcePolicy,
) -> Optional[Path]:
    """Pick largest installed LM allowed by memory tier."""
    root = Path(bundle_root)
    allowed = set(policy.available_lm_models)
    for name in LM_DIR_CANDIDATES:
        if name not in allowed:
            continue
        candidate = root / name
        if (candidate / "model.safetensors").is_file():
            return candidate
    for name in LM_DIR_CANDIDATES:
        candidate = root / name
        if (candidate / "model.safetensors").is_file():
            return candidate
    return None
