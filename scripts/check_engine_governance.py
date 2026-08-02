#!/usr/bin/env python3
"""Unified engine architecture governance gates (imports, layout, family patterns, registry)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "backend" / "engine"
FAMILIES = ENGINE / "families"
COMMON = ENGINE / "common"
COMMON_SUBPACKAGES = frozenset({"ops", "model", "bundle", "codecs"})
FORBIDDEN_COMMON_SUBDIRS = frozenset({
    "platform",
    "features",
    "vae",
    "text_encoders",
    "bundle_weights",
    "weights",
})
ALLOWLIST = ROOT / "scripts" / "engine_governance_allowlist.txt"
REGISTRY = ROOT / "default_config" / "models_registry.json"

ALLOWLIST_SECTIONS = ("imports", "mlx-torch", "bundle-python", "layout", "primitives", "attention")

MLX_IMPORT_PREFIXES = (
    "import mlx",
    "from mlx",
)
TORCH_IMPORT_PREFIXES = (
    "import torch",
    "from torch",
)
IMPORT_FORBIDDEN_PREFIXES = MLX_IMPORT_PREFIXES + TORCH_IMPORT_PREFIXES
# Sole non-*_cuda.py torch site: CUDA RuntimeContext implementation.
_TORCH_IMPORT_RUNTIME_ALLOW = frozenset({"backend/engine/runtime/cuda.py"})
LAYOUT_FORBIDDEN_DIRS = {"mlx", "torch", "runtime", "common"}
FORBIDDEN_ENGINE_CODEC_DIRS = (
    "backend/engine/vae_codecs",
    "backend/engine/video_codecs",
)
FORBIDDEN_COMMON_CODEC_PATHS = (
    "backend/engine/common/codecs/vae/qwen_image",
)
PIPELINE_FAMILY_BRANCH_RE = re.compile(r'\bfamily\s*(!=|==)\s*["\']')

PRIMITIVE_PATTERNS = (
    re.compile(r"^\s*class\s+SelfAttention\s*[\(:]", re.M),
    re.compile(r"^\s*class\s+RMSNorm\s*[\(:]", re.M),
    re.compile(r"^\s*class\s+_RMSNorm\s*[\(:]", re.M),
)
ATTENTION_RE = re.compile(r"\bctx\.attention\s*\(")
SDPA_PATTERNS = (
    re.compile(r"\bmx\.fast\.scaled_dot_product_attention\s*\("),
    re.compile(r"^\s*from\s+mlx\.core\.fast\s+import\s+scaled_dot_product_attention\b", re.M),
    re.compile(r"\bF\.scaled_dot_product_attention\s*\("),
    re.compile(r"\btorch\.nn\.functional\.scaled_dot_product_attention\s*\("),
    re.compile(r"(?<![A-Za-z0-9_\.])scaled_dot_product_attention\s*\("),
)
ROPE_PATTERNS = (
    re.compile(r"^\s*def\s+_apply_rope_bshd\s*\(", re.M),
    re.compile(r"^\s*def\s+_apply_rope_qwen\s*\(", re.M),
    re.compile(r"^\s*def\s+_apply_rotary\s*\(\s*self\s*,\s*x\s*,\s*freqs_cis\s*\)", re.M),
)
MODULATION_PATTERNS = (
    re.compile(r"modulation\.chunk\(\s*6\s*,\s*dim\s*=\s*1\s*\)"),
    re.compile(r"modulation\.shape\[-1\]\s*//\s*4"),
    re.compile(r"gate_msa\s*=\s*v\[:,\s*:D\]\s*\[:,\s*None,\s*:\]"),
    re.compile(r"scale\s*=\s*v\[\.\.\.,\s*:D\]"),
)

HUNYUAN_REQUIRED_IDS = (
    "hunyuan-video-1.5-t2v-distill",
    "hunyuan-video-1.5-1080p-sr",
)
REMOVED_SHARED_MODEL_IDS = (
    "hunyuan-video-1.5-shared",
    "wan-2.2-14b-shared",
)
WAN_STEP_DISTILL_IDS = (
    "wan-2.2-i2v-14b-distill",
    "wan-2.2-t2v-14b-distill",
    "wan-2.2-i2v-14b-turbo",
    "wan-2.2-t2v-14b-turbo-720p",
    "wan-2.2-t2v-14b-turbo-480p",
    "wan-2.2-t2v-1.3b-turbo",
)
HUNYUAN_STEP_DISTILL_IDS = (
    "hunyuan-video-1.5-t2v-distill",
)
WAN_ENCODER_REPO_ID = "Wan-AI/Wan2.2-T2V-A14B"
ADAPTER_CATEGORIES = frozenset({"controlnets", "loras"})
ADAPTER_FORBIDDEN_BUNDLE_PREFIXES = (
    "models/Image/",
    "models/Video/",
    "models/Text/",
)
ADAPTER_FORBIDDEN_BUNDLE_REPOS = frozenset({
    "AI-ModelScope/FLUX.1-dev",
    "black-forest-labs/FLUX.1-dev",
    "Tongyi-MAI/Z-Image-Turbo",
})
HUNYUAN_REPO_ID = "Tencent-Hunyuan/HunyuanVideo-1.5"
MAX_FAMILY_UNITS = 8
DOCS = ROOT / "docs"
ALLOWED_ENGINE_DOCS = frozenset({"engine_architecture.md"})

ALL_RULES = (
    "imports",
    "mlx-torch",
    "bundle-python",
    "layout",
    "primitives",
    "attention",
    "sdpa",
    "rope",
    "modulation",
    "registry",
    "docs",
    "pipeline-family",
    "ltx-vendored",
)
RULE_CHOICES = ALL_RULES + ("parity",)


def _section_marker(name: str) -> str:
    return f"# --- {name} ---"


def load_allowlist() -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {k: [] for k in ALLOWLIST_SECTIONS}
    if not ALLOWLIST.is_file():
        return sections
    current: str | None = None
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for name in ALLOWLIST_SECTIONS:
            if stripped.lower() == _section_marker(name).lower():
                current = name
                break
        else:
            if not stripped or stripped.startswith("#"):
                continue
            if current:
                sections[current].append(stripped.rstrip("/"))
    return sections


def _write_allowlist_section(section: str, paths: list[str]) -> None:
    markers = {name: _section_marker(name) for name in ALLOWLIST_SECTIONS}
    blocks: dict[str, list[str]] = {name: [] for name in ALLOWLIST_SECTIONS}
    preamble: list[str] = []
    if ALLOWLIST.is_file():
        current: str | None = None
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            matched = False
            for name in ALLOWLIST_SECTIONS:
                if stripped.lower() == markers[name].lower():
                    current = name
                    matched = True
                    break
            if matched:
                continue
            if not stripped or stripped.startswith("#"):
                if current is None:
                    preamble.append(line)
                else:
                    blocks[current].append(line)
                continue
            if current:
                blocks[current].append(stripped.rstrip("/"))
    blocks[section] = paths
    out: list[str] = list(preamble) if preamble else [
        "# Engine governance allowlists (shrink over time; do not grow without review).",
        "# Sections: imports | layout | primitives | attention",
        "# Paths are repo-relative. Prefix matching for layout/primitives/attention.",
        "",
    ]
    for name in ALLOWLIST_SECTIONS:
        if out and out[-1] != "":
            out.append("")
        out.append(markers[name])
        out.extend(blocks[name])
        if blocks[name]:
            out.append("")
    ALLOWLIST.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {len(paths)} paths to {ALLOWLIST} [{section}]")


def _path_allowed(rel: str, allowlist: list[str]) -> bool:
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in allowlist)


def _mlx_import_allowed(rel: str) -> bool:
    if rel.startswith("backend/engine/runtime/"):
        return True
    name = Path(rel).name
    return name.endswith("_mlx.py") or name.endswith("_cuda.py")


def _torch_import_allowed(rel: str) -> bool:
    if rel in _TORCH_IMPORT_RUNTIME_ALLOW:
        return True
    return Path(rel).name.endswith("_cuda.py")


def _import_line_allowed(rel: str, line: str) -> bool:
    s = line.strip()
    if s.startswith(MLX_IMPORT_PREFIXES):
        return _mlx_import_allowed(rel)
    if s.startswith(TORCH_IMPORT_PREFIXES):
        return _torch_import_allowed(rel)
    return True


_MLX_TORCH_BRIDGE_IMPORT_RE = re.compile(
    r"^\s*(from\s+\S*_cuda\S*\s+import|import\s+\S*_cuda\S*)"
)
_MLX_TORCH_INDIRECT_RE = re.compile(
    r"importlib\.import_module\(\s*[\"']torch"
    r"|from\s+safetensors\.torch\s+import"
    r"|import\s+safetensors\.torch"
)
_BUNDLE_PYTHON_PATTERNS = (
    re.compile(r"\bsys\.path\.insert\s*\("),
    re.compile(r"\bspec_from_file_location\s*\("),
    re.compile(r"\bexec_module\s*\("),
)


def check_mlx_torch(allow: dict[str, list[str]]) -> list[str]:
    """``*_mlx.py`` must not import ``*_cuda`` modules (MLX hot path cannot depend on torch)."""
    violations: list[str] = []
    exempt = set(allow.get("mlx-torch", []))
    for path in sorted(ENGINE.rglob("*_mlx.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel in exempt:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or not s:
                continue
            if s.startswith(TORCH_IMPORT_PREFIXES):
                violations.append(f"{rel}:{i}: {s[:80]}")
                continue
            if _MLX_TORCH_BRIDGE_IMPORT_RE.match(s):
                violations.append(f"{rel}:{i}: {s[:80]}")
                continue
            if _MLX_TORCH_INDIRECT_RE.search(s):
                violations.append(f"{rel}:{i}: {s[:80]}")
    return violations


def check_bundle_python(allow: dict[str, list[str]]) -> list[str]:
    """Engine code must not inject arbitrary bundle Python; only allowlisted bootstrap sites."""
    violations: list[str] = []
    exempt = set(allow.get("bundle-python", []))
    for path in sorted(ENGINE.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel in exempt:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for pat in _BUNDLE_PYTHON_PATTERNS:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                snippet = text.splitlines()[line_no - 1].strip()
                violations.append(f"{rel}:{line_no}: {snippet[:80]}")
    return violations


def _scan_family_py() -> list[Path]:
    if not FAMILIES.is_dir():
        return []
    return sorted(FAMILIES.rglob("*.py"))


def _violations_regex(
    paths: list[Path],
    patterns: tuple[re.Pattern[str], ...],
    *,
    allowlist: list[str] | None = None,
    line_filter: Callable[[str], bool] | None = None,
) -> list[str]:
    violations: list[str] = []
    for path in paths:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if allowlist and _path_allowed(rel, allowlist):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        lines = text.splitlines()
        for pat in patterns:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                snippet = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else "<unknown>"
                if line_filter and not line_filter(snippet):
                    continue
                violations.append(f"{rel}:{line_no}: {snippet}")
    return violations


def check_imports(allow: dict[str, list[str]]) -> list[str]:
    violations: list[str] = []
    exempt = set(allow["imports"])
    for path in sorted(ENGINE.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel in exempt:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or not s:
                continue
            if s.startswith(IMPORT_FORBIDDEN_PREFIXES) and not _import_line_allowed(rel, s):
                violations.append(f"{rel}:{i}: {s[:80]}")
    return violations


def check_layout(allow: dict[str, list[str]]) -> list[str]:
    violations: list[str] = []
    if FAMILIES.is_dir():
        for family_dir in sorted(
            p for p in FAMILIES.iterdir() if p.is_dir() and not p.name.startswith("_")
        ):
            for path in sorted(family_dir.rglob("*")):
                if not path.is_dir() or path.name not in LAYOUT_FORBIDDEN_DIRS:
                    continue
                rel = str(path.relative_to(ROOT)).replace("\\", "/")
                if _path_allowed(rel, allow["layout"]):
                    continue
                violations.append(
                    f"{rel}: forbidden family subtree directory '{path.name}' "
                    "(use common/ or *_mlx.py/*_cuda.py hooks instead)"
                )
    for rel in FORBIDDEN_ENGINE_CODEC_DIRS:
        if (ROOT / rel).is_dir():
            violations.append(
                f"{rel}/: forbidden engine codec wrapper directory "
                "(register handlers in vae_codec_registry.py / video_codec_registry.py)"
            )
    if COMMON.is_dir():
        for path in sorted(COMMON.iterdir()):
            name = path.name
            if name.startswith("_") or name == "__pycache__":
                continue
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            if path.is_file() and path.suffix == ".py" and name != "__init__.py":
                violations.append(
                    f"{rel}: forbidden module at common/ root "
                    "(use common/{{ops,model,bundle,codecs}}/ or engine/{{cache,lineage,contracts,codecs}}.py facade)"
                )
            elif path.is_dir() and name not in COMMON_SUBPACKAGES:
                if name in FORBIDDEN_COMMON_SUBDIRS:
                    violations.append(
                        f"{rel}/: stale or forbidden common/ subtree '{name}' "
                        f"(use common/{{ops,model,bundle,codecs}}/ or families/<id>/; see docs/engine_architecture.md §3)"
                    )
                else:
                    violations.append(
                        f"{rel}/: unexpected directory under common/ "
                        f"(allowed: {', '.join(sorted(COMMON_SUBPACKAGES))})"
                    )
    for rel in FORBIDDEN_COMMON_CODEC_PATHS:
        if (ROOT / rel).exists():
            violations.append(
                f"{rel}: Qwen VAE belongs in families/qwen/vae/ (stale codec subtree)"
            )
    return violations


def check_pipeline_family(_allow: dict[str, list[str]]) -> list[str]:
    violations: list[str] = []
    pipelines = ROOT / "backend" / "engine" / "pipelines"
    if not pipelines.is_dir():
        return violations
    for path in sorted(pipelines.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or not s:
                continue
            if PIPELINE_FAMILY_BRANCH_RE.search(s):
                violations.append(f"{rel}:{i}: {s[:120]}")
    return violations


def check_primitives(allow: dict[str, list[str]]) -> list[str]:
    return _violations_regex(_scan_family_py(), PRIMITIVE_PATTERNS, allowlist=allow["primitives"])


def check_attention(allow: dict[str, list[str]]) -> list[str]:
    violations: list[str] = []
    for path in _scan_family_py():
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if _path_allowed(rel, allow["attention"]):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ATTENTION_RE.search(s):
                violations.append(f"{rel}:{i}: {s[:120]}")
    return violations


def check_sdpa() -> list[str]:
    return _violations_regex(_scan_family_py(), SDPA_PATTERNS)


def check_rope() -> list[str]:
    return _violations_regex(_scan_family_py(), ROPE_PATTERNS)


def check_modulation() -> list[str]:
    return _violations_regex(_scan_family_py(), MODULATION_PATTERNS)


def check_registry() -> list[str]:
    from backend.catalog.loader import expand_catalog_document

    with REGISTRY.open(encoding="utf-8") as f:
        raw = json.load(f)
    models = expand_catalog_document(raw).get("models") or {}
    failures: list[str] = []

    for mid in HUNYUAN_REQUIRED_IDS:
        if mid not in models:
            failures.append(f"missing required Hunyuan model: {mid}")

    for mid in HUNYUAN_REQUIRED_IDS:
        model = models.get(mid)
        if not isinstance(model, dict):
            continue
        if model.get("source") != "modelscope":
            failures.append(f"{mid}: source must be modelscope")

    for mid in REMOVED_SHARED_MODEL_IDS:
        if mid in models:
            failures.append(f"{mid}: shared install-only card removed (use self-contained model bundles)")

    for mid, model in models.items():
        if not isinstance(model, dict):
            continue
        deps = model.get("dependencies")
        if isinstance(deps, list) and deps:
            failures.append(f"{mid}: distribution.dependencies must be empty (use bundle_repos)")

    def _bundle_has_repo(ver: dict, repo_id: str) -> bool:
        bundle = ver.get("bundle_repos")
        if not isinstance(bundle, list):
            return False
        for item in bundle:
            if isinstance(item, dict) and item.get("repo_id") == repo_id:
                return True
        return False

    for mid in WAN_STEP_DISTILL_IDS:
        model = models.get(mid)
        if not isinstance(model, dict):
            continue
        versions = model.get("versions")
        if not isinstance(versions, dict):
            failures.append(f"{mid}: versions required for self-contained Wan bundle")
            continue
        ok = False
        for ver in versions.values():
            if isinstance(ver, dict) and _bundle_has_repo(ver, WAN_ENCODER_REPO_ID):
                ok = True
                break
        if not ok:
            failures.append(
                f"{mid}: bundle_repos must include {WAN_ENCODER_REPO_ID} encoders"
            )

    for mid, model in models.items():
        if not isinstance(model, dict):
            continue
        cat = str(model.get("category") or "")
        if cat not in ADAPTER_CATEGORIES:
            continue
        if not str(model.get("base_model") or "").strip():
            failures.append(f"{mid}: {cat} requires catalog.base_model (install base separately)")
        versions = model.get("versions")
        if not isinstance(versions, dict):
            continue
        for vk, ver in versions.items():
            if not isinstance(ver, dict):
                continue
            bundle = ver.get("bundle_repos")
            if not isinstance(bundle, list):
                continue
            for entry in bundle:
                if not isinstance(entry, dict):
                    continue
                repo = str(entry.get("repo_id") or "")
                lp = str(entry.get("local_path") or "")
                if repo in ADAPTER_FORBIDDEN_BUNDLE_REPOS or any(
                    lp.startswith(prefix) for prefix in ADAPTER_FORBIDDEN_BUNDLE_PREFIXES
                ):
                    failures.append(
                        f"{mid}: versions.{vk} must not bundle base model ({repo or lp}); "
                        "download adapter repo only"
                    )

    distill = models.get("hunyuan-video-1.5-t2v-distill")
    if isinstance(distill, dict):
        versions = distill.get("versions")
        if not isinstance(versions, dict) or "bf16" not in versions:
            failures.append("hunyuan-video-1.5-t2v-distill: versions.bf16 is required")
        else:
            ver = versions["bf16"]
            if not isinstance(ver, dict):
                failures.append("hunyuan-video-1.5-t2v-distill: versions.bf16 must be object")
            elif not ver.get("hunyuan_distill_variant"):
                failures.append("hunyuan-video-1.5-t2v-distill: versions.bf16.hunyuan_distill_variant is required")
            elif ver.get("repo_id") != "lightx2v/Hy1.5-Distill-Models":
                failures.append("hunyuan-video-1.5-t2v-distill: repo_id must be lightx2v/Hy1.5-Distill-Models")
            else:
                bundle = ver.get("bundle_repos")
                if not isinstance(bundle, list) or len(bundle) < 4:
                    failures.append(
                        "hunyuan-video-1.5-t2v-distill: bundle_repos must include DiT + Hunyuan + Qwen + ByT5"
                    )
                elif not _bundle_has_repo(ver, HUNYUAN_REPO_ID):
                    failures.append(
                        f"hunyuan-video-1.5-t2v-distill: bundle_repos must include {HUNYUAN_REPO_ID} encoders"
                    )
        params = distill.get("parameters", {})
        if params.get("text_encoder_qwen_local") != "models/Text/qwen2.5-vl-7b-instruct":
            failures.append("hunyuan-video-1.5-t2v-distill: parameters.text_encoder_qwen_local mismatch")
        if not params.get("text_encoder_release_after_encode"):
            failures.append("hunyuan-video-1.5-t2v-distill: parameters.text_encoder_release_after_encode must be true")

    sr = models.get("hunyuan-video-1.5-1080p-sr")
    if isinstance(sr, dict):
        versions = sr.get("versions")
        if not isinstance(versions, dict) or "bf16" not in versions:
            failures.append("hunyuan-video-1.5-1080p-sr: versions.bf16 is required")
        else:
            ver = versions["bf16"]
            if not isinstance(ver, dict):
                failures.append("hunyuan-video-1.5-1080p-sr: versions.bf16 must be object")
            elif not ver.get("hunyuan_ms_variant"):
                failures.append("hunyuan-video-1.5-1080p-sr: versions.bf16.hunyuan_ms_variant is required")
            elif ver.get("hunyuan_ms_variant") != "1080p_sr_distilled":
                failures.append("hunyuan-video-1.5-1080p-sr: hunyuan_ms_variant must be 1080p_sr_distilled")
            patterns = ver.get("allow_patterns") if isinstance(ver, dict) else None
            if not isinstance(patterns, list) or not patterns:
                failures.append("hunyuan-video-1.5-1080p-sr: versions.bf16.allow_patterns is required")
            elif "transformer/1080p_sr_distilled/**" not in patterns:
                failures.append(
                    "hunyuan-video-1.5-1080p-sr: allow_patterns must include transformer/1080p_sr_distilled/**"
                )
            elif "model_index.json" in patterns:
                failures.append(
                    "hunyuan-video-1.5-1080p-sr: allow_patterns must not include generated model_index.json"
                )
            bundle = ver.get("bundle_repos")
            if not isinstance(bundle, list) or len(bundle) < 3:
                failures.append(
                    "hunyuan-video-1.5-1080p-sr: bundle_repos must include Hunyuan + Qwen + ByT5"
                )
            elif isinstance(bundle, list) and bundle:
                first = bundle[0] if isinstance(bundle[0], dict) else {}
                if first.get("repo_id") != HUNYUAN_REPO_ID:
                    failures.append(f"hunyuan-video-1.5-1080p-sr: first bundle repo must be {HUNYUAN_REPO_ID}")

    i2v_distill = models.get("hunyuan-video-1.5-i2v-720p-distill")
    if isinstance(i2v_distill, dict):
        versions = i2v_distill.get("versions")
        if not isinstance(versions, dict) or "bf16" not in versions:
            failures.append("hunyuan-video-1.5-i2v-720p-distill: versions.bf16 is required")
        else:
            ver = versions["bf16"]
            if not isinstance(ver, dict):
                failures.append("hunyuan-video-1.5-i2v-720p-distill: versions.bf16 must be object")
            elif ver.get("hunyuan_ms_variant") != "720p_i2v_distilled_sparse":
                failures.append(
                    "hunyuan-video-1.5-i2v-720p-distill: hunyuan_ms_variant must be 720p_i2v_distilled_sparse"
                )
            patterns = ver.get("allow_patterns") if isinstance(ver, dict) else None
            if isinstance(patterns, list) and "image_encoder/**" in patterns:
                failures.append(
                    "hunyuan-video-1.5-i2v-720p-distill: image_encoder/** must come from "
                    "FLUX.1-Redux-dev bundle_repos, not Tencent-Hunyuan allow_patterns"
                )
            bundle = ver.get("bundle_repos") if isinstance(ver, dict) else None
            if not isinstance(bundle, list) or len(bundle) < 4:
                failures.append(
                    "hunyuan-video-1.5-i2v-720p-distill: bundle_repos must include "
                    "Hunyuan + SigLIP (FLUX Redux) + Qwen + ByT5"
                )
            elif isinstance(bundle, list):
                siglip = next(
                    (
                        item
                        for item in bundle
                        if isinstance(item, dict)
                        and item.get("repo_id") == "black-forest-labs/FLUX.1-Redux-dev"
                    ),
                    None,
                )
                siglip_patterns = siglip.get("allow_patterns") if isinstance(siglip, dict) else None
                if not isinstance(siglip_patterns, list) or "image_encoder/**" not in siglip_patterns:
                    failures.append(
                        "hunyuan-video-1.5-i2v-720p-distill: FLUX.1-Redux-dev bundle_repos entry "
                        "must include image_encoder/**"
                    )
        params = i2v_distill.get("parameters", {})
        if params.get("supports_guidance") is not False:
            failures.append("hunyuan-video-1.5-i2v-720p-distill: parameters.supports_guidance must be false")
        actions = i2v_distill.get("actions") or {}
        if "animate" not in actions:
            failures.append("hunyuan-video-1.5-i2v-720p-distill: actions must include animate")

    for mid in HUNYUAN_STEP_DISTILL_IDS + WAN_STEP_DISTILL_IDS:
        model = models.get(mid)
        if not isinstance(model, dict):
            continue
        params = model.get("parameters", {})
        if params.get("supports_guidance") is not False:
            failures.append(f"{mid}: parameters.supports_guidance must be false")
        if not params.get("step_distill"):
            failures.append(f"{mid}: parameters.step_distill must be true")
        if params.get("negative_prompt_support") is not False:
            failures.append(f"{mid}: parameters.negative_prompt_support must be false")
        if "guide_scale" in params:
            failures.append(f"{mid}: parameters.guide_scale must not be present")
        actions = model.get("actions") or {}
        if mid.endswith("i2v-step-distill") and "animate" not in actions:
            failures.append(f"{mid}: actions must include animate")
        if mid.endswith("t2v-distill") and "create" not in actions:
            failures.append(f"{mid}: actions must include create")
        if mid.endswith("i2v-14b-distill") and "animate" not in actions:
            failures.append(f"{mid}: actions must include animate")
        if mid.endswith("t2v-14b-distill") and "create" not in actions:
            failures.append(f"{mid}: actions must include create")
        if mid.endswith("-turbo") and "create" not in actions and "animate" not in actions:
            failures.append(f"{mid}: turbo model actions must include create and/or animate")
        if mid.endswith("i2v-14b-turbo") and "animate" not in actions:
            failures.append(f"{mid}: actions must include animate")
        if mid.endswith("t2v-14b-turbo") and "create" not in actions:
            failures.append(f"{mid}: actions must include create")
        if mid.endswith("t2v-1.3b-turbo") and "create" not in actions:
            failures.append(f"{mid}: actions must include create")

    mid = "hunyuan-video-1.5-1080p-sr"
    model = models.get(mid)
    if isinstance(model, dict):
        params = model.get("parameters", {})
        if not params.get("vae_spatial_tiling"):
            failures.append(f"{mid}: parameters.vae_spatial_tiling must be true")
        if params.get("text_encoder_qwen_local") != "models/Text/qwen2.5-vl-7b-instruct":
            failures.append(f"{mid}: parameters.text_encoder_qwen_local mismatch")
        if not params.get("text_encoder_release_after_encode"):
            failures.append(f"{mid}: parameters.text_encoder_release_after_encode must be true")

    successor_edges: dict[str, str] = {}
    for mid, model in models.items():
        if not isinstance(model, dict):
            continue
        successor = model.get("successor")
        if successor is None:
            continue
        if not isinstance(successor, str) or not successor.strip():
            failures.append(f"{mid}: successor must be a non-empty string")
            continue
        target = successor.strip()
        if target == mid:
            failures.append(f"{mid}: successor must not point to itself")
            continue
        if target not in models:
            failures.append(f"{mid}: successor {target!r} is not in models registry")
            continue
        successor_edges[mid] = target

    for mid, target in successor_edges.items():
        seen = {mid}
        cursor = target
        while cursor in successor_edges:
            if cursor in seen:
                failures.append(f"{mid}: successor chain contains a cycle")
                break
            seen.add(cursor)
            cursor = successor_edges[cursor]

    distilled_from_edges: dict[str, str] = {}
    for mid, model in models.items():
        if not isinstance(model, dict):
            continue
        for field in ("distilled_from", "distilled_variant"):
            value = model.get(field)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                failures.append(f"{mid}: {field} must be a non-empty string")
                continue
            target = value.strip()
            if target == mid:
                failures.append(f"{mid}: {field} must not point to itself")
                continue
            if target not in models:
                failures.append(f"{mid}: {field} {target!r} is not in models registry")
                continue
            if field == "distilled_from":
                distilled_from_edges[mid] = target

        base_id = model.get("distilled_from")
        variant_id = model.get("distilled_variant")
        if isinstance(base_id, str) and base_id.strip() and isinstance(variant_id, str) and variant_id.strip():
            failures.append(f"{mid}: cannot set both distilled_from and distilled_variant")

    for mid, base_id in distilled_from_edges.items():
        variant = models.get(base_id, {}).get("distilled_variant")
        if isinstance(variant, str) and variant.strip() and variant.strip() != mid:
            failures.append(
                f"{mid}: distilled_from {base_id!r} has distilled_variant {variant.strip()!r} (expected {mid!r})"
            )

    for mid, model in models.items():
        if not isinstance(model, dict):
            continue
        variant_id = model.get("distilled_variant")
        if not isinstance(variant_id, str) or not variant_id.strip():
            continue
        target = variant_id.strip()
        back_ref = models.get(target, {}).get("distilled_from")
        if isinstance(back_ref, str) and back_ref.strip() and back_ref.strip() != mid:
            failures.append(
                f"{mid}: distilled_variant {target!r} has distilled_from {back_ref.strip()!r} (expected {mid!r})"
            )

    return failures


def _family_dirs() -> list[Path]:
    if not FAMILIES.is_dir():
        return []
    return sorted(p for p in FAMILIES.iterdir() if p.is_dir() and not p.name.startswith("_"))


def _family_logical_units(family_dir: Path) -> int:
    """Count logical units: top-level stems + one unit per subpackage directory."""
    stems: set[str] = set()
    skip_subdirs = frozenset({"data", "__pycache__"})

    for path in family_dir.glob("*.py"):
        if path.name == "__init__.py":
            continue
        stem = path.stem
        if stem.endswith("_mlx"):
            stem = stem[:-4]
        elif stem.endswith("_cuda"):
            stem = stem[:-5]
        stems.add(stem)

    for sub in family_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("_") or sub.name in skip_subdirs:
            continue
        if any(sub.glob("*.py")):
            stems.add(sub.name)

    return len(stems)


def _report_family(mode: str) -> list[str]:
    if mode == "registry":
        if not REGISTRY.is_file():
            return ["models_registry.json missing"]
        from backend.core.registry_profiles import audit_registry_document

        return audit_registry_document(json.loads(REGISTRY.read_text(encoding="utf-8")))

    lines: list[str] = []
    for family_dir in _family_dirs():
        if mode == "budget":
            units = _family_logical_units(family_dir)
            status = "OK" if units <= MAX_FAMILY_UNITS else "OVER"
            lines.append(f"{family_dir.name}: {units} logical units [{status}]")
            continue
        common_imports = 0
        inference_refs = 0
        total = 0
        for path in family_dir.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            total += len(text.splitlines())
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped.startswith(("import ", "from ")):
                    continue
                if "backend.engine.common" in line:
                    common_imports += 1
                if "backend.engine.inference" in line:
                    inference_refs += 1
        rate = common_imports / max(total, 1)
        lines.append(
            f"{family_dir.name}: common={common_imports}/{total} ({rate:.1%}), "
            f"inference_refs={inference_refs}"
        )
    return lines


def check_parity() -> list[str]:
    from backend.engine import _transformer_registry as tr
    from backend.engine._transformer_registry import get_transformer_class
    from backend.engine.config.model_configs import get_config_class
    from backend.engine.runtime.mlx import MLXContext

    failures: list[str] = []
    for family in sorted(tr._TRANSFORMER.keys()):
        try:
            config = get_config_class(family)()
            # Shape C family_generator stems are placeholders without DiT _param_map.
            if str(getattr(config, "image_pipeline_shape", "") or "") == "family_generator":
                continue
            ctx = MLXContext()
            model = get_transformer_class(family)(config, ctx)
            if hasattr(model, "_build_param_map"):
                model._build_param_map()
            param_map = getattr(model, "_param_map", None)
            if not isinstance(param_map, dict) or not param_map:
                failures.append(f"{family}: empty or missing _param_map")
        except Exception as exc:
            failures.append(f"{family}: failed to collect param keys: {exc}")
    return failures


def check_docs(_allow: dict[str, list[str]]) -> list[str]:
    violations: list[str] = []
    if not DOCS.is_dir():
        violations.append("docs/: missing engine architecture doc (expected docs/engine_architecture.md)")
        return violations
    for path in sorted(DOCS.iterdir()):
        if path.name.startswith("."):
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if path.is_dir():
            violations.append(f"{rel}/: no subdirectories under docs/ (single architecture doc only)")
        elif path.suffix == ".md" and path.name not in ALLOWED_ENGINE_DOCS:
            violations.append(
                f"{rel}: forbidden docs/ file (only docs/engine_architecture.md allowed; merge into it)"
            )
    if not (DOCS / "engine_architecture.md").is_file():
        violations.append("docs/engine_architecture.md: required single engine architecture document")
    return violations


LTX_VENDORED_MARKERS = (
    "import ltx_core_mlx",
    "from ltx_core_mlx",
    "import ltx_pipelines_mlx",
    "from ltx_pipelines_mlx",
    "ltx-core-mlx",
    "ltx-pipelines-mlx",
)


def check_ltx_vendored(_allow: dict[str, list[str]]) -> list[str]:
    """Ban third-party LTX MLX pipeline packages — engine uses in-repo families/ltx/."""
    violations: list[str] = []
    for path in sorted((ROOT / "backend").rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or not s:
                continue
            if any(marker in s for marker in LTX_VENDORED_MARKERS):
                violations.append(f"{rel}:{i}: {s[:100]}")
    return violations


RULE_RUNNERS: dict[str, Callable[[dict[str, list[str]]], list[str]]] = {
    "imports": check_imports,
    "mlx-torch": check_mlx_torch,
    "bundle-python": check_bundle_python,
    "layout": check_layout,
    "primitives": check_primitives,
    "attention": check_attention,
    "sdpa": lambda _a: check_sdpa(),
    "rope": lambda _a: check_rope(),
    "modulation": lambda _a: check_modulation(),
    "registry": lambda _a: check_registry(),
    "parity": lambda _a: check_parity(),
    "docs": lambda _a: check_docs({}),
    "pipeline-family": lambda _a: check_pipeline_family({}),
    "ltx-vendored": check_ltx_vendored,
}

RULE_HINTS: dict[str, str] = {
    "imports": "Move mlx imports to *_mlx.py / *_cuda.py / runtime/; torch only in *_cuda.py "
    f"(+ runtime/cuda.py), or shrink allowlist in {ALLOWLIST} [imports]",
    "mlx-torch": "MLX hot path (*_mlx.py) must not import torch or *_cuda modules; use native MLX "
    f"or dispatch from text_encoder.py / fail loud; shrink allowlist in {ALLOWLIST} [mlx-torch]",
    "bundle-python": "Runtime bundle Python bootstrap only in allowlisted files (model download dirs); "
    f"see {ALLOWLIST} [bundle-python]",
    "layout": f"Flatten family layout or shrink allowlist in {ALLOWLIST} [layout]",
    "primitives": f"Reuse common primitives or shrink allowlist in {ALLOWLIST} [primitives]",
    "attention": "Use backend/engine/common/ops/attention.py helpers or shrink allowlist "
    f"in {ALLOWLIST} [attention]",
    "sdpa": "Use backend/engine/common/ops/attention.py helpers",
    "rope": "Use backend/engine/common/ops/embeddings.py helpers",
    "modulation": "Use backend/engine/common/ops/norm.py helpers",
    "registry": "Fix default_config/models_registry.json Hunyuan contracts",
    "parity": "Fix remap_* vs Transformer _param_map key mismatch (make check-engine-governance --rule parity)",
    "docs": "Keep only docs/engine_architecture.md; merge or delete other docs/*.md files",
    "pipeline-family": "Use registry/config dispatch instead of family == in pipelines (see docs/engine_architecture.md §5)",
    "ltx-vendored": "Remove ltx_core_mlx / ltx_pipelines_mlx imports; implement under backend/engine/families/ltx/",
}


def _write_allowlist_for_rule(rule: str) -> int:
    if rule == "imports":
        found: list[str] = []
        for path in sorted(ENGINE.rglob("*.py")):
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("#") or not s:
                    continue
                if s.startswith(IMPORT_FORBIDDEN_PREFIXES) and not _import_line_allowed(rel, s):
                    found.append(rel)
                    break
        _write_allowlist_section("imports", sorted(set(found)))
        return 0

    if rule == "layout":
        found = []
        if FAMILIES.is_dir():
            for family_dir in sorted(
                p for p in FAMILIES.iterdir() if p.is_dir() and not p.name.startswith("_")
            ):
                for path in sorted(family_dir.rglob("*")):
                    if path.is_dir() and path.name in LAYOUT_FORBIDDEN_DIRS:
                        found.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        _write_allowlist_section("layout", found)
        return 0

    if rule == "mlx-torch":
        found: list[str] = []
        for path in sorted(ENGINE.rglob("*_mlx.py")):
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("#") or not s:
                    continue
                if s.startswith(TORCH_IMPORT_PREFIXES) or _MLX_TORCH_BRIDGE_IMPORT_RE.match(s):
                    found.append(rel)
                    break
        _write_allowlist_section("mlx-torch", sorted(set(found)))
        return 0

    if rule == "bundle-python":
        found: list[str] = []
        for path in sorted(ENGINE.rglob("*.py")):
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if any(p.search(text) for p in _BUNDLE_PYTHON_PATTERNS):
                found.append(rel)
        _write_allowlist_section("bundle-python", sorted(set(found)))
        return 0

    if rule == "primitives":
        found = []
        for path in _scan_family_py():
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if any(p.search(text) for p in PRIMITIVE_PATTERNS):
                found.append(rel)
        _write_allowlist_section("primitives", found)
        return 0

    if rule == "attention":
        found = []
        for path in _scan_family_py():
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if ATTENTION_RE.search(text):
                found.append(rel)
        _write_allowlist_section("attention", found)
        return 0

    print(f"--write-allowlist not supported for rule '{rule}'", file=sys.stderr)
    return 2


def run_rules(rules: tuple[str, ...]) -> int:
    allow = load_allowlist()
    failed = False
    for rule in rules:
        violations = RULE_RUNNERS[rule](allow)
        if violations:
            failed = True
            print(f"[{rule}] governance failed ({len(violations)}):", file=sys.stderr)
            for item in violations[:40]:
                print(f"  - {item}", file=sys.stderr)
            if len(violations) > 40:
                print(f"  … and {len(violations) - 40} more", file=sys.stderr)
            print(f"  → {RULE_HINTS[rule]}", file=sys.stderr)
    if failed:
        return 1
    if len(rules) == 1:
        print(f"Engine governance [{rules[0]}] OK")
    else:
        print("Engine governance OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rule",
        choices=RULE_CHOICES,
        action="append",
        help="Run one rule (repeatable). Default: all engine rules.",
    )
    ap.add_argument(
        "--write-allowlist",
        choices=("imports", "mlx-torch", "bundle-python", "layout", "primitives", "attention"),
        help="Regenerate allowlist section from current tree (migration utility).",
    )
    ap.add_argument(
        "--report",
        choices=("registry", "family-budget", "reuse"),
        help="Print non-blocking audit report (registry shrink hints, family budget, reuse).",
    )
    args = ap.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    if args.report:
        reporters = {
            "registry": lambda: _report_family("registry"),
            "family-budget": lambda: _report_family("budget"),
            "reuse": lambda: _report_family("reuse"),
        }
        lines = reporters[args.report]()
        for line in lines:
            print(line)
        print(f"Report [{args.report}]: {len(lines)} line(s)")
        return 0

    if args.write_allowlist:
        return _write_allowlist_for_rule(args.write_allowlist)

    rules = tuple(args.rule) if args.rule else ALL_RULES
    return run_rules(rules)


if __name__ == "__main__":
    raise SystemExit(main())
