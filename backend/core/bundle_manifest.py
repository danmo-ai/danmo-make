"""Bundle install manifest: scan, write, validate (fail loud at load time)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "bundle.manifest.json"
SCHEMA_VERSION = 1
LORA_REGISTRY_CATEGORY = "loras"
CONTROLNET_REGISTRY_CATEGORY = "controlnets"

# ACE-Step DiT lives at bundle root or under acestep-v15-*/ (see ace_step.generation.resolve_dit_bundle).
_ACE_STEP_DIT_SUBDIRS = frozenset(
    {
        "acestep-v15-xl-sft",
        "acestep-v15-xl-turbo",
        "acestep-v15-xl-base",
        "acestep-v15-sft",
        "acestep-v15-turbo",
        "acestep-v15-base",
    }
)


def _ace_step_dit_safetensors(rel_lower: str, name_lower: str) -> bool:
    if name_lower != "model.safetensors":
        return False
    if rel_lower.count("/") == 0:
        return True
    top = rel_lower.split("/", 1)[0]
    return top in _ACE_STEP_DIT_SUBDIRS


def is_registry_lora_category(category: str | None) -> bool:
    """True for registry ``catalog.category=loras`` rows (single-file adapter bundles)."""
    return (category or "").strip().lower() == LORA_REGISTRY_CATEGORY


def is_registry_controlnet_category(category: str | None) -> bool:
    """True for registry ``catalog.category=controlnets`` rows (ControlNet adapter bundles)."""
    return (category or "").strip().lower() == CONTROLNET_REGISTRY_CATEGORY


def skips_full_family_bundle_contract(category: str | None) -> bool:
    """LoRA / ControlNet registry rows ship a single adapter weight bundle, not a full family stack."""
    return is_registry_lora_category(category) or is_registry_controlnet_category(category)


@dataclass(frozen=True)
class FamilyBundleContract:
    required: frozenset[str]
    optional: frozenset[str] = frozenset()


# Required component names align with scan_components keys.
FAMILY_BUNDLE_CONTRACTS: dict[str, FamilyBundleContract] = {
    "flux1": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder"}),
        optional=frozenset({"vae", "tokenizer"}),
    ),
    "flux2": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "z_image": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "qwen_image": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer", "processor"}),
    ),
    "fibo": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "ernie_image": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer", "scheduler"}),
    ),
    "cogview4": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer", "scheduler"}),
    ),
    "seedvr2": FamilyBundleContract(
        required=frozenset({"transformer", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "esrgan": FamilyBundleContract(
        required=frozenset({"transformer"}),
        optional=frozenset(),
    ),
    "ltx": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "ltx25": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "wan": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "hunyuan": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer", "image_encoder"}),
    ),
    "ace_step": FamilyBundleContract(
        required=frozenset({"transformer", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "diffrhythm": FamilyBundleContract(
        required=frozenset({"transformer"}),
        optional=frozenset({"vae"}),
    ),
    "qwen2": FamilyBundleContract(
        required=frozenset({"transformer", "tokenizer"}),
        optional=frozenset(),
    ),
    "qwen3": FamilyBundleContract(
        required=frozenset({"transformer", "tokenizer"}),
        optional=frozenset(),
    ),
    "qwen3_vl": FamilyBundleContract(
        required=frozenset({"transformer", "tokenizer"}),
        optional=frozenset(),
    ),
    "longcat": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae"}),
        optional=frozenset({"tokenizer"}),
    ),
    "longcat_avatar": FamilyBundleContract(
        required=frozenset({"transformer", "text_encoder", "vae", "audio_encoder"}),
        optional=frozenset({"tokenizer"}),
    ),
    "hidream_o1": FamilyBundleContract(
        required=frozenset({"transformer", "tokenizer"}),
        optional=frozenset(),
    ),
    "boogu_image": FamilyBundleContract(
        required=frozenset({"transformer", "vae"}),
        optional=frozenset({"text_encoder", "scheduler", "tokenizer"}),
    ),
}


def get_family_bundle_contract(family: str) -> FamilyBundleContract | None:
    return FAMILY_BUNDLE_CONTRACTS.get(family)


def require_family_bundle_contract(family: str) -> FamilyBundleContract:
    contract = get_family_bundle_contract(family)
    if contract is None:
        raise RuntimeError(
            f"No FamilyBundleContract registered for family={family!r}. "
            "Add an entry to bundle_manifest.FAMILY_BUNDLE_CONTRACTS before enabling bundle validation."
        )
    return contract


def _rel_path(bundle_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(bundle_root))
    except ValueError:
        return str(path)


def missing_safetensor_shards(
    bundle_root: Path,
    *,
    index_name: str = "model.safetensors.index.json",
) -> list[str]:
    """Return shard filenames listed in the safetensors index but absent on disk."""
    if not bundle_root.is_dir():
        return []
    index_path = bundle_root / index_name
    if not index_path.is_file():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return []
    expected = sorted({str(name) for name in weight_map.values() if str(name).strip()})
    return [name for name in expected if not (bundle_root / name).is_file()]


def scan_components(bundle_root: Path) -> dict[str, list[str]]:
    """Scan bundle directory and classify files by component (convention-based)."""
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"Bundle root not found: {bundle_root}")

    components: dict[str, list[str]] = {
        "transformer": [],
        "text_encoder": [],
        "vae": [],
        "tokenizer": [],
        "scheduler": [],
        "image_encoder": [],
        "audio_encoder": [],
    }

    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        rel_lower = _rel_path(bundle_root, path).lower()
        name_lower = path.name.lower()

        if name_lower == "tokenizer.json" or rel_lower.startswith("tokenizer/"):
            components["tokenizer"].append(_rel_path(bundle_root, path))
            continue

        suffix_lower = path.suffix.lower()
        weight_suffixes = (".safetensors", ".bin", ".json", ".pth")

        if rel_lower.startswith("mllm/") or "/mllm/" in f"/{rel_lower}/":
            if suffix_lower in weight_suffixes:
                components["text_encoder"].append(_rel_path(bundle_root, path))
            continue

        if rel_lower.startswith("processor/") or "/processor/" in f"/{rel_lower}/":
            if suffix_lower in weight_suffixes:
                components["tokenizer"].append(_rel_path(bundle_root, path))
            continue

        if rel_lower.startswith("scheduler/") or "/scheduler/" in f"/{rel_lower}/":
            if suffix_lower in weight_suffixes:
                components["scheduler"].append(_rel_path(bundle_root, path))
            continue

        # Wan / ModelScope flat bundles: models_t5*.pth + Wan2.2_VAE.pth at bundle root.
        if suffix_lower == ".pth":
            if name_lower.startswith("models_t5") or "text_encoder" in rel_lower:
                components["text_encoder"].append(_rel_path(bundle_root, path))
                continue
            if (
                name_lower.endswith("_vae.pth")
                or (name_lower.startswith("wan") and "_vae" in name_lower)
                or "/vae/" in f"/{rel_lower}/"
                or rel_lower.startswith("vae/")
            ):
                components["vae"].append(_rel_path(bundle_root, path))
                continue

        if "text_encoder" in rel_lower or name_lower.startswith("text_encoder"):
            if suffix_lower in weight_suffixes:
                components["text_encoder"].append(_rel_path(bundle_root, path))
            continue

        if "image_encoder" in rel_lower or name_lower.startswith("image_encoder"):
            if suffix_lower in weight_suffixes:
                components["image_encoder"].append(_rel_path(bundle_root, path))
            continue

        if "audio_encoder" in rel_lower or name_lower.startswith("audio_encoder"):
            if suffix_lower in weight_suffixes:
                components["audio_encoder"].append(_rel_path(bundle_root, path))
            continue

        if "/vae/" in f"/{rel_lower}/" or rel_lower.startswith("vae/") or name_lower.startswith("vae."):
            if suffix_lower in weight_suffixes:
                components["vae"].append(_rel_path(bundle_root, path))
            continue

        if path.suffix.lower() == ".safetensors":
            if _ace_step_dit_safetensors(rel_lower, name_lower):
                components["transformer"].append(_rel_path(bundle_root, path))
            elif (
                "t5_encoder" in name_lower
                or name_lower.startswith("umt5")
                or "text_encoder" in rel_lower
                or name_lower.startswith("connector")
            ):
                components["text_encoder"].append(_rel_path(bundle_root, path))
            elif (
                "vae" in rel_lower
                or name_lower.startswith("vae_")
                or "latent_upsampler" in name_lower
                or name_lower.startswith("vocoder")
                or name_lower.startswith("audio_vae")
            ):
                components["vae"].append(_rel_path(bundle_root, path))
            elif (
                "transformer" in rel_lower
                or rel_lower.startswith("dit/")
                or "/dit/" in f"/{rel_lower}/"
                or "diffusion" in name_lower
                or "unet" in name_lower
                or rel_lower.count("/") == 0
                or rel_lower.startswith("transformer/")
            ):
                components["transformer"].append(_rel_path(bundle_root, path))

    # Drop empty lists for cleaner manifest
    return {k: v for k, v in components.items() if v}


def missing_required_components(
    components: dict[str, list[str]],
    contract: FamilyBundleContract,
) -> list[str]:
    return sorted(name for name in contract.required if not components.get(name))


def build_manifest_payload(
    *,
    model_id: str,
    family: str,
    bundle_root: Path,
    contract: FamilyBundleContract | None = None,
) -> dict[str, Any]:
    components = scan_components(bundle_root)
    contract = contract or get_family_bundle_contract(family)
    missing: list[str] = []
    if contract is not None:
        missing = missing_required_components(components, contract)

    safetensors_count = sum(1 for p in bundle_root.rglob("*.safetensors"))
    missing_shards = missing_safetensor_shards(bundle_root)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "family": family,
        "bundle_root": str(bundle_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "detected": {
            "safetensors_count": safetensors_count,
            "weight_format": "safetensors" if safetensors_count else "unknown",
        },
        "status": "complete" if not missing and not missing_shards else "incomplete",
    }
    if missing:
        payload["missing_components"] = missing
    if missing_shards:
        payload["missing_shards"] = missing_shards
    return payload


def write_bundle_manifest(
    bundle_root: Path,
    *,
    model_id: str,
    family: str,
) -> Path:
    contract = get_family_bundle_contract(family)
    payload = build_manifest_payload(
        model_id=model_id,
        family=family,
        bundle_root=bundle_root,
        contract=contract,
    )
    out_path = bundle_root / MANIFEST_FILENAME
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def read_bundle_manifest(bundle_root: Path) -> dict[str, Any] | None:
    path = bundle_root / MANIFEST_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assert_bundle_ready_for_family(
    bundle_root: Path,
    *,
    family: str,
    model_id: str,
    registry_entry: Any | None = None,
    project_root: Path | None = None,
) -> None:
    """Fail loud when required bundle components are missing."""
    _ = registry_entry, project_root
    if not bundle_root.is_dir():
        raise RuntimeError(
            f"Model {model_id!r} (family={family}): bundle directory missing at {bundle_root}"
        )

    contract = require_family_bundle_contract(family)
    components = scan_components(bundle_root)
    missing = missing_required_components(components, contract)
    missing_shards = missing_safetensor_shards(bundle_root)

    if missing:
        raise RuntimeError(
            f"Model {model_id!r} (family={family}): bundle at {bundle_root} is missing "
            f"required components: {missing}. Re-download or repair the model bundle."
        )
    if missing_shards:
        preview = ", ".join(missing_shards[:3])
        suffix = f" (+{len(missing_shards) - 3} more)" if len(missing_shards) > 3 else ""
        raise RuntimeError(
            f"Model {model_id!r} (family={family}): bundle at {bundle_root} is missing "
            f"weight shard(s): {preview}{suffix}. Re-download or repair the model bundle."
        )


def bundle_component_status(
    bundle_root: Path,
    *,
    family: str,
    version_config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    """Component presence for download center; None when family has no contract."""
    _ = version_config, project_root
    contract = get_family_bundle_contract(family)
    if contract is None or not bundle_root.is_dir():
        return None

    components = scan_components(bundle_root)

    missing = missing_required_components(components, contract)
    missing_shards = missing_safetensor_shards(bundle_root)
    tracked = sorted(contract.required | contract.optional)
    present = sorted(name for name in tracked if components.get(name))
    return {
        "complete": not missing and not missing_shards,
        "missing": missing,
        "missing_shards": missing_shards,
        "present": present,
        "components": {name: bool(components.get(name)) for name in tracked},
    }
