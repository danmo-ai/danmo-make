"""Migrate the model registry: dense download + local `q4-mixed` conversion.

Every downloadable base model keeps exactly one dense version (the only
download) plus a `q4-mixed` derived version that the installer builds on the
machine. Third-party pre-quantized packages are removed.

LoRA / ControlNet adapters are single-file and are not quantized; only their
version key is normalized to the real dtype.
"""

import json
import sys

REGISTRY = "default_config/models_registry.json"
DERIVED_KEY = "q4-mixed"

# model id -> recipe id
RECIPE = {
    "qwen3-4b-thinking-2507": "qwen3-q4-mixed-v1",
    "qwen3.5-4b": "qwen3-q4-mixed-v1",
    "qwen3.6-27b": "qwen3-q4-mixed-v1",
    "qwen3-vl-4b-instruct": "qwen3-vl-q4-mixed-v1",
    "flux2-klein-4b": "flux2-q4-mixed-v1",
    "flux2-klein-9b": "flux2-q4-mixed-v1",
    "flux2-klein-base-4b": "flux2-q4-mixed-v1",
    "flux2-klein-base-9b": "flux2-q4-mixed-v1",
    "z-image": "z-image-q4-mixed-v1",
    "z-image-turbo": "z-image-q4-mixed-v1",
    "fibo-edit-rmbg": "fibo-q4-mixed-v1",
    "qwen-image": "qwen-image-q4-mixed-v1",
    "qwen-image-edit": "qwen-image-q4-mixed-v1",
    "ernie-image-turbo": "ernie-image-q4-mixed-v1",
    "seedvr2-7b": "seedvr2-q4-mixed-v1",
    "seedvr2-video-7b": "seedvr2-q4-mixed-v1",
    "real-esrgan-x4plus": "seedvr2-q4-mixed-v1",
    "wan-2.2-t2v-14b": "wan-q4-mixed-v1",
    "wan-2.2-i2v-14b": "wan-q4-mixed-v1",
    "wan-2.2-i2v-14b-distill": "wan-q4-mixed-v1",
    "wan-2.2-ti2v-5b": "wan-q4-mixed-v1",
    "ace-step-xl-sft": "ace-step-q4-mixed-v1",
    "ace-step-xl-turbo": "ace-step-q4-mixed-v1",
    "diffrhythm-v2": "ace-step-q4-mixed-v1",
    "longcat-avatar-1.5": "longcat-avatar-q4-mixed-v1",
    "minimax-h3-fl2va": "minimax-h3-q4-mixed-v1",
    "ltx-2.5-distilled": "ltx25-q4-mixed-v1",
}

# Models whose published name is not a dtype key: pin the canonical key and the
# official dense repository. The dense entry keeps bundle_repos/allow_patterns.
DENSE_OVERRIDE = {
    "qwen3-4b-thinking-2507": ("bf16", "Qwen/Qwen3-4B-Thinking-2507"),
    "qwen3.5-4b": ("bf16", "Qwen/Qwen3.5-4B"),
    "qwen3.6-27b": ("bf16", "Qwen/Qwen3.6-27B"),
    "qwen3-vl-4b-instruct": ("bf16", "Qwen/Qwen3-VL-4B-Instruct"),
    "ltx-2.5-distilled": ("bf16", "Lightricks/LTX-2.5"),
    "ace-step-xl-sft": ("fp16", "ACE-Step/Ace-Step1.5"),
}

# Layout converters a family already owns. The derived version reuses the hook
# against the downloaded dense pack instead of the generic in-place converter.
HOOK_DERIVED = {
    "ltx-2.5-distilled": {"type": "ltx25_ingest", "quantize": 4, "group_size": 64},
}

# Adapter cards: (old key, new keys). Adapters are never quantized.
ADAPTER_KEYS = {
    "z-image-turbo-fun-controlnet-union": ("8steps", ["fp16"]),
    "wan2.2-t2v-lightning-lora": ("v1217", ["fp16"]),
    "wan2.2-i2v-lightning-lora": ("v1022", ["fp16"]),
    "qwen-image-2512-lightning-lora": ("v4-bf16", ["bf16-4step"]),
    "qwen-image-edit-2511-lightning-lora": (
        None,
        ["bf16-4step", "bf16-8step"],
    ),
    "ltx25-msr-lora": ("v2", ["fp16"]),
    "minimax-h3-turbo-lora": ("v4-ema", ["fp16"]),
    "ace-step-xl-turbo": ("xl-turbo", ["fp16"]),
}

STANDARD_DTYPE_KEYS = {"fp16", "bf16", "fp32", "original", "fp8"}

# Models with no full-precision version at all: add the official dense source.
LONE_DENSE = {
    "longcat-avatar-1.5": (
        "bf16",
        "mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
        "models/Video/longcat-avatar-1.5-bf16",
    ),
}

# minimax-h3-fl2va had no dense entry: its DiT came from a pre-quantized repo.
# Rebuild the bundle from the official full-precision sources.
MINIMAX_FL2VA_DENSE = {
    "name": {"zh": "BF16 完整版", "en": "BF16 Full"},
    "size": "76GB",
    "min_unified_memory_gb": 64,
    "source": "huggingface",
    "source_type": "full",
    "kind": "dense",
    "local_path": "models/Video/minimax-h3-fl2va-bf16",
    "notes": {
        "zh": "官方 FL2VA 全精度 DiT 与双 VAE，配 Comfy-Org 截断 Qwen3-VL 文本编码器（H3 条件维度，含视觉塔）。",
        "en": "Official FL2VA full-precision DiT and VAEs with the Comfy-Org truncated Qwen3-VL text encoder (H3 conditioning layout with vision).",
    },
    "bundle_repos": [
        {
            "repo_id": "MiniMaxAI/MiniMax-H3",
            "local_path": "models/Video/minimax-h3-fl2va-bf16",
            "name": "FL2VA full",
            "source": "huggingface",
            "size": "65GB",
            "allow_patterns": [
                "FL2VA/transformer/**",
                "FL2VA/video_vae/**",
                "FL2VA/audio_vae/**",
                "FL2VA/processor/**",
                "FL2VA/tokenizer/**",
                "FL2VA/model_index.json",
            ],
        },
        {
            "repo_id": "Comfy-Org/MiniMax-H3",
            "local_path": "models/Video/minimax-h3-fl2va-bf16",
            "name": {
                "zh": "H3 截断文本编码器",
                "en": "H3 truncated text encoder",
            },
            "source": "huggingface",
            "size": "52GB",
            "allow_patterns": [
                "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
            ],
        },
    ],
}


def dense_stem(local_path: str) -> str:
    return local_path.rsplit("-mlx-q4", 1)[0].rsplit("-mlx-q8", 1)[0]


def normalize_dense(model_id, entry, key):
    """Strip quantization leftovers from a dense entry, keep download fields."""
    body = json.loads(json.dumps(entry))
    body.pop("install_hooks", None)
    body.pop("quantization", None)
    body.pop("quantized", None)
    body["source_type"] = "full"
    body["kind"] = "dense"
    body["default"] = False
    body["name"] = body.get("name") or {"zh": "完整版", "en": "Full"}
    if model_id in DENSE_OVERRIDE:
        _, repo = DENSE_OVERRIDE[model_id]
        body["repo_id"] = repo
        for repo_row in body.get("bundle_repos", []):
            pass
    local = body.get("local_path", "")
    for stale in ("-mlx-q4", "-mlx-q8", "-mlx-q4-dmd", "-mlx-q8-dmd"):
        if local.endswith(stale):
            local = local[: -len(stale)]
    body["local_path"] = f"{local}-{key}" if not local.endswith(f"-{key}") else local
    return body


def derived_version(base_key, recipe, local_path, source, hook, memory_gb=None):
    body = {
        "name": {"zh": "4-bit 混合量化版", "en": "4-bit Mixed Quantized"},
        "default": True,
        "source_type": "derived",
        "kind": "derived",
        "base": base_key,
        "recipe": recipe,
        "local_path": local_path,
        "source": source,
        "quantization": {"bits": 4, "scheme": "mlx_affine"},
    }
    if hook:
        body["install_hooks"] = [dict(hook, source=".")]
    if memory_gb:
        body["min_unified_memory_gb"] = max(8, int(memory_gb * 0.45))
    return body


def main():
    with open(REGISTRY, encoding="utf-8") as fh:
        doc = json.load(fh)
    models = doc["models"]
    report = []

    for model_id, model in models.items():
        versions = model.get("distribution", {}).get("versions")
        if not versions:
            continue
        source = model.get("catalog", {}).get("source", "modelscope")

        if model_id in ADAPTER_KEYS:
            old, new_keys = ADAPTER_KEYS[model_id]
            items = list(versions.items())
            if old is None:
                # rename each version in order of appearance
                rebuilt = {}
                for (old_key, body), new_key in zip(items, new_keys):
                    body = json.loads(json.dumps(body))
                    body["kind"] = "dense"
                    body["source_type"] = "full"
                    body["default"] = False
                    body.setdefault("source", source)
                    rebuilt[new_key] = body
                versions.clear()
                versions.update(rebuilt)
                report.append(f"{model_id}: -> {new_keys}")
                continue
            body = versions.pop(old)
            body = json.loads(json.dumps(body))
            body["kind"] = "dense"
            body["source_type"] = "full"
            body["default"] = False
            body.setdefault("source", source)
            versions.clear()
            for new_key in new_keys:
                row = json.loads(json.dumps(body))
                if len(new_keys) > 1:
                    row["name"] = {
                        "zh": f"{'4' if '4step' in new_key else '8'} 步 BF16",
                        "en": f"{'4' if '4step' in new_key else '8'}-Step BF16",
                    }
                versions[new_key] = row
            report.append(f"{model_id}: {old} -> {new_keys}")
            continue

        recipe = RECIPE.get(model_id)
        if not recipe:
            continue

        # ---- pick the dense entry
        full_keys = [
            key
            for key, body in versions.items()
            if body.get("source_type") == "full"
        ]
        if model_id == "minimax-h3-fl2va":
            dense_key = "bf16"
            dense = json.loads(json.dumps(MINIMAX_FL2VA_DENSE))
            dense["source_type"] = "full"
            dense["kind"] = "dense"
            dense["default"] = False
        elif model_id in DENSE_OVERRIDE:
            dense_key, _ = DENSE_OVERRIDE[model_id]
            template = full_keys[0] if full_keys else next(iter(versions))
            dense = normalize_dense(model_id, versions[template], dense_key)
        elif full_keys:
            dense_key = full_keys[0]
            dense = normalize_dense(model_id, versions[full_keys[0]], dense_key)
        elif model_id in LONE_DENSE:
            dense_key, repo, local = LONE_DENSE[model_id]
            dense = {
                "name": {"zh": "完整版", "en": "Full"},
                "default": False,
                "repo_id": repo,
                "local_path": local,
                "source_type": "full",
                "kind": "dense",
                "source": source,
            }
        else:
            raise SystemExit(
                f"{model_id}: no full-precision version to use as the dense base"
            )

        dense_local = dense["local_path"]
        removed = sorted(set(versions) - {dense_key})
        versions.clear()
        versions[dense_key] = dense

        hook = HOOK_DERIVED.get(model_id)
        versions[DERIVED_KEY] = derived_version(
            dense_key,
            recipe,
            f"{dense_local}-{DERIVED_KEY}",
            dense.get("source", source),
            hook,
            dense.get("min_unified_memory_gb"),
        )
        report.append(f"{model_id}: dense={dense_key} removed={removed}")

    with open(REGISTRY, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    for row in report:
        print(row)
    print(f"\n{len(report)} models migrated")


if __name__ == "__main__":
    sys.exit(main())
