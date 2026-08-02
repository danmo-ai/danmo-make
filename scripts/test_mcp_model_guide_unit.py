#!/usr/bin/env python3
"""Unit tests for MCP model cards (sort + simplified params)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.mcp.model_guide import (
    enrich_model_list,
    normalize_list_action,
    summarize_model,
)


def main() -> int:
    assert normalize_list_action("create") == "generate"
    assert normalize_list_action("create", media="audio") == "create_music"
    assert normalize_list_action("rewrite") == "edit"

    cfg_old = {
        "name": {"en": "Old"},
        "media": "image",
        "type": "diffusion",
        "commercial_use_allowed": True,
        "successor": "new-model",
        "actions": {"create": {}},
        "versions": {"fp16": {"default": True}},
        "parameters": {
            "resolution_presets": {"default": "1024x1024", "options": ["1024x1024"]},
        },
    }
    cfg_base = {
        "name": {"en": "Base"},
        "media": "image",
        "type": "diffusion",
        "commercial_use_allowed": True,
        "actions": {"create": {}, "rewrite": {}},
        "versions": {"fp16": {"default": True}},
        "parameters": {
            "resolution_presets": {"default": "1024x1024", "options": ["512x512", "1024x1024"]},
            "steps": {"type": "int", "default": 20, "min": 1, "max": 50},
        },
    }
    cfg_distill = {
        "name": {"en": "Distill"},
        "media": "image",
        "type": "diffusion",
        "commercial_use_allowed": True,
        "distilled_from": "base-model",
        "actions": {"create": {}},
        "versions": {
            "fp16": {"default": True},
            "int4": {"quantization": {"bits": 4}},
        },
        "parameters": {
            "resolution_presets": {"default": "768x1280", "options": ["768x1280"]},
        },
    }
    cfg_nc = {
        "name": {"en": "NonCommercial"},
        "media": "image",
        "type": "diffusion",
        "commercial_use_allowed": False,
        "distilled_from": "x",
        "actions": {"create": {}},
        "versions": {"int4": {"quantization": {"bits": 4}}},
        "parameters": {
            "resolution_presets": {"default": "1024x1024", "options": ["1024x1024"]},
        },
    }
    cfg_edit = {
        "name": {"en": "EditOnly"},
        "media": "image",
        "type": "diffusion",
        "commercial_use_allowed": True,
        "actions": {"rewrite": {}},
        "versions": {"fp16": {"default": True}},
        "parameters": {
            "resolution_presets": {"default": "1024x1024", "options": ["1024x1024"]},
        },
    }
    cfg_lora = {
        "name": {"en": "StarFace"},
        "media": "image",
        "type": "lora",
        "category": "loras",
        "commercial_use_allowed": True,
        "actions": {},
        "versions": {"fp16": {"default": True}},
    }

    card = summarize_model(
        "distill",
        index_row={"media": "image", "installed": True, "actions": ["generate"]},
        config=cfg_distill,
        full=True,
    )
    assert card["parameters"]["size"]["options"] == ["768x1280"]
    assert card["preferred_version"] == "int4"
    assert card["actions"] == ["generate"]
    assert "lora_support" not in card.get("parameters", {})

    out = enrich_model_list(
        {
            "models": {
                "old-model": {"media": "image", "installed": True, "actions": ["generate"]},
                "base-model": {
                    "media": "image",
                    "installed": True,
                    "actions": ["edit", "generate"],
                },
                "distill-model": {
                    "media": "image",
                    "installed": True,
                    "actions": ["generate"],
                },
                "nc-model": {"media": "image", "installed": True, "actions": ["generate"]},
                "edit-only": {"media": "image", "installed": True, "actions": ["edit"]},
                "lora-adapter": {"media": "image", "installed": True, "actions": []},
            }
        },
        {
            "old-model": cfg_old,
            "base-model": cfg_base,
            "distill-model": cfg_distill,
            "nc-model": cfg_nc,
            "edit-only": cfg_edit,
            "lora-adapter": cfg_lora,
        },
        require_action="generate",
    )
    ids = [c["id"] for c in out["image"]]
    assert "lora-adapter" not in ids
    assert "edit-only" not in ids
    # commercial first → nc last; among commercial: no successor, distill+smaller quant wins
    assert ids[-1] == "nc-model"
    assert ids[0] == "distill-model"
    assert "old-model" in ids
    assert ids.index("old-model") > ids.index("distill-model")
    assert all("generate" in c["actions"] for c in out["image"])
    print("ok: mcp model_guide sort+simplified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
