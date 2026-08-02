"""Optional VLM audits for LoRA dataset health and training progress previews."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Literal

from backend.engine.llm.vision_mlx import analyze_image_file

Severity = Literal["info", "warning", "error"]
AuditKind = Literal["concept", "style"]

MAX_VLM_DATASET_IMAGES = 48

_CONCEPT_DATASET_AUDIT_INSTRUCTION = """You audit ONE image for DreamBooth **person/face** LoRA training at **512×512 cover crop**.
Score harshly — attractive photos can still be useless for face LoRA if the face is tiny after crop.

Score 1-2 when: full-body/distant shot, landscape/wide framing with small face, short edge under ~800px, multiple people, heavy blur/occlusion/watermark.
Score 4-5 only for: single person, face large and sharp (bust/headshot/close portrait).

Output EXACTLY this format (no markdown, no extra lines):
SCORE: 1-5
ISSUES: comma-separated tags from blurry,small_face,tiny_face_in_crop,landscape_framing,full_body,low_resolution,multiple_people,heavy_occlusion,wrong_framing,low_detail,watermark,text_overlay,good
REASON: one concise sentence (use Chinese if the photo is Chinese-centric)"""

_STYLE_DATASET_AUDIT_INSTRUCTION = """You audit ONE image for DreamBooth **style** LoRA training (artistic look, rendering, palette — NOT a specific person's identity).
Focus on: style clarity, visual consistency, composition, detail, cleanliness — do NOT penalize missing faces.
Output EXACTLY this format (no markdown, no extra lines):
SCORE: 1-5
ISSUES: comma-separated tags from blurry,inconsistent_style,cluttered,low_detail,watermark,text_overlay,off_theme,noisy,good
REASON: one concise sentence (use Chinese if appropriate)"""

_CONCEPT_PROGRESS_AUDIT_INSTRUCTION = """You audit a LoRA training progress preview for **person/concept** likeness.
Training caption/prompt: {progress_prompt}
Output EXACTLY this format (no markdown, no extra lines):
LIKENESS: 1-5
ISSUES: comma-separated tags from blurry,face_not_visible,wrong_subject,artifact_heavy,underfit,overfit,good
REASON: one concise sentence (use Chinese if appropriate)"""

_STYLE_PROGRESS_AUDIT_INSTRUCTION = """You audit a LoRA training progress preview for **style** transfer quality.
Training caption/prompt: {progress_prompt}
Output EXACTLY this format (no markdown, no extra lines):
STYLE_MATCH: 1-5
ISSUES: comma-separated tags from blurry,style_not_visible,wrong_palette,artifact_heavy,underfit,overfit,good
REASON: one concise sentence (use Chinese if appropriate)"""


def build_dataset_audit_instruction(audit_kind: str = "concept") -> str:
    if audit_kind == "style":
        return _STYLE_DATASET_AUDIT_INSTRUCTION
    return _CONCEPT_DATASET_AUDIT_INSTRUCTION


def build_progress_audit_instruction(progress_prompt: str, *, audit_kind: str = "concept") -> str:
    template = (
        _STYLE_PROGRESS_AUDIT_INSTRUCTION
        if audit_kind == "style"
        else _CONCEPT_PROGRESS_AUDIT_INSTRUCTION
    )
    prompt = (progress_prompt or "").strip()
    if not prompt:
        raise RuntimeError(
            "progress_prompt is required for VLM progress audit; set the training preview prompt first."
        )
    return template.format(progress_prompt=prompt)


def resolve_audit_paths(image_paths: list[Path], *, max_samples: int = 0) -> tuple[list[Path], bool]:
    """Return paths to audit. max_samples=0 means all images (capped)."""
    if not image_paths:
        return [], False
    if max_samples > 0:
        return sample_evenly(image_paths, max_samples), False
    if len(image_paths) <= MAX_VLM_DATASET_IMAGES:
        return list(image_paths), False
    return list(image_paths[:MAX_VLM_DATASET_IMAGES]), True


def pick_progress_preview_paths(preview_paths: list[Path]) -> list[Path]:
    if not preview_paths:
        return []
    sorted_paths = sorted(preview_paths)
    picks = [sorted_paths[0]]
    if len(sorted_paths) > 1:
        picks.append(sorted_paths[-1])
    return picks

_ISSUE_TAG_RE = re.compile(r"[a-z_]+")


def _hint(code: str, severity: Severity, **params: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "params": params, "source": "vlm"}


def parse_vlm_audit_output(text: str) -> dict[str, Any]:
    """Parse SCORE/LIKENESS + ISSUES + REASON blocks from VLM output."""
    fields: dict[str, str] = {}
    for line in (text or "").strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        fields[key.strip().upper()] = val.strip()

    score_raw = fields.get("SCORE") or fields.get("LIKENESS") or fields.get("STYLE_MATCH") or ""
    score_match = re.search(r"(\d+(?:\.\d+)?)", score_raw)
    score = float(score_match.group(1)) if score_match else None

    issues_raw = fields.get("ISSUES", "")
    issues = {
        tag
        for tag in _ISSUE_TAG_RE.findall(issues_raw.lower())
        if tag and tag != "good"
    }
    reason = fields.get("REASON", "").strip()
    return {"score": score, "issues": sorted(issues), "reason": reason, "raw": text.strip()}


def sample_evenly(items: list[Any], max_samples: int) -> list[Any]:
    if not items or max_samples <= 0:
        return []
    if len(items) <= max_samples:
        return list(items)
    if max_samples == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (max_samples - 1)
    return [items[int(round(i * step))] for i in range(max_samples)]


def _evenly_sample(items: list[Any], max_samples: int) -> list[Any]:
    return sample_evenly(items, max_samples)


def compile_dataset_vlm_report(
    sample_paths: list[Path],
    texts: list[str],
    *,
    file_keys: list[str] | None = None,
    audit_kind: str = "concept",
    truncated: bool = False,
    total_images: int = 0,
) -> dict[str, Any]:
    """Build dataset VLM audit dict from subprocess batch texts."""
    keys = file_keys or [p.name for p in sample_paths]
    vlm_by_key: dict[str, dict[str, Any]] = {}
    for idx, (path, raw) in enumerate(zip(sample_paths, texts, strict=False)):
        parsed = parse_vlm_audit_output(raw)
        key = keys[idx] if idx < len(keys) else path.name
        vlm_by_key[key] = parsed
    return _finalize_dataset_audit_report(
        vlm_by_key=vlm_by_key,
        audit_kind=audit_kind,
        truncated=truncated,
        vlm_sample_count=len(sample_paths),
        total_images=total_images or len(sample_paths),
    )


def compile_portrait_dataset_audit(
    all_paths: list[Path],
    vlm_paths: list[Path],
    vlm_texts: list[str],
    *,
    all_file_keys: list[str] | None = None,
    vlm_file_keys: list[str] | None = None,
    audit_kind: str = "concept",
    truncated: bool = False,
    total_images: int = 0,
    training_resolution: tuple[int, int] = (512, 512),
) -> dict[str, Any]:
    """VLM on ``vlm_paths`` plus pixel heuristics on every ``all_paths`` (concept only)."""
    if audit_kind != "concept":
        keys = vlm_file_keys or [p.name for p in vlm_paths]
        return compile_dataset_vlm_report(
            vlm_paths,
            vlm_texts,
            file_keys=keys,
            audit_kind=audit_kind,
            truncated=truncated,
            total_images=total_images or len(all_paths),
        )

    from backend.engine.training.portrait_lora_suitability import (
        analyze_portrait_training_image,
        merge_vlm_and_heuristic_sample,
    )

    all_keys = all_file_keys or [p.name for p in all_paths]
    vlm_keys = vlm_file_keys or [p.name for p in vlm_paths]
    vlm_by_key: dict[str, dict[str, Any]] = {}
    for idx, (path, raw) in enumerate(zip(vlm_paths, vlm_texts, strict=False)):
        key = vlm_keys[idx] if idx < len(vlm_keys) else path.name
        vlm_by_key[key] = parse_vlm_audit_output(raw)

    heuristic_by_key: dict[str, dict[str, Any]] = {}
    for path, key in zip(all_paths, all_keys, strict=False):
        heuristic_by_key[key] = analyze_portrait_training_image(
            path,
            training_resolution=training_resolution,
        )

    merged_by_key: dict[str, dict[str, Any]] = {}
    for key in all_keys:
        merged_by_key[key] = merge_vlm_and_heuristic_sample(
            file_key=key,
            vlm_parsed=vlm_by_key.get(key),
            heuristic=heuristic_by_key.get(key),
        )

    return _finalize_dataset_audit_report(
        vlm_by_key=merged_by_key,
        audit_kind=audit_kind,
        truncated=truncated,
        vlm_sample_count=len(vlm_paths),
        total_images=total_images or len(all_paths),
        heuristic_scored_count=len(all_paths),
    )


def _finalize_dataset_audit_report(
    *,
    vlm_by_key: dict[str, dict[str, Any]],
    audit_kind: str,
    truncated: bool,
    vlm_sample_count: int,
    total_images: int,
    heuristic_scored_count: int = 0,
) -> dict[str, Any]:
    samples = list(vlm_by_key.values())
    issue_counts: dict[str, int] = {}
    scores: list[float] = []
    unsuitable = 0
    for sample in samples:
        score = sample.get("score")
        if score is not None:
            scores.append(float(score))
            if float(score) < 3.0:
                unsuitable += 1
        for tag in sample.get("issues") or []:
            issue_counts[tag] = issue_counts.get(tag, 0) + 1

    hints = _dataset_vlm_hints(
        samples,
        scores,
        issue_counts,
        audit_kind=audit_kind,
        unsuitable_count=unsuitable,
        total_scored=len(samples),
    )
    if truncated and total_images > vlm_sample_count:
        hints.insert(
            0,
            _hint(
                "vlm_truncated",
                "warning",
                audited=vlm_sample_count,
                total=total_images,
                cap=MAX_VLM_DATASET_IMAGES,
            ),
        )
    if heuristic_scored_count > vlm_sample_count and audit_kind == "concept":
        hints.insert(
            1 if truncated and total_images > vlm_sample_count else 0,
            _hint(
                "vlm_heuristic_all_images",
                "info",
                heuristic_count=heuristic_scored_count,
                vlm_count=vlm_sample_count,
            ),
        )

    avg_score = sum(scores) / len(scores) if scores else None
    return {
        "audit_kind": audit_kind,
        "samples": samples,
        "avg_score": round(avg_score, 2) if avg_score is not None else None,
        "hints": hints,
        "audited_count": len(samples),
        "vlm_sample_count": vlm_sample_count,
        "heuristic_scored_count": heuristic_scored_count or len(samples),
        "unsuitable_count": unsuitable,
    }


def compile_progress_vlm_report(
    sample_paths: list[Path],
    texts: list[str],
    *,
    audit_kind: str = "concept",
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    scores: list[float] = []
    issue_counts: dict[str, int] = {}
    for path, raw in zip(sample_paths, texts, strict=False):
        parsed = parse_vlm_audit_output(raw)
        parsed["file"] = path.name
        samples.append(parsed)
        if parsed.get("score") is not None:
            scores.append(float(parsed["score"]))
        for tag in parsed.get("issues") or []:
            issue_counts[tag] = issue_counts.get(tag, 0) + 1
    hints = _progress_vlm_hints(samples, scores, issue_counts, audit_kind=audit_kind)
    avg = sum(scores) / len(scores) if scores else None
    return {
        "audit_kind": audit_kind,
        "samples": samples,
        "avg_likeness": round(avg, 2) if avg is not None else None,
        "hints": hints,
    }


def audit_dataset_images_with_vlm(
    image_paths: list[Path],
    *,
    model_dir: Path,
    max_samples: int = 4,
    analyze_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Run VLM portrait audit on a spread of dataset images."""
    analyze = analyze_fn or (
        lambda path, instruction: analyze_image_file(
            path, model_dir, instruction=instruction, max_tokens=200, temperature=0.2
        )
    )
    targets = sample_evenly(image_paths, max_samples)
    samples: list[dict[str, Any]] = []
    issue_counts: dict[str, int] = {}
    scores: list[float] = []

    for path in targets:
        if not path.is_file():
            continue
        raw = analyze(path, build_dataset_audit_instruction("concept"))
        parsed = parse_vlm_audit_output(raw)
        parsed["file"] = path.name
        samples.append(parsed)
        if parsed.get("score") is not None:
            scores.append(float(parsed["score"]))
        for tag in parsed.get("issues") or []:
            issue_counts[tag] = issue_counts.get(tag, 0) + 1

    hints = _dataset_vlm_hints(samples, scores, issue_counts)
    avg_score = sum(scores) / len(scores) if scores else None
    return {
        "samples": samples,
        "avg_score": round(avg_score, 2) if avg_score is not None else None,
        "hints": hints,
    }


def _dataset_vlm_hints(
    samples: list[dict[str, Any]],
    scores: list[float],
    issue_counts: dict[str, int],
    *,
    audit_kind: str = "concept",
    unsuitable_count: int = 0,
    total_scored: int = 0,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    if not samples:
        hints.append(_hint("vlm_no_samples", "warning"))
        return hints

    if scores:
        avg = sum(scores) / len(scores)
        if audit_kind == "style":
            if avg < 2.5:
                hints.append(_hint("vlm_low_style_score", "error", avg=round(avg, 1), count=len(scores)))
            elif avg < 3.5:
                hints.append(_hint("vlm_low_style_score", "warning", avg=round(avg, 1), count=len(scores)))
            elif avg >= 4.0:
                hints.append(_hint("vlm_good_style_score", "info", avg=round(avg, 1), count=len(scores)))
        else:
            if avg < 2.5:
                hints.append(_hint("vlm_low_portrait_score", "error", avg=round(avg, 1), count=len(scores)))
            elif avg < 3.5:
                hints.append(_hint("vlm_low_portrait_score", "warning", avg=round(avg, 1), count=len(scores)))
            elif avg >= 4.0:
                hints.append(_hint("vlm_good_portrait_score", "info", avg=round(avg, 1), count=len(scores)))

    style_error_tags = {"inconsistent_style", "off_theme", "cluttered", "noisy"}
    concept_error_tags = {
        "small_face",
        "tiny_face_in_crop",
        "landscape_framing",
        "full_body",
        "low_resolution",
        "multiple_people",
        "heavy_occlusion",
        "low_detail",
        "face_soft_in_crop",
    }

    if audit_kind == "concept" and total_scored > 0 and unsuitable_count > 0:
        severity: Severity = "error" if unsuitable_count >= max(2, total_scored // 4) else "warning"
        hints.append(
            _hint(
                "vlm_training_unsuitable",
                severity,
                count=unsuitable_count,
                total=total_scored,
            )
        )

    for tag, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        code = f"vlm_{tag}"
        severity: Severity = "warning"
        if audit_kind == "style":
            if tag in style_error_tags and count >= 2:
                severity = "error"
        elif tag in concept_error_tags and count >= 2:
            severity = "error"
        hints.append(_hint(code, severity, count=count, samples=len(samples)))

    weak = [s for s in samples if s.get("reason") and (s.get("score") or 5) < 3.5]
    if weak:
        hints.append(
            _hint(
                "vlm_per_image_notes",
                "info",
                count=len(weak),
                notes="; ".join(f"{s.get('file')}: {s['reason']}" for s in weak[:5]),
            )
        )
    return hints


def audit_progress_previews_with_vlm(
    preview_paths: list[Path],
    *,
    progress_prompt: str,
    model_dir: Path,
    analyze_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Audit first/last training progress preview images."""
    analyze = analyze_fn or (
        lambda path, instruction: analyze_image_file(
            path, model_dir, instruction=instruction, max_tokens=200, temperature=0.2
        )
    )
    if not preview_paths:
        return {"samples": [], "hints": [_hint("vlm_no_progress_images", "warning")]}

    sorted_paths = sorted(preview_paths)
    picks = [sorted_paths[0]]
    if len(sorted_paths) > 1:
        picks.append(sorted_paths[-1])

    instruction = build_progress_audit_instruction(progress_prompt, audit_kind="concept")
    samples: list[dict[str, Any]] = []
    scores: list[float] = []
    issue_counts: dict[str, int] = {}

    for path in picks:
        raw = analyze(path, instruction)
        parsed = parse_vlm_audit_output(raw)
        parsed["file"] = path.name
        samples.append(parsed)
        if parsed.get("score") is not None:
            scores.append(float(parsed["score"]))
        for tag in parsed.get("issues") or []:
            issue_counts[tag] = issue_counts.get(tag, 0) + 1

    hints = _progress_vlm_hints(samples, scores, issue_counts)
    avg = sum(scores) / len(scores) if scores else None
    return {
        "samples": samples,
        "avg_likeness": round(avg, 2) if avg is not None else None,
        "hints": hints,
    }


def _progress_vlm_hints(
    samples: list[dict[str, Any]],
    scores: list[float],
    issue_counts: dict[str, int],
    *,
    audit_kind: str = "concept",
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    if scores:
        avg = sum(scores) / len(scores)
        if audit_kind == "style":
            if avg < 2.5:
                hints.append(_hint("vlm_progress_low_style_match", "error", avg=round(avg, 1)))
            elif avg < 3.5:
                hints.append(_hint("vlm_progress_low_style_match", "warning", avg=round(avg, 1)))
            elif avg >= 4.0:
                hints.append(_hint("vlm_progress_good_style_match", "info", avg=round(avg, 1)))
        else:
            if avg < 2.5:
                hints.append(_hint("vlm_progress_low_likeness", "error", avg=round(avg, 1)))
            elif avg < 3.5:
                hints.append(_hint("vlm_progress_low_likeness", "warning", avg=round(avg, 1)))
            elif avg >= 4.0:
                hints.append(_hint("vlm_progress_good_likeness", "info", avg=round(avg, 1)))

    for tag, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        code = f"vlm_progress_{tag}"
        severity: Severity = "warning"
        if tag in {"underfit", "face_not_visible", "wrong_subject"}:
            severity = "error" if count >= 1 else "warning"
        hints.append(_hint(code, severity, count=count))

    notes = [s["reason"] for s in samples if s.get("reason")]
    if notes:
        hints.append(_hint("vlm_progress_notes", "info", notes="; ".join(notes[:2])))
    return hints


def merge_vlm_hints(
    base: dict[str, Any],
    vlm: dict[str, Any],
    *,
    downgrade: bool = True,
) -> dict[str, Any]:
    """Merge VLM hints into a health or training quality report."""
    merged = dict(base)
    existing = list(merged.get("hints") or [])
    vlm_hints = list(vlm.get("hints") or [])
    merged["hints"] = existing + vlm_hints
    merged["vlm"] = {
        "audit_kind": vlm.get("audit_kind"),
        "avg_score": vlm.get("avg_score") or vlm.get("avg_likeness"),
        "audited_count": vlm.get("audited_count") or len(vlm.get("samples") or []),
        "samples": vlm.get("samples") or [],
    }

    if downgrade and vlm_hints:
        level = str(merged.get("level") or "fair")
        has_error = any(h.get("severity") == "error" for h in vlm_hints)
        has_warning = any(h.get("severity") == "warning" for h in vlm_hints)
        if has_error:
            merged["level"] = "poor"
        elif has_warning and level == "good":
            merged["level"] = "fair"
        score = int(merged.get("score") or 100)
        merged["score"] = max(0, score - (25 if has_error else 10 if has_warning else 0))

    return merged


def collect_dataset_image_paths(workspace_root: Path, dataset_id: str) -> list[Path]:
    from backend.engine.training import dataset_store

    ds = dataset_store.get_dataset(workspace_root, dataset_id)
    root = dataset_store.datasets_root(workspace_root) / dataset_id
    paths: list[Path] = []
    for img in ds.get("images") or []:
        if not img.get("exists"):
            continue
        file_rel = str(img.get("file") or "")
        if not file_rel:
            continue
        path = root / file_rel
        if path.is_file():
            paths.append(path)
    return paths


def resolve_dataset_audit_kind(workspace_root: Path, dataset_id: str, override: str | None = None) -> str:
    if override in ("concept", "style"):
        return override
    from backend.engine.training import dataset_store

    try:
        ds = dataset_store.get_dataset(workspace_root, dataset_id)
        kind = str(ds.get("kind") or "concept").strip().lower()
        return kind if kind in ("concept", "style") else "concept"
    except FileNotFoundError:
        return "concept"
