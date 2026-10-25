"""Heuristic dataset health + post-training quality hints for LoRA DreamBooth."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from backend.engine.training import dataset_store

Level = Literal["good", "fair", "poor"]
Severity = Literal["info", "warning", "error"]

# Tunable thresholds (concept LoRA portraits; aligned with yz vs cyq observations).
_MIN_IMAGES_IDEAL = 10
_MIN_IMAGES_WARN = 6
_SHORT_EDGE_SMALL = 600
_SHORT_EDGE_TINY = 512
_MEDIAN_SHORT_EDGE_WARN = 720
_MEDIAN_SHORT_EDGE_POOR = 560
_VAL_GAP_WARN = 0.08
_VAL_REGRESSION_WARN = 0.08
# dHash Hamming distance at or below which two photos count as the same shot (burst / re-export).
_NEAR_DUP_HAMMING = 6


def _hint(code: str, severity: Severity, **params: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "params": params}


def _level_from_score(score: int, *, force_poor: bool = False, force_fair: bool = False) -> Level:
    if force_poor or score < 45:
        return "poor"
    if force_fair or score < 72:
        return "fair"
    return "good"


def _suspicious_filename(name: str) -> bool:
    lower = name.lower()
    return ".heic" in lower or ".heif" in lower or lower.endswith(".png.png")


def _probe_image(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    try:
        # Header size only. Decoding every portrait here stacked with YuNet and dHash
        # and stalled the API on large identity sets.
        with Image.open(path) as img:
            w, h = img.size
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}
    short_edge = min(w, h)
    return {"ok": True, "width": w, "height": h, "short_edge": short_edge}


def _dhash(path: Path, *, size: int = 8) -> int:
    """Difference hash (row-adjacent gradients on a 9×8 grayscale thumbnail)."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((size + 1, size), Image.BILINEAR)
        px = list(gray.tobytes())
    bits = 0
    for y in range(size):
        row = px[y * (size + 1) : (y + 1) * (size + 1)]
        for x in range(size):
            bits = (bits << 1) | (1 if row[x] > row[x + 1] else 0)
    return bits


def find_near_duplicates(paths: list[Path], *, max_distance: int = _NEAR_DUP_HAMMING) -> list[tuple[str, str]]:
    """Pairs of files whose dHash differs by at most ``max_distance`` bits (same shot / burst)."""
    hashes: list[tuple[str, int]] = []
    for p in paths:
        try:
            h = _dhash(p)
        except Exception:
            continue
        # Near-uniform hashes (flat backgrounds, solid colours) collide trivially; skip them.
        ones = bin(h).count("1")
        if ones < 8 or ones > 56:
            continue
        hashes.append((p.name, h))
    pairs: list[tuple[str, str]] = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i][1] ^ hashes[j][1]).count("1") <= max_distance:
                pairs.append((hashes[i][0], hashes[j][0]))
    return pairs


def analyze_dataset_health(
    workspace_root: Path,
    dataset_id: str,
    *,
    face_audit: bool = True,
    training_resolution: tuple[int, int] = (512, 512),
) -> dict[str, Any]:
    """Scan dataset images for resolution / integrity / face-framing / caption issues before training."""
    ds = dataset_store.get_dataset(workspace_root, dataset_id)
    path = dataset_store.datasets_root(workspace_root) / dataset_id
    rows = ds.get("images") or []
    kind = str(ds.get("kind") or "concept").strip().lower()
    trigger = str(ds.get("trigger_word") or "").strip()

    missing_count = 0
    broken_count = 0
    suspicious_count = 0
    small_600 = 0
    small_512 = 0
    empty_caption = 0
    short_edges: list[int] = []
    readable_paths: list[Path] = []
    caption_pairs: list[tuple[Path, str]] = []

    for row in rows:
        file_rel = str(row.get("file") or "")
        caption = str(row.get("prompt") or "").strip()
        if not caption:
            empty_caption += 1
        if _suspicious_filename(file_rel):
            suspicious_count += 1
        img_path = path / file_rel
        probe = _probe_image(img_path)
        if not probe.get("ok"):
            if probe.get("reason") == "missing":
                missing_count += 1
            else:
                broken_count += 1
            continue
        readable_paths.append(img_path)
        caption_pairs.append((img_path, caption))
        se = int(probe["short_edge"])
        short_edges.append(se)
        if se < _SHORT_EDGE_SMALL:
            small_600 += 1
        if se < _SHORT_EDGE_TINY:
            small_512 += 1

    total = len(rows)
    readable = len(short_edges)
    median_short = 0
    min_short = 0
    if short_edges:
        sorted_edges = sorted(short_edges)
        mid = len(sorted_edges) // 2
        median_short = sorted_edges[mid] if len(sorted_edges) % 2 else (
            (sorted_edges[mid - 1] + sorted_edges[mid]) // 2
        )
        min_short = sorted_edges[0]

    hints: list[dict[str, Any]] = []
    score = 100
    force_poor = False
    force_fair = False

    if missing_count:
        hints.append(_hint("missing_images", "error", count=missing_count, total=total))
        score -= min(40, missing_count * 15)
        force_poor = True
    if broken_count:
        hints.append(_hint("broken_images", "error", count=broken_count, total=total))
        score -= min(35, broken_count * 12)
        force_poor = True
    if suspicious_count:
        hints.append(_hint("suspicious_filenames", "warning", count=suspicious_count))
        score -= min(20, suspicious_count * 6)
        force_fair = True

    if total < _MIN_IMAGES_WARN:
        hints.append(_hint("too_few_images", "error", count=total, recommended=_MIN_IMAGES_IDEAL))
        score -= 25
        force_poor = True
    elif total < _MIN_IMAGES_IDEAL:
        hints.append(_hint("few_images", "warning", count=total, recommended=_MIN_IMAGES_IDEAL))
        score -= 10
        force_fair = True

    if readable and small_600:
        ratio = small_600 / readable
        if ratio >= 0.4 or small_600 >= 5:
            hints.append(
                _hint("many_small_600", "warning", count=small_600, total=readable, pct=round(ratio * 100))
            )
            score -= min(25, int(ratio * 30) + small_600)
            force_fair = True
    if readable and small_512:
        hints.append(_hint("many_small_512", "warning", count=small_512, total=readable))
        score -= min(20, small_512 * 4)
        if small_512 >= 3:
            force_fair = True

    if readable and median_short < _MEDIAN_SHORT_EDGE_POOR:
        hints.append(_hint("low_resolution_median", "error", median=median_short))
        score -= 25
        force_poor = True
    elif readable and median_short < _MEDIAN_SHORT_EDGE_WARN:
        hints.append(_hint("low_resolution_median", "warning", median=median_short))
        score -= 12
        force_fair = True

    if empty_caption:
        hints.append(_hint("empty_captions", "warning", count=empty_caption, total=total))
        score -= min(15, empty_caption * 3)

    faces_report: dict[str, Any] | None = None
    caption_stats: dict[str, Any] | None = None
    caption_mode_auto = ""
    if kind == "concept":
        if face_audit and readable_paths:
            from backend.engine.training.face_crop import audit_dataset_faces

            faces_report = audit_dataset_faces(
                readable_paths,
                project_root=workspace_root,
                resolution=training_resolution,
            )
            if not faces_report.get("available"):
                hints.append(_hint("face_detector_unavailable", "info"))
            else:
                counts = faces_report.get("counts") or {}
                none_count = int(counts.get("none", 0))
                tiny_count = int(counts.get("tiny", 0))
                crop_count = int(counts.get("crop", 0))
                multi_count = int(counts.get("multi", 0))
                if none_count:
                    # Every unusable identity image is a training blocker, not a score hint.
                    severity = "error"
                    hints.append(_hint("faces_not_detected", severity, count=none_count, total=readable))
                    score -= min(25, none_count * 8)
                    force_poor = True
                if tiny_count:
                    hints.append(
                        _hint(
                            "faces_too_small",
                            "error",
                            count=tiny_count,
                            total=readable,
                            px=int(faces_report.get("tiny_face_px") or 0),
                        )
                    )
                    score -= min(20, tiny_count * 5)
                    force_poor = True
                if multi_count:
                    hints.append(_hint("multiple_faces", "error", count=multi_count, total=readable))
                    score -= min(12, multi_count * 4)
                    force_poor = True
                if int(counts.get("error", 0)):
                    hints.append(_hint("faces_not_detected", "error", count=int(counts["error"]), total=readable))
                    force_poor = True
                if crop_count:
                    hints.append(_hint("faces_auto_crop", "info", count=crop_count, total=readable))

        dup_pairs = find_near_duplicates(readable_paths) if readable_paths else []
        if dup_pairs:
            hints.append(_hint("near_duplicates", "warning", count=len(dup_pairs)))
            score -= min(12, len(dup_pairs) * 3)

        if trigger and caption_pairs:
            caption_stats = dataset_store.concept_caption_stats(caption_pairs, trigger=trigger)
            caption_mode_auto = dataset_store.detect_caption_mode(
                caption_pairs, dataset_meta={"kind": kind, "trigger_word": trigger}
            )
            if caption_stats["missing_trigger"]:
                hints.append(
                    _hint(
                        "captions_missing_trigger",
                        "warning",
                        count=caption_stats["missing_trigger"],
                        total=caption_stats["non_empty"],
                        trigger=trigger,
                    )
                )
                score -= min(15, caption_stats["missing_trigger"] * 3)
            if caption_stats["long"]:
                hints.append(
                    _hint("captions_long", "warning", count=caption_stats["long"], total=caption_stats["non_empty"])
                )
                score -= min(10, caption_stats["long"] * 2)
            trigger_only = int(caption_stats.get("trigger_only") or 0)
            if trigger_only >= 3 and caption_stats["non_empty"] and trigger_only * 2 >= caption_stats["non_empty"]:
                hints.append(
                    _hint(
                        "captions_trigger_only",
                        "warning",
                        count=trigger_only,
                        total=caption_stats["non_empty"],
                    )
                )
                score -= min(12, trigger_only)
                force_fair = True
            if dataset_store.trigger_looks_like_personal_name(trigger):
                hints.append(_hint("trigger_looks_like_name", "info", trigger=trigger))
            if caption_mode_auto == "per_image":
                hints.append(_hint("caption_mode_per_image", "info", count=caption_stats["anchored_short"]))
            elif caption_stats["unique"] > 1 and not caption_stats["long"] and not caption_stats["missing_trigger"]:
                hints.append(
                    _hint(
                        "caption_mode_unified_small",
                        "info",
                        count=total,
                        recommended=dataset_store.CONCEPT_PER_IMAGE_MIN_IMAGES,
                    )
                )
        elif not trigger:
            hints.append(_hint("missing_trigger_word", "error"))
            score -= 20
            force_poor = True

    blocking = [h for h in hints if h["severity"] != "info"]
    if not blocking and total >= _MIN_IMAGES_IDEAL and readable == total:
        hints.append(_hint("dataset_healthy", "info", count=total, median=median_short))

    score = max(0, min(100, score))
    level = _level_from_score(score, force_poor=force_poor, force_fair=force_fair)

    stats: dict[str, Any] = {
        "image_count": total,
        "readable_count": readable,
        "missing_count": missing_count,
        "broken_count": broken_count,
        "suspicious_filename_count": suspicious_count,
        "small_600_count": small_600,
        "small_512_count": small_512,
        "empty_caption_count": empty_caption,
        "median_short_edge": median_short,
        "min_short_edge": min_short,
    }
    if faces_report and faces_report.get("available"):
        for key, value in (faces_report.get("counts") or {}).items():
            stats[f"face_{key}_count"] = int(value)
    if caption_stats:
        stats["caption_missing_trigger_count"] = caption_stats["missing_trigger"]
        stats["caption_long_count"] = caption_stats["long"]
        stats["caption_anchored_short_count"] = caption_stats["anchored_short"]
        stats["caption_trigger_only_count"] = int(caption_stats.get("trigger_only") or 0)

    report: dict[str, Any] = {
        "level": level,
        "score": score,
        "kind": kind,
        "stats": stats,
        "hints": hints,
    }
    if caption_mode_auto:
        report["caption_mode_auto"] = caption_mode_auto
    if faces_report is not None:
        report["faces"] = {
            "available": bool(faces_report.get("available")),
            "reason": faces_report.get("reason") or "",
            "resolution": list(training_resolution),
            "rows": faces_report.get("rows") or [],
            "counts": dict(faces_report.get("counts") or {}),
            "tiny_face_px": faces_report.get("tiny_face_px"),
        }
    return report


_FACE_COUNT_KEYS = ("crop", "keep", "tiny", "none", "error", "multi")
_FACE_HINT_CODES = {"faces_too_small", "multiple_faces", "faces_not_detected", "faces_auto_crop", "faces_user_skipped"}


def apply_training_skip_overlay(report: dict[str, Any], skipped_rels: list[str]) -> dict[str, Any]:
    """Drop user-skipped images from gate counts without forgetting why they failed.

    The cached audit still describes every file. Skipped rows keep their action and gain
    ``skipped`` so the wizard can offer delete or restore.
    """
    import copy

    out = copy.deepcopy(report)
    skipped_full = {str(rel).replace("\\", "/") for rel in skipped_rels}
    skipped_names = {Path(rel).name for rel in skipped_full}
    hints = [h for h in (out.get("hints") or []) if h.get("code") not in _FACE_HINT_CODES]
    faces = out.get("faces")
    if isinstance(faces, dict) and faces.get("rows"):
        counts = {key: 0 for key in _FACE_COUNT_KEYS}
        for row in faces["rows"]:
            file = str(row.get("file") or "")
            base = Path(file).name
            if file in skipped_full or base in skipped_names or f"images/{base}" in skipped_full:
                row["skipped"] = True
                continue
            action = str(row.get("action") or "")
            if action in counts:
                counts[action] += 1
            if row.get("multi") and action != "multi":
                counts["multi"] += 1
        faces["counts"] = counts
        readable = sum(counts[key] for key in ("crop", "keep", "tiny", "none", "error", "multi"))
        if faces.get("available"):
            if counts["none"]:
                hints.append(_hint("faces_not_detected", "error", count=counts["none"], total=readable))
            if counts["error"]:
                hints.append(_hint("faces_not_detected", "error", count=counts["error"], total=readable))
            if counts["tiny"]:
                hints.append(
                    _hint(
                        "faces_too_small",
                        "error",
                        count=counts["tiny"],
                        total=readable,
                        px=int(faces.get("tiny_face_px") or 0),
                    )
                )
            if counts["multi"]:
                hints.append(_hint("multiple_faces", "error", count=counts["multi"], total=readable))
            if counts["crop"]:
                hints.append(_hint("faces_auto_crop", "info", count=counts["crop"], total=readable))
        stats = out.setdefault("stats", {})
        if isinstance(stats, dict):
            for key, value in counts.items():
                stats[f"face_{key}_count"] = value
    if skipped_full:
        hints.append(_hint("faces_user_skipped", "info", count=len(skipped_full)))
    out["hints"] = hints
    out["skipped_images"] = list(skipped_rels)
    errors = [h for h in hints if h.get("severity") == "error"]
    warnings = [h for h in hints if h.get("severity") == "warning"]
    if not errors and out.get("level") == "poor":
        out["level"] = "fair" if warnings else "good"
    elif errors and any(h.get("code") in {"faces_too_small", "multiple_faces", "faces_not_detected"} for h in errors):
        out["level"] = "poor"
    return out


def _health_for_training(workspace_root: Path, dataset_id: str, report: dict[str, Any]) -> dict[str, Any]:
    return apply_training_skip_overlay(report, dataset_store.active_skipped_rels(workspace_root, dataset_id))


_HEALTH_CACHE_LOCK = threading.Lock()
_HEALTH_CACHE: dict[str, dict[str, Any]] = {}
_HEALTH_INFLIGHT: dict[str, threading.Event] = {}
_HEALTH_CACHE_LIMIT = 24


def analyze_dataset_health_shared(
    workspace_root: Path,
    dataset_id: str,
    *,
    training_resolution: tuple[int, int] = (512, 512),
) -> dict[str, Any]:
    """One in-flight audit per dataset snapshot; later callers reuse that result.

    The wizard fires health on select and again on “check faces”. Overlapping YuNet
    calls on the shared detector stalled the API.
    """
    pairs = dataset_store.load_dataset_pairs(workspace_root, dataset_id)
    meta = dataset_store._dataset_meta(workspace_root, dataset_id)
    fingerprint = dataset_store.training_dataset_fingerprint(pairs)
    key = (
        f"{dataset_id}|{fingerprint}|{int(training_resolution[0])}x{int(training_resolution[1])}"
        f"|{meta.get('kind')}|{meta.get('trigger_word')}"
    )
    with _HEALTH_CACHE_LOCK:
        cached = _HEALTH_CACHE.get(key)
        if cached is not None:
            return _health_for_training(workspace_root, dataset_id, cached)
        event = _HEALTH_INFLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _HEALTH_INFLIGHT[key] = event
            is_owner = True
        else:
            is_owner = False
    if not is_owner:
        if not event.wait(timeout=180):
            raise TimeoutError("Face preflight is still running for this dataset.")
        with _HEALTH_CACHE_LOCK:
            cached = _HEALTH_CACHE.get(key)
        if cached is None:
            raise RuntimeError("Face preflight failed; retry the check.")
        return _health_for_training(workspace_root, dataset_id, cached)
    try:
        report = analyze_dataset_health(
            workspace_root, dataset_id, training_resolution=training_resolution
        )
        with _HEALTH_CACHE_LOCK:
            _HEALTH_CACHE[key] = report
            while len(_HEALTH_CACHE) > _HEALTH_CACHE_LIMIT:
                _HEALTH_CACHE.pop(next(iter(_HEALTH_CACHE)))
        return _health_for_training(workspace_root, dataset_id, report)
    finally:
        with _HEALTH_CACHE_LOCK:
            _HEALTH_INFLIGHT.pop(key, None)
        event.set()


def _loss_points(loss_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in loss_history:
        try:
            step = float(row.get("step") or 0)
            loss = float(row.get("loss") or 0)
        except (TypeError, ValueError):
            continue
        if step <= 0 or loss <= 0:
            continue
        pt: dict[str, float] = {"step": step, "loss": loss}
        if row.get("val_loss") is not None:
            try:
                pt["val_loss"] = float(row["val_loss"])
            except (TypeError, ValueError):
                pass
        out.append(pt)
    return sorted(out, key=lambda p: p["step"])


def analyze_training_quality(
    loss_history: list[dict[str, Any]],
    *,
    dataset_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize loss diagnostics without treating loss as a likeness score."""
    points = _loss_points(loss_history)
    hints: list[dict[str, Any]] = []
    score = 100
    force_fair = False

    metrics: dict[str, Any] = {
        "steps_logged": len(points),
        "initial_loss": None,
        "final_loss": None,
        "loss_drop_ratio": None,
        "final_val_loss": None,
        "best_val_loss": None,
    }

    if len(points) < 2:
        hints.append(_hint("insufficient_loss_data", "warning", steps=len(points)))
        return {
            "level": "fair",
            "score": 50,
            "metrics": metrics,
            "hints": hints,
            "dataset_health": dataset_health,
        }

    # Single-sample random-sigma loss is noisy; use short windows for diagnostics.
    window = max(1, min(5, len(points) // 4 or 1))
    initial = sum(p["loss"] for p in points[:window]) / window
    final = sum(p["loss"] for p in points[-window:]) / window
    drop_ratio = (initial - final) / initial if initial > 0 else 0.0
    metrics.update(
        {
            "initial_loss": round(initial, 4),
            "final_loss": round(final, 4),
            "loss_drop_ratio": round(drop_ratio, 4),
        }
    )

    hints.append(
        _hint(
            "loss_curve_diagnostic_only",
            "info",
            initial=round(initial, 4),
            final=round(final, 4),
        )
    )

    val_points = [p for p in points if "val_loss" in p]
    if val_points:
        final_val = val_points[-1]["val_loss"]
        metrics["final_val_loss"] = round(final_val, 4)
        best_val = min(p["val_loss"] for p in val_points)
        metrics["best_val_loss"] = round(best_val, 4)
        if final_val - final > _VAL_GAP_WARN:
            hints.append(
                _hint(
                    "val_loss_gap",
                    "warning",
                    val_loss=round(final_val, 4),
                    train_loss=round(final, 4),
                )
            )
            score -= 8
            force_fair = True
        if final_val - best_val > _VAL_REGRESSION_WARN:
            hints.append(
                _hint(
                    "val_loss_regressed",
                    "warning",
                    best_val_loss=round(best_val, 4),
                    final_val_loss=round(final_val, 4),
                )
            )
            score -= 8
            force_fair = True

    if dataset_health:
        ds_level = str(dataset_health.get("level") or "")
        if ds_level == "poor":
            hints.append(_hint("dataset_health_poor", "warning"))
            score -= 20
            force_fair = True
        elif ds_level == "fair":
            hints.append(_hint("dataset_health_fair", "info"))

    score = max(0, min(100, score))
    level = _level_from_score(score, force_fair=force_fair)

    return {
        "level": level,
        "score": score,
        "metrics": metrics,
        "hints": hints,
        "dataset_health": dataset_health,
    }
