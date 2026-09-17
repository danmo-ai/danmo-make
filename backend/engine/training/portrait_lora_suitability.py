"""Heuristic portrait LoRA training suitability (512 cover crop simulation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_TRAINING_RESOLUTION = (512, 512)


def score_100_to_1_5(score_100: int) -> float:
    score = max(0, min(100, int(score_100)))
    if score >= 72:
        return 5.0
    if score >= 58:
        return 4.0
    if score >= 45:
        return 3.0
    if score >= 30:
        return 2.0
    return 1.0


def _laplacian_variance(gray: Any) -> float:
    import numpy as np

    if gray.size == 0:
        return 0.0
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(np.var(lap))


def _simulate_training_cover_crop(path: Path, resolution: tuple[int, int]) -> tuple[Any, int, int]:
    """Return float RGB array (0–1) and original width/height."""
    from backend.engine.training.dataset_store import open_rgb_image, resize_rgb_image

    img = open_rgb_image(path)
    src_w, src_h = img.size
    crop = resize_rgb_image(path, resolution, augmentation_index=0, resize_mode="cover")
    return crop, src_w, src_h


def resolve_face_model_for_audit(project_root: Path | None) -> Path | None:
    """YuNet path when OpenCV + model are available locally (no download during audits)."""
    if project_root is None:
        return None
    from backend.engine.training.face_crop import opencv_face_detector_available, resolve_face_detector_path

    ok, _why = opencv_face_detector_available()
    if not ok:
        return None
    try:
        return resolve_face_detector_path(project_root, allow_download=False)
    except RuntimeError:
        return None


def _face_geometry_issues(
    path: Path,
    *,
    face_model_path: Path,
    src_w: int,
    src_h: int,
    training_resolution: tuple[int, int],
) -> tuple[int, list[str], list[str], dict[str, Any]]:
    """Penalties from the real face box: none / tiny (unfixable) / small (auto-crop fixes) / multi."""
    from backend.engine.training.face_crop import (
        FACE_LARGE_ENOUGH_FRACTION,
        detect_faces,
        min_usable_face_px,
    )

    faces = detect_faces(path, face_model_path)
    stats: dict[str, Any] = {"face_count": len(faces)}
    if not faces:
        return 40, ["face_not_detected", "small_face"], ["未检测到人脸（人物 LoRA 每张图都应清晰露脸）"], stats
    face = faces[0]
    aspect = training_resolution[0] / float(training_resolution[1])
    cover_h = max(1.0, min(float(src_h), src_w / aspect))
    frac = face.h / cover_h
    stats.update({"face_px": int(round(face.h)), "face_frac": round(frac, 3)})
    penalty = 0
    issues: list[str] = []
    reasons: list[str] = []
    if face.h < min_usable_face_px(int(training_resolution[1])):
        penalty += 40
        issues.extend(["tiny_face_in_crop", "small_face"])
        reasons.append(f"人脸仅 {int(face.h)}px，即使自动人脸裁切放大也会模糊")
    elif frac < FACE_LARGE_ENOUGH_FRACTION:
        penalty += 8
        issues.append("small_face")
        reasons.append(f"人脸占画面 {frac:.0%}，训练时将自动人脸裁切放大")
    if len(faces) >= 2 and faces[1].h >= 0.4 * face.h:
        penalty += 15
        issues.append("multiple_people")
        reasons.append("画面中有多张人脸，身份可能混淆")
    return penalty, issues, reasons, stats


def analyze_portrait_training_image(
    path: Path,
    *,
    training_resolution: tuple[int, int] = DEFAULT_TRAINING_RESOLUTION,
    face_model_path: Path | None = None,
) -> dict[str, Any]:
    """Score one image for person/concept LoRA training at ``training_resolution``.

    With ``face_model_path`` (YuNet) the face size comes from a real detection; otherwise the
    legacy Laplacian "face band" energy of the simulated cover crop is used.
    """
    if not path.is_file():
        return {
            "score_100": 0,
            "score_1_5": 1.0,
            "issues": ["missing_file"],
            "reason": "图片文件不存在或不可读",
            "stats": {},
        }

    try:
        crop, src_w, src_h = _simulate_training_cover_crop(path, training_resolution)
    except Exception as exc:
        return {
            "score_100": 0,
            "score_1_5": 1.0,
            "issues": ["broken_image"],
            "reason": f"无法解码或裁切：{exc}",
            "stats": {},
        }

    import numpy as np

    gray = (crop.mean(axis=2) * 255.0).astype(np.float32)
    h, w = gray.shape
    y0, y1 = int(h * 0.12), int(h * 0.42)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    face_band = gray[y0:y1, x0:x1]
    face_energy = _laplacian_variance(face_band)
    overall_energy = _laplacian_variance(gray)

    short_edge = min(src_w, src_h)
    aspect = src_w / max(src_h, 1)
    landscape = src_w > src_h * 1.05

    score = 100
    issues: list[str] = []
    reasons: list[str] = []

    if short_edge < 512:
        score -= 45
        issues.append("low_resolution")
        reasons.append(f"短边仅 {short_edge}px，512 裁切后面部细节不足")
    elif short_edge < 800:
        score -= 20
        issues.append("low_resolution")
        reasons.append(f"短边 {short_edge}px 偏低，训练裁切后五官可能偏糊")

    face_stats: dict[str, Any] = {}
    face_detected = False
    if face_model_path is not None:
        try:
            penalty, f_issues, f_reasons, face_stats = _face_geometry_issues(
                path,
                face_model_path=face_model_path,
                src_w=src_w,
                src_h=src_h,
                training_resolution=training_resolution,
            )
            score -= penalty
            issues.extend(f_issues)
            reasons.extend(f_reasons)
            face_detected = True
        except Exception:
            face_stats = {}
            face_detected = False

    if not face_detected:
        # Framing penalties only matter when we cannot measure (and auto-crop) the real face.
        if landscape:
            score -= 12
            issues.append("landscape_framing")
            reasons.append("横图构图，512 cover 裁切后人物/面部占比通常过小")
        if aspect >= 1.6:
            score -= 8
            issues.append("wrong_framing")
            reasons.append("超宽画幅，主体在训练分辨率下过小")
        if face_energy < 35:
            score -= 40
            issues.extend(["tiny_face_in_crop", "small_face"])
            reasons.append("模拟 512 裁切后面部区域过小或过于模糊")
        elif face_energy < 70:
            score -= 22
            issues.append("face_soft_in_crop")
            reasons.append("模拟 512 裁切后面部清晰度一般")
        elif face_energy < 100:
            score -= 10
            issues.append("low_detail")
            reasons.append("面部细节略弱，近景胸像/半身照更佳")

    if overall_energy < 25 and "blurry" not in issues:
        score -= 15
        issues.append("blurry")
        reasons.append("整图偏糊")

    score = max(0, min(100, score))
    unique_issues = sorted(dict.fromkeys(issues))
    reason = "；".join(reasons[:3]) if reasons else "适合人物 LoRA 训练"

    stats: dict[str, Any] = {
        "src_width": src_w,
        "src_height": src_h,
        "short_edge": short_edge,
        "aspect_ratio": round(aspect, 3),
        "face_crop_energy": round(face_energy, 1),
        "overall_crop_energy": round(overall_energy, 1),
        "landscape": landscape,
        "face_detector": face_detected,
    }
    stats.update(face_stats)
    return {
        "score_100": score,
        "score_1_5": score_100_to_1_5(score),
        "issues": unique_issues,
        "reason": reason,
        "stats": stats,
    }


def merge_vlm_and_heuristic_sample(
    *,
    file_key: str,
    vlm_parsed: dict[str, Any] | None,
    heuristic: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine VLM output with pixel heuristics; final score is the minimum."""
    h_score = float(heuristic["score_1_5"]) if heuristic and heuristic.get("score_1_5") is not None else None
    v_score = float(vlm_parsed["score"]) if vlm_parsed and vlm_parsed.get("score") is not None else None

    issues: set[str] = set()
    reasons: list[str] = []
    if heuristic:
        issues.update(heuristic.get("issues") or [])
        if heuristic.get("reason"):
            reasons.append(str(heuristic["reason"]))
    if vlm_parsed:
        issues.update(vlm_parsed.get("issues") or [])
        if vlm_parsed.get("reason"):
            reasons.append(str(vlm_parsed["reason"]))
    issues.discard("good")

    if v_score is not None and h_score is not None:
        final = min(v_score, h_score)
        source = "vlm+heuristic" if abs(final - v_score) > 1e-6 or abs(final - h_score) > 1e-6 else "vlm"
    elif v_score is not None:
        final = v_score
        source = "vlm"
    elif h_score is not None:
        final = h_score
        source = "heuristic"
    else:
        final = None
        source = "none"

    sample: dict[str, Any] = {
        "file": file_key,
        "score": final,
        "vlm_score": v_score,
        "heuristic_score": h_score,
        "issues": sorted(issues),
        "reason": "；".join(dict.fromkeys(r for r in reasons if r))[:500],
        "source": source,
        "suitable_for_training": final is not None and final >= 3.0,
    }
    if vlm_parsed and vlm_parsed.get("raw"):
        sample["vlm_raw"] = vlm_parsed["raw"]
    if heuristic and heuristic.get("stats"):
        sample["heuristic_stats"] = heuristic["stats"]
    return sample
