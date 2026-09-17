"""Face-aware training crops for identity (concept) LoRA datasets.

At 512² a full-body or waist-up photo leaves the face at 40–80 px (5–10 latent tokens), far
too small for the DiT to bind identity. When a face is detected and is small relative to the
plain cover crop, the crop window is re-centred on the face and tightened so the face fills
roughly a third of the frame (head-and-shoulders framing), then scaled to the training size.

Detection uses OpenCV's YuNet (``cv2.FaceDetectorYN``), a 230 KB ONNX model fetched once into
``{workspace}/models/_aux/face_detection/``. OpenCV is a source-install dependency
(``opencv-python-headless``) but is excluded from packaged desktop sidecars; ``face_crop=auto``
therefore logs and keeps the portrait-biased cover crop when the detector is unavailable, while
``face_crop=on`` fails loud.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

FACE_DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
FACE_DETECTOR_REL = Path("models/_aux/face_detection") / FACE_DETECTOR_FILE
FACE_DETECTOR_HF_REPO = "opencv/face_detection_yunet"
FACE_DETECTOR_URLS: tuple[str, ...] = (
    f"https://huggingface.co/{FACE_DETECTOR_HF_REPO}/resolve/main/{FACE_DETECTOR_FILE}",
    f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{FACE_DETECTOR_FILE}",
)
# Anything smaller is an HTML error page, not the model.
_FACE_DETECTOR_MIN_BYTES = 100_000

FACE_CROP_MODES: tuple[str, ...] = ("auto", "on", "off")

# Target face height as a fraction of the crop height (≈ head-and-shoulders framing).
FACE_TARGET_FRACTION = 0.30
# Faces already at least this tall (relative to the plain cover crop) are left alone.
FACE_LARGE_ENOUGH_FRACTION = 0.22
# Vertical anchor of the face centre inside the crop (eyes above the middle).
FACE_ANCHOR_Y = 0.40
# Tight windows may be smaller than the training size; allow at most this much up-sampling
# (LANCZOS) before preferring a looser frame — a blurry 512² is worse than a 21% face.
MAX_UPSAMPLE = 1.5
MIN_FACE_SCORE = 0.70
_DETECT_MAX_SIDE = 640


@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass(frozen=True)
class FaceCropPlan:
    """Per-image decision: ``window`` is ``(left, top, w, h)`` in source pixels or None."""

    window: tuple[int, int, int, int] | None
    face: FaceBox | None
    reason: str


def normalize_face_crop_mode(value: Any) -> str:
    if value is None or value == "":
        return "auto"
    if value is True:
        return "on"
    if value is False:
        return "off"
    mode = str(value).strip().lower()
    if mode not in FACE_CROP_MODES:
        raise RuntimeError(f"face_crop must be one of {'|'.join(FACE_CROP_MODES)} (got {value!r})")
    return mode


def opencv_face_detector_available() -> tuple[bool, str]:
    try:
        import cv2
    except ImportError:
        return False, "opencv-python-headless is not installed (excluded from packaged desktop builds)"
    if not hasattr(cv2, "FaceDetectorYN"):
        return False, f"OpenCV {getattr(cv2, '__version__', '?')} lacks cv2.FaceDetectorYN (needs >= 4.5.4)"
    return True, ""


def _fetch_via_huggingface() -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(repo_id=FACE_DETECTOR_HF_REPO, filename=FACE_DETECTOR_FILE))
    except Exception:
        return None


def _fetch_via_url(url: str, dest: Path) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https URLs
            data = resp.read()
    except Exception:
        return False
    if len(data) < _FACE_DETECTOR_MIN_BYTES:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def resolve_face_detector_path(project_root: Path, *, allow_download: bool = True) -> Path:
    """Return the local YuNet ONNX path, downloading it into the workspace on first use."""
    local = Path(project_root) / FACE_DETECTOR_REL
    if local.is_file() and local.stat().st_size >= _FACE_DETECTOR_MIN_BYTES:
        return local
    if allow_download:
        fetched = _fetch_via_huggingface()
        if fetched is not None and fetched.is_file() and fetched.stat().st_size >= _FACE_DETECTOR_MIN_BYTES:
            import shutil

            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fetched, local)
            return local
        for url in FACE_DETECTOR_URLS:
            if _fetch_via_url(url, local):
                return local
    raise RuntimeError(
        f"Face-aware cropping needs the YuNet face detector ({FACE_DETECTOR_FILE}, ~230KB). "
        f"Download failed; place the file at {local} "
        f"(HF: huggingface.co/{FACE_DETECTOR_HF_REPO} · GitHub: opencv/opencv_zoo) "
        "or set face_crop=off."
    )


@lru_cache(maxsize=4)
def _detector(model_path: str) -> Any:
    import cv2

    return cv2.FaceDetectorYN.create(model_path, "", (320, 320), MIN_FACE_SCORE, 0.3, 5000)


@lru_cache(maxsize=512)
def _detect_cached(path_key: str, mtime_ns: int, model_path: str) -> tuple[FaceBox, ...]:
    import numpy as np

    from backend.engine.training.dataset_store import open_rgb_image

    img = open_rgb_image(Path(path_key))
    src_w, src_h = img.size
    scale = min(1.0, _DETECT_MAX_SIDE / float(max(src_w, src_h)))
    det_w = max(32, int(round(src_w * scale)))
    det_h = max(32, int(round(src_h * scale)))
    if (det_w, det_h) != (src_w, src_h):
        from PIL import Image

        img = img.resize((det_w, det_h), Image.BILINEAR)
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    det = _detector(model_path)
    det.setInputSize((det_w, det_h))
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return ()
    # YuNet rows: x, y, w, h, 5 landmarks (10 values), score.
    boxes: list[FaceBox] = []
    for row in faces:
        x, y, w, h = (float(v) for v in row[:4])
        score = float(row[14]) if len(row) > 14 else 1.0
        if score < MIN_FACE_SCORE or w <= 0 or h <= 0:
            continue
        boxes.append(FaceBox(x / scale, y / scale, w / scale, h / scale, score))
    boxes.sort(key=lambda b: b.w * b.h, reverse=True)
    return tuple(boxes)


def detect_faces(path: Path, model_path: Path) -> tuple[FaceBox, ...]:
    """All confident faces in source-pixel coordinates, largest first."""
    p = Path(path)
    return _detect_cached(str(p.resolve()), p.stat().st_mtime_ns, str(model_path))


def detect_primary_face(path: Path, model_path: Path) -> FaceBox | None:
    """Largest confident face in source-pixel coordinates (None when no face)."""
    faces = detect_faces(path, model_path)
    return faces[0] if faces else None


def face_crop_window(
    src_w: int,
    src_h: int,
    face: FaceBox,
    target: tuple[int, int],
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Pure geometry: crop window (target aspect) that frames ``face``, or None if not needed.

    The window up-samples at most ``MAX_UPSAMPLE``× (unless the image itself is smaller) and
    never exceeds the plain cover crop (so large faces keep the existing behaviour).
    """
    target_w, target_h = int(target[0]), int(target[1])
    aspect = target_w / float(target_h)
    cover_h = min(float(src_h), src_w / aspect)
    if cover_h <= 0:
        return None, "degenerate_image"
    if face.h / cover_h >= FACE_LARGE_ENOUGH_FRACTION:
        return None, "face_large_enough"

    crop_h = face.h / FACE_TARGET_FRACTION
    crop_h = max(crop_h, min(cover_h, target_h / MAX_UPSAMPLE))
    crop_h = min(crop_h, cover_h)
    crop_w = crop_h * aspect
    if crop_w > src_w:
        crop_w = float(src_w)
        crop_h = crop_w / aspect

    left = face.cx - crop_w / 2.0
    top = face.cy - FACE_ANCHOR_Y * crop_h
    left = min(max(0.0, left), src_w - crop_w)
    top = min(max(0.0, top), src_h - crop_h)
    w_i = max(1, int(round(crop_w)))
    h_i = max(1, int(round(crop_h)))
    l_i = min(max(0, int(round(left))), max(0, src_w - w_i))
    t_i = min(max(0, int(round(top))), max(0, src_h - h_i))
    return (l_i, t_i, w_i, h_i), "face_crop"


def plan_face_crop(path: Path, model_path: Path, target: tuple[int, int]) -> FaceCropPlan:
    from backend.engine.training.dataset_store import open_rgb_image

    face = detect_primary_face(path, model_path)
    if face is None:
        return FaceCropPlan(None, None, "no_face")
    src_w, src_h = open_rgb_image(Path(path)).size
    window, reason = face_crop_window(src_w, src_h, face, target)
    return FaceCropPlan(window, face, reason)


# A face shorter than this (source px) cannot be re-framed to FACE_TARGET_FRACTION without
# exceeding MAX_UPSAMPLE at 512² — the crop would be blurry, so the image is flagged instead.
def min_usable_face_px(target_h: int) -> float:
    return FACE_TARGET_FRACTION * target_h / MAX_UPSAMPLE


def audit_dataset_faces(
    paths: list[Path],
    *,
    project_root: Path,
    resolution: tuple[int, int] = (512, 512),
    allow_download: bool = True,
) -> dict[str, Any]:
    """Pre-flight face report for an identity dataset (used by dataset health, not training).

    Returns ``{"available": bool, "reason": str, "rows": [...], "counts": {...}}`` where each row
    is ``{file, width, height, faces, face_px, face_frac, action, window}``; ``action`` is one of
    ``crop`` (auto face crop will re-frame), ``keep`` (already close-up), ``tiny`` (face too small
    even for face crop), ``none`` (no face) or ``error``.
    """
    ok, why = opencv_face_detector_available()
    if not ok:
        return {"available": False, "reason": why, "rows": [], "counts": {}}
    try:
        model_path = resolve_face_detector_path(project_root, allow_download=allow_download)
    except RuntimeError as exc:
        return {"available": False, "reason": str(exc), "rows": [], "counts": {}}

    from backend.engine.training.dataset_store import open_rgb_image

    target_h = int(resolution[1])
    tiny_px = min_usable_face_px(target_h)
    rows: list[dict[str, Any]] = []
    counts = {"crop": 0, "keep": 0, "tiny": 0, "none": 0, "error": 0, "multi": 0}
    for path in paths:
        row: dict[str, Any] = {"file": path.name, "faces": 0, "action": "none", "window": None}
        try:
            src_w, src_h = open_rgb_image(path).size
            row["width"], row["height"] = src_w, src_h
            faces = detect_faces(path, model_path)
        except Exception as exc:
            row["action"] = "error"
            row["error"] = str(exc)
            counts["error"] += 1
            rows.append(row)
            continue
        row["faces"] = len(faces)
        if len(faces) >= 2:
            # Only count a second face when it is a real competitor (≥ 40% of the primary).
            primary, second = faces[0], faces[1]
            if second.h >= 0.4 * primary.h:
                counts["multi"] += 1
                row["multi"] = True
        if not faces:
            counts["none"] += 1
            rows.append(row)
            continue
        face = faces[0]
        aspect = resolution[0] / float(resolution[1])
        cover_h = max(1.0, min(float(src_h), src_w / aspect))
        row["face_px"] = int(round(face.h))
        row["face_frac"] = round(face.h / cover_h, 3)
        window, reason = face_crop_window(src_w, src_h, face, resolution)
        if face.h < tiny_px:
            row["action"] = "tiny"
            counts["tiny"] += 1
            row["window"] = list(window) if window else None
        elif window is not None and reason == "face_crop":
            row["action"] = "crop"
            counts["crop"] += 1
            row["window"] = list(window)
        else:
            row["action"] = "keep"
            counts["keep"] += 1
        rows.append(row)
    return {"available": True, "reason": "", "rows": rows, "counts": counts, "tiny_face_px": int(tiny_px)}


def face_crop_cache_key(plans: dict[Path, FaceCropPlan]) -> str:
    """Stable digest of the applied windows (for the latent-cache fingerprint)."""
    applied = sorted(
        (str(Path(p).name), list(plan.window))
        for p, plan in plans.items()
        if plan.window is not None
    )
    if not applied:
        return ""
    raw = json.dumps(applied, separators=(",", ":"), sort_keys=True)
    return "face:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def plan_training_face_crops(
    pairs: list[tuple[Path, str]],
    *,
    project_root: Path,
    resolution: tuple[int, int],
    mode: str,
    dataset_meta: dict[str, Any] | None,
    on_log: Callable[[str, str], None] | None = None,
) -> dict[Path, FaceCropPlan]:
    """Decide the crop window per training image and log a summary.

    ``auto`` enables face crops for concept (identity) datasets only, and degrades to the plain
    cover crop — with an explicit warning — when OpenCV / YuNet is unavailable. ``on`` requires
    the detector (fail loud). ``off`` disables.
    """
    mode = normalize_face_crop_mode(mode)
    log = on_log or (lambda _level, _msg: None)
    kind = str((dataset_meta or {}).get("kind") or "concept").strip().lower()
    if mode == "off" or (mode == "auto" and kind != "concept"):
        return {}

    ok, why = opencv_face_detector_available()
    if not ok:
        if mode == "on":
            raise RuntimeError(f"face_crop=on but face detection is unavailable: {why}")
        log("warning", f"Face-aware crop skipped ({why}); using portrait-biased cover crop.")
        return {}
    try:
        model_path = resolve_face_detector_path(project_root)
    except RuntimeError as exc:
        if mode == "on":
            raise
        log("warning", f"Face-aware crop skipped: {exc}")
        return {}

    plans: dict[Path, FaceCropPlan] = {}
    counts = {"face_crop": 0, "face_large_enough": 0, "no_face": 0}
    for img_path, _prompt in pairs:
        try:
            plan = plan_face_crop(img_path, model_path, resolution)
        except Exception as exc:
            if mode == "on":
                raise RuntimeError(f"Face detection failed for {img_path.name}: {exc}") from exc
            log("warning", f"Face detection failed for {img_path.name}: {exc}")
            plan = FaceCropPlan(None, None, "detect_error")
        plans[img_path] = plan
        counts[plan.reason] = counts.get(plan.reason, 0) + 1
        if plan.window is not None and plan.face is not None:
            _l, _t, w, h = plan.window
            log(
                "info",
                f"Face crop {img_path.name}: face {int(plan.face.w)}×{int(plan.face.h)}px → "
                f"window {w}×{h}px (face ≈ {plan.face.h / max(1, h):.0%} of frame)",
            )
    total = len(pairs)
    log(
        "info",
        f"Face-aware crop: {counts.get('face_crop', 0)}/{total} re-framed, "
        f"{counts.get('face_large_enough', 0)} already close-up, {counts.get('no_face', 0)} no face detected",
    )
    no_face = counts.get("no_face", 0)
    if no_face >= 2 and no_face * 2 >= total:
        log(
            "warning",
            "Face not detected in at least half of the dataset; for an identity LoRA make sure "
            "each image shows the subject's face clearly.",
        )
    return plans
