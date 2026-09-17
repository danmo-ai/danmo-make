"""Workspace dataset storage for LoRA training (mlx-examples ``train.jsonl`` format)."""

from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

MIN_DATASET_IMAGES = 3

_HEIF_OPENER_READY = False


def _ensure_heif_opener() -> bool:
    """Register HEIC/HEIF with Pillow when pillow-heif is installed."""
    global _HEIF_OPENER_READY
    if _HEIF_OPENER_READY:
        return True
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_OPENER_READY = True
        return True
    except ImportError:
        return False


def _is_heif_payload(data: bytes) -> bool:
    if len(data) < 12:
        return False
    brand = data[8:12]
    return data[4:8] == b"ftyp" and brand in {
        b"heic",
        b"heix",
        b"hevc",
        b"heif",
        b"mif1",
        b"msf1",
    }


def open_rgb_image(path: Path) -> Image.Image:
    """Open a dataset/training image as RGB (JPEG/PNG/WebP/HEIC)."""
    from io import BytesIO

    _ensure_heif_opener()
    try:
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")
    except Exception as exc:
        try:
            raw = path.read_bytes()
        except OSError:
            raise RuntimeError(f"Cannot read image file {path}") from exc
        if _is_heif_payload(raw):
            if not _ensure_heif_opener():
                raise RuntimeError(
                    f"Image {path.name!r} is HEIC/HEIF but pillow-heif is not installed "
                    "(pip install pillow-heif or re-upload as JPEG/PNG)."
                ) from exc
            with Image.open(BytesIO(raw)) as img:
                img.load()
                return img.convert("RGB")
        raise RuntimeError(
            f"Cannot identify image file {path} "
            f"({exc}). Re-upload as JPEG/PNG or install pillow-heif for HEIC."
        ) from exc


def _normalize_dataset_image_bytes(filename: str, data: bytes) -> tuple[str, bytes]:
    """Decode upload bytes and store as JPEG (or PNG when source is non-HEIC PNG)."""
    from io import BytesIO

    if not data:
        raise ValueError(f"Uploaded image {filename!r} is empty")
    _ensure_heif_opener()
    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            rgb = img.convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Cannot read uploaded image {filename!r}: {exc}. "
            "Use JPEG, PNG, WebP, or HEIC."
        ) from exc

    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    while stem.lower().endswith((".heic", ".heif")):
        stem = Path(stem).stem

    if ext == ".png" and not _is_heif_payload(data):
        out = BytesIO()
        rgb.save(out, format="PNG")
        safe = f"{stem}.png" if stem else "image.png"
        return safe, out.getvalue()

    out = BytesIO()
    rgb.save(out, format="JPEG", quality=92)
    safe = f"{stem}.jpg" if stem else "image.jpg"
    return safe, out.getvalue()
MAX_DATASET_IMAGES = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def datasets_root(workspace_root: Path) -> Path:
    return workspace_root / "datasets"


def _validate_dataset_id(dataset_id: str) -> str:
    dataset_id = (dataset_id or "").strip()
    if (
        not dataset_id
        or dataset_id in (".", "..")
        or dataset_id.startswith("_")
        or "/" in dataset_id
        or "\\" in dataset_id
        or ".." in dataset_id
    ):
        raise ValueError(f"Invalid dataset id {dataset_id!r}")
    return dataset_id


def _validate_dataset_file_rel(file_rel: str) -> str:
    file_rel = (file_rel or "").strip().replace("\\", "/")
    if not file_rel or file_rel.startswith("/") or ".." in file_rel.split("/"):
        raise ValueError(f"Invalid file path {file_rel!r}")
    return file_rel


def _dataset_dir(root: Path, dataset_id: str) -> Path:
    return root / _validate_dataset_id(dataset_id)


def new_dataset_id() -> str:
    return "ds_" + secrets.token_hex(8)


def list_datasets(workspace_root: Path) -> list[dict[str, Any]]:
    root = datasets_root(workspace_root)
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items.append(_summarize_dataset(path.name, path, meta))
    return items


def _summarize_dataset(dataset_id: str, path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    rows = _read_train_jsonl(path / "train.jsonl")
    cover_image = ""
    for row in rows:
        img_rel = row.get("image") or ""
        if img_rel and (path / img_rel).is_file():
            cover_image = img_rel
            break
    return {
        "id": dataset_id,
        "name": meta.get("name") or dataset_id,
        "kind": meta.get("kind") or "concept",
        "trigger_word": meta.get("trigger_word") or "",
        "default_prompt": meta.get("default_prompt") or "",
        "nsfw": bool(meta.get("nsfw", False)),
        "image_count": len(rows),
        "cover_image": cover_image,
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
    }


def _read_train_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append({"image": str(obj.get("image", "")), "prompt": str(obj.get("prompt", ""))})
    return rows


def _write_train_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"image": row["image"], "prompt": row["prompt"]}, ensure_ascii=False) + "\n")


def get_dataset(workspace_root: Path, dataset_id: str) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = _read_train_jsonl(path / "train.jsonl")
    images = []
    for row in rows:
        img_rel = row["image"]
        img_path = path / img_rel
        images.append(
            {
                "file": img_rel,
                "prompt": row["prompt"],
                "exists": img_path.is_file(),
            }
        )
    summary = _summarize_dataset(dataset_id, path, meta)
    summary["images"] = images
    return summary


def create_dataset(
    workspace_root: Path,
    *,
    name: str,
    kind: str = "concept",
    trigger_word: str = "",
    default_prompt: str = "",
    nsfw: bool = False,
) -> dict[str, Any]:
    dataset_id = new_dataset_id()
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(exist_ok=True)
    meta = {
        "name": name.strip() or dataset_id,
        "kind": kind,
        "trigger_word": trigger_word.strip(),
        "default_prompt": default_prompt.strip(),
        "nsfw": nsfw,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_train_jsonl(path / "train.jsonl", [])
    return get_dataset(workspace_root, dataset_id)


def update_dataset_meta(workspace_root: Path, dataset_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("name", "kind", "trigger_word", "default_prompt", "nsfw"):
        if key in patch:
            meta[key] = patch[key]
    meta["updated_at"] = _now_iso()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_dataset(workspace_root, dataset_id)


def add_dataset_images(
    workspace_root: Path,
    dataset_id: str,
    files: list[tuple[str, bytes]],
    *,
    default_prompt: str | None = None,
    per_file_prompts: list[str] | None = None,
) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    prompt = (default_prompt if default_prompt is not None else meta.get("default_prompt") or "").strip()
    if not prompt and meta.get("trigger_word"):
        prompt = str(meta["trigger_word"]).strip()
    rows = _read_train_jsonl(path / "train.jsonl")
    images_dir = path / "images"
    images_dir.mkdir(exist_ok=True)
    for idx, (filename, data) in enumerate(files):
        try:
            safe_name, data = _normalize_dataset_image_bytes(
                Path(filename).name or f"image_{len(rows):04d}.jpg",
                data,
            )
        except ValueError:
            raise
        dest = images_dir / safe_name
        counter = 0
        while dest.exists():
            counter += 1
            stem = Path(safe_name).stem
            dest = images_dir / f"{stem}_{counter}{Path(safe_name).suffix}"
        dest.write_bytes(data)
        rel = f"images/{dest.name}"
        row_prompt = prompt
        if per_file_prompts is not None and idx < len(per_file_prompts):
            row_prompt = str(per_file_prompts[idx] or prompt)
        rows.append({"image": rel, "prompt": row_prompt})
    if len(rows) > MAX_DATASET_IMAGES:
        raise ValueError(f"Dataset exceeds maximum of {MAX_DATASET_IMAGES} images")
    _write_train_jsonl(path / "train.jsonl", rows)
    meta["updated_at"] = _now_iso()
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_dataset(workspace_root, dataset_id)


def update_dataset_captions(
    workspace_root: Path,
    dataset_id: str,
    captions: list[dict[str, str]],
) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    rows = _read_train_jsonl(path / "train.jsonl")
    by_file = {row["image"]: row for row in rows}
    for item in captions:
        file_key = item.get("file") or item.get("image")
        if not file_key or file_key not in by_file:
            continue
        by_file[file_key]["prompt"] = str(item.get("prompt") or "")
    _write_train_jsonl(path / "train.jsonl", list(by_file.values()))
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    meta["updated_at"] = _now_iso()
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_dataset(workspace_root, dataset_id)


def remove_dataset_image(workspace_root: Path, dataset_id: str, file_rel: str) -> dict[str, Any]:
    file_rel = _validate_dataset_file_rel(file_rel)
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    rows = _read_train_jsonl(path / "train.jsonl")
    rows = [r for r in rows if r["image"] != file_rel]
    img_path = path / file_rel
    if img_path.is_file():
        img_path.unlink()
    _write_train_jsonl(path / "train.jsonl", rows)
    return get_dataset(workspace_root, dataset_id)


def delete_dataset(workspace_root: Path, dataset_id: str) -> None:
    """Remove a training dataset directory from the workspace."""
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    shutil.rmtree(path)


def validate_dataset_for_training(workspace_root: Path, dataset_id: str) -> None:
    ds = get_dataset(workspace_root, dataset_id)
    count = int(ds.get("image_count") or 0)
    if count < MIN_DATASET_IMAGES:
        raise ValueError(
            f"Dataset needs at least {MIN_DATASET_IMAGES} images (has {count})"
        )


def _normalize_asset_id(ref: str) -> str:
    ref = ref.strip()
    if ref.startswith("asset:"):
        return ref[len("asset:") :]
    return ref


def import_dataset_from_assets(
    workspace_root: Path,
    dataset_id: str,
    asset_ids: list[str],
    *,
    resolve_asset_path: Callable[[str], Path],
    default_prompt: str | None = None,
    asset_captions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Copy gallery/canvas assets into a training dataset."""
    if not asset_ids:
        raise ValueError("asset_ids must not be empty")
    files: list[tuple[str, bytes]] = []
    prompts: list[str] = []
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Dataset {dataset_id!r} not found")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    fallback = (default_prompt if default_prompt is not None else meta.get("default_prompt") or "").strip()
    caps = asset_captions or {}
    for raw in asset_ids:
        aid = _normalize_asset_id(raw)
        if not aid:
            continue
        src = resolve_asset_path(aid)
        if not src.is_file():
            raise FileNotFoundError(f"Asset {aid!r} not found at {src}")
        suffix = src.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"Asset {aid!r} is not an image ({suffix or 'unknown'})")
        cap = caps.get(raw) or caps.get(aid) or caps.get(f"asset:{aid}") or fallback
        if not str(cap or "").strip():
            raise ValueError(
                f"Asset {aid!r}: no caption. Set dataset default_prompt or provide asset_captions."
            )
        files.append((src.name, src.read_bytes()))
        prompts.append(cap)
    if not files:
        raise ValueError("No valid image assets to import")
    return add_dataset_images(
        workspace_root,
        dataset_id,
        files,
        default_prompt=fallback,
        per_file_prompts=prompts,
    )


def bundled_dog6_example_dir(default_config_root: Path) -> Path:
    """Shipped DreamBooth dog6 example (Google DreamBooth, bundled in ``default_config/``)."""
    return default_config_root / "lora_examples" / "dog6"


def import_dog6_example(
    workspace_root: Path,
    *,
    bundled_root: Path | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Copy bundled DreamBooth dog6 example into the workspace datasets directory."""
    if bundled_root is None:
        raise RuntimeError("Dog6 example bundle path is not configured")
    bundled_root = Path(bundled_root)
    src_jsonl = bundled_root / "train.jsonl"
    if not src_jsonl.is_file():
        raise RuntimeError(f"Dog6 example bundle missing train.jsonl at {src_jsonl}")
    if on_log:
        on_log("info", f"Importing bundled Dog6 example from {bundled_root} …")

    ds = create_dataset(
        workspace_root,
        name="DreamBooth Dog6 (example)",
        kind="concept",
        trigger_word="sks dog",
        default_prompt="A photo of sks dog",
    )
    dataset_id = ds["id"]
    files: list[tuple[str, bytes]] = []
    for line in src_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rel = str(row.get("image") or "").strip()
        if not rel:
            continue
        src_img = bundled_root / rel
        if not src_img.is_file():
            src_img = bundled_root / "images" / Path(rel).name
        if not src_img.is_file():
            raise RuntimeError(f"Dog6 example bundle missing image {rel!r} (expected {src_img})")
        files.append((Path(rel).name, src_img.read_bytes()))
    if len(files) < MIN_DATASET_IMAGES:
        raise RuntimeError(
            f"Dog6 example bundle has {len(files)} images; need at least {MIN_DATASET_IMAGES}"
        )
    return add_dataset_images(workspace_root, dataset_id, files, default_prompt="A photo of sks dog")


def load_training_pairs(workspace_root: Path, dataset_id: str) -> list[tuple[Path, str]]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    rows = _read_train_jsonl(path / "train.jsonl")
    pairs: list[tuple[Path, str]] = []
    for row in rows:
        img_path = path / row["image"]
        if img_path.is_file():
            pairs.append((img_path, row["prompt"]))
    return pairs


def _dataset_meta(workspace_root: Path, dataset_id: str) -> dict[str, Any]:
    path = _dataset_dir(datasets_root(workspace_root), dataset_id)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _caption_contains_trigger(caption: str, trigger: str) -> bool:
    trigger = (trigger or "").strip()
    if not trigger:
        return True
    return trigger.lower() in str(caption or "").lower()


def _validate_concept_trigger_captions(
    pairs: list[tuple[Path, str]],
    *,
    dataset_meta: dict[str, Any],
) -> None:
    if str(dataset_meta.get("kind") or "concept").strip().lower() != "concept":
        return
    trigger = (dataset_meta.get("trigger_word") or "").strip()
    if not trigger:
        raise RuntimeError("Concept LoRA training requires trigger_word; set it on the dataset first.")
    missing = [path.name for path, caption in pairs if not _caption_contains_trigger(caption, trigger)]
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise RuntimeError(
            f"Concept LoRA training captions must visibly include trigger_word {trigger!r}; "
            f"{len(missing)}/{len(pairs)} caption(s) are missing it: {preview}{more}. "
            "Run VLM auto-caption again or edit captions before training."
        )


def resolve_dreambooth_caption(
    pairs: list[tuple[Path, str]],
    *,
    progress_prompt: str,
    dataset_meta: dict[str, Any],
) -> str:
    """Single caption for DreamBooth — must match inference / progress preview prompt."""
    unified = (progress_prompt or "").strip()
    if not unified:
        unified = (dataset_meta.get("default_prompt") or "").strip()
    if not unified:
        raise RuntimeError(
            "DreamBooth training requires progress_prompt or dataset default_prompt; "
            "set one in the training form or dataset settings."
        )
    return unified


def resolve_per_image_captions(
    pairs: list[tuple[Path, str]],
    *,
    progress_prompt: str,
    dataset_meta: dict[str, Any],
) -> list[tuple[Path, str]]:
    """Per-image captions exactly as stored; blanks may use progress/default prompt only."""
    fallback = (progress_prompt or "").strip() or (dataset_meta.get("default_prompt") or "").strip()
    out: list[tuple[Path, str]] = []
    missing: list[str] = []
    for path, cap in pairs:
        text = str(cap or "").strip() or fallback
        if not text:
            missing.append(path.name)
        out.append((path, text))
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise RuntimeError(
            f"{len(missing)}/{len(pairs)} training caption(s) are empty: {preview}{more}. "
            "Fill per-image captions or set progress_prompt / dataset default_prompt."
        )
    return out


def detect_caption_mode(
    pairs: list[tuple[Path, str]],
    *,
    dataset_meta: dict[str, Any] | None = None,
) -> str:
    """Auto-detect caption mode when the training request leaves ``caption_mode`` on auto.

    **Concept / face LoRA** (``kind=concept``): always ``unified`` — bind identity to the
    trigger / ``progress_prompt`` only. VLM per-image captions (outfit, background, pose)
    dilute the trigger in Qwen3 embeddings and prevent face memorization.

    **Style / other kinds**: ``per_image`` when most captions differ (typical VLM output).
    """
    kind = str((dataset_meta or {}).get("kind") or "concept").strip().lower()
    if kind == "concept":
        return "unified"
    if len(pairs) < 2:
        return "unified"
    captions = [str(p or "").strip() for _, p in pairs]
    non_empty = [c for c in captions if c]
    if not non_empty:
        return "unified"
    unique = set(non_empty)
    if len(unique) > max(1, len(non_empty) * 0.5):
        return "per_image"
    return "unified"


def load_training_pairs_unified(
    workspace_root: Path,
    dataset_id: str,
    *,
    progress_prompt: str,
    caption_mode: str = "unified",
) -> tuple[list[tuple[Path, str]], str]:
    """Load training pairs; ``unified`` uses one caption, ``per_image`` keeps row captions.

    When *caption_mode* is ``None`` or ``"auto"``, the function inspects the
    dataset to decide: if most images carry distinct captions (e.g. from VLM
    auto-captioning), ``per_image`` is selected; otherwise ``unified``.
    """
    pairs_raw = load_training_pairs(workspace_root, dataset_id)
    if not pairs_raw:
        return [], ""
    meta = _dataset_meta(workspace_root, dataset_id)
    preview_caption = resolve_dreambooth_caption(
        pairs_raw,
        progress_prompt=progress_prompt,
        dataset_meta=meta,
    )
    raw_mode = (caption_mode or "auto").strip().lower()
    if raw_mode in ("auto", "none", ""):
        mode = detect_caption_mode(pairs_raw, dataset_meta=meta)
    else:
        mode = raw_mode
    if mode == "per_image":
        pairs = resolve_per_image_captions(
            pairs_raw,
            progress_prompt=progress_prompt,
            dataset_meta=meta,
        )
        _validate_concept_trigger_captions(pairs, dataset_meta=meta)
        return pairs, preview_caption
    pairs = [(path, preview_caption) for path, _ in pairs_raw]
    _validate_concept_trigger_captions(pairs, dataset_meta=meta)
    return pairs, preview_caption


def _augmentation_rng(path: Path, augmentation_index: int) -> "random.Random":
    """Deterministic per-image jitter RNG (``hash(str)`` is salted per process; crc32 is not)."""
    import random
    import zlib

    salt = zlib.crc32(str(path.resolve()).encode("utf-8")) % 100_000
    return random.Random(augmentation_index * 7919 + salt)


def resize_rgb_image(
    path: Path,
    resolution: tuple[int, int],
    *,
    augmentation_index: int = 0,
    resize_mode: str = "cover",
    allow_flip: bool = True,
    crop_window: tuple[int, int, int, int] | None = None,
) -> Any:
    """Resize for LoRA training.

    ``cover`` (default): scale-to-fill then crop. Portrait images keep the upper region
    (face-first) instead of center crop, which often removed faces in the previous build.
    Augmentations > 0 apply a small random crop jitter on the scaled image.
    ``stretch``: legacy fit — squish to target (distorts but keeps full frame).
    ``allow_flip=False`` disables horizontal mirroring for every aspect ratio (faces are not
    symmetric; concept / identity datasets pass False — the portrait heuristic alone missed
    the common 1:1 face crop).
    ``crop_window`` ``(left, top, w, h)`` in source pixels (from ``face_crop.plan_face_crop``)
    replaces the cover crop: the window is jittered slightly per augmentation, cropped, then
    scaled to ``resolution``.
    """
    import math
    import random

    import numpy as np

    target_w, target_h = resolution
    mode = (resize_mode or "cover").strip().lower()
    img = open_rgb_image(path)
    src_w, src_h = img.size

    if crop_window is not None and mode == "cover":
        img = _crop_source_window(img, crop_window, augmentation_index, path)
        img = img.resize((target_w, target_h), Image.LANCZOS)
        if augmentation_index > 0:
            img = _photometric_jitter(img, _augmentation_rng(path, augmentation_index))
        return np.array(img).astype("float32") / 255.0

    if mode == "stretch":
        img = img.resize((target_w, target_h), Image.LANCZOS)
        return np.array(img).astype("float32") / 255.0

    if mode == "letterbox":
        scale = min(target_h / src_h, target_w / src_w)
        new_w = max(1, math.ceil(src_w * scale))
        new_h = max(1, math.ceil(src_h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), color=(127, 127, 127))
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        canvas.paste(img, (left, top))
        return np.array(canvas).astype("float32") / 255.0

    scale = max(target_h / src_h, target_w / src_w)
    new_w = math.ceil(src_w * scale)
    new_h = math.ceil(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    max_left = max(0, new_w - target_w)
    max_top = max(0, new_h - target_h)
    portrait = src_h > src_w * 1.05

    aug_rng: random.Random | None = None
    if augmentation_index > 0 and (max_left > 0 or max_top > 0):
        aug_rng = _augmentation_rng(path, augmentation_index)
        if portrait:
            top = aug_rng.randint(0, max(max_top // 3, 0))
        else:
            top = aug_rng.randint(0, max_top)
        left = aug_rng.randint(0, max_left)
    elif portrait and max_top > 0:
        top = max_top // 4
        left = max_left // 2
    else:
        left = max_left // 2
        top = max_top // 2

    img = img.crop((left, top, left + target_w, top + target_h))
    if augmentation_index > 0:
        if aug_rng is None:
            aug_rng = _augmentation_rng(path, augmentation_index)
        # Horizontal flip harms face memorization (faces are not symmetric). Portrait aspect
        # ratios are always spared; identity datasets disable it entirely via allow_flip.
        if allow_flip and not portrait and aug_rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = _photometric_jitter(img, aug_rng)
    return np.array(img).astype("float32") / 255.0


def _photometric_jitter(img: Any, rng: "random.Random") -> Any:
    from PIL import ImageEnhance

    brightness = 1.0 + (rng.random() - 0.5) * 0.2
    contrast = 1.0 + (rng.random() - 0.5) * 0.2
    img = ImageEnhance.Brightness(img).enhance(brightness)
    return ImageEnhance.Contrast(img).enhance(contrast)


_FACE_WINDOW_JITTER = 0.06


def _crop_source_window(
    img: Any,
    window: tuple[int, int, int, int],
    augmentation_index: int,
    path: Path,
) -> Any:
    """Crop ``window`` from the source image; augmentations shift it by up to 6% (in bounds)."""
    src_w, src_h = img.size
    left, top, w, h = (int(v) for v in window)
    w = max(1, min(w, src_w))
    h = max(1, min(h, src_h))
    if augmentation_index > 0:
        rng = _augmentation_rng(path, augmentation_index)
        left += int(round((rng.random() - 0.5) * 2 * _FACE_WINDOW_JITTER * w))
        top += int(round((rng.random() - 0.5) * 2 * _FACE_WINDOW_JITTER * h))
    left = min(max(0, left), src_w - w)
    top = min(max(0, top), src_h - h)
    return img.crop((left, top, left + w, top + h))
