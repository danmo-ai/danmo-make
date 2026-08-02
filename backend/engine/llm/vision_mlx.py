"""Optional MLX-VLM helpers for canvas node vision description."""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path
from typing import Any

from backend.core.contracts import ChatMessage
from backend.engine.llm.prompts.system import (
    IMAGE_TO_PROMPT_INSTRUCTION,
    VIDEO_FRAME_TO_PROMPT_INSTRUCTION,
    VISION_DESCRIBE_PROMPT,
)

logger = logging.getLogger(__name__)

VLM_MAX_IMAGE_EDGE = 768


def mlx_vlm_importable() -> bool:
    try:
        import mlx_vlm  # noqa: F401

        return True
    except ImportError:
        return False


def vision_weights_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return (
        (model_dir / "model.safetensors").is_file()
        or any(f.suffix == ".safetensors" for f in model_dir.rglob("*") if f.is_file())
        or any(f.suffix == ".bin" for f in model_dir.rglob("*") if f.is_file())
    )


def prepare_image_for_vlm(image_path: Path, *, max_edge: int = VLM_MAX_IMAGE_EDGE) -> tuple[Path, bool]:
    """Downscale large inputs before VLM to reduce Metal memory pressure. Returns (path, is_temp)."""
    from PIL import Image

    with Image.open(image_path) as img:
        img.load()
        w, h = img.size
        if min(w, h) <= max_edge:
            return image_path, False
        scale = max_edge / min(w, h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        rgb = img.convert("RGB")
        resized = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        fd, tmp_name = tempfile.mkstemp(suffix=".jpg", prefix="dq_vlm_")
        import os

        os.close(fd)
        tmp_path = Path(tmp_name)
        resized.save(tmp_path, format="JPEG", quality=88)
        return tmp_path, True


def _release_vlm_model(model: Any, processor: Any) -> None:
    del model
    del processor
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def analyze_image_files_batch_messages(
    image_paths: list[Path],
    model_dir: Path,
    *,
    messages: list[ChatMessage],
    metadata_hint: str = "",
    max_tokens: int = 256,
    temperature: float = 0.4,
) -> list[str]:
    """Run batch VLM with OpenAI-style messages (system + user text; images from paths)."""
    from backend.engine.llm.message_content import extract_vision_instruction

    instruction = extract_vision_instruction(messages)
    return analyze_image_files_batch(
        image_paths,
        model_dir,
        instruction=instruction,
        metadata_hint=metadata_hint,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def analyze_image_file_messages(
    image_path: Path,
    model_dir: Path,
    *,
    messages: list[ChatMessage],
    metadata_hint: str = "",
    max_tokens: int = 256,
    temperature: float = 0.4,
) -> str:
    texts = analyze_image_files_batch_messages(
        [image_path],
        model_dir,
        messages=messages,
        metadata_hint=metadata_hint,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return texts[0]


def analyze_images_multi(
    image_paths: list[Path],
    model_dir: Path,
    *,
    instruction: str,
    metadata_hint: str = "",
    max_tokens: int = 256,
    temperature: float = 0.4,
) -> str:
    """Single VLM inference with multiple images in one prompt."""
    if not image_paths:
        raise RuntimeError("analyze_images_multi requires at least one image")
    if not vision_weights_ready(model_dir):
        raise RuntimeError(f"Vision model weights not found under {model_dir}")

    try:
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm is not installed. Install with: pip install mlx-vlm"
        ) from exc

    prompt = instruction.strip()
    if metadata_hint.strip():
        prompt += f"\n\nOptional metadata context:\n{metadata_hint.strip()}"

    model, processor = load(str(model_dir))
    config = load_config(str(model_dir))
    temp_paths: list[Path] = []
    try:
        image_refs: list[str] = []
        for image_path in image_paths:
            if not image_path.is_file():
                raise RuntimeError(f"Vision analyze image not found: {image_path}")
            use_path, is_temp = prepare_image_for_vlm(image_path)
            if is_temp:
                temp_paths.append(use_path)
            image_refs.append(str(use_path))
        formatted = apply_chat_template(
            processor,
            config,
            prompt,
            num_images=len(image_refs),
        )
        raw = generate(
            model,
            processor,
            formatted,
            image_refs,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        text = _coerce_vlm_output_text(raw)
        if not text:
            raise RuntimeError("Vision model returned empty output for multi-image request")
        try:
            mx.clear_cache()
        except Exception:
            pass
        return text
    finally:
        _release_vlm_model(model, processor)
        for tmp in temp_paths:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def analyze_image_files_batch(
    image_paths: list[Path],
    model_dir: Path,
    *,
    instruction: str,
    metadata_hint: str = "",
    max_tokens: int = 256,
    temperature: float = 0.4,
) -> list[str]:
    """Load VLM once and analyze multiple images (reduces crash risk vs per-image load)."""
    if not image_paths:
        return []
    if not vision_weights_ready(model_dir):
        raise RuntimeError(f"Vision model weights not found under {model_dir}")

    try:
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm is not installed. Install with: pip install mlx-vlm"
        ) from exc

    prompt = instruction.strip()
    if metadata_hint.strip():
        prompt += f"\n\nOptional metadata context:\n{metadata_hint.strip()}"

    model, processor = load(str(model_dir))
    config = load_config(str(model_dir))
    temp_paths: list[Path] = []
    outputs: list[str] = []
    try:
        for image_path in image_paths:
            if not image_path.is_file():
                raise RuntimeError(f"Vision analyze image not found: {image_path}")
            use_path, is_temp = prepare_image_for_vlm(image_path)
            if is_temp:
                temp_paths.append(use_path)
            images = [str(use_path)]
            formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
            raw = generate(
                model,
                processor,
                formatted,
                images,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            text = _coerce_vlm_output_text(raw)
            if not text:
                raise RuntimeError(f"Vision model returned empty output for {image_path.name}")
            outputs.append(text)
            try:
                mx.clear_cache()
            except Exception:
                pass
    finally:
        _release_vlm_model(model, processor)
        for tmp in temp_paths:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return outputs


def analyze_image_file(
    image_path: Path,
    model_dir: Path,
    *,
    instruction: str,
    metadata_hint: str = "",
    max_tokens: int = 256,
    temperature: float = 0.4,
) -> str:
    """Run a single-image VLM task with a custom instruction. Raises RuntimeError on failure."""
    texts = analyze_image_files_batch(
        [image_path],
        model_dir,
        instruction=instruction,
        metadata_hint=metadata_hint,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return texts[0]


def _coerce_vlm_output_text(output: Any) -> str:
    """mlx-vlm 0.3+ returns GenerationResult; older versions may return str."""
    if isinstance(output, str):
        return output.strip()
    text = getattr(output, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    draft = getattr(output, "draft_text", None)
    if isinstance(draft, str) and draft.strip():
        return draft.strip()
    return str(output).strip()


def describe_image_file(
    image_path: Path,
    model_dir: Path,
    *,
    metadata_hint: str = "",
    max_tokens: int = 256,
) -> str:
    """Canvas node note via VLM."""
    return analyze_image_file(
        image_path,
        model_dir,
        instruction=VISION_DESCRIBE_PROMPT,
        metadata_hint=metadata_hint,
        max_tokens=max_tokens,
    )
