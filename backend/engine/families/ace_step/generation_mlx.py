"""
ACE-Step text-to-music — pure MLX (no PyTorch).

Conditioning, DiT diffusion, and VAE decode all run on MLX.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from backend.engine.runtime.mlx_runtime import random_normal, seeded_random_normal
from backend.engine.families.ace_step.audio.condition_mlx import (
    load_condition_encoder_mlx,
    load_qwen3_embedding_mlx,
    prepare_condition_mlx,
)
from backend.engine.families.ace_step.generation import (
    COVER_DIT_INSTRUCTION,
    DEFAULT_DIT_INSTRUCTION,
    LATENT_HOP_SAMPLES,
    SAMPLE_RATE,
    SFT_GEN_PROMPT,
    VAE_TILE_CHUNK_SIZE,
    VAE_TILE_OVERLAP,
    diffusion_retry_seed,
    duration_to_latent_frames,
    estimate_hum_ratio,
    estimate_mains_correlation,
    capture_inference_lyrics,
    ensure_vocal_caption_hint,
    finalize_lyrics_for_inference,
    resolve_vocal_language,
    vocal_language_mismatch_warning,
    warn_weak_vocal_lyrics,
    warn_lyrics_token_truncation,
    LYRIC_TOKEN_MAX_LENGTH,
    format_lyrics,
    format_metadata,
    latents_collapsed_to_silence,
    latent_cosine_similarity,
    latent_diff_mean,
    load_silence_latent_numpy,
    normalize_waveform,
    resolve_dit_bundle,
    resolve_silence_latent_path,
    snap_latent_frames_for_inference,
)
from backend.engine.families.ace_step.transformer import AceStepTransformer
from backend.engine.families.ace_step.transformer_mlx import _CrossAttentionCache
from backend.engine.families.ace_step.vae.vae import AceStepVAE
from backend.engine.families.ace_step.weights_mlx import load_decoder_safetensors_for_mlx

logger = logging.getLogger(__name__)


class AceStepMlxGenerator:
    """MLX-only ACE-Step generator."""

    def __init__(
        self,
        ctx: Any,
        bundle_root: Path,
        *,
        entry: Any | None = None,
        version_key: str | None = None,
    ):
        self._ctx = ctx
        self._bundle_root = Path(bundle_root)
        self._registry_entry = entry
        self._version_key = version_key
        self._condition_encoder: Any = None
        self._audio_codec: Any = None
        self._dit: AceStepTransformer | None = None
        self._vae: AceStepVAE | None = None
        self._text_encoder: Any = None
        self._text_tokenizer: Any = None
        self._silence_latent: np.ndarray | None = None
        self._model_config: Any = None
        self.last_latent_frames: int = 0
        self.last_hum_ratio: float = 0.0
        self.last_mains_acf: float = 0.0
        self.last_latent_cos: float = 0.0
        self.last_latent_diff_mean: float = 0.0
        self.last_decode_mode: str = "mlx_vae"
        self.last_lm_expanded: bool = False
        self.last_lyrics_capture: Any = None
        self.last_quality: Any = None
        self.last_audio_code_indices: tuple[int, ...] = ()
        self.last_pmi: Any = None
        self._lm_formatter: Any = None
        self._lm_formatter_key: tuple[Any, ...] | None = None

    @property
    def model_config(self) -> Any:
        if self._model_config is None:
            raise RuntimeError("ACE-Step model not loaded")
        return self._model_config

    def load(self) -> None:
        import json

        from transformers import AutoConfig, AutoTokenizer

        bundle = self._bundle_root
        dit_subdir = None
        if self._registry_entry is not None:
            from backend.engine.families.ace_step.weights import ace_step_dit_subdir_for_model

            dit_subdir = ace_step_dit_subdir_for_model(str(getattr(self._registry_entry, "id", "") or ""))
        dit_bundle = resolve_dit_bundle(bundle, dit_subdir=dit_subdir)
        cfg_path = dit_bundle / "config.json"
        if not cfg_path.is_file():
            raise RuntimeError(f"ACE-Step config.json missing under {dit_bundle}")

        with open(cfg_path, encoding="utf-8") as f:
            self._model_config = json.loads(f.read())

        logger.info("Loading ACE-Step MLX condition encoder + DiT + VAE from %s", dit_bundle)
        self._condition_encoder = load_condition_encoder_mlx(
            dit_bundle, eval_fn=self._ctx.eval, array_fn=self._ctx.array
        )
        from backend.engine.families.ace_step.audio.audio_codec_mlx import load_audio_codec_mlx

        self._audio_codec = load_audio_codec_mlx(
            dit_bundle, eval_fn=self._ctx.eval, array_fn=self._ctx.array
        )

        cfg = self._model_config
        from backend.engine.common.bundle.quant_inference import resolve_dit_inference_weight_mode
        from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized

        dit_path = dit_bundle / "model.safetensors"
        dit_weights = load_decoder_safetensors_for_mlx(
            str(dit_path), array_fn=self._ctx.array
        )
        weight_items = list(dit_weights)
        weight_keys = frozenset(k for k, _ in weight_items)
        bundle_affine_bits = read_bundle_affine_bits_if_quantized(dict(weight_items), dit_path)
        inference_mode = resolve_dit_inference_weight_mode(
            self._ctx,
            entry=self._registry_entry,
            version_key=self._version_key,
            weight_keys=weight_keys,
            bundle_affine_bits=bundle_affine_bits,
        )

        self._dit = AceStepTransformer(
            self._ctx,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            rms_norm_eps=cfg["rms_norm_eps"],
            attention_bias=cfg["attention_bias"],
            in_channels=cfg["in_channels"],
            audio_acoustic_hidden_dim=cfg["audio_acoustic_hidden_dim"],
            patch_size=cfg["patch_size"],
            sliding_window=cfg.get("sliding_window", 128),
            layer_types=tuple(cfg["layer_types"]) if cfg.get("layer_types") else None,
            rope_theta=cfg["rope_theta"],
            max_position_embeddings=cfg["max_position_embeddings"],
        )
        self._dit.load_weights(
            weight_items,
            strict=False,
            ctx=self._ctx,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=inference_mode,
        )
        setattr(self._dit, "_dq_inference_mode", inference_mode)
        self._ctx.eval(self._dit.parameters())

        enc_dir = bundle / "Qwen3-Embedding-0.6B"
        if not enc_dir.is_dir():
            raise RuntimeError(
                f"Qwen3-Embedding-0.6B not found at {enc_dir}. "
                "Download it from Hugging Face: Qwen/Qwen3-Embedding-0.6B"
            )
        self._text_tokenizer = AutoTokenizer.from_pretrained(str(enc_dir))
        self._text_encoder = load_qwen3_embedding_mlx(
            enc_dir,
            eval_fn=self._ctx.eval,
            load_fn=getattr(self._ctx, "load_weights", None),
        )

        sp = resolve_silence_latent_path(bundle, dit_bundle)
        self._silence_latent = load_silence_latent_numpy(sp)

        vae_dir = bundle / "vae"
        self._vae = AceStepVAE(self._ctx, vae_dir=str(vae_dir))

    @staticmethod
    def _lm_enabled() -> bool:
        import os

        return os.environ.get("ACESTEP_USE_LM", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def _ensure_lm_formatter(
        self,
        *,
        quantize_bits: Optional[int] = None,
        lm_dir: Optional[Path] = None,
    ) -> Any:
        key = (quantize_bits, str(lm_dir) if lm_dir else "")
        if self._lm_formatter is not None and self._lm_formatter_key == key:
            return self._lm_formatter
        from backend.engine.families.ace_step.lm.lm_format_mlx import AceStepLmFormatterMlx
        from backend.engine.families.ace_step.quality.resource_policy import resolve_lm_dir_for_policy, resolve_resource_policy

        policy = resolve_resource_policy()
        resolved_dir = lm_dir or resolve_lm_dir_for_policy(self._bundle_root, policy)
        fmt = AceStepLmFormatterMlx.from_bundle(
            self._bundle_root,
            ctx=self._ctx,
            lm_dir=resolved_dir,
            quantize_bits=quantize_bits if quantize_bits is not None else policy.lm_quantize_bits,
        )
        if fmt is None:
            return None
        fmt.load()
        self._lm_formatter = fmt
        self._lm_formatter_key = key
        return fmt

    def _silence_latent_slice(self, length: int) -> np.ndarray:
        available = self._silence_latent.shape[1]
        if length <= available:
            return self._silence_latent[0, :length, :]
        repeats = (length + available - 1) // available
        tiled = np.tile(self._silence_latent[0], (repeats, 1))
        return tiled[:length, :]

    def _infer_refer_latent(self) -> tuple[np.ndarray, np.ndarray]:
        refer = self._silence_latent[:, :750, :]
        order = np.array([0], dtype=np.int32)
        return refer, order

    @staticmethod
    def _is_turbo(config: dict) -> bool:
        return bool(config.get("is_turbo", False))

    def _decode_latents_mlx(self, latents_nlc: Any) -> Any:
        latents = self._ctx.array(np.asarray(latents_nlc, dtype=np.float32))
        if latents.ndim == 2:
            latents = mx.expand_dims(latents, 0)
        t_len = int(latents.shape[1])
        if t_len <= VAE_TILE_CHUNK_SIZE:
            return self._vae.decode(latents)

        logger.info(
            "ACE-Step MLX VAE tiled decode: T=%d chunk=%d overlap=%d",
            t_len,
            VAE_TILE_CHUNK_SIZE,
            VAE_TILE_OVERLAP,
        )
        chunk_size = VAE_TILE_CHUNK_SIZE
        overlap = VAE_TILE_OVERLAP
        stride = chunk_size - 2 * overlap
        num_steps = (t_len + stride - 1) // stride
        chunks: list[Any] = []
        upsample: float | None = None

        for step_idx in range(num_steps):
            core_start = step_idx * stride
            core_end = min(core_start + stride, t_len)
            win_start = max(0, core_start - overlap)
            win_end = min(t_len, core_end + overlap)
            latent_chunk = latents[:, win_start:win_end, :]
            audio_chunk = self._vae.decode(latent_chunk)
            if upsample is None:
                upsample = float(audio_chunk.shape[1]) / float(latent_chunk.shape[1])
            trim_start = int(round((core_start - win_start) * upsample))
            trim_end = int(round((win_end - core_end) * upsample))
            audio_len = int(audio_chunk.shape[1])
            end_idx = audio_len - trim_end if trim_end > 0 else audio_len
            chunks.append(audio_chunk[:, trim_start:end_idx, :])

        return mx.concatenate(chunks, axis=1)

    def generate_waveform(
        self,
        *,
        prompt: str,
        lyrics: str = "[Instrumental]",
        vocal_language: str = "en",
        duration: float = 30.0,
        steps: int = 50,
        guidance: float = 3.0,
        seed: int = 0,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        shift: float = 3.0,
        instrumental: bool = False,
        lm_enabled: bool = True,
        lm_quantize_bits: Optional[int] = None,
    ) -> np.ndarray:
        del steps, guidance  # turbo schedule is shift-driven
        from backend.engine.families.ace_step.quality.quality_score import assess_generation_quality

        if self._condition_encoder is None or self._dit is None or self._vae is None:
            raise RuntimeError("ACE-Step MLX generator not loaded; call load() first")

        duration = max(10.0, min(600.0, float(duration)))
        requested_samples = int(round(duration * SAMPLE_RATE))
        latent_frames = duration_to_latent_frames(duration)
        max_latent_length = snap_latent_frames_for_inference(latent_frames)
        self.last_latent_frames = max_latent_length
        if max_latent_length != latent_frames:
            logger.info(
                "ACE-Step: latent frames %d → %d (even length avoids turbo DiT collapse)",
                latent_frames,
                max_latent_length,
            )

        silence_tiled = self._silence_latent_slice(max_latent_length)
        instruction = DEFAULT_DIT_INSTRUCTION
        if not instruction.endswith(":"):
            instruction = instruction + ":"
        caption = (prompt or "").strip()
        lyrics_input = (lyrics or "").strip()
        lyrics_use = lyrics_input or "[Instrumental]"
        lang_use = resolve_vocal_language(lyrics_use or lyrics_input, vocal_language)
        bpm_use = bpm
        key_use = key_scale or ""
        ts_use = time_signature or ""
        from backend.engine.families.ace_step.lm.lm_format import is_instrumental_lyrics

        self.last_lm_expanded = False
        self.last_audio_code_indices = ()
        self.last_pmi = None
        mismatch = vocal_language_mismatch_warning(lyrics_use, lang_use)
        if mismatch:
            logger.warning("ACE-Step: %s", mismatch)

        if lm_enabled and self._lm_enabled():
            lm_fmt = self._ensure_lm_formatter(
                quantize_bits=lm_quantize_bits,
            )
        else:
            lm_fmt = None

        def _lm_stage() -> None:
            nonlocal caption, lyrics_use, duration, requested_samples, max_latent_length
            nonlocal silence_tiled, bpm_use, key_use, ts_use, lang_use
            if lm_fmt is None:
                if lm_enabled and self._lm_enabled():
                    logger.warning(
                        "ACE-Step 5Hz LM not found in bundle; caption/lyrics expansion skipped"
                    )
                return
            try:
                logger.info("ACE-Step: 5Hz LM format_sample (MLX)...")
                expanded = lm_fmt.format_sample(
                    caption=caption or "instrumental music",
                    lyrics=lyrics_use,
                    duration=duration,
                    bpm=bpm_use,
                    keyscale=key_use,
                    timesignature=ts_use,
                    language=lang_use,
                )
                caption = expanded.caption
                lyrics_use = expanded.lyrics
                if expanded.bpm is not None:
                    bpm_use = expanded.bpm
                if expanded.duration is not None:
                    duration = float(expanded.duration)
                    requested_samples = int(round(duration * SAMPLE_RATE))
                    latent_frames = duration_to_latent_frames(duration)
                    max_latent_length = snap_latent_frames_for_inference(latent_frames)
                    self.last_latent_frames = max_latent_length
                    silence_tiled = self._silence_latent_slice(max_latent_length)
                key_use = expanded.keyscale or key_use
                ts_use = expanded.timesignature or ts_use
                lang_use = expanded.language or lang_use
                self.last_lm_expanded = True
                self.last_audio_code_indices = tuple(expanded.audio_code_indices)
                self.last_pmi = getattr(lm_fmt, "last_pmi", None)
                if expanded.audio_code_indices:
                    logger.info(
                        "ACE-Step: LM produced %d audio codes for DiT hints",
                        len(expanded.audio_code_indices),
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"ACE-Step 5Hz LM expansion failed: {exc}"
                ) from exc

        def _dit_stage(_stage1: Any) -> np.ndarray:
            nonlocal caption, lyrics_use, lang_use, shift

            lyrics_use = finalize_lyrics_for_inference(
                lyrics_use,
                instrumental=instrumental,
                lm_expanded=self.last_lm_expanded,
            )
            caption = ensure_vocal_caption_hint(caption, lyrics_use, lang_use)
            self.last_lyrics_capture = capture_inference_lyrics(
                lyrics_input=lyrics_input,
                lyrics_effective=lyrics_use,
                caption_effective=caption,
                lm_expanded=self.last_lm_expanded,
            )

            vocal_warn = warn_weak_vocal_lyrics(lyrics_use)
            if vocal_warn:
                logger.warning("ACE-Step: %s", vocal_warn)

            if not is_instrumental_lyrics(lyrics_use):
                logger.info(
                    "ACE-Step vocals enabled (lyrics %d chars, lang=%s)",
                    len(lyrics_use),
                    lang_use,
                )
            else:
                logger.info("ACE-Step instrumental mode (no vocals)")

            metadata = format_metadata(
                duration=duration,
                bpm=bpm_use,
                key_scale=key_use,
                time_signature=ts_use,
            )
            text_prompt = SFT_GEN_PROMPT.format(
                instruction=instruction,
                caption=caption,
                metadata=metadata,
            )
            lyrics_text = format_lyrics(lyrics_use, lang_use)

            text_tok = self._text_tokenizer(
                text_prompt,
                padding="longest",
                truncation=True,
                max_length=256,
                return_tensors="np",
            )
            lyric_len_tok = int(
                self._text_tokenizer(lyrics_text, truncation=False, return_tensors="np")["input_ids"].shape[1]
            )
            trunc_warn = warn_lyrics_token_truncation(lyric_len_tok)
            if trunc_warn:
                logger.warning("ACE-Step: %s", trunc_warn)
            lyric_tok = self._text_tokenizer(
                lyrics_text,
                padding="longest",
                truncation=True,
                max_length=LYRIC_TOKEN_MAX_LENGTH,
                return_tensors="np",
            )
            text_ids = self._ctx.array(text_tok["input_ids"].astype(np.int32))
            text_mask = self._ctx.array(text_tok["attention_mask"].astype(np.float32))
            lyric_ids = self._ctx.array(lyric_tok["input_ids"].astype(np.int32))
            lyric_mask = self._ctx.array(lyric_tok["attention_mask"].astype(np.float32))

            text_hidden = self._text_encoder.encode(text_ids, text_mask)
            lyric_hidden = self._text_encoder.token_embed(lyric_ids)

            refer_np, refer_order = self._infer_refer_latent()
            refer_packed = self._ctx.array(refer_np.astype(np.float32))
            refer_order_mx = self._ctx.array(refer_order.astype(np.int32))

            src_latents_np = silence_tiled.astype(np.float32)[np.newaxis, ...]
            src_latents = self._ctx.array(src_latents_np)
            latent_dim = int(src_latents.shape[-1])
            chunk_masks = mx.ones((1, max_latent_length, latent_dim), dtype=mx.float32)
            is_covers = mx.zeros((1,), dtype=mx.int32)
            attention_mask = mx.ones((1, max_latent_length), dtype=mx.float32)
            silence_latent_mx = self._ctx.array(self._silence_latent.astype(np.float32))
            audio_code_mx = None
            if self.last_audio_code_indices:
                pool = int(getattr(self._audio_codec, "pool_window_size", 5))
                max_codes = max(1, max_latent_length // pool)
                idx = np.array(self.last_audio_code_indices[:max_codes], dtype=np.int32)
                audio_code_mx = mx.array(idx.reshape(1, -1, 1), dtype=mx.int32)
                is_covers = mx.ones((1,), dtype=mx.int32)

            if self._is_turbo(self._model_config) and shift >= 2.5:
                shift = 1.0

            enc_hs, _, ctx = prepare_condition_mlx(
                self._condition_encoder,
                text_hidden_states=text_hidden,
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden,
                lyric_attention_mask=lyric_mask,
                refer_packed=refer_packed,
                refer_order=refer_order_mx,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                silence_latent=silence_latent_mx,
                attention_mask=attention_mask,
                audio_codec=self._audio_codec,
                audio_code_indices=audio_code_mx,
            )
            enc_np = np.array(enc_hs, dtype=np.float32)
            ctx_np = np.array(ctx, dtype=np.float32)
            src_shape = tuple(src_latents_np.shape)

            latents = None
            latent_cos = 1.0
            latent_diff = 0.0
            max_attempts = 6 if self.last_lm_expanded else 3
            run_seed = int(seed)

            for attempt in range(max_attempts):
                out = mlx_generate_diffusion(
                    self._dit._model,
                    encoder_hidden_states_np=enc_np,
                    context_latents_np=ctx_np,
                    src_latents_shape=src_shape,
                    seed=run_seed,
                    infer_method="ode",
                    shift=shift,
                    eval_fn=self._ctx.eval,
                    array_fn=self._ctx.array,
                    randn_fn=getattr(self._ctx, "randn", None),
                    seeded_randn_fn=getattr(self._ctx, "seeded_randn", None),
                )
                latents = out["target_latents"]
                collapsed, latent_cos, latent_diff = latents_collapsed_to_silence(
                    latents, src_latents_np,
                )
                if not collapsed:
                    if attempt > 0:
                        logger.info(
                            "ACE-Step MLX diffusion recovered on attempt %d "
                            "(cos=%.4f, diff=%.4f, seed=%d)",
                            attempt + 1,
                            latent_cos,
                            latent_diff,
                            run_seed,
                        )
                    break
                if attempt < max_attempts - 1:
                    next_seed = diffusion_retry_seed(int(seed), attempt + 1)
                    logger.warning(
                        "ACE-Step MLX latents near silence (cos=%.4f, diff=%.4f); "
                        "retry seed %d (attempt %d/%d)",
                        latent_cos,
                        latent_diff,
                        next_seed,
                        attempt + 1,
                        max_attempts,
                    )
                    run_seed = next_seed
            else:
                logger.warning(
                    "ACE-Step MLX latents remain near silence after %d seeds "
                    "(cos=%.4f, diff=%.4f)",
                    max_attempts,
                    latent_cos,
                    latent_diff,
                )

            self.last_latent_cos = latent_cos
            self.last_latent_diff_mean = latent_diff
            logger.info(
                "ACE-Step MLX diffusion done (cos=%.4f, diff=%.4f)",
                latent_cos,
                latent_diff,
            )

            audio_nlc = self._decode_latents_mlx(latents)
            wf = np.array(audio_nlc)
            if wf.ndim == 3:
                wf = wf[0]
            if wf.ndim == 2 and wf.shape[0] <= 8 and wf.shape[0] < wf.shape[1]:
                wf = wf.T
            wf = normalize_waveform(wf.astype(np.float32))
            if wf.shape[0] > requested_samples:
                wf = wf[:requested_samples]

            hum_ratio = estimate_hum_ratio(wf)
            mains_acf = estimate_mains_correlation(wf)
            self.last_hum_ratio = hum_ratio
            self.last_mains_acf = mains_acf

            collapsed, _, _ = latents_collapsed_to_silence(latents, src_latents_np)
            if collapsed and mains_acf > 0.55:
                raise RuntimeError(
                    "ACE-Step MLX produced near-silence latents that decode as tonal hum "
                    f"(latent_cos={latent_cos:.4f}, latent_diff={latent_diff:.4f}, "
                    f"mains_acf={mains_acf:.3f}). Try another seed or a more specific prompt."
                )
            if mains_acf > 0.4:
                logger.warning(
                    "ACE-Step MLX output may contain mains hum (acf=%.3f); try another seed",
                    mains_acf,
                )
            self.last_quality = assess_generation_quality(
                hum_ratio=hum_ratio,
                mains_acf=mains_acf,
                latent_cos=latent_cos,
                latent_diff=latent_diff,
                lm_expanded=self.last_lm_expanded,
                near_silence=collapsed,
                pmi_bonus=self.last_pmi.quality_bonus() if self.last_pmi is not None else 0.0,
            )
            return wf

        if lm_fmt is not None:
            from backend.engine.inference.two_stage import TwoStageBundle, TwoStageInference

            return TwoStageInference().run(
                TwoStageBundle(stage1_fn=_lm_stage, stage2_fn=_dit_stage)
            )["latents"]

        _lm_stage()
        return _dit_stage(None)

    def _encode_reference_latents(self, reference_waveform: np.ndarray) -> np.ndarray:
        wf = np.asarray(reference_waveform, dtype=np.float32)
        if wf.ndim == 1:
            wf = wf[:, None]
        if wf.shape[0] <= 8 and wf.shape[0] < wf.shape[1]:
            wf = wf.T
        audio = self._ctx.array(wf.astype(np.float32)[np.newaxis, ...])
        latents = self._vae._vae.encode_mean(audio)
        self._ctx.eval(latents)
        out = np.array(latents, dtype=np.float32)
        if out.ndim == 2:
            out = out[np.newaxis, ...]
        return out

    def generate_cover_waveform(
        self,
        *,
        reference_waveform: np.ndarray,
        prompt: str = "",
        lyrics: str = "[Instrumental]",
        vocal_language: str = "en",
        duration: Optional[float] = None,
        seed: int = 0,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        shift: float = 3.0,
        audio_cover_strength: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        from backend.engine.families.ace_step.quality.quality_score import assess_generation_quality

        if (
            self._condition_encoder is None
            or self._dit is None
            or self._vae is None
            or self._audio_codec is None
        ):
            raise RuntimeError("ACE-Step MLX generator not loaded; call load() first")

        ref_latents_np = self._encode_reference_latents(reference_waveform)
        ref_len = int(ref_latents_np.shape[1])
        if duration is not None:
            max_latent_length = snap_latent_frames_for_inference(
                duration_to_latent_frames(float(duration))
            )
        else:
            max_latent_length = snap_latent_frames_for_inference(ref_len)
        max_latent_length = max(max_latent_length, ref_len)
        self.last_latent_frames = max_latent_length

        if ref_len < max_latent_length:
            pad = self._silence_latent_slice(max_latent_length - ref_len)
            src_latents_np = np.concatenate(
                [ref_latents_np[0], pad.astype(np.float32)],
                axis=0,
            )[np.newaxis, ...]
        else:
            src_latents_np = ref_latents_np[:, :max_latent_length, :].copy()

        refer_len = min(750, int(src_latents_np.shape[1]))
        refer_packed = self._ctx.array(src_latents_np[:, :refer_len, :].astype(np.float32))
        refer_order_mx = self._ctx.array(np.array([0], dtype=np.int32))

        duration_sec = max_latent_length * LATENT_HOP_SAMPLES / SAMPLE_RATE
        requested_samples = int(round(duration_sec * SAMPLE_RATE))
        caption = (prompt or "").strip() or "cover generation"
        lyrics_use = (lyrics or "").strip() or "[Instrumental]"
        lang_use = resolve_vocal_language(lyrics_use, vocal_language)
        self.last_lm_expanded = False
        self.last_lyrics_capture = capture_inference_lyrics(
            lyrics_input=lyrics_use,
            lyrics_effective=lyrics_use,
            caption_effective=caption,
            lm_expanded=False,
        )

        instruction = COVER_DIT_INSTRUCTION
        if not instruction.endswith(":"):
            instruction = instruction + ":"
        metadata = format_metadata(
            duration=duration_sec,
            bpm=bpm,
            key_scale=key_scale or "",
            time_signature=time_signature or "",
        )
        text_prompt = SFT_GEN_PROMPT.format(
            instruction=instruction,
            caption=caption,
            metadata=metadata,
        )
        lyrics_text = format_lyrics(lyrics_use, lang_use)

        text_tok = self._text_tokenizer(
            text_prompt,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="np",
        )
        lyric_tok = self._text_tokenizer(
            lyrics_text,
            padding="longest",
            truncation=True,
            max_length=LYRIC_TOKEN_MAX_LENGTH,
            return_tensors="np",
        )
        text_ids = self._ctx.array(text_tok["input_ids"].astype(np.int32))
        text_mask = self._ctx.array(text_tok["attention_mask"].astype(np.float32))
        lyric_ids = self._ctx.array(lyric_tok["input_ids"].astype(np.int32))
        lyric_mask = self._ctx.array(lyric_tok["attention_mask"].astype(np.float32))

        text_hidden = self._text_encoder.encode(text_ids, text_mask)
        lyric_hidden = self._text_encoder.token_embed(lyric_ids)

        src_latents = self._ctx.array(src_latents_np.astype(np.float32))
        latent_dim = int(src_latents.shape[-1])
        chunk_masks = mx.ones((1, max_latent_length, latent_dim), dtype=mx.float32)
        is_covers = mx.ones((1,), dtype=mx.int32)
        attention_mask = mx.ones((1, max_latent_length), dtype=mx.float32)
        silence_latent_mx = self._ctx.array(self._silence_latent.astype(np.float32))

        if self._is_turbo(self._model_config) and shift >= 2.5:
            shift = 1.0

        strength = float(max(0.0, min(1.0, audio_cover_strength)))

        enc_hs, _, ctx = prepare_condition_mlx(
            self._condition_encoder,
            text_hidden_states=text_hidden,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hidden,
            lyric_attention_mask=lyric_mask,
            refer_packed=refer_packed,
            refer_order=refer_order_mx,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=silence_latent_mx,
            attention_mask=attention_mask,
            audio_codec=self._audio_codec,
        )

        enc_hs_nc, ctx_nc = None, None
        if strength < 1.0:
            silence_tiled = self._silence_latent_slice(max_latent_length)
            silence_src = self._ctx.array(silence_tiled.astype(np.float32)[np.newaxis, ...])
            is_covers_zero = mx.zeros((1,), dtype=mx.int32)
            enc_hs_nc, _, ctx_nc = prepare_condition_mlx(
                self._condition_encoder,
                text_hidden_states=text_hidden,
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden,
                lyric_attention_mask=lyric_mask,
                refer_packed=refer_packed,
                refer_order=refer_order_mx,
                src_latents=silence_src,
                chunk_masks=chunk_masks,
                is_covers=is_covers_zero,
                silence_latent=silence_latent_mx,
                attention_mask=attention_mask,
                audio_codec=self._audio_codec,
            )

        enc_np = np.array(enc_hs, dtype=np.float32)
        ctx_np = np.array(ctx, dtype=np.float32)
        enc_np_nc = np.array(enc_hs_nc, dtype=np.float32) if enc_hs_nc is not None else None
        ctx_np_nc = np.array(ctx_nc, dtype=np.float32) if ctx_nc is not None else None
        src_shape = tuple(src_latents_np.shape)

        run_seed = int(seed)
        out = mlx_generate_diffusion(
            self._dit._model,
            encoder_hidden_states_np=enc_np,
            context_latents_np=ctx_np,
            src_latents_shape=src_shape,
            seed=run_seed,
            infer_method="ode",
            shift=shift,
            audio_cover_strength=strength,
            encoder_hidden_states_non_cover_np=enc_np_nc,
            context_latents_non_cover_np=ctx_np_nc,
            eval_fn=self._ctx.eval,
            array_fn=self._ctx.array,
            randn_fn=getattr(self._ctx, "randn", None),
            seeded_randn_fn=getattr(self._ctx, "seeded_randn", None),
        )
        latents = out["target_latents"]
        latent_cos = latent_cosine_similarity(latents, src_latents_np)
        latent_diff = latent_diff_mean(latents, src_latents_np)
        self.last_latent_cos = latent_cos
        self.last_latent_diff_mean = latent_diff

        audio_nlc = self._decode_latents_mlx(latents)
        wf = np.array(audio_nlc)
        if wf.ndim == 3:
            wf = wf[0]
        if wf.ndim == 2 and wf.shape[0] <= 8 and wf.shape[0] < wf.shape[1]:
            wf = wf.T
        wf = normalize_waveform(wf.astype(np.float32))
        if wf.shape[0] > requested_samples:
            wf = wf[:requested_samples]

        self.last_hum_ratio = estimate_hum_ratio(wf)
        self.last_mains_acf = estimate_mains_correlation(wf)
        collapsed, _, _ = latents_collapsed_to_silence(latents, src_latents_np)
        self.last_quality = assess_generation_quality(
            hum_ratio=self.last_hum_ratio,
            mains_acf=self.last_mains_acf,
            latent_cos=latent_cos,
            latent_diff=latent_diff,
            lm_expanded=False,
            near_silence=collapsed,
        )
        return wf


# --- MLX diffusion loop (merged from diffusion_mlx) ---

VALID_SHIFTS = [1.0, 2.0, 3.0]

SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [
        1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
        0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222,
    ],
    3.0: [
        1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
        0.6428571428571429, 0.5, 0.3,
    ],
}


def get_timestep_schedule(
    shift: float = 3.0,
    timesteps: Optional[list] = None,
) -> List[float]:
    """Timestep schedule for turbo DiT (matches upstream ``generate_audio``)."""
    if timesteps is not None:
        ts_list = list(timesteps)
        while ts_list and ts_list[-1] == 0:
            ts_list.pop()
        if len(ts_list) >= 1:
            valid = [
                1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
                0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
                0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
                0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
            ]
            return [min(valid, key=lambda x, t=t: abs(x - t)) for t in ts_list[:20]]

    original_shift = shift
    shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
    if original_shift != shift:
        logger.warning(
            "ACE-Step shift=%.2f rounded to nearest valid shift=%.1f",
            original_shift,
            shift,
        )
    return SHIFT_TIMESTEPS[shift]


def mlx_generate_diffusion(
    mlx_decoder: Any,
    encoder_hidden_states_np: np.ndarray,
    context_latents_np: np.ndarray,
    src_latents_shape: Tuple[int, ...],
    seed: Optional[Union[int, List[int]]] = None,
    infer_method: str = "ode",
    shift: float = 3.0,
    timesteps: Optional[list] = None,
    audio_cover_strength: float = 1.0,
    encoder_hidden_states_non_cover_np: Optional[np.ndarray] = None,
    context_latents_non_cover_np: Optional[np.ndarray] = None,
    eval_fn: Optional[Callable[..., None]] = None,
    array_fn: Optional[Callable[..., Any]] = None,
    randn_fn: Optional[Callable[..., Any]] = None,
    seeded_randn_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, object]:
    """Run MLX flow-matching diffusion via L2 ``FlowMatchingInference``."""
    from backend.engine.inference import (
        AudioInferenceBundle,
        FlowMatchingInference,
        FlowMatchingSpec,
    )

    total_start = time.time()
    if eval_fn is None:
        eval_fn = mx.eval
    if array_fn is None:
        array_fn = mx.array

    enc_hs = array_fn(encoder_hidden_states_np)
    ctx_lat = array_fn(context_latents_np)
    enc_hs_nc = (
        array_fn(encoder_hidden_states_non_cover_np)
        if encoder_hidden_states_non_cover_np is not None
        else None
    )
    ctx_nc = (
        array_fn(context_latents_non_cover_np)
        if context_latents_non_cover_np is not None
        else None
    )

    bsz = int(src_latents_shape[0])
    t_len = int(src_latents_shape[1])
    channels = int(src_latents_shape[2])
    noise_shape = (bsz, t_len, channels)

    t_schedule_list = get_timestep_schedule(shift, timesteps)
    num_steps = len(t_schedule_list)
    cover_steps = int(num_steps * audio_cover_strength)

    # -- Build closures for AudioInferenceBundle --

    # Mutable state bag (shared between model_forward and before_step_fn)
    _state_enc_hs = [enc_hs]
    _state_ctx = [ctx_lat]

    def _model_forward(xt: Any, t_curr_val: float, state: dict) -> Any:
        t_curr = mx.full((bsz,), t_curr_val)
        vt, new_cache = mlx_decoder(
            hidden_states=xt,
            timestep=t_curr,
            timestep_r=t_curr,
            encoder_hidden_states=_state_enc_hs[0],
            context_latents=_state_ctx[0],
            cache=state.get("cache"),
            use_cache=True,
        )
        state["cache"] = new_cache
        return vt

    def _init_noise(shape: tuple, _seed: int) -> Any:
        if seed is None:
            return random_normal(randn_fn, shape)
        if isinstance(seed, list):
            parts = []
            for s in seed:
                if s is None or s < 0:
                    parts.append(random_normal(randn_fn, (1, t_len, channels)))
                else:
                    parts.append(seeded_random_normal(seeded_randn_fn, (1, t_len, channels), int(s)))
            return mx.concatenate(parts, axis=0)
        return seeded_random_normal(seeded_randn_fn, shape, int(seed))

    def _euler_step(x: Any, v: Any, t_curr: float, t_next: float | None, step_idx: int) -> Any:
        if t_next is None:
            # Last step: use full timestep as dt
            t_unsq = mx.expand_dims(mx.expand_dims(mx.full((bsz,), t_curr), axis=-1), axis=-1)
            return x - v * t_unsq
        if infer_method == "sde":
            t_unsq = mx.expand_dims(mx.expand_dims(mx.full((bsz,), t_curr), axis=-1), axis=-1)
            pred_clean = x - v * t_unsq
            return t_next * random_normal(randn_fn, tuple(x.shape)) + (1.0 - t_next) * pred_clean
        dt = t_curr - t_next
        return x - v * mx.full((bsz, 1, 1), dt)

    def _before_step(step_idx: int, state: dict) -> None:
        if step_idx >= cover_steps and enc_hs_nc is not None:
            _state_enc_hs[0] = enc_hs_nc
            _state_ctx[0] = ctx_nc
            state["cache"] = _CrossAttentionCache()

    bundle = AudioInferenceBundle(
        ctx=None,
        model_forward=_model_forward,
        latent_shape=noise_shape,
        seed=int(seed) if isinstance(seed, int) else 0,
        eval_fn=eval_fn,
        flow=FlowMatchingSpec(
            timestep_schedule=t_schedule_list,
            init_noise_fn=_init_noise,
            euler_step_fn=_euler_step,
            cache_init_fn=lambda: _CrossAttentionCache(),
            before_step_fn=_before_step if enc_hs_nc is not None else None,
        ),
    )

    result = FlowMatchingInference().run(bundle)
    xt = result["latents"]

    total_end = time.time()
    time_costs = result.get("time_costs", {})
    time_costs["total_time_cost"] = total_end - total_start

    return {
        "target_latents": np.array(xt),
        "time_costs": time_costs,
    }
