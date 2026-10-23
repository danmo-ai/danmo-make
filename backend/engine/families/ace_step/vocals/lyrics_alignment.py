"""Lyric timestamps: DiT cross-attention alignment (upstream parity) + structure estimate fallback."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

_SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")
DIT_ALIGNMENT_MODE = "dit_cross_attention"
LYRIC_HEADER_TEMPLATE = "# Languages\n{language}\n\n# Lyric\n"
QWEN_ENDOFTEXT_ID = 151643
_SINGABLE_LINE_RE = re.compile(r"^\s*(?!\[).+")


@dataclass(frozen=True)
class LyricsAlignmentSegment:
    label: str
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class LyricsAlignmentResult:
    mode: str
    segments: tuple[LyricsAlignmentSegment, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "lyrics_alignment": self.mode,
            "lyrics_segments": [
                {
                    "label": s.label,
                    "start_sec": round(s.start_sec, 2),
                    "end_sec": round(s.end_sec, 2),
                    "text": s.text[:200],
                }
                for s in self.segments
            ],
        }


def estimate_lyrics_alignment(
    lyrics: str,
    *,
    duration_sec: float,
    instrumental: bool = False,
) -> LyricsAlignmentResult:
    """Evenly allocate non-empty lyric lines across track duration by section."""
    if instrumental:
        return LyricsAlignmentResult(mode="instrumental", segments=())

    text = (lyrics or "").strip()
    if not text:
        return LyricsAlignmentResult(mode="empty", segments=())

    sections: list[tuple[str, list[str]]] = []
    current_label = "body"
    current_lines: list[str] = []

    for raw in text.splitlines():
        line = raw.rstrip()
        sec = _SECTION_RE.match(line)
        if sec:
            if current_lines:
                sections.append((current_label, current_lines))
            current_label = sec.group(1).strip() or "section"
            current_lines = []
            continue
        if _SINGABLE_LINE_RE.match(line) and line.strip():
            current_lines.append(line.strip())
    if current_lines:
        sections.append((current_label, current_lines))

    singable = [(label, lines) for label, lines in sections if lines]
    if not singable:
        return LyricsAlignmentResult(mode="conditioning_only", segments=())

    total_lines = sum(len(lines) for _, lines in singable)
    if total_lines <= 0 or duration_sec <= 0:
        return LyricsAlignmentResult(mode="conditioning_only", segments=())

    sec_duration = duration_sec / max(len(singable), 1)
    segments: list[LyricsAlignmentSegment] = []
    t = 0.0
    for label, lines in singable:
        block_end = min(duration_sec, t + sec_duration)
        line_dur = (block_end - t) / max(len(lines), 1)
        for i, ln in enumerate(lines):
            start = t + i * line_dur
            end = min(block_end, start + line_dur)
            segments.append(
                LyricsAlignmentSegment(
                    label=label,
                    start_sec=start,
                    end_sec=end,
                    text=ln,
                )
            )
        t = block_end

    return LyricsAlignmentResult(mode="structure_estimate", segments=tuple(segments))


def format_lrc(alignment: LyricsAlignmentResult) -> Optional[str]:
    if not alignment.segments:
        return None
    lines: list[str] = []
    for seg in alignment.segments:
        mm = int(seg.start_sec // 60)
        ss = seg.start_sec % 60
        ts = f"[{mm:02d}:{ss:05.2f}]"
        lines.append(f"{ts}{seg.text}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# DiT cross-attention alignment (``acestep/core/scoring/dit_alignment.py``)
# ---------------------------------------------------------------------------


def parse_layers_config(raw: Any) -> Optional[dict[int, list[int]]]:
    """``{"3": [18, 27], ...}`` from DiT ``config.json`` → ``{3: [18, 27]}``; ``None`` if absent."""
    if not raw or not isinstance(raw, dict):
        return None
    out = {int(k): [int(h) for h in v] for k, v in raw.items()}
    return dict(sorted(out.items())) or None


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def consensus_alignment_matrix(weights: np.ndarray, violence_level: float = 2.0) -> np.ndarray:
    """Bidirectional consensus denoise of ``[heads, tokens, frames]`` → ``[tokens, frames]``."""
    w = np.asarray(weights, dtype=np.float32)
    processed = _softmax(w, axis=-1) * _softmax(w, axis=-2)
    processed = np.maximum(
        processed - violence_level * np.quantile(processed, 0.5, axis=-1, keepdims=True), 0.0
    )
    processed = np.maximum(
        processed - violence_level * np.quantile(processed, 0.5, axis=-2, keepdims=True), 0.0
    )
    processed = processed**2
    mean = processed.mean()
    std = processed.std()
    return ((processed - mean) / (std + 1e-9)).mean(axis=0)


def dtw_path(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic DTW (upstream ``dtw_cpu`` tie-breaking), vectorized along anti-diagonals."""
    x = np.asarray(cost, dtype=np.float64)
    n, m = x.shape
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    trace = np.full((n + 1, m + 1), -1, dtype=np.int8)
    acc[0, 0] = 0.0
    for d in range(2, n + m + 1):
        i = np.arange(max(1, d - m), min(n, d - 1) + 1)
        j = d - i
        c0 = acc[i - 1, j - 1]
        c1 = acc[i - 1, j]
        c2 = acc[i, j - 1]
        diag = (c0 < c1) & (c0 < c2)
        up = ~diag & (c1 < c0) & (c1 < c2)
        best = np.where(diag, c0, np.where(up, c1, c2))
        acc[i, j] = x[i - 1, j - 1] + best
        trace[i, j] = np.where(diag, 0, np.where(up, 1, 2))
    trace[0, :] = 2
    trace[:, 0] = 1
    ti: list[int] = []
    tj: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        ti.append(i - 1)
        tj.append(j - 1)
        t = trace[i, j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        elif t == 2:
            j -= 1
        else:
            break
    return np.array(ti[::-1], dtype=np.int64), np.array(tj[::-1], dtype=np.int64)


def decode_tokens_incrementally(token_ids: Sequence[int], decode: Callable[[list[int]], str]) -> list[str]:
    """Per-token text that stays valid for multi-byte (CJK) byte-level tokens."""
    out: list[str] = []
    prev = b""
    for i in range(len(token_ids)):
        cur = decode(list(token_ids[: i + 1])).encode("utf-8", errors="surrogatepass")
        if len(cur) >= len(prev):
            try:
                out.append(cur[len(prev):].decode("utf-8"))
            except UnicodeDecodeError:
                out.append("")
        else:
            out.append("")
        prev = cur
    return out


def align_lyrics_from_attention(
    attention: np.ndarray,
    lyric_token_ids: Sequence[int],
    *,
    total_duration_sec: float,
    decode: Callable[[list[int]], str],
    violence_level: float = 2.0,
) -> LyricsAlignmentResult:
    """``attention`` ``[heads, lyric_tokens, frames]`` → sentence segments (one per lyric line)."""
    ids = list(lyric_token_ids)
    if not ids:
        return LyricsAlignmentResult(mode="empty", segments=())
    if attention.ndim != 3 or attention.shape[1] != len(ids):
        raise ValueError(
            f"attention shape {tuple(attention.shape)} does not match {len(ids)} lyric tokens"
        )
    calc = consensus_alignment_matrix(attention, violence_level=violence_level)
    text_idx, time_idx = dtw_path(-calc)
    sec_per_frame = float(total_duration_sec) / float(calc.shape[-1])
    pieces = decode_tokens_incrementally(ids, decode)

    starts: list[float] = []
    ends: list[float] = []
    for k in range(len(ids)):
        mask = text_idx == k
        if not np.any(mask):
            s = ends[-1] if ends else 0.0
            starts.append(s)
            ends.append(s)
            continue
        times = time_idx[mask] * sec_per_frame
        starts.append(float(times[0]))
        ends.append(max(float(times[-1]), float(times[0])))

    segments: list[LyricsAlignmentSegment] = []
    label = "body"
    group: list[int] = []

    def _flush() -> None:
        nonlocal label
        if not group:
            return
        text = decode([ids[k] for k in group]).strip()
        group.clear()
        if not text:
            return
        sec = _SECTION_RE.match(text)
        if sec:
            label = sec.group(1).strip() or "section"
            return
        k0, k1 = first_last[0], first_last[1]
        segments.append(
            LyricsAlignmentSegment(
                label=label,
                start_sec=round(starts[k0], 3),
                end_sec=round(max(ends[k1], starts[k0]), 3),
                text=text,
            )
        )

    first_last = [0, 0]
    for k, piece in enumerate(pieces):
        if not group:
            first_last[0] = k
        group.append(k)
        first_last[1] = k
        if "\n" in piece:
            _flush()
    _flush()
    return LyricsAlignmentResult(mode=DIT_ALIGNMENT_MODE, segments=tuple(segments))
