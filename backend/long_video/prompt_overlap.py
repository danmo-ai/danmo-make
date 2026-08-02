"""Language-agnostic prompt text overlap helpers (CJK + Latin tokens)."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}")


def prompt_token_set(text: str, *, min_len: int = 2) -> set[str]:
    parts = _TOKEN_RE.findall(text or "")
    return {p.lower() if p.isascii() else p for p in parts if len(p) >= min_len}


def _cjk_bigram_set(text: str) -> set[str]:
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "")
    out: set[str] = set()
    if len(t) == 1:
        out.add(t)
        return out
    for i in range(len(t) - 1):
        out.add(t[i : i + 2])
    return out


def _fuzzy_token_hit(needle_token: str, hay_tokens: set[str]) -> bool:
    if needle_token in hay_tokens:
        return True
    for hay in hay_tokens:
        if len(needle_token) >= 2 and len(hay) >= 2 and (needle_token in hay or hay in needle_token):
            return True
    return False


def prompt_token_coverage(haystack: str, needle: str) -> float:
    """Fraction of needle tokens also present in haystack (0..1)."""
    needle_tokens = prompt_token_set(needle)
    if not needle_tokens:
        return 1.0
    hay_tokens = prompt_token_set(haystack)
    if not hay_tokens:
        return 0.0
    hit = sum(1 for t in needle_tokens if t in hay_tokens)
    return hit / len(needle_tokens)


def prompt_token_coverage_fuzzy(haystack: str, needle: str) -> float:
    """Token coverage allowing CJK substring matches (e.g. 公寓 in 城郊旧公寓)."""
    needle_tokens = prompt_token_set(needle)
    if not needle_tokens:
        return 1.0
    hay_tokens = prompt_token_set(haystack)
    if not hay_tokens:
        return 0.0
    hit = sum(1 for t in needle_tokens if _fuzzy_token_hit(t, hay_tokens))
    return hit / len(needle_tokens)


def cjk_bigram_token_coverage(haystack: str, needle: str) -> float:
    """Fraction of needle CJK bigrams present in haystack (paraphrase-tolerant)."""
    needle_bigrams = _cjk_bigram_set(needle)
    if not needle_bigrams:
        return 1.0
    hay_bigrams = _cjk_bigram_set(haystack)
    if not hay_bigrams:
        return 0.0
    hit = sum(1 for g in needle_bigrams if g in hay_bigrams)
    return hit / len(needle_bigrams)


def prompt_narrative_coverage(haystack: str, needle: str) -> float:
    """Best-effort narrative overlap for beat coverage checks."""
    return max(
        prompt_token_coverage(haystack, needle),
        prompt_token_coverage_fuzzy(haystack, needle),
        cjk_bigram_token_coverage(haystack, needle),
    )
