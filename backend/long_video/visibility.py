"""Character visibility progression helpers."""
from __future__ import annotations

_VISIBILITY_RANK: dict[str, int] = {
    "invisible": 0,
    "silhouette": 1,
    "partial": 2,
    "full_face": 3,
}

_ORDER = ("invisible", "silhouette", "partial", "full_face")


def vis_rank(value: str | None) -> int:
    return _VISIBILITY_RANK.get(str(value or "invisible").strip(), 0)


def vis_label(value: str | None) -> str:
    v = str(value or "invisible").strip()
    return v if v in _VISIBILITY_RANK else "invisible"


def next_vis(value: str | None, *, steps: int = 1) -> str:
    r = min(3, vis_rank(value) + max(0, steps))
    return _ORDER[r]


def clamp_vis_progression(prev_end: str | None, cur_start: str | None) -> str:
    """Return a valid start visibility at most one step after *prev_end*."""
    pe = vis_rank(prev_end)
    cs = vis_rank(cur_start)
    if cs <= pe + 1:
        return vis_label(cur_start)
    return next_vis(prev_end, steps=1)


def vis_short(value: str | None) -> str:
    m = {
        "invisible": "Inv",
        "silhouette": "Sil",
        "partial": "Part",
        "full_face": "Face",
    }
    return m.get(vis_label(value), "Inv")
