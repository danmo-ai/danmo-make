"""Map RunTrace spans onto declarative pipeline graph manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from backend.observability.trace import RunTrace, Span

_GRAPHS_DIR = Path(__file__).resolve().parent / "graphs"

_KIND_TO_GRAPH: dict[str, str] = {
    "image.generation": "image_create",
    "image.edit": "image_edit",
    "image.upscale": "image_upscale",
    "video.generation": "video_create",
    "video.long_generation": "video_long_generation",
    "video.edit": "video_create",
    "audio.generation": "audio_create",
    "audio.edit": "audio_create",
    "lora.training": "lora_training",
    "lora.evaluation": "lora_training",
    "tools.z_image_merge": "tools_z_image_merge",
}


def graph_id_for_task_kind(kind: str) -> str:
    return _KIND_TO_GRAPH.get(kind, "image_create")


@dataclass
class GraphNodeView:
    id: str
    label: dict[str, str]
    status: str
    duration_ms: float | None = None
    error_code: str | None = None


@dataclass
class GraphSnapshot:
    graph_id: str
    nodes: list[GraphNodeView]
    edges: list[tuple[str, str]]
    active_node: str | None
    progress: float


def load_graph_manifest(graph_id: str) -> dict[str, Any]:
    path = _GRAPHS_DIR / f"{graph_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"pipeline graph manifest not found: {graph_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid graph manifest: {graph_id}")
    return data


def _span_status_for_node(spans: list[Span], node_id: str) -> tuple[str, float | None, str | None]:
    matched = [s for s in spans if s.name == node_id]
    if not matched:
        return "pending", None, None
    last = matched[-1]
    err = None
    if last.status == "failed":
        err = "failed"
    return last.status, last.duration_ms(), err


def snapshot(trace: RunTrace | None, graph_id: str, *, locale: str = "zh") -> GraphSnapshot:
    manifest = load_graph_manifest(graph_id)
    nodes_def = manifest.get("nodes") or []
    edges_raw = manifest.get("edges") or []
    spans = trace.spans if trace else []

    nodes: list[GraphNodeView] = []
    active: str | None = None
    ok_count = 0
    for raw in nodes_def:
        if not isinstance(raw, dict):
            continue
        nid = str(raw.get("id", ""))
        label = raw.get("label") if isinstance(raw.get("label"), dict) else {"en": nid}
        status, dur, err = _span_status_for_node(spans, nid)
        if status == "running":
            active = nid
        if status == "ok":
            ok_count += 1
        nodes.append(GraphNodeView(id=nid, label=label, status=status, duration_ms=dur, error_code=err))

    edges: list[tuple[str, str]] = []
    for pair in edges_raw:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            edges.append((str(pair[0]), str(pair[1])))

    total = max(1, len(nodes))
    progress = ok_count / total
    return GraphSnapshot(
        graph_id=graph_id,
        nodes=nodes,
        edges=edges,
        active_node=active,
        progress=progress,
    )


def snapshot_to_dict(snap: GraphSnapshot, *, locale: str = "zh") -> dict[str, Any]:
    loc = "en" if locale.startswith("en") else "zh"

    def _label(node: GraphNodeView) -> str:
        return node.label.get(loc) or node.label.get("en") or node.id

    return {
        "graph_id": snap.graph_id,
        "nodes": [
            {
                "id": n.id,
                "label": _label(n),
                "status": n.status,
                "duration_ms": n.duration_ms,
                "error_code": n.error_code,
            }
            for n in snap.nodes
        ],
        "edges": [{"from": a, "to": b} for a, b in snap.edges],
        "active_node": snap.active_node,
        "progress": snap.progress,
    }
