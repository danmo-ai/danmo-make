"""LoRA search across ModelScope, Hugging Face, and CivitAI."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

import aiohttp

from backend.core.registry_format import resolve_registry_label
from backend.third_party.civitai_client import CivitAIClient

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
MS_ENDPOINT = "https://www.modelscope.cn"

LORA_SEARCH_DEFAULT_LIMIT = 500
LORA_SEARCH_MAX_LIMIT = 500
MS_PAGE_SIZE = 40


@dataclass
class LoraSearchItem:
    id: str
    source: str
    name: str
    description: str = ""
    preview_url: str = ""
    base_model_label: str = ""
    hub_base_model: str = ""
    tags: List[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    nsfw: bool = False
    creator: str = ""
    repo_id: str = ""
    filename: str = ""
    download_url: str = ""
    civitai_model_id: Optional[int] = None
    civitai_version_id: Optional[int] = None
    versions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "name": self.name,
            "description": self.description,
            "preview_url": self.preview_url,
            "base_model_label": self.base_model_label,
            "hub_base_model": self.hub_base_model,
            "tags": list(self.tags),
            "downloads": self.downloads,
            "likes": self.likes,
            "nsfw": self.nsfw,
            "creator": self.creator,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "download_url": self.download_url,
            "civitai_model_id": self.civitai_model_id,
            "civitai_version_id": self.civitai_version_id,
            "versions": self.versions,
        }


def _catalog(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.get("catalog"):
        return cfg["catalog"]
    return cfg


def _ui_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ui = cfg.get("ui") or {}
    if ui.get("parameters"):
        return ui["parameters"]
    return cfg.get("parameters") or {}


def list_lora_base_models(registry: Dict[str, Any], *, locale: str = "zh") -> List[Dict[str, str]]:
    """Registry models that declare ``lora_support`` (image + video bases)."""
    out: List[Dict[str, str]] = []
    for model_id, cfg in registry.items():
        if not _ui_params(cfg).get("lora_support"):
            continue
        cat = _catalog(cfg).get("category") or cfg.get("category") or ""
        if cat not in ("base_models", "video_models"):
            continue
        name = resolve_registry_label(_catalog(cfg).get("name"), model_id, locale=locale)
        out.append({"id": model_id, "name": name})
    out.sort(key=lambda row: row["name"].lower())
    return out


def resolve_lora_browse_queries(
    registry: Dict[str, Any],
    base_model_id: str,
) -> List[str]:
    """Hub-facing browse queries from registry ``ui.parameters.lora_search``."""
    mid = (base_model_id or "").split(":", 1)[0].strip()
    if not mid:
        return ["lora"]

    cfg = (registry or {}).get(mid) or {}
    params = _ui_params(cfg)
    lora_search = params.get("lora_search")
    if isinstance(lora_search, dict):
        explicit = lora_search.get("queries")
        if isinstance(explicit, list):
            queries = [str(q).strip() for q in explicit if str(q).strip()]
            if queries:
                return queries
        terms = lora_search.get("terms")
        suffix = str(lora_search.get("suffix") or "lora").strip()
        if isinstance(terms, list):
            built: List[str] = []
            for term in terms:
                text = str(term).strip()
                if not text:
                    continue
                built.append(f"{text} {suffix}".strip() if suffix else text)
            if built:
                return built

    legacy = params.get("lora_search_terms")
    if isinstance(legacy, list):
        queries = [str(q).strip() for q in legacy if str(q).strip()]
        if queries:
            return queries

    cat = _catalog(cfg)
    name = resolve_registry_label(cat.get("name"), mid, locale="en")
    if name and name != mid:
        return [f"{name} lora"]
    return [f"{mid} lora"]


def build_search_query(
    registry: Dict[str, Any],
    base_model_id: str,
    user_query: str,
) -> str:
    """Primary hub query: user keyword, else first registry browse query."""
    q = (user_query or "").strip()
    if q:
        return q
    queries = resolve_lora_browse_queries(registry, base_model_id)
    return queries[0]


def _append_unique_items(
    dest: List[LoraSearchItem],
    batch: Sequence[LoraSearchItem],
    *,
    seen: set[str],
    limit: int,
) -> None:
    for item in batch:
        key = (item.id or item.repo_id or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dest.append(item)
        if len(dest) >= limit:
            return


def _hub_base_from_tags(tags: Optional[List[Any]]) -> str:
    for tag in tags or []:
        text = str(tag).strip()
        lower = text.lower()
        if lower.startswith("base_model:"):
            return text.split(":", 1)[-1].strip()
    return ""


def _normalize_hub_tags(tags: Optional[List[Any]]) -> List[str]:
    out: List[str] = []
    for tag in tags or []:
        text = str(tag).strip()
        if not text or text.lower().startswith("license:"):
            continue
        if text.lower().startswith("custom_tag:"):
            out.append(text.split(":", 1)[-1].strip())
            continue
        if text.lower().startswith("task:"):
            out.append(text.split(":", 1)[-1].strip())
            continue
        if text.lower().startswith("library:"):
            lib = text.split(":", 1)[-1].strip()
            if lib and lib.lower() not in ("lora", "safetensors"):
                out.append(lib)
            continue
        out.append(text)
    deduped: List[str] = []
    seen: set[str] = set()
    for label in out:
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(label)
        if len(deduped) >= 8:
            break
    return deduped


def _modelscope_row_is_lora(row: Dict[str, Any]) -> bool:
    tags = row.get("tags") or []
    if any("lora" in str(t).lower() for t in tags):
        return True
    blob = f"{row.get('id') or ''} {row.get('display_name') or ''}".lower()
    return "lora" in blob


async def search_huggingface(
    *,
    query: str,
    base_model_id: str = "",
    limit: int = LORA_SEARCH_DEFAULT_LIMIT,
    hf_token: Optional[str] = None,
) -> List[LoraSearchItem]:
    headers = {"User-Agent": "huggingface_hub"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    params = {
        "search": query,
        "filter": "lora",
        "limit": min(limit, LORA_SEARCH_MAX_LIMIT),
        "sort": "downloads",
        "direction": -1,
    }
    url = f"{HF_ENDPOINT}/api/models"
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Hugging Face search failed ({resp.status}): {text[:200]}")
            data = await resp.json()

    items: List[LoraSearchItem] = []
    for row in data if isinstance(data, list) else []:
        repo_id = row.get("modelId") or row.get("id") or ""
        if not repo_id:
            continue
        tags = row.get("tags") or []
        base_label = _hub_base_from_tags(tags)
        name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        hub_tags = _normalize_hub_tags(tags)
        items.append(
            LoraSearchItem(
                id=f"huggingface:{repo_id}",
                source="huggingface",
                name=name,
                description="",
                preview_url="",
                base_model_label=base_label,
                hub_base_model=base_label,
                tags=hub_tags,
                downloads=int(row.get("downloads") or 0),
                likes=int(row.get("likes") or 0),
                creator=repo_id.split("/")[0] if "/" in repo_id else "",
                repo_id=repo_id,
            )
        )
        if len(items) >= limit:
            break
    return items


async def search_modelscope(
    *,
    query: str,
    base_model_id: str = "",
    limit: int = LORA_SEARCH_DEFAULT_LIMIT,
    page: int = 1,
) -> List[LoraSearchItem]:
    url = f"{MS_ENDPOINT}/openapi/v1/models"
    page_size = MS_PAGE_SIZE
    max_pages = max(1, (limit + page_size - 1) // page_size)
    items: List[LoraSearchItem] = []
    seen_repos: set[str] = set()
    async with aiohttp.ClientSession(headers={"User-Agent": "modelscope"}) as session:
        for offset in range(max_pages):
            page_number = page + offset
            if len(items) >= limit:
                break
            params = {
                "search": query,
                "page_size": page_size,
                "page_number": page_number,
                "sort": "downloads",
            }
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"ModelScope search failed ({resp.status}): {text[:200]}")
                payload = await resp.json()

            data = payload.get("data") if isinstance(payload, dict) else {}
            models = data.get("models") if isinstance(data, dict) else []
            if not models:
                break

            for row in models:
                repo_id = row.get("id") or ""
                if not repo_id or repo_id in seen_repos:
                    continue
                seen_repos.add(repo_id)
                if not _modelscope_row_is_lora(row):
                    continue
                tags = row.get("tags") or []
                name = row.get("display_name") or repo_id.split("/")[-1]
                hub_tags = _normalize_hub_tags(tags)
                tasks = row.get("tasks") or []
                for task in tasks:
                    task_label = str(task).strip()
                    if task_label and task_label not in hub_tags:
                        hub_tags.append(task_label)
                items.append(
                    LoraSearchItem(
                        id=f"modelscope:{repo_id}",
                        source="modelscope",
                        name=name,
                        description=(row.get("description") or "")[:500],
                        preview_url="",
                        base_model_label="",
                        hub_base_model=_hub_base_from_tags(tags),
                        tags=hub_tags[:8],
                        downloads=int(row.get("downloads") or 0),
                        likes=int(row.get("likes") or 0),
                        creator=repo_id.split("/")[0] if "/" in repo_id else "",
                        repo_id=repo_id,
                    )
                )
                if len(items) >= limit:
                    break

            if len(models) < page_size:
                break
    return items[:limit]


def _civitai_model_to_item(model: Any, *, base_model_id: str) -> Optional[LoraSearchItem]:
    versions = []
    for v in model.model_versions:
        files = [
            {
                "name": f.name,
                "download_url": f.download_url,
                "size_kb": f.size_kb,
                "format": f.format,
                "primary": f.primary,
            }
            for f in v.files
        ]
        versions.append(
            {
                "id": v.id,
                "name": v.name,
                "base_model": v.base_model,
                "download_url": v.download_url,
                "files": files,
                "images": v.images,
            }
        )
    if not versions:
        return None
    preview = ""
    if versions[0].get("images"):
        preview = versions[0]["images"][0].get("url") or ""
    primary_version = versions[0]
    hub_base = str(primary_version.get("base_model") or "").strip()
    primary_file = next(
        (f for f in primary_version.get("files") or [] if f.get("primary")),
        (primary_version.get("files") or [{}])[0],
    )
    return LoraSearchItem(
        id=f"civitai:{model.id}",
        source="civitai",
        name=model.name,
        description=(model.description or "")[:500],
        preview_url=preview,
        base_model_label=hub_base,
        hub_base_model=hub_base,
        tags=[str(t) for t in (model.tags or [])[:8]],
        downloads=int((model.stats or {}).get("downloadCount") or 0),
        likes=int((model.stats or {}).get("thumbsUpCount") or 0),
        nsfw=bool(model.nsfw),
        creator=(model.creator or {}).get("username") or "",
        download_url=primary_file.get("download_url") or "",
        filename=primary_file.get("name") or "",
        civitai_model_id=model.id,
        civitai_version_id=primary_version.get("id"),
        versions=versions,
    )


async def search_civitai(
    *,
    query: str,
    base_model_id: str,
    limit: int = LORA_SEARCH_DEFAULT_LIMIT,
    page: int = 1,
    cursor: Optional[str] = None,
    civitai_token: Optional[str] = None,
    nsfw: Optional[bool] = None,
) -> tuple[List[LoraSearchItem], Optional[str]]:
    client = CivitAIClient(api_key=civitai_token)
    items: List[LoraSearchItem] = []
    seen_ids: set[int] = set()
    next_cursor: Optional[str] = cursor
    request_page = page
    try:
        while len(items) < limit:
            batch_limit = min(100, limit - len(items))
            result = await client.search(
                query=query,
                types=["LORA"],
                limit=batch_limit,
                page=request_page,
                cursor=next_cursor,
                sort="Highest Rated",
                nsfw=nsfw,
            )
            batch = result.get("items") or []
            if not batch:
                break
            for model in batch:
                if model.id in seen_ids:
                    continue
                seen_ids.add(model.id)
                item = _civitai_model_to_item(model, base_model_id=base_model_id)
                if item is not None:
                    items.append(item)
                if len(items) >= limit:
                    break

            metadata = result.get("metadata") or {}
            next_cursor = metadata.get("nextCursor")
            if query:
                if not next_cursor:
                    break
            else:
                request_page += 1
                if len(batch) < batch_limit:
                    break
    finally:
        await client.close()

    return items[:limit], next_cursor


async def search_loras(
    *,
    query: str,
    base_model_id: str,
    source: str = "all",
    limit: int = LORA_SEARCH_DEFAULT_LIMIT,
    page: int = 1,
    cursor: Optional[str] = None,
    hf_token: Optional[str] = None,
    civitai_token: Optional[str] = None,
    nsfw: Optional[bool] = None,
    registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search LoRAs from one or all sources."""
    limit = min(max(1, limit), LORA_SEARCH_MAX_LIMIT)
    registry_map = registry or {}
    user_q = (query or "").strip()
    browse_queries = [user_q] if user_q else resolve_lora_browse_queries(registry_map, base_model_id)
    search_q = browse_queries[0]
    src = (source or "all").strip().lower()
    next_cursor: Optional[str] = None
    errors: Dict[str, str] = {}

    async def _collect_hf() -> List[LoraSearchItem]:
        local: List[LoraSearchItem] = []
        seen: set[str] = set()
        for browse_q in browse_queries:
            if len(local) >= limit:
                break
            batch = await search_huggingface(
                query=browse_q,
                base_model_id=base_model_id,
                limit=limit,
                hf_token=hf_token,
            )
            _append_unique_items(local, batch, seen=seen, limit=limit)
        return local

    async def _collect_ms() -> List[LoraSearchItem]:
        local: List[LoraSearchItem] = []
        seen: set[str] = set()
        for browse_q in browse_queries:
            if len(local) >= limit:
                break
            batch = await search_modelscope(
                query=browse_q,
                base_model_id=base_model_id,
                limit=limit,
                page=page,
            )
            _append_unique_items(local, batch, seen=seen, limit=limit)
        return local

    async def _collect_civit() -> tuple[List[LoraSearchItem], Optional[str]]:
        local: List[LoraSearchItem] = []
        seen: set[str] = set()
        cursor_out: Optional[str] = cursor
        for browse_q in browse_queries:
            if len(local) >= limit:
                break
            batch, cursor_out = await search_civitai(
                query=browse_q,
                base_model_id=base_model_id,
                limit=limit,
                page=page,
                cursor=cursor_out,
                civitai_token=civitai_token,
                nsfw=nsfw,
            )
            _append_unique_items(local, batch, seen=seen, limit=limit)
        return local, cursor_out

    items: List[LoraSearchItem] = []
    if src == "all":
        hf_items: List[LoraSearchItem] = []
        ms_items: List[LoraSearchItem] = []
        civit_items: List[LoraSearchItem] = []

        async def _hf() -> None:
            nonlocal hf_items
            try:
                hf_items = await _collect_hf()
            except Exception as e:
                errors["huggingface"] = str(e)

        async def _ms() -> None:
            nonlocal ms_items
            try:
                ms_items = await _collect_ms()
            except Exception as e:
                errors["modelscope"] = str(e)

        async def _civit() -> None:
            nonlocal civit_items, next_cursor
            try:
                civit_items, next_cursor = await _collect_civit()
            except Exception as e:
                errors["civitai"] = str(e)

        await asyncio.gather(_hf(), _ms(), _civit())
        merged_seen: set[str] = set()
        for batch in (hf_items, ms_items, civit_items):
            _append_unique_items(items, batch, seen=merged_seen, limit=limit)
        items.sort(key=lambda x: x.downloads, reverse=True)
        items = items[:limit]
    elif src == "huggingface":
        try:
            items = await _collect_hf()
        except Exception as e:
            errors["huggingface"] = str(e)
    elif src == "modelscope":
        try:
            items = await _collect_ms()
        except Exception as e:
            errors["modelscope"] = str(e)
    elif src == "civitai":
        try:
            items, next_cursor = await _collect_civit()
        except Exception as e:
            errors["civitai"] = str(e)
    else:
        raise ValueError(f"Unknown LoRA search source: {source}")

    if not items and errors and len(errors) >= (3 if src == "all" else 1):
        joined = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(joined)

    return {
        "items": [item.to_dict() for item in items],
        "query": search_q,
        "browse_queries": browse_queries,
        "next_cursor": next_cursor,
        "errors": errors,
    }


def resolve_huggingface_lora_url(repo_id: str, filename: str) -> str:
    return f"{HF_ENDPOINT}/{repo_id}/resolve/main/{quote(filename, safe='/')}"


async def resolve_huggingface_lora_filename(
    repo_id: str,
    *,
    hf_token: Optional[str] = None,
) -> str:
    headers = {"User-Agent": "huggingface_hub"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    url = f"{HF_ENDPOINT}/api/models/{repo_id}/tree/main"
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to list Hugging Face repo files: {resp.status} {text[:200]}")
            files = await resp.json()
    candidates = [
        row.get("path")
        for row in files
        if isinstance(row, dict) and str(row.get("path", "")).endswith(".safetensors")
    ]
    if not candidates:
        raise RuntimeError(f"No .safetensors file found in Hugging Face repo {repo_id}")
    preferred = next((c for c in candidates if "lora" in c.lower()), candidates[0])
    return preferred


def resolve_modelscope_lora_filename(repo_id: str) -> str:
    from modelscope.hub.api import HubApi

    api = HubApi()
    files = api.get_model_files(repo_id, recursive=True)
    candidates = [
        f.get("Name")
        for f in files
        if str(f.get("Name", "")).endswith(".safetensors")
    ]
    if not candidates:
        raise RuntimeError(f"No .safetensors file found in ModelScope repo {repo_id}")
    preferred = next((c for c in candidates if "lora" in c.lower()), candidates[0])
    return preferred


def resolve_modelscope_lora_url(
    repo_id: str,
    filename: str,
    *,
    revision: str = "master",
) -> str:
    from modelscope.hub.file_download import get_file_download_url

    return get_file_download_url(repo_id, filename, revision)
