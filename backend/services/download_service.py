"""
Download service implementation - supports HuggingFace and HTTP dual-source downloads.
"""

import os
# Configure HF mirror site
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import logging
import threading
import time
import uuid
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import asyncio
import shutil

from backend.core.bundle_repos import (
    bundle_local_paths,
    bundle_repos_from_version,
    primary_and_follow_ups,
    version_primary_local_path,
)
from backend.core.dependency_specs import DependencySpec, parse_dependencies
from backend.core.install_hooks import install_hooks_from_version, run_install_hooks


def _run_skip_completeness_check(target: str, dest: Path) -> bool:
    """Invoke a hook's ``skip_completeness_if`` callback (``module:function``)."""
    import importlib

    mod_name, fn_name = target.rsplit(":", 1)
    fn = getattr(importlib.import_module(mod_name), fn_name, None)
    if fn is None or not callable(fn):
        raise RuntimeError(f"skip_completeness_if target not found: {target}")
    return bool(fn(dest))
from backend.core.bundle_manifest import skips_full_family_bundle_contract, write_bundle_manifest
from backend.core.interfaces import (
    IDownloadService, IPathResolver, IConfigStore,
    DownloadTask, DownloadProgress, TaskStatus, ConversionTask
)
from backend.core.i18n import t as tt, get_locale
from backend.core.registry_format import resolve_registry_label
from backend.core.downloaders import HTTPDownloader
from backend.services.hf_repo_resolve import resolve_huggingface_repo_id

logger = logging.getLogger(__name__)


def _calc_download_dir_bytes(root: Path) -> int:
    """Bytes on disk for a model install dir (incl. ModelScope ``._____temp`` partials)."""
    if not root.is_dir():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _format_download_speed(bytes_per_sec: float) -> str:
    if bytes_per_sec <= 0:
        return ""
    if bytes_per_sec > 1024 ** 3:
        return f"{bytes_per_sec / (1024 ** 3):.1f} GB/s"
    if bytes_per_sec > 1024 ** 2:
        return f"{bytes_per_sec / (1024 ** 2):.1f} MB/s"
    if bytes_per_sec > 1024:
        return f"{bytes_per_sec / 1024:.1f} KB/s"
    return f"{bytes_per_sec:.1f} B/s"


class _ModelScopeProgressTracker:
    """Thread-safe byte counter fed by ModelScope ``ProgressCallback`` during snapshot_download."""

    def __init__(self, *, baseline_bytes: int = 0) -> None:
        self._lock = threading.Lock()
        self._baseline = int(baseline_bytes)
        self._session_bytes = 0
        self._active_file = ""
        self._last_activity_time = time.time()

    def note_activity(self) -> None:
        with self._lock:
            self._last_activity_time = time.time()

    def callback_class(self) -> type:
        tracker = self

        from modelscope.hub.callback import ProgressCallback

        class _TrackedCallback(ProgressCallback):
            def __init__(self, filename: str, file_size: int) -> None:
                super().__init__(filename, file_size)
                with tracker._lock:
                    tracker._active_file = filename
                tracker.note_activity()

            def update(self, size: int) -> None:
                with tracker._lock:
                    tracker._session_bytes += int(size)
                tracker.note_activity()

            def end(self) -> None:
                tracker.note_activity()

        return _TrackedCallback

    def downloaded_bytes(self, root: Path) -> int:
        disk = _calc_download_dir_bytes(root)
        with self._lock:
            tracked = self._baseline + self._session_bytes
        return max(disk, tracked)

    @property
    def active_file(self) -> str:
        with self._lock:
            return self._active_file

    @property
    def last_activity_time(self) -> float:
        with self._lock:
            return self._last_activity_time


def _modelscope_stall_timeout_seconds(
    *,
    estimated_bytes: int = 0,
    allow_patterns: list[str] | None = None,
) -> int:
    """Idle timeout before aborting a ModelScope snapshot (large shards need longer handshakes)."""
    if estimated_bytes >= 50 * 1024 ** 3:
        return 900
    if estimated_bytes >= 20 * 1024 ** 3:
        return 600
    if allow_patterns and any(
        str(p).endswith((".safetensors", ".pth")) for p in allow_patterns
    ):
        return 600
    return _DOWNLOAD_STALL_SECONDS


def _modelscope_connect_grace_seconds(*, estimated_bytes: int = 0) -> int:
    """Extra idle budget while still fetching early metadata / first multi-GB shard."""
    if estimated_bytes >= 20 * 1024 ** 3:
        return 600
    if estimated_bytes >= 5 * 1024 ** 3:
        return 300
    return 120


_MODELSCOPE_CONNECT_GRACE_BYTE_CAP = 256 * 1024 ** 2


def _modelscope_stall_watch_kwargs(
    *,
    estimated_bytes: int,
    allow_patterns: list[str] | None,
    tracker: _ModelScopeProgressTracker,
) -> dict[str, Any]:
    return {
        "stall_seconds": _modelscope_stall_timeout_seconds(
            estimated_bytes=estimated_bytes,
            allow_patterns=allow_patterns,
        ),
        "connect_grace_seconds": _modelscope_connect_grace_seconds(
            estimated_bytes=estimated_bytes,
        ),
        "connect_grace_byte_cap": _MODELSCOPE_CONNECT_GRACE_BYTE_CAP,
        "last_activity_time": tracker.last_activity_time,
        "active_file": tracker.active_file,
    }


def _modelscope_workers_for_patterns(patterns: list[str] | None) -> int:
    """Large shard repos: download sequentially to reduce disk peak and network errors."""
    if not patterns:
        return 4
    if any(str(p).endswith((".safetensors", ".pth")) for p in patterns):
        return 1
    return 4


def _split_modelscope_allow_patterns(patterns: list[str] | None) -> list[list[str]]:
    """Split mixed subtree + repo-root patterns for ModelScope ``snapshot_download``.

    ModelScope ``extract_root_from_patterns`` picks e.g. ``google`` from ``google/**`` and
    then only lists files under that subtree — root-level shards (T5/VAE) never download.
    """
    if not patterns:
        return []
    normalized = [str(p).strip() for p in patterns if str(p).strip()]
    if not normalized:
        return []

    try:
        from modelscope.hub.utils.utils import extract_root_from_patterns

        root = extract_root_from_patterns(allow_patterns=normalized)
    except Exception:
        return [normalized]

    if not root:
        return [normalized]

    root = root.strip("/")
    prefix = f"{root}/"
    in_root: list[str] = []
    outside: list[str] = []
    for pat in normalized:
        bare = pat.rstrip("/")
        if bare.endswith("/**") and bare[:-3].rstrip("/") == root:
            in_root.append(pat)
        elif pat.startswith(prefix):
            in_root.append(pat)
        elif "/" not in pat:
            outside.append(pat)
        else:
            outside.append(pat)

    if not outside:
        return [in_root]
    if not in_root:
        return _split_modelscope_allow_patterns(outside)

    return [in_root] + _split_modelscope_allow_patterns(outside)


def _min_bytes_for_pattern(pattern: str) -> int:
    name = pattern.rsplit("/", 1)[-1]
    if name.endswith(".safetensors"):
        lower = name.lower()
        if "vae" in lower or lower.startswith("ae."):
            # Flux VAE ae.safetensors ~335MB, LTX-2.5 audio VAE ~348MB,
            # LTX-2.5 video VAE conv ~1.45GB — do not use the 1GB DiT/T5 floor.
            return 300 * 1024 ** 2
        if "upscaler" in lower or "upsampler" in lower:
            # LTX-2.5 spatial latent upsampler ~950MB.
            return 800 * 1024 ** 2
        return 1024 ** 3
    if name.endswith(".pth"):
        lower = name.lower()
        if "vae" in lower:
            # Wan2.1_VAE.pth is ~484MB on ModelScope; do not use the 1GB DiT/T5 floor.
            return 400 * 1024 ** 2
        if "t5" in lower or name.startswith("models_t5"):
            return 1024 ** 3
        return 100 * 1024 ** 2
    if name.endswith(".json"):
        return 32
    return 1024


def _bundle_repo_is_complete(dest: Path, patterns: list[str] | None) -> bool:
    """True when every allow_pattern target exists with plausible size."""
    return not _missing_bundle_patterns(dest, patterns)


def _missing_bundle_patterns(dest: Path, patterns: list[str] | None) -> list[str]:
    if not patterns or not dest.is_dir():
        return list(patterns or [])
    import fnmatch

    missing: list[str] = []
    for raw in patterns:
        pat = str(raw).strip()
        if not pat:
            continue
        min_sz = _min_bytes_for_pattern(pat)
        if pat.endswith("/**"):
            sub = dest / pat[:-3].rstrip("/")
            if not sub.is_dir():
                missing.append(pat)
                continue
            if not any(f.is_file() and f.stat().st_size >= 1024 for f in sub.rglob("*")):
                missing.append(pat)
            continue
        if "**" in pat:
            suffix = pat.split("**/")[-1]
            hits = [p for p in dest.rglob(suffix) if p.is_file()]
        elif any(ch in pat for ch in "*?[]"):
            hits = [p for p in dest.rglob("*") if p.is_file() and fnmatch.fnmatch(
                str(p.relative_to(dest)), pat
            )]
        else:
            fp = dest / pat
            hits = [fp] if fp.is_file() else []
        if not hits or not any(h.stat().st_size >= min_sz for h in hits):
            missing.append(pat)
    return missing


_DOWNLOAD_STALL_SECONDS = 180


class DownloadService(IDownloadService):
    """Download service implementation.

    Automatically selects the download method based on the source field in the model registry:
    - huggingface: uses huggingface_hub.snapshot_download, polls directory size for progress
    - modelscope: uses modelscope.snapshot_download with ProgressCallback + disk polling
    - civitai / http: uses HTTPDownloader (aiohttp)
    """

    def __init__(self, path_resolver: IPathResolver, config_store: Optional[IConfigStore] = None):
        self._path_resolver = path_resolver
        self._config = config_store
        self._downloads: Dict[str, DownloadTask] = {}
        self._progress: Dict[str, DownloadProgress] = {}
        self._conversions: Dict[str, ConversionTask] = {}
        self._conversion_events: Dict[str, asyncio.Event] = {}
        self._persist_path = path_resolver.get_workspace_config_dir() / ".download_tasks.json"

        # Read tokens from config
        hf_token = None
        civitai_token = None
        if self._config:
            settings = self._config.load()
            hf_token = settings.huggingface_token or None
            civitai_token = settings.civitai_token or None

        # Initialize downloaders
        self._http_downloader = HTTPDownloader(
            civitai_token=civitai_token,
            huggingface_token=hf_token,
        )
        self._cancelled_downloads: set = set()
        self._token = hf_token

        # Active model download dedup: (model_name, version) -> task_id
        self._active_model_downloads: Dict[tuple, str] = {}
        # In-flight download workers (same model+version must not run concurrently)
        self._inflight_model_downloads: set[tuple[str, str | None]] = set()
        # Prevent dependency install recursion loops
        self._dependency_install_chain: set[tuple[str, str | None]] = set()
        # Concurrency lock protecting dedup checks and task creation
        self._download_lock = asyncio.Lock()

        # Load persisted download tasks
        self._load_persisted_downloads()

    def _persist_downloads(self) -> None:
        """Persist download tasks to JSON file (retains all states, including completed and failed)."""
        try:
            data = {}
            for task_id, task in self._downloads.items():
                progress = self._progress.get(task_id)
                data[task_id] = {
                    "id": task.id,
                    "url": task.url,
                    "target_path": task.target_path,
                    "status": task.status.value,
                    "progress": task.progress,
                    "total_size": progress.total_size if progress else 0,
                    "downloaded_size": progress.downloaded_size if progress else 0,
                    "speed": progress.speed if progress else "",
                    "filename": progress.filename if progress else task.url,
                    "error_message": progress.error_message if progress else "",
                    "model_name": getattr(task, '_model_name', None),
                    "version": getattr(task, '_version', None),
                    "is_lora": getattr(task, '_is_lora', False),
                    "lora_filename": getattr(task, '_filename', None),
                }
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DownloadService] Failed to persist download tasks: {e}")

    def _load_persisted_downloads(self) -> None:
        """Load persisted download tasks from JSON file.

        After process restart, tasks in running state are marked as paused,
        waiting for the user to manually resume.
        """
        if not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for task_id, item in data.items():
                task = DownloadTask(
                    id=item["id"],
                    url=item["url"],
                    target_path=item["target_path"],
                )
                raw_status = item.get("status", "running")
                # After process restart, tasks that were running become paused
                if raw_status == "running":
                    task.status = TaskStatus.PAUSED
                else:
                    task.status = TaskStatus(raw_status)
                task.progress = item.get("progress", 0)
                # Restore metadata
                if item.get("model_name"):
                    task._model_name = item["model_name"]
                if "version" in item:
                    task._version = item["version"]
                if item.get("is_lora"):
                    task._is_lora = item["is_lora"]
                if item.get("lora_filename"):
                    task._filename = item["lora_filename"]
                self._downloads[task_id] = task
                self._progress[task_id] = DownloadProgress(
                    task_id=task_id,
                    status=task.status.value,
                    progress=item.get("progress", 0),
                    total_size=item.get("total_size", 0),
                    downloaded_size=item.get("downloaded_size", 0),
                    speed=item.get("speed", ""),
                    filename=item.get("filename", task.url),
                    error_message=item.get("error_message", "")
                )
        except Exception as e:
            print(f"[DownloadService] Failed to load persisted download tasks: {e}")
        else:
            self._reconcile_duplicate_model_downloads()
            self._inflight_model_downloads.clear()

    def _find_active_model_download(self, dedup_key: tuple[str, str | None]) -> str | None:
        """Return task_id for an active model download (running, paused, or failed)."""
        tid = self._active_model_downloads.get(dedup_key)
        if tid and tid in self._downloads:
            task = self._downloads[tid]
            if task.status in (TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.FAILED):
                return tid
        model_name, version = dedup_key
        best_id: str | None = None
        best_bytes = -1
        for candidate_id, task in self._downloads.items():
            if getattr(task, "_model_name", None) != model_name:
                continue
            if getattr(task, "_version", None) != version:
                continue
            if task.status not in (TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.FAILED):
                continue
            prog = self._progress.get(candidate_id)
            nbytes = prog.downloaded_size if prog else 0
            if nbytes > best_bytes:
                best_bytes = nbytes
                best_id = candidate_id
        return best_id

    async def _emit_task_progress(
        self,
        task_id: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]],
    ) -> None:
        if not progress_callback:
            return
        progress = self._progress.get(task_id)
        if progress:
            await self._async_callback(progress_callback, progress)
            return
        task = self._downloads.get(task_id)
        if not task:
            return
        await self._async_callback(
            progress_callback,
            DownloadProgress(
                task_id=task_id,
                status=task.status.value,
                progress=task.progress,
                filename=getattr(task, "url", "") or task_id,
            ),
        )

    def _reconcile_duplicate_model_downloads(self) -> None:
        """Collapse duplicate paused/running tasks for the same model+version after restart."""
        groups: dict[tuple[str, str | None], list[str]] = {}
        for task_id, task in self._downloads.items():
            model_name = getattr(task, "_model_name", None)
            if not model_name:
                continue
            if task.status not in (TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.FAILED):
                continue
            key = (model_name, getattr(task, "_version", None))
            groups.setdefault(key, []).append(task_id)

        changed = False
        for key, task_ids in groups.items():
            if len(task_ids) == 1:
                self._active_model_downloads[key] = task_ids[0]
                continue

            def _score(tid: str) -> tuple[int, float]:
                prog = self._progress.get(tid)
                if not prog:
                    return (0, 0.0)
                return (prog.downloaded_size, prog.progress)

            keep_id = max(task_ids, key=_score)
            self._active_model_downloads[key] = keep_id
            for tid in task_ids:
                if tid == keep_id:
                    continue
                logger.info(
                    "Removing duplicate download task %s for %s (keeping %s)",
                    tid,
                    key,
                    keep_id,
                )
                del self._downloads[tid]
                self._progress.pop(tid, None)
                changed = True

        if changed:
            self._persist_downloads()


    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry."""
        registry_path = self._path_resolver.get_models_registry_path()
        if not registry_path.exists():
            return {}
        try:
            from backend.catalog.loader import expand_catalog_document

            with open(registry_path, "r", encoding="utf-8") as f:
                return expand_catalog_document(json.load(f)).get("models", {})
        except Exception as e:
            print(f"[DownloadService] Failed to load model registry: {e}")
            return {}

    def get_model_download_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get download configuration info for a model."""
        registry = self._load_registry()
        return registry.get(model_name)

    @staticmethod
    def _resolve_download_version_key(
        config: dict[str, Any],
        version: str | None,
    ) -> str | None:
        """Pick explicit version or registry default (``default: true``)."""
        from backend.core.version_keys import resolve_registry_version_key

        versions = config.get("versions")
        if not isinstance(versions, dict) or not versions:
            return (version or "").strip() or None
        key = (version or "").strip()
        resolved = resolve_registry_version_key(versions, key or None)
        if resolved and isinstance(versions.get(resolved), dict):
            return resolved
        return None

    @staticmethod
    def _check_hf_connectivity(timeout: float = 10.0) -> bool:
        """Check if HuggingFace mirror is accessible."""
        import urllib.request
        try:
            endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            req = urllib.request.Request(
                endpoint,
                method="HEAD",
                headers={"User-Agent": "huggingface_hub"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _check_modelscope_connectivity(timeout: float = 10.0) -> bool:
        """Check if ModelScope is accessible."""
        import urllib.request
        try:
            endpoint = "https://www.modelscope.cn"
            req = urllib.request.Request(
                endpoint,
                method="HEAD",
                headers={"User-Agent": "modelscope"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _parse_size_to_bytes(size_str: str) -> int:
        """Parse size string from registry (e.g. '23.8GB') into bytes."""
        if not size_str:
            return 0
        size_str = size_str.strip().upper()
        units = {
            'GB': 1024 ** 3,
            'MB': 1024 ** 2,
            'KB': 1024,
            'B': 1,
        }
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                try:
                    value = float(size_str[:-len(unit)])
                    return int(value * multiplier)
                except ValueError:
                    return 0
        return 0

    def _bundle_estimated_bytes(self, ver_config: dict[str, Any] | None) -> int:
        entries = bundle_repos_from_version(ver_config)
        if not entries:
            return 0
        total = 0
        for spec in entries:
            sz = spec.get("size")
            if sz:
                total += self._parse_size_to_bytes(str(sz))
        return total

    def _require_bundle_repo_complete(
        self,
        dest: Path,
        spec: dict[str, Any],
        *,
        label: str,
        ver_config: dict[str, Any] | None = None,
    ) -> None:
        variant = self._hunyuan_ms_variant_for_spec(spec, ver_config)
        if variant:
            self._require_hunyuan_ms_bundle_complete(dest, variant, label=label)
            return
        # Registry-driven "converted bundle is the real artifact" hooks: when a
        # hook declares ``skip_completeness_if``, the converted bundle's readiness
        # supersedes the raw-download completeness check. If neither the converted
        # bundle nor the raw files are present, fail with a reinstall hint.
        for hook in install_hooks_from_version(ver_config):
            check = hook.get("skip_completeness_if")
            if not check:
                continue
            if _run_skip_completeness_check(check, dest):
                return
            patterns = spec.get("allow_patterns")
            raw_missing = _missing_bundle_patterns(dest, patterns if isinstance(patterns, list) else None)
            if raw_missing:
                repo = str(spec.get("repo_id") or label)
                raise RuntimeError(
                    f"{label} incomplete after download ({repo}): missing {', '.join(raw_missing)}. "
                    "The converted bundle is also incomplete — reinstall to re-download the "
                    "source files and re-run conversion."
                )
            return
        patterns = spec.get("allow_patterns")
        missing = _missing_bundle_patterns(dest, patterns if isinstance(patterns, list) else None)
        if missing:
            repo = str(spec.get("repo_id") or label)
            raise RuntimeError(
                f"{label} incomplete after download ({repo}): missing {', '.join(missing)}"
            )

    @staticmethod
    def _hunyuan_ms_variant_for_spec(
        spec: dict[str, Any] | None,
        ver_config: dict[str, Any] | None = None,
    ) -> str | None:
        """Resolve ModelScope Hunyuan variant for assembly/readiness checks.

        ``hunyuan_ms_variant`` on the version entry applies only to repos that
        download the native ``transformer/<variant>/`` tree (``allow_patterns``
        with a ``transformer/`` prefix). Do not inherit it for sibling
        ``bundle_repos`` such as Qwen / ByT5 text encoders.
        """
        if not isinstance(spec, dict):
            return None
        own = spec.get("hunyuan_ms_variant")
        if own:
            return str(own)
        patterns = spec.get("allow_patterns")
        if isinstance(patterns, list) and any(
            str(p).startswith("transformer/") for p in patterns
        ):
            if isinstance(ver_config, dict):
                inherited = ver_config.get("hunyuan_ms_variant")
                if inherited:
                    return str(inherited)
        return None

    def _require_hunyuan_ms_bundle_complete(
        self,
        dest: Path,
        variant: str,
        *,
        label: str,
    ) -> None:
        from backend.services.hunyuan_ms_bundle import (
            ensure_hunyuan_ms_bundle_assembled,
            hunyuan_assembled_bundle_patterns,
            hunyuan_raw_download_patterns,
            is_hunyuan_ms_bundle_assembled,
        )

        if not dest.is_dir():
            raise RuntimeError(f"{label} incomplete: bundle directory missing ({dest})")
        if is_hunyuan_ms_bundle_assembled(dest):
            missing = _missing_bundle_patterns(dest, hunyuan_assembled_bundle_patterns())
        else:
            missing = _missing_bundle_patterns(dest, hunyuan_raw_download_patterns(variant))
            if missing:
                raise RuntimeError(
                    f"{label} incomplete after download (Tencent-Hunyuan/HunyuanVideo-1.5): "
                    f"missing {', '.join(missing)}"
                )
            ensure_hunyuan_ms_bundle_assembled(dest, variant)
            missing = _missing_bundle_patterns(dest, hunyuan_assembled_bundle_patterns())
        if missing:
            raise RuntimeError(
                f"{label} incomplete after HunyuanVideo assembly ({dest}): "
                f"missing {', '.join(missing)}"
            )

    def _hunyuan_ms_bundle_is_ready(self, dest: Path, variant: str) -> bool:
        from backend.services.hunyuan_ms_bundle import (
            ensure_hunyuan_ms_bundle_assembled,
            hunyuan_assembled_bundle_patterns,
            hunyuan_bundle_ready_patterns,
            hunyuan_raw_download_patterns,
            is_hunyuan_ms_bundle_assembled,
        )

        if not dest.is_dir():
            return False
        try:
            if is_hunyuan_ms_bundle_assembled(dest):
                return not _missing_bundle_patterns(dest, hunyuan_bundle_ready_patterns(variant))
            if _missing_bundle_patterns(dest, hunyuan_raw_download_patterns(variant)):
                return False
            ensure_hunyuan_ms_bundle_assembled(dest, variant)
            return not _missing_bundle_patterns(dest, hunyuan_bundle_ready_patterns(variant))
        except RuntimeError:
            return False

    def _check_download_stall(
        self,
        *,
        downloaded: int,
        last_growth_bytes: int,
        last_growth_time: float,
        label: str,
        stall_seconds: int | None = None,
        last_activity_time: float | None = None,
        connect_grace_seconds: int = 0,
        connect_grace_byte_cap: int = 0,
        active_file: str = "",
    ) -> None:
        now = time.time()
        if downloaded > last_growth_bytes:
            return
        timeout = int(stall_seconds or _DOWNLOAD_STALL_SECONDS)
        if (
            connect_grace_seconds > 0
            and connect_grace_byte_cap > 0
            and downloaded < connect_grace_byte_cap
            and now - last_growth_time < connect_grace_seconds
        ):
            return
        activity = last_activity_time if last_activity_time is not None else last_growth_time
        idle_for = now - max(last_growth_time, activity)
        if idle_for < timeout:
            return
        file_hint = f" (last file: {os.path.basename(active_file)})" if active_file else ""
        raise RuntimeError(
            f"{label} stalled: no new bytes for {int(idle_for)}s "
            f"(limit {timeout}s, downloaded {downloaded / (1024 ** 3):.2f} GB){file_hint}. "
            "ModelScope may be slow starting large shards (e.g. T5 ~10GB). "
            "Check network to modelscope.cn, cancel and retry, or try again off-peak."
        )

    def _resolve_registry_path(self, local_path: str) -> Path:
        return self._path_resolver.resolve_registry_local_path(local_path)

    def _assert_disk_headroom(self, dest: Path, required_bytes: int) -> None:
        import shutil

        dest.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(str(dest)).free
        if free < required_bytes:
            raise RuntimeError(
                f"Insufficient disk space under {dest}: need at least "
                f"{required_bytes / (1024 ** 3):.0f} GB free, have {free / (1024 ** 3):.0f} GB."
            )

    def _modelscope_snapshot(
        self,
        *,
        model_id: str,
        local_dir: str,
        allow_patterns: list[str] | None,
        progress_callback_cls: type | None = None,
        max_workers: int | None = None,
        retries: int = 3,
    ) -> str:
        from modelscope import snapshot_download as ms_snapshot

        if not self._check_modelscope_connectivity():
            raise ConnectionError(
                "Cannot connect to ModelScope, please check network connection"
            )
        pattern_groups = _split_modelscope_allow_patterns(allow_patterns)
        if not pattern_groups:
            pattern_groups = [allow_patterns]

        ms_callbacks = [progress_callback_cls] if progress_callback_cls else None
        last_exc: Exception | None = None
        result_path = local_dir
        for group in pattern_groups:
            workers = max_workers if max_workers is not None else _modelscope_workers_for_patterns(
                group
            )
            for attempt in range(1, retries + 1):
                try:
                    result_path = ms_snapshot(
                        model_id=model_id,
                        local_dir=local_dir,
                        allow_patterns=group if group else None,
                        progress_callbacks=ms_callbacks,
                        max_workers=workers,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= retries:
                        break
                    wait_s = min(2 ** attempt, 30)
                    logger.warning(
                        "ModelScope download %s failed (attempt %d/%d, patterns=%s): %s; retry in %ds",
                        model_id,
                        attempt,
                        retries,
                        group,
                        exc,
                        wait_s,
                    )
                    time.sleep(wait_s)
            if last_exc is not None:
                break

        if last_exc is not None:
            hint = (
                " Large safetensor shards download one at a time; ensure stable network "
                "to modelscope.cn and enough free disk (~65GB for Wan 720p distill MoE)."
            )
            raise RuntimeError(
                f"ModelScope download failed for {model_id}: {last_exc}.{hint}"
            ) from last_exc
        return result_path

    def _maybe_assemble_hunyuan_ms_bundle(self, target: Path, ver_config: dict[str, Any] | None) -> None:
        if not ver_config:
            return
        variant = ver_config.get("hunyuan_ms_variant")
        if not variant:
            return
        from backend.services.hunyuan_ms_bundle import ensure_hunyuan_ms_bundle_assembled

        ensure_hunyuan_ms_bundle_assembled(target, str(variant))

    def _dependency_is_ready(self, spec: DependencySpec) -> bool:
        dep_cfg = self.get_model_download_config(spec.model_id) or {}
        versions = dep_cfg.get("versions") or {}
        version_keys: list[str] = []
        if spec.version:
            version_keys.append(spec.version)
            if spec.version == "shared":
                from backend.core.version_keys import resolve_full_bundle_version_key

                base = resolve_full_bundle_version_key(versions)
                if base:
                    version_keys.append(base)
        else:
            version_keys = list(versions.keys())

        for version_key in version_keys:
            ver = versions.get(version_key)
            if not isinstance(ver, dict):
                continue
            try:
                local_path = version_primary_local_path(ver)
            except ValueError:
                continue
            root = self._path_resolver.resolve_registry_local_path(local_path)
            variant = self._hunyuan_ms_variant_for_spec(ver, None)
            if variant:
                if self._hunyuan_ms_bundle_is_ready(root, variant):
                    return True
                continue
            patterns = ver.get("allow_patterns")
            if isinstance(patterns, list) and patterns:
                if _bundle_repo_is_complete(root, [str(p) for p in patterns]):
                    return True
                continue
            if root.is_dir() and _calc_download_dir_bytes(root) >= 1024 ** 3:
                return True
        return False

    def _version_local_artifacts_ready(
        self,
        *,
        ver_config: dict[str, Any],
        target: Path,
        family: str | None = None,
        category: str | None = None,
    ) -> bool:
        bundle_entries = bundle_repos_from_version(ver_config)
        if bundle_entries:
            for spec in bundle_entries:
                dest = self._resolve_registry_path(str(spec["local_path"]))
                variant = self._hunyuan_ms_variant_for_spec(spec, ver_config)
                if variant:
                    if not self._hunyuan_ms_bundle_is_ready(dest, variant):
                        return False
                    continue
                patterns = spec.get("allow_patterns")
                pat_list = [str(p) for p in patterns] if isinstance(patterns, list) else None
                if pat_list is None:
                    from backend.services.services import SettingsService

                    if not dest.is_dir() or not SettingsService._path_has_bundle_weights(dest):
                        return False
                    continue
                if not _bundle_repo_is_complete(dest, pat_list):
                    return False
            return True

        variant = self._hunyuan_ms_variant_for_spec(ver_config, None)
        if variant:
            return self._hunyuan_ms_bundle_is_ready(target, variant)

        patterns = ver_config.get("allow_patterns")
        if isinstance(patterns, list) and patterns:
            return _bundle_repo_is_complete(target, [str(p) for p in patterns])

        family_key = str(family or "").strip()
        if family_key and target.is_dir():
            from backend.core.bundle_manifest import (
                bundle_component_status,
                skips_full_family_bundle_contract,
            )

            if not skips_full_family_bundle_contract(str(category or "").strip()):
                status = bundle_component_status(target, family=family_key)
                if status is not None:
                    return bool(status.get("complete"))

        return target.is_dir() and _calc_download_dir_bytes(target) >= 1024 ** 3

    async def _try_finish_if_already_installed(
        self,
        *,
        model_name: str,
        version: str | None,
        config: dict[str, Any],
        ver_config: dict[str, Any] | None,
        target: Path,
        task_id: str,
        display_name: str,
        total_size: int,
        on_progress: Callable[[DownloadProgress], Any],
    ) -> bool:
        if not ver_config or not self._version_local_artifacts_ready(
            ver_config=ver_config,
            target=target,
            family=str(config.get("family") or "").strip() or None,
            category=str(config.get("category") or "").strip() or None,
        ):
            return False

        for spec in parse_dependencies(config.get("dependencies")):
            if ver_config and ver_config.get("skip_dependencies"):
                break
            if not self._dependency_is_ready(spec):
                return False

        await self._finalize_version_install(
            model_name=model_name,
            version=version,
            ver_config=ver_config,
            target=target,
            task_id=task_id,
            display_name=display_name,
            on_progress=on_progress,
        )
        final_downloaded = _calc_download_dir_bytes(target)
        await on_progress(
            DownloadProgress(
                task_id=task_id,
                status="completed",
                progress=1.0,
                total_size=total_size,
                downloaded_size=final_downloaded,
                speed="",
                filename=display_name,
            )
        )
        return True

    async def _ensure_dependencies_installed(
        self,
        *,
        model_name: str,
        config: dict[str, Any],
        progress_callback: Callable[[DownloadProgress], None] | None,
    ) -> None:
        deps = parse_dependencies(config.get("dependencies"))
        if not deps:
            return
        for spec in deps:
            if self._dependency_is_ready(spec):
                continue
            dep_key = (spec.model_id, spec.version)
            if dep_key in self._dependency_install_chain:
                raise RuntimeError(
                    f"Circular model dependency while installing {model_name!r}: {spec.model_id!r}"
                )
            self._dependency_install_chain.add(dep_key)
            try:
                logger.info(
                    "Installing dependency %s:%s for %s",
                    spec.model_id,
                    spec.version or "default",
                    model_name,
                )
                await self.download_model(
                    spec.model_id,
                    version=spec.version,
                    progress_callback=progress_callback,
                )
            finally:
                self._dependency_install_chain.discard(dep_key)
            if not self._dependency_is_ready(spec):
                loc = get_locale()
                raise RuntimeError(
                    tt(
                        "error.missing_dependencies",
                        loc,
                        deps=f"{spec.model_id}"
                        + (f":{spec.version}" if spec.version else ""),
                    )
                )

    def _maybe_assemble_hunyuan_distill_bundle(self, target: Path, ver_config: dict[str, Any] | None) -> None:
        if not ver_config:
            return
        variant = ver_config.get("hunyuan_distill_variant")
        if not variant:
            return
        from backend.services.hunyuan_distill_shared import finalize_hunyuan_distill_bundle

        finalize_hunyuan_distill_bundle(distill_root=target, variant=str(variant))
        from backend.services.hunyuan_distill_bundle import assemble_hunyuan_distill_bundle

        assemble_hunyuan_distill_bundle(target, str(variant))

    def _maybe_assemble_wan_distill_bundle(self, target: Path, ver_config: dict[str, Any] | None) -> None:
        if not ver_config:
            return
        variant = ver_config.get("wan_distill_variant")
        if not variant:
            return
        from backend.services.wan_distill_shared import finalize_wan_distill_bundle

        finalize_wan_distill_bundle(distill_root=target, variant=str(variant))
        from backend.services.wan_distill_bundle import assemble_wan_distill_bundle

        assemble_wan_distill_bundle(target, str(variant))

    def _maybe_assemble_bernini_bundle(self, target: Path, ver_config: dict[str, Any] | None) -> None:
        """Hoist Diffusers Bernini-R shards in-place; never symlink Wan shared encoders."""
        if not ver_config:
            return
        variant = ver_config.get("bernini_variant")
        if not variant:
            return
        if str(ver_config.get("source_type") or "").lower() == "prequantized":
            return
        root = Path(target)
        if not (root / "transformer").is_dir() and not (root / "transformer_2").is_dir():
            return
        from backend.services.bernini_bundle import assemble_bernini_bundle

        assemble_bernini_bundle(root, str(variant))

    async def _finalize_version_install(
        self,
        *,
        model_name: str,
        version: str | None,
        ver_config: dict[str, Any] | None,
        target: Path,
        task_id: str,
        display_name: str,
        on_progress: Callable[[DownloadProgress], Any],
    ) -> None:
        """Post-download steps: Hunyuan MS assembly + install_hooks + bundle manifest."""
        if ver_config and ver_config.get("hunyuan_ms_variant"):
            self._maybe_assemble_hunyuan_ms_bundle(target, ver_config)
        if ver_config and ver_config.get("hunyuan_distill_variant"):
            self._maybe_assemble_hunyuan_distill_bundle(target, ver_config)
        if ver_config and ver_config.get("wan_distill_variant"):
            self._maybe_assemble_wan_distill_bundle(target, ver_config)
        if ver_config and ver_config.get("bernini_variant"):
            self._maybe_assemble_bernini_bundle(target, ver_config)

        hooks = install_hooks_from_version(ver_config)
        if hooks:
            logger.info(
                "Running %d install hook(s) for %s:%s at %s",
                len(hooks),
                model_name,
                version or "default",
                target,
            )
            await on_progress(
                DownloadProgress(
                    task_id=task_id,
                    status="running",
                    progress=0.99,
                    filename=display_name,
                    speed="",
                )
            )
            loop = asyncio.get_event_loop()

            def _run_hooks() -> None:
                run_install_hooks(
                    model_name=model_name,
                    version_key=version,
                    ver_config=ver_config,
                    bundle_root=target,
                )

            await loop.run_in_executor(None, _run_hooks)

        config = self.get_model_download_config(model_name) or {}
        family = str(config.get("family") or "").strip()
        category = str(config.get("category") or "").strip()
        if family and target.is_dir() and not skips_full_family_bundle_contract(category):
            from backend.core.bundle_manifest import assert_bundle_ready_for_family

            try:
                assert_bundle_ready_for_family(
                    target,
                    family=family,
                    model_id=model_name,
                )
                manifest_path = write_bundle_manifest(
                    target,
                    model_id=model_name,
                    family=family,
                )
                logger.info("Wrote bundle manifest: %s", manifest_path)
            except Exception as exc:
                logger.error(
                    "Bundle validation or manifest write failed for %s at %s: %s",
                    model_name,
                    target,
                    exc,
                )
                raise RuntimeError(
                    f"Post-download bundle validation failed for {model_name!r}: {exc}"
                ) from exc

    def _sync_download_bundle_repo(
        self,
        spec: dict[str, Any],
        *,
        default_source: str,
        progress_callback_cls: type | None = None,
        ver_config: dict[str, Any] | None = None,
    ) -> str:
        """Download one ``bundle_repos`` follow-up entry."""
        repo = str(spec["repo_id"]).strip()
        dest = self._resolve_registry_path(str(spec["local_path"]).strip())
        dest.mkdir(parents=True, exist_ok=True)
        source = str(spec.get("source") or default_source).strip().lower()
        variant = self._hunyuan_ms_variant_for_spec(spec, ver_config)
        if variant and self._hunyuan_ms_bundle_is_ready(dest, variant):
            logger.info("Bundle repo %s already present under %s; skipping download", repo, dest)
            return str(dest)
        patterns = spec.get("allow_patterns")
        if isinstance(patterns, list) and _bundle_repo_is_complete(dest, patterns):
            logger.info("Bundle repo %s already present under %s; skipping download", repo, dest)
            return str(dest)

        if source == "huggingface":
            from huggingface_hub import snapshot_download as hf_snapshot

            if not self._check_hf_connectivity():
                raise ConnectionError(
                    "Cannot connect to model download server (HF_ENDPOINT=%s), please check network connection"
                    % os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
                )
            return hf_snapshot(
                repo_id=resolve_huggingface_repo_id(repo),
                local_dir=str(dest),
                allow_patterns=patterns if patterns else None,
                token=self._token,
            )

        if source == "modelscope":
            if isinstance(patterns, list) and any(
                str(p).endswith((".safetensors", ".pth")) for p in patterns
            ):
                self._assert_disk_headroom(dest, 130 * 1024 ** 3)
            return self._modelscope_snapshot(
                model_id=repo,
                local_dir=str(dest),
                allow_patterns=patterns if patterns else None,
                progress_callback_cls=progress_callback_cls,
            )

        raise RuntimeError(f"Unsupported bundle repo source {source!r} for {repo!r}")

    async def download_model(self, model_name: str, version: Optional[str] = None,
                            progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                            existing_task_id: Optional[str] = None) -> str:
        """Download model by registry model name (supports base models and LoRA)."""
        config = self.get_model_download_config(model_name)
        if not config:
            loc = get_locale()
            raise ValueError(tt("error.model_not_in_registry", loc, name=model_name))
        if config.get("stub_no_download"):
            loc = get_locale()
            raise ValueError(tt("error.audio_stub_no_download", loc, name=model_name))

        loc = get_locale()
        version = self._resolve_download_version_key(config, version)

        source = config.get("source", "huggingface")
        repo_id = config.get("repo_id")
        download_url = config.get("download_url")
        local_path = config.get("local_path", f"models/{model_name}")

        # Build friendly display name: model name + version name
        model_display_name = resolve_registry_label(
            config.get("name"), model_name, locale=loc
        )

        version_display_name = ""
        ver_config = None
        if version and config.get("versions"):
            ver_config = config["versions"].get(version)
            if ver_config:
                version_display_name = resolve_registry_label(
                    ver_config.get("name"), version, locale=loc
                )
                if not repo_id:
                    repo_id = ver_config.get("repo_id")
                if not download_url:
                    download_url = ver_config.get("download_url")
                if ver_config.get("local_path"):
                    local_path = ver_config["local_path"]
                vs = ver_config.get("source")
                if isinstance(vs, str) and vs.strip():
                    source = vs.strip().lower()
                primary_spec, _follow = primary_and_follow_ups(ver_config)
                if primary_spec:
                    repo_id = primary_spec["repo_id"]
                    local_path = primary_spec["local_path"]
                    if primary_spec.get("source"):
                        source = str(primary_spec["source"]).strip().lower()

        if source == "huggingface" and repo_id:
            repo_id = resolve_huggingface_repo_id(repo_id)

        # Parse estimated size for this version from registry (fallback if HF API fails)
        estimated_size = 0
        bundle_bytes = self._bundle_estimated_bytes(ver_config)
        if bundle_bytes > 0:
            estimated_size = bundle_bytes
        elif ver_config and ver_config.get("size"):
            estimated_size = self._parse_size_to_bytes(ver_config["size"])
        elif config.get("size"):
            estimated_size = self._parse_size_to_bytes(config["size"])

        # Unified display name
        display_name = model_display_name
        if version_display_name:
            display_name = f"{model_display_name} {version_display_name}"

        target = self._path_resolver.resolve_registry_local_path(local_path)
        target.mkdir(parents=True, exist_ok=True)

        dedup_key = (model_name, version)

        async with self._download_lock:
            if dedup_key in self._inflight_model_downloads:
                inflight_id = self._active_model_downloads.get(dedup_key)
                if inflight_id and inflight_id in self._downloads:
                    await self._emit_task_progress(inflight_id, progress_callback)
                    return inflight_id

            existing_active_id = self._find_active_model_download(dedup_key)
            if existing_active_id and existing_active_id in self._downloads:
                existing_task = self._downloads[existing_active_id]
                if existing_task.status == TaskStatus.RUNNING:
                    await self._emit_task_progress(existing_active_id, progress_callback)
                    return existing_active_id
                if existing_task.status in (TaskStatus.PAUSED, TaskStatus.FAILED):
                    if not existing_task_id or existing_task_id != existing_active_id:
                        existing_task_id = existing_active_id

            if existing_task_id and existing_task_id in self._downloads:
                task_id = existing_task_id
                task = self._downloads[task_id]
                task.status = TaskStatus.RUNNING
                task.target_path = str(target)
            else:
                task_id = str(uuid.uuid4())
                task = DownloadTask(
                    id=task_id,
                    url=repo_id or download_url or "",
                    target_path=str(target),
                )
            task._model_name = model_name
            task._version = version
            self._downloads[task_id] = task
            self._active_model_downloads[dedup_key] = task_id
            task.status = TaskStatus.RUNNING
            self._inflight_model_downloads.add(dedup_key)

        # Internal progress callback (store progress + external callback + persist)
        async def on_progress(progress: DownloadProgress):
            self._progress[task_id] = progress
            self._persist_downloads()
            if progress_callback:
                await self._async_callback(progress_callback, progress)

        # When resuming download, preserve existing progress; don't reset to 0
        existing_progress = self._progress.get(task_id)
        if progress_callback:
            if existing_progress and existing_progress.downloaded_size > 0:
                await self._async_callback(progress_callback, existing_progress)
            else:
                display_name = model_display_name
                if version_display_name:
                    display_name = f"{model_display_name} {version_display_name}"
                init_total = estimated_size if estimated_size > 0 else (
                    existing_progress.total_size if existing_progress else 0
                )
                init_downloaded = (
                    existing_progress.downloaded_size
                    if existing_progress and existing_progress.downloaded_size > 0
                    else _calc_download_dir_bytes(target)
                )
                await self._async_callback(progress_callback, DownloadProgress(
                    task_id=task_id,
                    status="running",
                    progress=(init_downloaded / init_total) if init_total > 0 else 0,
                    total_size=init_total,
                    downloaded_size=init_downloaded,
                    speed="",
                    error_message="",
                    filename=display_name
                ))

        try:
            if not (ver_config and ver_config.get("skip_dependencies")):
                await self._ensure_dependencies_installed(
                    model_name=model_name,
                    config=config,
                    progress_callback=progress_callback,
                )

            if ver_config and await self._try_finish_if_already_installed(
                model_name=model_name,
                version=version,
                config=config,
                ver_config=ver_config,
                target=target,
                task_id=task_id,
                display_name=display_name,
                total_size=estimated_size,
                on_progress=on_progress,
            ):
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                self._active_model_downloads.pop(dedup_key, None)
                self._persist_downloads()
                return str(target)

            if source == "huggingface" and repo_id:
                # Inline HuggingFace download: snapshot_download + poll directory size for progress
                from huggingface_hub import snapshot_download

                def calc_downloaded_for(p: Path) -> int:
                    if not p.exists():
                        return 0
                    total = 0
                    for dirpath, _dirnames, filenames in os.walk(p):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            try:
                                total += os.path.getsize(fp)
                            except OSError:
                                pass
                    return total

                def format_speed(bytes_per_sec):
                    if bytes_per_sec > 1024 * 1024 * 1024:
                        return f"{bytes_per_sec / (1024 * 1024 * 1024):.1f} GB/s"
                    elif bytes_per_sec > 1024 * 1024:
                        return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
                    elif bytes_per_sec > 1024:
                        return f"{bytes_per_sec / 1024:.1f} KB/s"
                    else:
                        return f"{bytes_per_sec:.1f} B/s"

                total_size = estimated_size if estimated_size > 0 else (existing_progress.total_size if existing_progress else 0)

                loop = asyncio.get_event_loop()

                def do_download():
                    # Check network connectivity first; avoid snapshot_download silently falling back to existing dir
                    if not self._check_hf_connectivity():
                        raise ConnectionError(
                            "Cannot connect to model download server (HF_ENDPOINT=%s), please check network connection" % os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
                        )
                    allow_patterns = None
                    if ver_config:
                        from backend.services.hunyuan_ms_bundle import resolve_hunyuan_modelscope_allow_patterns

                        allow_patterns = resolve_hunyuan_modelscope_allow_patterns(ver_config)
                    kwargs: dict[str, Any] = {
                        "repo_id": repo_id,
                        "local_dir": str(target),
                        "token": self._token,
                    }
                    if isinstance(allow_patterns, list) and allow_patterns:
                        kwargs["allow_patterns"] = allow_patterns
                    return snapshot_download(**kwargs)

                download_future = loop.run_in_executor(None, do_download)

                # Poll directory size to calculate progress (HF)
                last_downloaded = calc_downloaded_for(target)
                last_report_time = time.time()

                try:
                    while not download_future.done():
                        await asyncio.sleep(2)  # Check every 2 seconds
                        if task_id in self._cancelled_downloads:
                            break

                        current_downloaded = calc_downloaded_for(target)
                        now = time.time()

                        # Calculate speed
                        dt = now - last_report_time
                        speed_str = ""
                        if dt >= 1:
                            speed_bytes = (current_downloaded - last_downloaded) / dt
                            speed_str = format_speed(speed_bytes)
                            last_downloaded = current_downloaded
                            last_report_time = now

                        progress = current_downloaded / total_size if total_size > 0 else 0

                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="running",
                            progress=min(progress, 1.0),
                            total_size=total_size,
                            downloaded_size=current_downloaded,
                            speed=speed_str,
                            filename=display_name
                        ))

                    result_path = await download_future

                    _, follow_ups = primary_and_follow_ups(ver_config)
                    if follow_ups:
                        label_parts = [display_name]
                        for spec in follow_ups:
                            repo_label = str(spec.get("name") or str(spec["repo_id"]).split("/")[-1])
                            label_parts.append(repo_label)
                            companion_label = " + ".join(label_parts)

                            def do_download_c(_spec=spec):
                                return self._sync_download_bundle_repo(
                                    _spec, default_source="huggingface",
                                )

                            c_future = loop.run_in_executor(None, do_download_c)
                            last_b = calc_downloaded_for(target)
                            last_report_time = time.time()
                            while not c_future.done():
                                await asyncio.sleep(2)
                                if task_id in self._cancelled_downloads:
                                    break
                                cur = calc_downloaded_for(target)
                                now = time.time()
                                dt = now - last_report_time
                                speed_str = ""
                                if dt >= 1:
                                    speed_bytes = (cur - last_b) / dt
                                    speed_str = format_speed(speed_bytes)
                                    last_b = cur
                                    last_report_time = now
                                prog = cur / total_size if total_size > 0 else 0.0
                                await on_progress(DownloadProgress(
                                    task_id=task_id,
                                    status="running",
                                    progress=min(prog, 1.0),
                                    total_size=total_size,
                                    downloaded_size=cur,
                                    speed=speed_str,
                                    filename=companion_label,
                                ))
                            await c_future
                        final_downloaded = calc_downloaded_for(target)
                        await self._finalize_version_install(
                            model_name=model_name,
                            version=version,
                            ver_config=ver_config,
                            target=target,
                            task_id=task_id,
                            display_name=companion_label,
                            on_progress=on_progress,
                        )
                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="completed",
                            progress=1.0,
                            total_size=total_size,
                            downloaded_size=final_downloaded,
                            speed="",
                            filename=companion_label,
                        ))
                        result = result_path
                    else:
                        final_downloaded = calc_downloaded_for(target)
                        await self._finalize_version_install(
                            model_name=model_name,
                            version=version,
                            ver_config=ver_config,
                            target=target,
                            task_id=task_id,
                            display_name=display_name,
                            on_progress=on_progress,
                        )
                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="completed",
                            progress=1.0,
                            total_size=total_size,
                            downloaded_size=final_downloaded,
                            speed="",
                            filename=display_name,
                        ))
                        result = result_path

                except asyncio.CancelledError:
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="cancelled",
                        progress=0,
                        filename=display_name
                    ))
                    raise
                except Exception as e:
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="failed",
                        progress=0,
                        error_message=str(e),
                        filename=display_name
                    ))
                    raise
            elif source == "modelscope" and repo_id:
                total_size = estimated_size if estimated_size > 0 else (
                    existing_progress.total_size if existing_progress else 0
                )
                baseline = _calc_download_dir_bytes(target)
                tracker = _ModelScopeProgressTracker(baseline_bytes=baseline)
                ms_allow_patterns: list[str] | None = None
                if ver_config:
                    _primary_spec, _ = primary_and_follow_ups(ver_config)
                    from backend.services.hunyuan_ms_bundle import resolve_hunyuan_modelscope_allow_patterns

                    ms_allow_patterns = resolve_hunyuan_modelscope_allow_patterns(
                        ver_config,
                        primary_spec=_primary_spec if isinstance(_primary_spec, dict) else None,
                    )

                loop = asyncio.get_event_loop()

                def do_download_ms():
                    allow_patterns = None
                    primary_spec = None
                    if ver_config:
                        primary_spec, _ = primary_and_follow_ups(ver_config)
                        from backend.services.hunyuan_ms_bundle import resolve_hunyuan_modelscope_allow_patterns

                        allow_patterns = resolve_hunyuan_modelscope_allow_patterns(
                            ver_config,
                            primary_spec=primary_spec if isinstance(primary_spec, dict) else None,
                        )
                    variant = self._hunyuan_ms_variant_for_spec(
                        primary_spec if isinstance(primary_spec, dict) else None,
                        ver_config,
                    )
                    if variant and self._hunyuan_ms_bundle_is_ready(target, variant):
                        logger.info(
                            "Model bundle already present under %s; skipping download",
                            target,
                        )
                        return str(target)
                    if isinstance(allow_patterns, list) and _bundle_repo_is_complete(
                        target, [str(p) for p in allow_patterns]
                    ):
                        logger.info(
                            "Model bundle already present under %s; skipping download",
                            target,
                        )
                        return str(target)
                    if isinstance(allow_patterns, list) and any(
                        str(p).endswith((".safetensors", ".pth")) for p in allow_patterns
                    ):
                        self._assert_disk_headroom(target, 130 * 1024 ** 3)
                    return self._modelscope_snapshot(
                        model_id=repo_id,
                        local_dir=str(target),
                        allow_patterns=allow_patterns if allow_patterns else None,
                        progress_callback_cls=tracker.callback_class(),
                    )

                download_future = loop.run_in_executor(None, do_download_ms)

                last_downloaded = tracker.downloaded_bytes(target)
                last_report_time = time.time()
                last_growth_bytes = last_downloaded
                last_growth_time = last_report_time

                def _ms_progress_filename(label: str, downloaded: int) -> str:
                    active = tracker.active_file
                    if active:
                        return f"{label} · {os.path.basename(active)}"
                    if downloaded <= baseline:
                        return f"{label} · 连接魔塔…"
                    return label

                try:
                    while not download_future.done():
                        await asyncio.sleep(1.0)
                        if task_id in self._cancelled_downloads:
                            break

                        current_downloaded = tracker.downloaded_bytes(target)
                        now = time.time()
                        if current_downloaded > last_growth_bytes:
                            last_growth_bytes = current_downloaded
                            last_growth_time = now
                        dt = now - last_report_time
                        speed_str = ""
                        if dt >= 1:
                            speed_str = _format_download_speed(
                                (current_downloaded - last_downloaded) / dt
                            )
                            last_downloaded = current_downloaded
                            last_report_time = now

                        progress = current_downloaded / total_size if total_size > 0 else 0

                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="running",
                            progress=min(progress, 1.0),
                            total_size=total_size,
                            downloaded_size=current_downloaded,
                            speed=speed_str,
                            filename=_ms_progress_filename(display_name, current_downloaded),
                        ))

                    result_path = await download_future

                    if ver_config and ver_config.get("allow_patterns") and not bundle_repos_from_version(
                        ver_config
                    ):
                        self._require_bundle_repo_complete(
                            target, ver_config, label=display_name, ver_config=ver_config
                        )

                    primary_spec, follow_ups = primary_and_follow_ups(ver_config)
                    if primary_spec:
                        self._require_bundle_repo_complete(
                            target,
                            primary_spec,
                            label=display_name,
                            ver_config=ver_config,
                        )
                    if follow_ups:
                        label_parts = [display_name]
                        for spec in follow_ups:
                            repo_label = str(spec.get("name") or str(spec["repo_id"]).split("/")[-1])
                            label_parts.append(repo_label)
                            companion_label = " + ".join(label_parts)
                            comp_tracker = _ModelScopeProgressTracker(
                                baseline_bytes=_calc_download_dir_bytes(target),
                            )

                            def do_download_ms_c(
                                _spec=spec,
                                _cb=comp_tracker.callback_class(),
                            ):
                                return self._sync_download_bundle_repo(
                                    _spec,
                                    default_source="modelscope",
                                    progress_callback_cls=_cb,
                                    ver_config=ver_config,
                                )

                            c_future = loop.run_in_executor(None, do_download_ms_c)
                            last_b = comp_tracker.downloaded_bytes(target)
                            last_report_time = time.time()
                            last_growth_bytes = last_b
                            last_growth_time = last_report_time
                            while not c_future.done():
                                await asyncio.sleep(1.0)
                                if task_id in self._cancelled_downloads:
                                    break
                                cur = comp_tracker.downloaded_bytes(target)
                                now = time.time()
                                if cur > last_growth_bytes:
                                    last_growth_bytes = cur
                                    last_growth_time = now
                                dt = now - last_report_time
                                speed_str = ""
                                if dt >= 1:
                                    speed_str = _format_download_speed((cur - last_b) / dt)
                                    last_b = cur
                                    last_report_time = now
                                prog = cur / total_size if total_size > 0 else 0.0
                                active = comp_tracker.active_file
                                fname = companion_label
                                if active:
                                    fname = f"{companion_label} · {os.path.basename(active)}"
                                elif cur <= last_growth_bytes:
                                    fname = f"{companion_label} · 连接魔塔…"
                                await on_progress(DownloadProgress(
                                    task_id=task_id,
                                    status="running",
                                    progress=min(prog, 1.0),
                                    total_size=total_size,
                                    downloaded_size=cur,
                                    speed=speed_str,
                                    filename=fname,
                                ))
                            await c_future
                            self._require_bundle_repo_complete(
                                target,
                                spec,
                                label=companion_label,
                                ver_config=ver_config,
                            )
                        final_downloaded = _calc_download_dir_bytes(target)
                        await self._finalize_version_install(
                            model_name=model_name,
                            version=version,
                            ver_config=ver_config,
                            target=target,
                            task_id=task_id,
                            display_name=companion_label,
                            on_progress=on_progress,
                        )
                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="completed",
                            progress=1.0,
                            total_size=total_size,
                            downloaded_size=final_downloaded,
                            speed="",
                            filename=companion_label,
                        ))
                        result = result_path
                    else:
                        final_downloaded = _calc_download_dir_bytes(target)
                        await self._finalize_version_install(
                            model_name=model_name,
                            version=version,
                            ver_config=ver_config,
                            target=target,
                            task_id=task_id,
                            display_name=display_name,
                            on_progress=on_progress,
                        )
                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="completed",
                            progress=1.0,
                            total_size=total_size,
                            downloaded_size=final_downloaded,
                            speed="",
                            filename=display_name,
                        ))
                        result = result_path

                except asyncio.CancelledError:
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="cancelled",
                        progress=0,
                        filename=display_name
                    ))
                    raise
                except Exception as e:
                    cur = tracker.downloaded_bytes(target)
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="failed",
                        progress=(cur / total_size) if total_size > 0 else 0,
                        total_size=total_size,
                        downloaded_size=cur,
                        error_message=str(e),
                        filename=display_name,
                    ))
                    raise
            elif download_url:
                # Download from download_url, target is a file under a directory
                # If files specifies specific files, download one by one
                files = config.get("files")
                if files and isinstance(files, list):
                    for i, file_pattern in enumerate(files):
                # Simplified: assume files is a list of specific filenames
                        file_url = f"{download_url}/{file_pattern}" if not file_pattern.startswith("http") else file_pattern
                        file_target = target / file_pattern
                        await self._http_downloader.download(
                            task_id=f"{task_id}_{i}",
                            url=file_url,
                            target=file_target,
                            progress_callback=on_progress
                        )
                    result = str(target)
                else:
                    # Single file download, infer filename from URL
                    filename = download_url.split("/")[-1].split("?")[0] or "model.safetensors"
                    file_target = target / filename
                    result = await self._http_downloader.download(
                        task_id=task_id,
                        url=download_url,
                        target=file_target,
                        progress_callback=on_progress
                    )
            else:
                loc = get_locale()
                raise ValueError(tt("error.model_no_download_source", loc, name=model_name))

            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            self._active_model_downloads.pop(dedup_key, None)
            self._persist_downloads()
            return result

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self._active_model_downloads.pop(dedup_key, None)
            self._persist_downloads()
            raise
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self._persist_downloads()
            raise
        finally:
            async with self._download_lock:
                self._inflight_model_downloads.discard(dedup_key)

    async def download_lora(self, url: str, filename: str,
                           progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                           existing_task_id: Optional[str] = None) -> str:
        """Download LoRA (generic HTTP)."""
        if existing_task_id and existing_task_id in self._downloads:
            task_id = existing_task_id
            task = self._downloads[task_id]
            task.status = TaskStatus.RUNNING
            target = Path(task.target_path)
        else:
            task_id = str(uuid.uuid4())
            target = self._path_resolver.get_loras_dir() / filename
            target.parent.mkdir(parents=True, exist_ok=True)

            task = DownloadTask(
                id=task_id,
                url=url,
                target_path=str(target)
            )
            task._is_lora = True
            task._filename = filename
            self._downloads[task_id] = task
            task.status = TaskStatus.RUNNING

        async def on_progress(progress: DownloadProgress):
            self._progress[task_id] = progress
            self._persist_downloads()
            if progress_callback:
                await self._async_callback(progress_callback, progress)

        # Send initial progress immediately so the API layer returns task_id without blocking the frontend
        if progress_callback:
            await self._async_callback(progress_callback, DownloadProgress(
                task_id=task_id,
                status="running",
                progress=0,
                total_size=0,
                downloaded_size=0,
                speed="",
                error_message="",
                filename=filename
            ))

        try:
            result = await self._http_downloader.download(
                task_id=task_id,
                url=url,
                target=target,
                progress_callback=on_progress
            )
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            self._persist_downloads()
            return result
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self._persist_downloads()
            raise
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self._persist_downloads()
            raise

    async def download_lora_from_hub(
        self,
        source: str,
        *,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        url: Optional[str] = None,
        civitai_version_id: Optional[int] = None,
        base_model: Optional[str] = None,
        display_name: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        existing_task_id: Optional[str] = None,
    ) -> str:
        """Download a remote LoRA from Hugging Face, ModelScope, CivitAI, or direct URL."""
        from backend.services.lora_search import (
            resolve_huggingface_lora_filename,
            resolve_huggingface_lora_url,
            resolve_modelscope_lora_filename,
            resolve_modelscope_lora_url,
        )
        from backend.third_party.civitai_client import CivitAIClient

        src = (source or "").strip().lower()
        download_url = (url or "").strip()
        target_name = (filename or "").strip()

        if src == "civitai":
            if not download_url and civitai_version_id:
                client = CivitAIClient(api_key=self._http_downloader._civitai_token)
                try:
                    version = await client.get_model_version(civitai_version_id)
                    primary = next((f for f in version.files if f.primary), version.files[0] if version.files else None)
                    if not primary:
                        raise RuntimeError("CivitAI version has no downloadable file")
                    download_url = primary.download_url
                    target_name = target_name or primary.name or f"civitai-{civitai_version_id}.safetensors"
                finally:
                    await client.close()
        elif src == "huggingface":
            if not repo_id:
                raise ValueError("repo_id is required for Hugging Face LoRA download")
            if not target_name:
                target_name = await resolve_huggingface_lora_filename(
                    repo_id, hf_token=self._token
                )
            if not download_url:
                download_url = resolve_huggingface_lora_url(repo_id, target_name)
        elif src == "modelscope":
            if not repo_id:
                raise ValueError("repo_id is required for ModelScope LoRA download")
            if not target_name:
                target_name = resolve_modelscope_lora_filename(repo_id)
            if not download_url:
                download_url = resolve_modelscope_lora_url(repo_id, target_name)
        elif src in ("http", "url"):
            if not download_url:
                raise ValueError("url is required for direct LoRA download")
        else:
            raise ValueError(f"Unsupported LoRA download source: {source}")

        if not download_url:
            raise RuntimeError("Could not resolve LoRA download URL")
        if not target_name:
            target_name = download_url.rsplit("/", 1)[-1].split("?")[0] or "lora.safetensors"
        if not target_name.endswith(".safetensors"):
            target_name = f"{target_name}.safetensors"

        result_path = await self.download_lora(
            download_url,
            target_name,
            progress_callback=progress_callback,
            existing_task_id=existing_task_id,
        )
        if base_model:
            self._register_remote_lora(
                result_path,
                base_model=base_model,
                display_name=display_name or repo_id or target_name,
                repo_id=repo_id or "",
                remote_hub_source=src,
            )
        return result_path

    def _register_remote_lora(
        self,
        absolute_path: str,
        *,
        base_model: str,
        display_name: str,
        repo_id: str,
        remote_hub_source: str,
    ) -> None:
        from backend.engine.training.user_lora_registry import list_user_loras, register_user_lora

        project_root = self._path_resolver.get_project_root()
        config_dir = self._path_resolver.get_workspace_config_dir()
        path = Path(absolute_path)
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            rel = Path("models/Lora") / path.name
        local_path = str(rel).replace("\\", "/")

        for item in list_user_loras(config_dir):
            if str(item.get("local_path") or "") == local_path:
                return
            if repo_id and str(item.get("repo_id") or "") == repo_id and str(item.get("base_model") or "") == base_model:
                return

        register_user_lora(
            config_dir,
            name=display_name.strip() or path.stem,
            base_model=base_model.split(":", 1)[0].strip(),
            local_path=local_path,
            source="remote_download",
            repo_id=repo_id,
            remote_hub_source=remote_hub_source,
        )

    def get_registry_models(self) -> Dict[str, Any]:
        """Expanded registry ``models`` map for download/search helpers."""
        return self._load_registry()

    def list_downloads(self) -> List[DownloadTask]:
        """List all download tasks."""
        return list(self._downloads.values())

    async def cancel_download(self, task_id: str) -> bool:
        """Cancel a download task."""
        task = self._downloads.get(task_id)
        if not task:
            return False

        self._cancelled_downloads.add(task_id)
        await self._http_downloader.cancel(task_id)

        task.status = TaskStatus.CANCELLED
        model_name = getattr(task, '_model_name', None)
        version = getattr(task, '_version', None)
        if model_name is not None:
            self._active_model_downloads.pop((model_name, version), None)
        self._persist_downloads()
        return True

    def delete_download(self, task_id: str) -> bool:
        """Delete a download task (remove from memory and persistence)."""
        task = self._downloads.get(task_id)
        if not task:
            return False

        # If the task is still running, cancel first
        if task.status == TaskStatus.RUNNING:
            self._cancelled_downloads.add(task_id)
            # Async cancel HTTP download
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._http_downloader.cancel(task_id))
            except RuntimeError:
                pass

        # Clean up dedup mapping
        model_name = getattr(task, '_model_name', None)
        version = getattr(task, '_version', None)
        if model_name is not None:
            self._active_model_downloads.pop((model_name, version), None)

        del self._downloads[task_id]
        self._progress.pop(task_id, None)
        self._persist_downloads()
        return True

    async def resume_download(self, task_id: str,
                              progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> str:
        """Resume a download task (after process restart).

        Determines whether this is a model download or LoRA download based on
        persisted metadata, and restarts the download reusing the original task_id.
        """
        task = self._downloads.get(task_id)
        if not task or task.status not in (TaskStatus.PAUSED, TaskStatus.FAILED):
            loc = get_locale()
            raise ValueError(tt("error.download_not_resumable", loc, id=task_id))

        task.error_message = ""

        model_name = getattr(task, '_model_name', None)
        version = getattr(task, '_version', None)
        is_lora = getattr(task, '_is_lora', False)
        filename = getattr(task, '_filename', None)

        if model_name:
            return await self.download_model(
                model_name, version=version,
                progress_callback=progress_callback,
                existing_task_id=task_id
            )
        elif is_lora and filename:
            return await self.download_lora(
                task.url, filename,
                progress_callback=progress_callback,
                existing_task_id=task_id
            )
        else:
            loc = get_locale()
            raise ValueError(tt("error.download_missing_metadata", loc, id=task_id))

    def get_progress(self, task_id: str) -> Optional[DownloadProgress]:
        """Get progress of a single download task."""
        return self._progress.get(task_id)

    async def convert_model(self, model_name: str, from_version: str, to_version: str,
                           progress_callback: Optional[Callable[[ConversionTask], None]] = None) -> str:
        """Build a derived weight layout via int4|int8 MLX quantization (``to_version`` names)."""
        config = self.get_model_download_config(model_name)
        if not config:
            loc = get_locale()
            raise ValueError(tt("error.model_not_in_registry", loc, name=model_name))

        versions = config.get("versions", {})
        from_ver_config = versions.get(from_version)
        to_ver_config = versions.get(to_version)

        if not from_ver_config:
            loc = get_locale()
            raise ValueError(tt("error.source_version_not_found", loc, version=from_version))
        if not to_ver_config:
            loc = get_locale()
            raise ValueError(tt("error.target_version_not_found", loc, version=to_version))

        loc = get_locale()
        if to_ver_config.get("source_type") != "derived":
            raise ValueError(tt("error.target_version_not_derived", loc, version=to_version))
        declared_parent = to_ver_config.get("from_version")
        if not declared_parent:
            raise ValueError(tt("error.derived_version_missing_parent", loc, version=to_version))
        if declared_parent != from_version:
            raise ValueError(
                tt(
                    "error.derived_version_parent_mismatch",
                    loc,
                    version=to_version,
                    expected=declared_parent,
                    got=from_version,
                )
            )

        from_path = self._resolve_version_path(from_ver_config)
        if not from_path.exists():
            loc = get_locale()
            raise ValueError(tt("error.source_version_not_ready", loc, version=from_version))

        to_path = self._resolve_version_path(to_ver_config)
        if to_path.exists():
            return str(to_path)

        task_id = str(uuid.uuid4())
        task = ConversionTask(
            id=task_id,
            model_name=model_name,
            from_version=from_version,
            to_version=to_version,
            status=TaskStatus.RUNNING,
            progress=0.0,
            stage="pending",
            output_path=str(to_path)
        )
        self._conversions[task_id] = task
        self._conversion_events[task_id] = asyncio.Event()

        async def update_progress(stage: str, progress: float):
            task.stage = stage
            task.progress = progress
            if progress_callback:
                await self._async_callback(progress_callback, task)

        if progress_callback:
            await self._async_callback(progress_callback, task)

        try:
            quantize_bits = (to_ver_config.get("quantization") or {}).get("bits")
            if not quantize_bits:
                raise ValueError(tt("error.cannot_determine_quantization", get_locale(), version=to_version))

            await update_progress("loading", 0.1)

            family = str(config.get("family") or "")
            from backend.core.derived_quant_layout import (
                copy_non_quantized_bundle,
                resolve_derived_quant_layout,
            )

            plan = resolve_derived_quant_layout(
                family=family,
                from_root=from_path,
                to_root=to_path,
                to_ver_config=to_ver_config,
            )

            def _quantize_target(
                load_paths: tuple[Path, ...],
                *,
                bits: int,
                output_dir: Path,
                shard_prefix: str,
                single_output_file: Path | None,
            ) -> int:
                from backend.core.derived_quant_mlx import quantize_and_save_sharded_inputs

                if not load_paths:
                    raise RuntimeError("No safetensors weight shards configured for quantization")
                return quantize_and_save_sharded_inputs(
                    load_paths,
                    bits=bits,
                    output_dir=output_dir,
                    shard_prefix=shard_prefix,
                    single_output_file=single_output_file,
                )

            def _do_conversion():
                """Execute quantization in a thread (blocking MLX computation)."""
                from backend.core.derived_quant_layout import copy_component_companion_files

                done = _quantize_target(
                    plan.load_paths,
                    bits=int(quantize_bits),
                    output_dir=plan.output_dir,
                    shard_prefix=plan.output_shard_prefix,
                    single_output_file=plan.single_output_file,
                )
                total = len([k for k in plan.load_paths])

                main_src = from_path / plan.output_dir.name
                if main_src.is_dir():
                    copy_component_companion_files(main_src, plan.output_dir)

                for target in plan.expert_quant_targets:
                    _quantize_target(
                        target.load_paths,
                        bits=int(quantize_bits),
                        output_dir=target.output_dir,
                        shard_prefix=target.output_shard_prefix,
                        single_output_file=target.single_output_file,
                    )
                    copy_component_companion_files(
                        from_path / target.subdir,
                        to_path / target.subdir,
                    )

                for target in plan.component_targets:
                    _quantize_target(
                        target.load_paths,
                        bits=int(target.bits),
                        output_dir=target.output_dir,
                        shard_prefix=target.output_shard_prefix,
                        single_output_file=target.single_output_file,
                    )
                    copy_component_companion_files(
                        from_path / target.subdir,
                        to_path / target.subdir,
                    )

                copy_non_quantized_bundle(from_path, to_path, plan)

                return done, total

            done, total = await asyncio.to_thread(_do_conversion)

            if self._conversion_events[task_id].is_set():
                if to_path.exists():
                    shutil.rmtree(to_path)
                raise asyncio.CancelledError("Conversion cancelled")

            await update_progress("completed", 1.0)

            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.stage = "completed"
            return str(to_path)

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.stage = "cancelled"
            if to_path.exists():
                shutil.rmtree(to_path)
            raise
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.stage = "error"
            task.error_message = str(e)
            if to_path.exists():
                shutil.rmtree(to_path)
            raise
        finally:
            self._conversion_events.pop(task_id, None)

    def _resolve_version_path(self, version_config: Dict[str, Any]) -> Path:
        """Resolve install root for a version (bundle_repos[0] or local_path)."""
        return self._path_resolver.resolve_registry_local_path(
            version_primary_local_path(version_config)
        )

    def list_conversions(self) -> List[ConversionTask]:
        """List all conversion tasks."""
        return list(self._conversions.values())

    async def cancel_conversion(self, task_id: str) -> bool:
        """Cancel a conversion task."""
        task = self._conversions.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False

        event = self._conversion_events.get(task_id)
        if event:
            event.set()

        task.status = TaskStatus.CANCELLED
        return True

    def get_conversion_progress(self, task_id: str) -> Optional[ConversionTask]:
        """Get progress of a single conversion task."""
        return self._conversions.get(task_id)

    async def delete_model(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Delete a model or specified version.

        Uses the local_path field in models_registry.json to find and delete the model file directory.
        """
        config = self.get_model_download_config(model_name)
        if not config:
            loc = get_locale()
            return {"success": False, "deleted_paths": [], "error": tt("error.model_not_in_registry", loc, name=model_name)}

        deleted_paths = []

        if version and config.get("versions"):
            ver_config = config["versions"].get(version)
            if not ver_config:
                loc = get_locale()
                return {"success": False, "deleted_paths": [], "error": tt("error.version_not_found", loc, version=version)}

            deleted_set: set[str] = set()
            paths = bundle_local_paths(ver_config)
            if not paths:
                paths = [version_primary_local_path(ver_config)]
            for lp in paths:
                cpath = self._path_resolver.resolve_registry_local_path(lp)
                key = str(cpath)
                if key in deleted_set:
                    continue
                deleted_set.add(key)
                if cpath.exists():
                    shutil.rmtree(cpath)
                    deleted_paths.append(key)
        else:
            versions = config.get("versions", {})
            if versions:
                deleted_set = set()
                for ver_key, ver_config in versions.items():
                    paths = bundle_local_paths(ver_config)
                    if not paths:
                        paths = [version_primary_local_path(ver_config)]
                    for lp in paths:
                        cpath = self._path_resolver.resolve_registry_local_path(lp)
                        key = str(cpath)
                        if key in deleted_set:
                            continue
                        deleted_set.add(key)
                        if cpath.exists():
                            shutil.rmtree(cpath)
                            deleted_paths.append(key)
            else:
                local_path = config.get("local_path", f"models/{model_name}")
                target = self._path_resolver.resolve_registry_local_path(local_path)
                if target.exists():
                    shutil.rmtree(target)
                    deleted_paths.append(str(target))

        return {"success": True, "deleted_paths": deleted_paths, "error": None}

    @staticmethod
    async def _async_callback(callback: Callable, progress):
        """Safely invoke an async callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)
        except Exception as e:
            print(f"[DownloadService] Callback exception: {e}")
