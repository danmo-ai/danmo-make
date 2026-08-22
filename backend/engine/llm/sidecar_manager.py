"""Lifecycle for the backend_llm mlx_vlm.server sidecar."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from shared.danqing_config.inference import load_llm_inference_config, resolve_builtin_base_url
from shared.danqing_config.paths import (
    control_plane_dir,
    llm_pid_file,
    llm_port_file,
    resolve_install_root,
)

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT_S = 120.0
_HEALTH_POLL_S = 0.5


class LlmSidecarManager:
    """Start/stop the local backend_llm process."""

    def __init__(self) -> None:
        self._subprocess: subprocess.Popen[Any] | None = None

    def configured_base_url(self) -> str | None:
        env = os.environ.get("DANQING_LLM_BASE_URL", "").strip().rstrip("/")
        if env:
            return env if env.endswith("/v1") else f"{env}/v1"
        port_path = llm_port_file()
        if port_path.is_file():
            try:
                port = int(port_path.read_text(encoding="utf-8").strip())
                host = os.environ.get("DANQING_LLM_HTTP_HOST", "127.0.0.1").strip()
                return f"http://{host}:{port}/v1"
            except (OSError, ValueError):
                pass
        return None

    def _sidecar_executable(self) -> list[str]:
        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).parent.resolve()
            candidate = exe_dir / "danqing-llm" / "danqing-llm"
            if sys.platform == "win32":
                candidate = candidate.with_suffix(".exe")
            if candidate.is_file():
                return [str(candidate)]
        repo = resolve_install_root()
        py = repo / ".venv" / "bin" / "python3"
        if py.is_file():
            return [str(py), "-m", "backend_llm"]
        return [sys.executable, "-m", "backend_llm"]

    def _spawn_cmd(self, *, host: str, port: int) -> list[str]:
        return [*self._sidecar_executable(), "--host", host, "--port", str(port)]

    def is_running(self) -> bool:
        if self._subprocess is not None and self._subprocess.poll() is None:
            return True
        pid_path = llm_pid_file()
        if not pid_path.is_file():
            return False
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def health_ok(self, *, base_url: str | None = None) -> bool:
        url_base = (base_url or self.configured_base_url() or "").rstrip("/")
        if not url_base:
            return False
        root = url_base[: -len("/v1")] if url_base.endswith("/v1") else url_base
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                resp = client.get(f"{root}/health")
                if resp.status_code != 200:
                    return False
                return str(resp.json().get("status", "")).lower() == "healthy"
        except Exception:
            return False

    def ensure_running(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
    ) -> str:
        """Start sidecar if needed; return ``/v1`` base URL."""
        from shared.danqing_config.settings import load_app_settings

        snap = load_app_settings()
        effective_port = port or snap.llm_builtin_port or 7801
        if self.health_ok(base_url=resolve_builtin_base_url(host=host, port=effective_port)):
            return resolve_builtin_base_url(host=host, port=effective_port)

        if self.is_running():
            deadline = time.time() + _HEALTH_TIMEOUT_S
            while time.time() < deadline:
                if self.health_ok(base_url=resolve_builtin_base_url(host=host, port=effective_port)):
                    return resolve_builtin_base_url(host=host, port=effective_port)
                time.sleep(_HEALTH_POLL_S)
            raise RuntimeError("LLM sidecar started but health check timed out")

        control_plane_dir().mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["DANQING_LLM_HTTP_HOST"] = host
        env["DANQING_LLM_HTTP_PORT"] = str(effective_port)
        repo = resolve_install_root()
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo) if not py_path else f"{repo}{os.pathsep}{py_path}"

        cmd = self._spawn_cmd(host=host, port=effective_port)
        logger.info("Starting LLM sidecar: %s", " ".join(cmd))
        self._subprocess = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(repo),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + _HEALTH_TIMEOUT_S
        while time.time() < deadline:
            if self._subprocess.poll() is not None:
                raise RuntimeError(
                    f"LLM sidecar exited early with code {self._subprocess.returncode}"
                )
            if self.health_ok(base_url=resolve_builtin_base_url(host=host, port=effective_port)):
                return resolve_builtin_base_url(host=host, port=effective_port)
            time.sleep(_HEALTH_POLL_S)
        raise RuntimeError("LLM sidecar failed to become healthy")

    def stop(self) -> None:
        if self._subprocess is not None and self._subprocess.poll() is None:
            self._subprocess.send_signal(signal.SIGTERM)
            try:
                self._subprocess.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._subprocess.kill()
            self._subprocess = None

        pid_path = llm_pid_file()
        if pid_path.is_file():
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
                os.kill(pid, signal.SIGTERM)
            except (OSError, ValueError, ProcessLookupError):
                pass

        for path in (llm_pid_file(), llm_port_file()):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def should_manage_sidecar(self, settings) -> bool:
        cfg = load_llm_inference_config(
            type(
                "Snap",
                (),
                {
                    "default_model_llm": settings.default_model_llm,
                    "default_model_vlm": settings.default_model_vlm,
                    "default_model_llm_think": settings.default_model_llm_think,
                    "llm_unload_each_request": settings.llm_unload_each_request,
                    "model_cache_ttl_minutes": settings.model_cache_ttl_minutes,
                    "llm_cache_ttl_minutes": settings.llm_cache_ttl_minutes,
                    "llm_inference_provider": settings.llm_inference_provider,
                    "llm_inference_base_url": settings.llm_inference_base_url,
                    "llm_inference_api_key": settings.llm_inference_api_key,
                    "llm_inference_api_key_hint": settings.llm_inference_api_key_hint,
                    "llm_inference_cloud_model": settings.llm_inference_cloud_model,
                    "llm_builtin_host": settings.llm_builtin_host,
                    "llm_builtin_port": settings.llm_builtin_port,
                    "llm_quantize_activations": settings.llm_quantize_activations,
                },
            )()
        )
        return cfg.provider == "builtin"
