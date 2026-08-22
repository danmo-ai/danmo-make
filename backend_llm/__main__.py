"""Entry point: ``python -m backend_llm`` / ``bin/danqing-llm``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _bootstrap_sys_path() -> None:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent.resolve()
        if sys.platform == "darwin" and exe_dir.name == "MacOS":
            resources = exe_dir.parent / "Resources"
            if resources.is_dir():
                root = resources
            else:
                root = exe_dir
        else:
            root = exe_dir
    else:
        root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main(argv: list[str] | None = None) -> int:
    _bootstrap_sys_path()

    parser = argparse.ArgumentParser(description="DanQing LLM sidecar (mlx_vlm.server)")
    parser.add_argument(
        "--host",
        default=os.environ.get("DANQING_LLM_HTTP_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DANQING_LLM_HTTP_PORT", "7801")),
    )
    parser.add_argument(
        "--preload",
        default="",
        help="Optional model directory to preload at startup",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("DANQING_LLM_LOG_LEVEL", "INFO"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from backend_llm.lifecycle import write_pid_port
    from backend_llm.server_launcher import launch_server, resolve_preload_model

    host = str(args.host).strip() or "127.0.0.1"
    port = max(1, int(args.port))
    write_pid_port(host=host, port=port)

    preload = (args.preload or "").strip() or resolve_preload_model()
    try:
        launch_server(host=host, port=port, preload_model=preload or None)
    finally:
        from backend_llm.lifecycle import clear_pid_port

        clear_pid_port()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
