"""Write llm.pid / llm.port under the control plane."""

from __future__ import annotations

import os
from pathlib import Path

from shared.danqing_config.paths import control_plane_dir, llm_pid_file, llm_port_file


def write_pid_port(*, host: str, port: int, control_plane: Path | None = None) -> None:
    control = (control_plane or control_plane_dir()).resolve()
    control.mkdir(parents=True, exist_ok=True)
    llm_pid_file(control).write_text(str(os.getpid()), encoding="utf-8")
    llm_port_file(control).write_text(str(int(port)), encoding="utf-8")


def clear_pid_port(control_plane: Path | None = None) -> None:
    control = (control_plane or control_plane_dir()).resolve()
    for path in (llm_pid_file(control), llm_port_file(control)):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
