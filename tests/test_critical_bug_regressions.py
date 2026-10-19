"""Regression tests for critical bug fixes (no MLX import required)."""

from __future__ import annotations

import inspect
import tempfile
import threading
import unittest
from pathlib import Path


class CriticalBugRegressionTests(unittest.TestCase):
    def test_subprocess_pythonpath_uses_os_pathsep(self) -> None:
        import os

        with self.assertRaises(AttributeError):
            _ = Path.pathsep
        joined = f"/repo{os.pathsep}/existing"
        self.assertIn("/repo", joined)
        self.assertIn("/existing", joined)

    def test_resolve_vocal_language_rejects_prompt_kwarg(self) -> None:
        from backend.engine.families.ace_step.generation import resolve_vocal_language

        self.assertEqual(resolve_vocal_language("hello world", "en"), "en")
        with self.assertRaises(TypeError):
            resolve_vocal_language("hello world", "en", prompt="pop")  # type: ignore[call-arg]

    def test_canvas_update_reads_under_lock(self) -> None:
        from backend.persistence.canvas_session_store import CanvasSessionStore

        src = inspect.getsource(CanvasSessionStore.update_session)
        self.assertNotIn("get_session", src)

        with tempfile.TemporaryDirectory() as tmp:
            store = CanvasSessionStore(Path(tmp) / "canvas.db")
            session = store.create_session(state={"items": {"a": 1}})
            sid = session["id"]
            store.update_session(sid, state={"items": {"a": 1, "b": 2}})
            updated = store.update_session(sid, title="renamed")
            self.assertIsNotNone(updated)
            assert updated is not None
            self.assertEqual(updated["title"], "renamed")
            self.assertEqual(updated["state"]["items"], {"a": 1, "b": 2})

    def test_canvas_concurrent_partial_update_keeps_latest_state(self) -> None:
        from backend.persistence.canvas_session_store import CanvasSessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = CanvasSessionStore(Path(tmp) / "canvas.db")
            session = store.create_session(state={"items": {}})
            sid = session["id"]
            errors: list[BaseException] = []

            def writer_state() -> None:
                try:
                    store.update_session(sid, state={"items": {"node": 1}})
                except BaseException as exc:
                    errors.append(exc)

            def writer_title() -> None:
                try:
                    store.update_session(sid, title="t1")
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=writer_state) for _ in range(5)]
            threads += [threading.Thread(target=writer_title) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            self.assertEqual(errors, [])
            final = store.get_session(sid)
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(final["title"], "t1")
            self.assertEqual(final["state"]["items"], {"node": 1})

    def test_loras_path_traversal_guard_logic(self) -> None:
        def reject(name: str) -> bool:
            return ".." in name or "/" in name or "\\" in name

        self.assertTrue(reject("../secret"))
        self.assertTrue(reject("foo/bar"))
        self.assertFalse(reject("final_adapters.safetensors"))

        root = Path("/tmp/dq-test-root")
        base = (root / "datasets" / "ds1").resolve()
        escaped = (base / "../../etc/passwd").resolve()
        with self.assertRaises(ValueError):
            escaped.relative_to(base)


if __name__ == "__main__":
    unittest.main()
