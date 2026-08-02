#!/usr/bin/env python3
"""Run backend engine unit tests (stdlib unittest, no GPU)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_ENGINE_TEST_MODULES = (
    "tests.engine_unit",
    "tests.test_catalog_unit",
    "tests.test_observability_unit",
    "tests.test_engine_sessions_unit",
    "tests.setup_unit",
    "tests.storyboard_unit",
    "tests.long_video_activity_unit",
    "tests.long_video_t2i_provenance_unit",
    "tests.llm_model_resolve_unit",
)


def main() -> int:
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for name in _ENGINE_TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
