"""MiniMax-H3 DiT block streaming (Phase 2) — placeholder.

Future: mmap safetensors shards and keep only N transformer blocks resident (h3.c
``--ssd-streaming`` analogue). Until implemented, ``h3_block_streaming=True`` fails
loud in ``MinimaxH3MlxGenerator.generate_and_save``.
"""
from __future__ import annotations

BLOCK_STREAMING_IMPLEMENTED = False
DEFAULT_RESIDENT_BLOCKS = 2
