#!/usr/bin/env python3
"""Unit tests for salted API-key hashing / verification."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.api.access_auth import (
    generate_api_key,
    hash_api_key,
    is_loopback_host,
    key_hint,
    verify_stored_key,
)


def main() -> int:
    plain = generate_api_key("http")
    assert plain.startswith("dmh_")
    stored = hash_api_key(plain)
    assert stored.startswith("v1:")
    assert verify_stored_key(plain, stored)
    assert not verify_stored_key(plain + "x", stored)
    other = hash_api_key(plain)  # new salt → different stored blob
    assert other != stored
    assert verify_stored_key(plain, other)

    # Non-v1 blobs must not authenticate.
    assert not verify_stored_key(plain, "sha256:" + hashlib.sha256(plain.encode()).hexdigest())
    assert not verify_stored_key("secret", "secret")
    assert not verify_stored_key(plain, "v1:bad")
    assert not verify_stored_key(plain, "")

    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert not is_loopback_host("192.168.1.1")

    hint = key_hint(plain)
    assert "…" in hint
    print("ok: access_auth salted HMAC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
