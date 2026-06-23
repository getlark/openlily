"""Shared fixtures for the test suite.

These tests cover pure config/validation logic only -- no real LLM/STT/TTS
calls, no audio hardware, and no network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from brains import overrides


@pytest.fixture(autouse=True)
def _clear_overrides_cache() -> Iterator[None]:
    """Reset the ``@lru_cache``d brains.yaml loader around every test.

    ``get_brain_overrides`` caches its first read, so without this a stale
    result (or path) would leak between tests.
    """
    overrides.get_brain_overrides.cache_clear()
    yield
    overrides.get_brain_overrides.cache_clear()


@pytest.fixture
def brains_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    """Return a writer that creates a temp ``brains.yaml`` the loader will read.

    Calling the returned function with YAML text writes it to a temp file and
    points ``overrides._OVERRIDE_PATH`` at it, so ``get_brain_overrides`` loads
    that file instead of any real one on disk.
    """

    def _write(contents: str) -> Path:
        path = tmp_path / "brains.yaml"
        path.write_text(contents)
        monkeypatch.setattr(overrides, "_OVERRIDE_PATH", path)
        return path

    return _write


@pytest.fixture
def no_brains_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the loader at a path that does not exist (the 'no file' case)."""
    path = tmp_path / "does-not-exist.yaml"
    monkeypatch.setattr(overrides, "_OVERRIDE_PATH", path)
    return path
