"""Tests for the CLI env-parsing helpers (``openlily/cli.py``).

Importing ``openlily.cli`` pulls in Pipecat (and runs ``load_dotenv`` when the
runner is imported); that's fine here since deps are installed and the helpers
read ``os.getenv`` at call time, so ``monkeypatch.setenv`` after import still
takes effect.
"""

from __future__ import annotations

import pytest

from openlily.agent import build_worker
from openlily.cli import _idle_timeout_secs, _wake_models
from openlily.config import DEFAULT_IDLE_TIMEOUT_SECS


def test_idle_timeout_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IDLE_TIMEOUT_SECS", raising=False)
    assert _idle_timeout_secs() == DEFAULT_IDLE_TIMEOUT_SECS


def test_idle_timeout_parses_valid_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDLE_TIMEOUT_SECS", "12.5")
    assert _idle_timeout_secs() == 12.5


def test_idle_timeout_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDLE_TIMEOUT_SECS", "not-a-number")
    assert _idle_timeout_secs() == DEFAULT_IDLE_TIMEOUT_SECS


def test_wake_models_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WAKE_MODELS", raising=False)
    assert _wake_models() == ["alexa"]


def test_wake_models_default_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAKE_MODELS", "   ,  ")
    assert _wake_models() == ["alexa"]


def test_wake_models_splits_and_trims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAKE_MODELS", " hey_jarvis , alexa ")
    assert _wake_models() == ["hey_jarvis", "alexa"]


def test_factory_exposes_build_worker() -> None:
    # Smoke check that the factory entry point is importable.
    assert callable(build_worker)
