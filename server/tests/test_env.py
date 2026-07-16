"""Tests for the fail-fast env helper (``openlily/env.py``)."""

from __future__ import annotations

import pytest

from openlily.env import require_env


def test_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENLILY_TEST_VAR", "hello")
    assert require_env("OPENLILY_TEST_VAR") == "hello"


def test_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENLILY_TEST_VAR", raising=False)
    with pytest.raises(RuntimeError) as exc:
        require_env("OPENLILY_TEST_VAR")
    assert "OPENLILY_TEST_VAR is required but is not set in the environment." in str(exc.value)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_raises_when_blank_or_whitespace(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("OPENLILY_TEST_VAR", blank)
    with pytest.raises(RuntimeError):
        require_env("OPENLILY_TEST_VAR")


def test_appends_callsite_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENLILY_TEST_VAR", raising=False)
    with pytest.raises(RuntimeError) as exc:
        require_env("OPENLILY_TEST_VAR", "Set it to use the openai_standard brain.")
    message = str(exc.value)
    assert message.endswith("Set it to use the openai_standard brain.")
    assert "OPENLILY_TEST_VAR is required" in message
