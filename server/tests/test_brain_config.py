"""Tests for brain selection (``server/brains/config.py`` + ``brains/__init__``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from brains import get_brain
from brains.base import BrainName
from brains.config import DEFAULT_BRAIN, get_brain_name


def test_default_when_no_file(no_brains_yaml: Path) -> None:
    assert get_brain_name() == DEFAULT_BRAIN == BrainName.CARTESIA_OPENAI


def test_uses_default_brain_from_file(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml("default_brain: openai_standard\n")
    assert get_brain_name() == BrainName.OPENAI_STANDARD


def test_get_brain_returns_registered_spec() -> None:
    spec = get_brain(BrainName.OPENAI_REALTIME)
    assert spec.name == BrainName.OPENAI_REALTIME
    assert spec.is_realtime is True


def test_get_brain_defaults_to_selected(no_brains_yaml: Path) -> None:
    assert get_brain().name == DEFAULT_BRAIN


def test_get_brain_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_brain("not_a_brain")  # type: ignore[arg-type]
