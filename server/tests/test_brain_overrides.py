"""Tests for brains.yaml parsing and validation (``server/brains/overrides.py``).

This is the core coverage for "brains.yaml values are picked up as overrides"
and for the fail-fast validation around a malformed file.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from brains.base import BrainName
from brains.overrides import BrainOverrides, get_brain_overrides
from tools.contracts import ToolName


def test_no_file_returns_empty_defaults(no_brains_yaml: Path) -> None:
    overrides = get_brain_overrides()
    assert isinstance(overrides, BrainOverrides)
    assert overrides.default_brain is None
    assert overrides.cartesia_openai.tts.voice is None
    assert overrides.cartesia_openai.llm.model is None
    # No 'tools' key -> no optional tools enabled.
    assert overrides.tools == []


def test_valid_file_parses_overrides(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml(
        """
        default_brain: cartesia_openai
        cartesia_openai:
          stt:
            model: ink-2
          llm:
            model: gpt-5.4-mini
          tts:
            model: sonic-3.5
            voice: my-custom-voice-id
        """
    )

    overrides = get_brain_overrides()

    assert overrides.default_brain is BrainName.CARTESIA_OPENAI
    assert overrides.cartesia_openai.stt.model == "ink-2"
    assert overrides.cartesia_openai.llm.model == "gpt-5.4-mini"
    assert overrides.cartesia_openai.tts.model == "sonic-3.5"
    assert overrides.cartesia_openai.tts.voice == "my-custom-voice-id"


def test_tools_list_parses(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml(
        """
        default_brain: cartesia_openai
        tools:
          - x
          - browser
        """
    )

    overrides = get_brain_overrides()

    assert overrides.tools == [ToolName.X, ToolName.BROWSER]


def test_tools_absent_defaults_empty(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml("default_brain: cartesia_openai\n")
    assert get_brain_overrides().tools == []


def test_unknown_tool_name_raises(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml(
        """
        default_brain: cartesia_openai
        tools:
          - not_a_real_tool
        """
    )
    with pytest.raises(RuntimeError) as exc:
        get_brain_overrides()
    assert "invalid settings" in str(exc.value)


def test_present_file_missing_default_brain_raises(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml(
        """
        cartesia_openai:
          llm:
            model: gpt-5.4-mini
        """
    )
    with pytest.raises(RuntimeError) as exc:
        get_brain_overrides()
    assert "default_brain" in str(exc.value)


def test_invalid_yaml_raises(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml("default_brain: [unterminated\n")
    with pytest.raises(RuntimeError) as exc:
        get_brain_overrides()
    assert "not valid YAML" in str(exc.value)


def test_non_mapping_top_level_raises(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml("- just\n- a\n- list\n")
    with pytest.raises(RuntimeError) as exc:
        get_brain_overrides()
    assert "must be a mapping" in str(exc.value)


def test_unknown_field_key_raises(brains_yaml: Callable[[str], Path]) -> None:
    # ``mdoel`` typo under tts -- extra="forbid" turns this into an error.
    brains_yaml(
        """
        default_brain: cartesia_openai
        cartesia_openai:
          tts:
            mdoel: sonic-3.5
        """
    )
    with pytest.raises(RuntimeError) as exc:
        get_brain_overrides()
    assert "invalid settings" in str(exc.value)


def test_unknown_brain_key_raises(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml(
        """
        default_brain: cartesia_openai
        not_a_real_brain:
          llm:
            model: x
        """
    )
    with pytest.raises(RuntimeError):
        get_brain_overrides()


def test_invalid_default_brain_value_raises(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml("default_brain: nonexistent_brain\n")
    with pytest.raises(RuntimeError):
        get_brain_overrides()


def test_loader_is_cached(brains_yaml: Callable[[str], Path]) -> None:
    brains_yaml("default_brain: openai_standard\n")
    first = get_brain_overrides()
    second = get_brain_overrides()
    assert first is second
