"""Tests for the system prompt builder (``server/prompt.py``)."""

from __future__ import annotations

from datetime import datetime

from prompt import build_system_instruction


def test_no_tools_leaves_no_bullets_or_blank_line() -> None:
    instruction = build_system_instruction()
    # The base prompt is always present.
    assert "<OutputRules>" in instruction
    # No tool bullet was injected after the "Use available tools as needed." line.
    assert "Use available tools as needed.\n- Speak outcomes clearly." in instruction


def test_tool_instructions_render_as_bullets() -> None:
    instruction = build_system_instruction(["You can search the web.", "You can send email."])
    assert "- You can search the web." in instruction
    assert "- You can send email." in instruction


def test_appends_todays_date() -> None:
    instruction = build_system_instruction()
    today = datetime.now().strftime("%A, %B %-d, %Y")
    assert f"Today's date is {today}." in instruction
    assert instruction.rstrip().endswith(f"Today's date is {today}.")
