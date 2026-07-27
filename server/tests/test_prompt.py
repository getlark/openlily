"""Tests for the system prompt builder (``openlily/prompt.py``)."""

from __future__ import annotations

from datetime import datetime

from openlily.prompt import build_system_instruction
from openlily.tools.bundle import ToolGuidance, render_tool_guidance


def test_no_tool_guidance_leaves_no_block_or_blank_line() -> None:
    instruction = build_system_instruction()
    # The base prompt is always present.
    assert "<OutputRules>" in instruction
    # No guidance block (or leftover blank line) was injected inside <Tools>.
    assert "<ToolGuidance>" not in instruction
    assert "technical details.\n</Tools>" in instruction


def test_tool_guidance_block_is_embedded() -> None:
    block = render_tool_guidance(
        [
            ToolGuidance(tool_names=("web_search", "web_fetch"), text="You can search the web."),
            ToolGuidance(tool_names=("send_email_to_user",), text="You can send email."),
        ]
    )
    instruction = build_system_instruction(block)
    assert '<Tool name="web_search, web_fetch">You can search the web.</Tool>' in instruction
    assert '<Tool name="send_email_to_user">You can send email.</Tool>' in instruction
    # The block sits inside the <Tools> section.
    assert instruction.index("<Tools>") < instruction.index("<ToolGuidance>")
    assert instruction.index("</ToolGuidance>") < instruction.index("</Tools>")


def test_render_tool_guidance_empty_returns_empty_string() -> None:
    assert render_tool_guidance([]) == ""


def test_render_tool_guidance_strips_snippet_whitespace() -> None:
    block = render_tool_guidance([ToolGuidance(tool_names=("a",), text="\nDo things.\n")])
    assert block == '<ToolGuidance>\n<Tool name="a">Do things.</Tool>\n</ToolGuidance>'


def test_appends_todays_date() -> None:
    instruction = build_system_instruction()
    today = datetime.now().strftime("%A, %B %-d, %Y")
    assert f"Today's date is {today}." in instruction
    assert instruction.rstrip().endswith(f"Today's date is {today}.")
