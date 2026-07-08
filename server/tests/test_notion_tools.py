"""Tests for Notion tool config and setup."""

from __future__ import annotations

import pytest
from pipecat.adapters.schemas.tools_schema import ToolsSchema

import tools as tools_pkg
from brains.base import ToolBundle, ToolName
from tools import setup_generic_tools
from tools.notion import NOTION_INSTRUCTION, setup_notion_tools
from tools.notion.config import (
    NOTION_ACCESS_TOKEN_ENV,
    build_notion_mcp_env,
    is_configured,
)


def test_notion_is_configured_when_token_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NOTION_ACCESS_TOKEN_ENV, "ntn_test")
    assert is_configured() is True


def test_notion_is_not_configured_when_token_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NOTION_ACCESS_TOKEN_ENV, "   ")
    assert is_configured() is False


def test_build_notion_mcp_env_maps_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    env = build_notion_mcp_env("ntn_secret")
    assert env["NOTION_TOKEN"] == "ntn_secret"
    assert NOTION_ACCESS_TOKEN_ENV not in env


async def test_notion_unconfigured_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NOTION_ACCESS_TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match=NOTION_ACCESS_TOKEN_ENV):
        await setup_notion_tools()


async def test_notion_configured_wires_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    monkeypatch.setenv(NOTION_ACCESS_TOKEN_ENV, "ntn_test")

    class FakeMCP:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def start(self) -> None:
            return None

        async def get_tools_schema(self) -> ToolsSchema:
            return ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name="post-search",
                        description="search",
                        properties={},
                        required=[],
                    )
                ]
            )

        async def close(self) -> None:
            return None

        async def register_tools_schema(self, tools: ToolsSchema, llm: object) -> None:
            return None

    monkeypatch.setattr("tools.notion.MCPClient", FakeMCP)

    bundle = await setup_notion_tools()
    assert len(bundle.standard_tools) == 1
    assert bundle.instructions == [NOTION_INSTRUCTION]
    assert bundle.registrations
    assert bundle.cleanups


async def test_notion_passes_tools_filter_to_mcp_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.notion as notion_pkg

    monkeypatch.setenv(NOTION_ACCESS_TOKEN_ENV, "ntn_test")
    monkeypatch.setattr(
        notion_pkg,
        "NOTION_MCP_TOOLS_FILTER",
        ["post-search", "retrieve-a-page"],
    )

    captured: dict[str, object] = {}

    class FakeMCP:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def start(self) -> None:
            return None

        async def get_tools_schema(self) -> ToolsSchema:
            return ToolsSchema(standard_tools=[])

        async def close(self) -> None:
            return None

    monkeypatch.setattr("tools.notion.MCPClient", FakeMCP)

    await setup_notion_tools()
    assert captured["tools_filter"] == ["post-search", "retrieve-a-page"]


async def test_generic_tools_notion_enabled_but_unconfigured_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_setup() -> ToolBundle:
        raise AssertionError("setup should not be called for an unconfigured tool")

    monkeypatch.setattr(tools_pkg, "get_enabled_tools", lambda: [ToolName.NOTION])
    monkeypatch.setitem(
        tools_pkg._OPTIONAL_TOOLS,
        ToolName.NOTION,
        tools_pkg._OptionalTool(
            setup=fail_setup,
            is_configured=lambda: False,
            requirement="NOTION_ACCESS_TOKEN",
        ),
    )

    with pytest.raises(RuntimeError) as exc:
        await setup_generic_tools()
    assert "NOTION_ACCESS_TOKEN" in str(exc.value)
