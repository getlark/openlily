"""Shared helpers for building MCP-backed ``ToolBundle``s."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import LLMService
from pipecat.services.mcp_service import MCPClient

from brains.base import ToolBundle


def mcp_tool_bundle(
    mcp: MCPClient,
    tools_schema: ToolsSchema,
    instructions: list[str],
    *,
    close_on_cleanup: bool,
    ready_log: str | None = None,
) -> ToolBundle:
    """Build a ``ToolBundle`` from a connected MCP client and its schema."""

    async def register(llm: LLMService) -> None:
        await mcp.register_tools_schema(tools_schema, llm)

    if close_on_cleanup:
        cleanups: list[Callable[[], Awaitable[None]]] = [mcp.close]
    else:

        async def _noop_cleanup() -> None:
            return None

        cleanups = [_noop_cleanup]

    if ready_log:
        logger.info(ready_log)

    return ToolBundle(
        standard_tools=list(tools_schema.standard_tools),
        instructions=instructions,
        registrations=[register],
        cleanups=cleanups,
    )


__all__ = ["mcp_tool_bundle"]
