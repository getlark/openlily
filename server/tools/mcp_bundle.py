"""Shared helpers for building MCP-backed ``ToolBundle``s."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import LLMService
from pipecat.services.mcp_service import MCPClient

from brains.base import ToolBundle


def prefix_tool_descriptions(tools_schema: ToolsSchema, label: str) -> ToolsSchema:
    """Return a copy of ``tools_schema`` with each tool's description tagged ``[label]``.

    MCP servers name their tools generically (e.g. the Notion server's
    ``API-post-search`` is described only as "Search by title"), which gives the
    LLM no signal about *which* service a tool belongs to. Tagging every
    description with its owning service (``[Notion]``, ``[X]``, ``[Browser]``, ...)
    disambiguates otherwise look-alike tools when several MCP servers are enabled.

    Only descriptions change; tool *names* are left untouched on purpose. Pipecat's
    ``MCPClient`` uses the name both to advertise a tool and to call back to the
    server (see ``MCPClient._tool_wrapper``), so renaming would break invocation.

    Idempotent: a description already carrying the ``[label]`` tag is left as-is,
    so re-tagging a shared/pooled schema won't double-prefix it.
    """
    tag = f"[{label}]"
    prefixed: list[FunctionSchema] = []
    for tool in tools_schema.standard_tools:
        description = tool.description or ""
        if description.startswith(tag):
            prefixed.append(tool)
            continue
        prefixed.append(
            FunctionSchema(
                name=tool.name,
                description=f"{tag} {description}".strip(),
                properties=tool.properties,
                required=tool.required,
                handler=tool.handler,
            )
        )
    return ToolsSchema(standard_tools=prefixed, custom_tools=tools_schema.custom_tools)


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


__all__ = ["mcp_tool_bundle", "prefix_tool_descriptions"]
