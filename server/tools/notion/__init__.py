"""Generic Notion tool, backed by the official Notion MCP server.

Like the browser tool, this is brain-agnostic: it's layered onto every brain
centrally in ``bot.py``. It launches ``@notionhq/notion-mcp-server`` locally
over stdio and exposes its full Notion API toolset (search, pages, databases,
comments, ...) so the agent can read and update connected workspace content.

Authentication uses a Notion internal integration token (``NOTION_ACCESS_TOKEN``
in ``.env``). Pages and databases must be connected to that integration in
Notion's settings before the agent can access them.

The Notion tool is opt-in via ``brains.yaml``; enabling it without
``NOTION_ACCESS_TOKEN`` is a startup error. If the MCP server fails to start
(missing Node.js/npx, bad token), the tools are skipped and the session continues.
"""

from __future__ import annotations

from loguru import logger
from mcp import StdioServerParameters
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_service import MCPClient

from brains.base import ToolBundle

from ..mcp_bundle import mcp_tool_bundle, prefix_tool_descriptions
from .config import (
    NOTION_ACCESS_TOKEN_ENV,
    NOTION_MCP_ARGS,
    NOTION_MCP_COMMAND,
    NOTION_MCP_TOOLS_FILTER,
    build_notion_mcp_env,
    get_notion_access_token,
)

# Prompt snippet describing the Notion capability. Attached to the bundle so the
# system prompt mentions Notion only when the tools are actually wired in.
NOTION_INSTRUCTION = """
You can access the user's Notion workspace to search for pages and databases, read page content, query database rows, and create or update pages when asked. Your Notion tools are the ones whose descriptions begin with the [Notion] tag; use them only to act on the user's Notion workspace, even though some are named generically (e.g. a "Search by title" tool). Use search before guessing page or database IDs. Prefer reading page content via markdown retrieval rather than walking individual blocks. Summarize Notion results in plain spoken language: do not read raw UUIDs, markdown syntax, or long lists aloud. When the user asks to change or delete content and their intent is ambiguous, confirm before making destructive edits.
"""


async def _connect_notion_mcp() -> tuple[MCPClient, ToolsSchema]:
    """Start the Notion MCP server and return the client plus tool schemas."""
    token = get_notion_access_token()
    if not token:
        raise RuntimeError(
            f"{NOTION_ACCESS_TOKEN_ENV} is required for Notion tools. Set it in .env, "
            "or remove 'notion' from the 'tools' list in brains.yaml."
        )

    tools_filter = NOTION_MCP_TOOLS_FILTER
    mcp = MCPClient(
        server_params=StdioServerParameters(
            command=NOTION_MCP_COMMAND,
            args=list(NOTION_MCP_ARGS),
            env=build_notion_mcp_env(token),
        ),
        tools_filter=tools_filter,
    )
    await mcp.start()
    tools = prefix_tool_descriptions(await mcp.get_tools_schema(), "Notion")
    return mcp, tools


async def setup_notion_tools() -> ToolBundle:
    """Start the Notion MCP server and bundle its tools.

    Called only when ``notion`` is listed in ``brains.yaml``; ``setup_generic_tools``
    already fail-fast if ``NOTION_ACCESS_TOKEN`` is missing, so this raises if the
    token is absent when invoked directly (e.g. tests).

    Discovers tool schemas up front (so the bundle can carry them and its prompt
    snippet before the LLM exists), and defers registering handlers onto the LLM
    via the bundle's ``registrations``. The bundle's cleanup closes the connection
    (stopping the server process). On MCP startup failure, logs a warning and
    returns an empty ``ToolBundle`` so the session continues without Notion tools.
    """
    token = get_notion_access_token()
    if not token:
        raise RuntimeError(
            f"{NOTION_ACCESS_TOKEN_ENV} is required for Notion tools. Set it in .env, "
            "or remove 'notion' from the 'tools' list in brains.yaml."
        )

    try:
        mcp, tools = await _connect_notion_mcp()
    except Exception:
        logger.exception(
            "Notion tools unavailable: failed to start Notion MCP server "
            f"({NOTION_MCP_COMMAND} {' '.join(NOTION_MCP_ARGS)}). Is Node.js/npx "
            f"installed, and is {NOTION_ACCESS_TOKEN_ENV} a valid integration token? "
            "Continuing without Notion tools."
        )
        return ToolBundle()

    tools_filter = NOTION_MCP_TOOLS_FILTER
    filter_note = f" (filtered to {len(tools.standard_tools)} tools)" if tools_filter else ""
    return mcp_tool_bundle(
        mcp,
        tools,
        [NOTION_INSTRUCTION],
        close_on_cleanup=True,
        ready_log=f"Notion tools ready: {len(tools.standard_tools)} Notion MCP tools{filter_note}",
    )


__all__ = ["NOTION_INSTRUCTION", "_connect_notion_mcp", "setup_notion_tools"]
