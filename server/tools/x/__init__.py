"""Generic X (Twitter) tool, backed by X's hosted MCP server.

Like the browser tool, this is brain-agnostic: it's layered onto every brain
centrally in ``bot.py``. It connects to X's hosted Streamable HTTP MCP server
(``https://api.x.com/mcp``) and exposes its toolset (search posts, look up
users, trends, news, ...) so the agent can pull live information from X.

Authentication uses the "app-only Bearer" route (see ``config.py``): a static
Bearer token in the ``Authorization`` header, no local bridge or OAuth login.
That means read-only, no-user-context access -- fine for search/lookup, but not
for writes like bookmarking or posting.

The X tool is opt-in: when ``X_APP_BEARER_TOKEN`` is unset it's skipped without
opening a connection. If the connection fails (bad token, network, X down), the
tools are skipped and the session runs without them.
"""

from __future__ import annotations

from loguru import logger
from mcp.client.session_group import StreamableHttpParameters
from pipecat.services.llm_service import LLMService
from pipecat.services.mcp_service import MCPClient

from brains.base import ToolBundle

from .config import (
    X_APP_BEARER_TOKEN_ENV,
    X_MCP_URL,
    build_x_mcp_headers,
    get_x_bearer_token,
)

# Prompt snippet describing the X capability. Attached to the bundle so the
# system prompt mentions X only when the tools are actually wired in (e.g. it's
# omitted when the connection fails or the token is unset).
X_INSTRUCTION = """
You can access X (formerly Twitter) to search posts, look up users and their recent posts, and check trends and news. Reach for it when the user asks what's happening on X, what someone posted, or for real-time reactions and trending topics. Access is read-only: you can search and read, but you cannot post, reply, like, or bookmark. When you relay a post, attribute it to its author and summarize it in plain spoken language rather than reading raw handles, links, or hashtags aloud.
"""


async def setup_x_tools() -> ToolBundle:
    """Connect to X's hosted MCP server and bundle its tools.

    The X tool is opt-in: if ``X_APP_BEARER_TOKEN`` is unset, returns an empty
    ``ToolBundle`` without opening a connection.

    Otherwise discovers the tool schemas up front (so the bundle can carry them
    and its prompt snippet before the LLM exists), and defers registering the
    handlers onto the LLM to the bundle's ``registrations`` -- run once the LLM
    is built. The bundle's cleanup closes the connection. On any startup failure
    (bad token, network, server error), logs a warning and returns an empty
    ``ToolBundle`` so the session continues without X tools.
    """
    token = get_x_bearer_token()
    if not token:
        logger.info(
            f"X tools skipped: {X_APP_BEARER_TOKEN_ENV} not set. Set it to your X app's "
            "App-only Bearer token (X developer portal) to enable."
        )
        return ToolBundle()

    mcp = MCPClient(
        server_params=StreamableHttpParameters(
            url=X_MCP_URL,
            headers=build_x_mcp_headers(token),
        ),
    )
    try:
        await mcp.start()
        tools = await mcp.get_tools_schema()
    except Exception:
        logger.exception(
            f"X tools unavailable: failed to connect to X's MCP server ({X_MCP_URL}). "
            f"Is {X_APP_BEARER_TOKEN_ENV} a valid App-only Bearer token, and is the "
            "network reachable? Continuing without X tools."
        )
        await mcp.close()
        return ToolBundle()

    async def register(llm: LLMService) -> None:
        await mcp.register_tools_schema(tools, llm)

    logger.info(f"X tools ready: {len(tools.standard_tools)} X MCP tools")
    return ToolBundle(
        standard_tools=list(tools.standard_tools),
        instructions=[X_INSTRUCTION],
        registrations=[register],
        cleanups=[mcp.close],
    )


__all__ = ["X_INSTRUCTION", "setup_x_tools"]
