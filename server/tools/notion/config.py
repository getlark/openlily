"""How the Notion tool connects to the official Notion MCP server.

The server runs locally over stdio via ``npx @notionhq/notion-mcp-server``.
openlily reads ``NOTION_ACCESS_TOKEN`` from the environment; the npm package
expects ``NOTION_TOKEN``, so ``build_notion_mcp_env`` maps between the two.

When ``NOTION_ACCESS_TOKEN`` is unset, the Notion tool is skipped entirely
(see ``setup_notion_tools``). To limit which MCP tools are advertised, set
``NOTION_MCP_TOOLS_FILTER`` in this module (Pipecat ``tools_filter``); ``None``
exposes all tools from the server.
"""

from __future__ import annotations

import os

NOTION_ACCESS_TOKEN_ENV = "NOTION_ACCESS_TOKEN"

NOTION_MCP_COMMAND = "npx"
NOTION_MCP_ARGS = ["-y", "@notionhq/notion-mcp-server@2.4.1"]

# Optional allowlist of MCP tool names passed to Pipecat's ``tools_filter``.
# Set to a list (e.g. ``["post-search", "retrieve-a-page"]``) to restrict; ``None`` = all.
NOTION_MCP_TOOLS_FILTER: list[str] | None = None

# Env var name the @notionhq/notion-mcp-server npm package reads for auth.
_NOTION_MCP_TOKEN_ENV = "NOTION_TOKEN"


def get_notion_access_token() -> str | None:
    """The configured Notion integration token, or ``None`` when unset/blank.

    Read fresh from the environment (not captured at import): this module is
    imported before ``load_dotenv()`` runs, so an env var from ``.env`` wouldn't
    be visible yet if captured at import time.
    """
    token = os.getenv(NOTION_ACCESS_TOKEN_ENV)
    return token.strip() if token and token.strip() else None


def is_configured() -> bool:
    """Whether the Notion tool's credential is present (``NOTION_ACCESS_TOKEN`` set).

    Used to fail fast when the tool is enabled in ``brains.yaml`` but its token
    is missing, before the (slower) MCP server is spawned.
    """
    return get_notion_access_token() is not None


def build_notion_mcp_env(token: str) -> dict[str, str]:
    """Build the child-process env for the Notion MCP server.

    openlily uses ``NOTION_ACCESS_TOKEN``; the npm MCP server expects
    ``NOTION_TOKEN``.
    """
    return {**os.environ, _NOTION_MCP_TOKEN_ENV: token}
