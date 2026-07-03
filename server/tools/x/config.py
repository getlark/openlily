"""How the X tool connects to X's hosted MCP server.

Unlike the browser tool (a local ``npx`` server over stdio), X hosts a
Streamable HTTP MCP server at ``https://api.x.com/mcp``. We use the "app-only
Bearer" route from https://docs.x.com/tools/mcp: point the client straight at
the URL with a static App-only Bearer token in the ``Authorization`` header --
no local ``xurl`` bridge and no browser OAuth login. That keeps setup to a
single env var, at the cost of read-only endpoints with no user context (it
can't act as you -- e.g. no bookmarking or posting).

The token comes from ``X_APP_BEARER_TOKEN`` (your X app's App-only Bearer token
from the developer portal). When it's unset, the X tool is skipped entirely
(see ``setup_x_tools``).
"""

from __future__ import annotations

import os

# X's hosted Streamable HTTP MCP endpoint (protocol 2025-06-18, serverInfo xmcp).
X_MCP_URL = "https://api.x.com/mcp"

# App-only Bearer token for X's API, injected as the Authorization header.
# Required to enable the X tool: when unset, ``setup_x_tools`` skips it and never
# opens a connection.
X_APP_BEARER_TOKEN_ENV = "X_APP_BEARER_TOKEN"


def get_x_bearer_token() -> str | None:
    """The configured App-only Bearer token, or ``None`` when unset/blank.

    Read fresh from the environment (not captured at import): this module is
    imported before ``load_dotenv()`` runs, so an env var from ``.env`` wouldn't
    be visible yet if captured at import time.
    """
    token = os.getenv(X_APP_BEARER_TOKEN_ENV)
    return token.strip() if token and token.strip() else None


def is_configured() -> bool:
    """Whether the X tool's credential is present (``X_APP_BEARER_TOKEN`` set).

    Used to fail fast when the tool is enabled in ``brains.yaml`` but its token
    is missing, before the (slower) MCP connection is attempted.
    """
    return get_x_bearer_token() is not None


def build_x_mcp_headers(token: str) -> dict[str, str]:
    """Build the auth headers for the app-only Bearer connection to X's MCP."""
    return {"Authorization": f"Bearer {token}"}
