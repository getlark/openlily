"""How the browser tool launches the Playwright MCP server.

The server runs locally over stdio via ``npx``. ``-y`` so npx doesn't prompt to
install the package on first run.
"""

from __future__ import annotations

import os

BROWSER_MCP_COMMAND = "npx"
BROWSER_MCP_BASE_ARGS = ["-y", "@playwright/mcp@latest"]

# Optional: a directory for a persistent browser profile. When set, the
# Playwright MCP server reuses this profile across sessions (``--user-data-dir``),
# so cookies, logins, and browsing history accumulate over time -- like a normal
# browser. When unset, the server falls back to its default non-persistent
# behavior (a throwaway profile per run).
#
# Caveat: a persistent profile can only be used by one browser instance at a
# time, so overlapping sessions sharing the same directory will conflict. Only
# safe when sessions don't run concurrently.
BROWSER_USER_DATA_DIR_ENV = "BROWSER_USER_DATA_DIR"


def build_browser_mcp_args() -> list[str]:
    """Build the Playwright MCP launch args, read fresh from the environment.

    Read at call time (not import time) because this module is imported before
    ``load_dotenv()`` runs, so an env var from ``.env`` wouldn't be visible yet
    if captured at import.
    """
    args = list(BROWSER_MCP_BASE_ARGS)
    user_data_dir = os.getenv(BROWSER_USER_DATA_DIR_ENV)
    if user_data_dir:
        args += ["--user-data-dir", user_data_dir]
    return args
