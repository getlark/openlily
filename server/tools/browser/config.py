"""How the browser tool launches the Playwright MCP server.

The server runs locally over stdio via ``npx``. ``-y`` so npx doesn't prompt to
install the package on first run.

Rather than launching (and later killing) its own browser, the MCP server
attaches to an already-running browser over the Chrome DevTools Protocol
(``--cdp-endpoint``). That browser is started and owned externally, so it
outlives the per-session MCP server: when a session ends and the server is torn
down, the browser stays open and the next session reconnects to it. The endpoint
comes from ``BROWSER_CDP_ENDPOINT`` (e.g. ``http://localhost:9222``); when it's
unset, browser tools are skipped entirely (see ``setup_browser_tools``).
"""

from __future__ import annotations

import os

BROWSER_MCP_COMMAND = "npx"
BROWSER_MCP_BASE_ARGS = ["-y", "@playwright/mcp@latest"]

# CDP endpoint of an already-running browser to attach to (e.g.
# ``http://localhost:9222``, exposed by launching Chrome with
# ``--remote-debugging-port=9222``). Required to enable browser tools: when it's
# unset, ``setup_browser_tools`` skips them and never spawns the MCP server.
BROWSER_CDP_ENDPOINT_ENV = "BROWSER_CDP_ENDPOINT"


def build_browser_mcp_args() -> list[str]:
    """Build the Playwright MCP launch args, read fresh from the environment.

    Attaches to the externally-managed browser at ``BROWSER_CDP_ENDPOINT`` via
    ``--cdp-endpoint``. The external browser owns its profile, so there's no
    ``--user-data-dir`` here.

    The caller guarantees ``BROWSER_CDP_ENDPOINT`` is set (browser tools are
    skipped otherwise -- see ``setup_browser_tools``). Read at call time (not
    import time) because this module is imported before ``load_dotenv()`` runs,
    so an env var from ``.env`` wouldn't be visible yet if captured at import.
    """
    endpoint = os.environ[BROWSER_CDP_ENDPOINT_ENV]
    return list(BROWSER_MCP_BASE_ARGS) + ["--cdp-endpoint", endpoint]
