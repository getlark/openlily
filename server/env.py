"""Small environment helpers shared across the app.

Kept separate from any one feature (brains, tools, transport) so anything that
reads configuration from the environment can reuse the same fail-fast behavior.
"""

from __future__ import annotations

import os


def require_env(name: str, msg: str | None = None) -> str:
    """Return a required env var's value, or raise if it's missing/blank.

    Critical config (API keys, etc.) should fail fast with a clear message
    rather than letting a service surface an opaque auth error later (or
    silently running with an empty value).

    Pass ``msg`` to add callsite-specific context to the error, e.g. which
    feature or brain needs the variable.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        detail = f" {msg}" if msg else ""
        raise RuntimeError(f"{name} is required but is not set in the environment.{detail}")
    return value
