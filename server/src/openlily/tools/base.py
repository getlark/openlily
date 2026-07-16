"""Generic contract for the agent's tool providers.

A *tool provider* bundles a related set of agent tools (web search, email,
calendar, ...) behind a single object that knows how to create them. Each
provider defines its own Pipecat function tools -- their names, parameters, and
docstrings -- so a tool's schema matches exactly what that provider supports.
Different provider types share this shape, so the wiring that selects a provider
and feeds its tools to the agent stays uniform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable


class ToolProvider(ABC):
    """A backend that supplies a set of the agent's tools.

    Implementations own any clients/credentials they need and define their tools
    as Pipecat direct functions. Credentials are reported via ``is_configured``
    -- checked *before* construction so an unconfigured provider can be skipped
    with a warning rather than raising.
    """

    @classmethod
    @abstractmethod
    def is_configured(cls) -> bool:
        """Whether this provider's required credentials are present in the env."""

    @abstractmethod
    def create_tools(self) -> list[Callable[..., Awaitable[None]]]:
        """Return this provider's Pipecat function tools (direct functions)."""
