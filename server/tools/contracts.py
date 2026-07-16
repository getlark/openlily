"""Lightweight shared contracts for tool definitions and selection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .bundle import ToolBundle


class ToolName(StrEnum):
    """Names users may enable in ``brains.yaml``."""

    BROWSER = "browser"
    EMAIL = "email"
    NOTION = "notion"
    X = "x"


class ToolId(StrEnum):
    """Canonical IDs for all concrete tool implementations."""

    SESSION = "session"
    BROWSER = "browser"
    EMAIL = "email"
    NOTION = "notion"
    X = "x"
    WEB_HOSTED = "web_hosted"
    WEB_EXA = "web_exa"


class ToolActivation(StrEnum):
    ALWAYS = "always"
    CONFIGURED = "configured"
    BRAIN = "brain"


class ToolBackend(StrEnum):
    LOCAL = "local"
    HOSTED = "hosted"
    MCP = "mcp"


SetupFactory = Callable[[], Awaitable[ToolBundle]]
ConfigCheck = Callable[[], bool]
MCPConnector = Callable[[], Awaitable[tuple[Any, Any]]]
InstructionsFactory = Callable[[], list[str]]


@dataclass(frozen=True)
class ToolSpec:
    id: ToolId
    activation: ToolActivation
    backend: ToolBackend
    setup: SetupFactory
    configurable_name: ToolName | None = None
    is_configured: ConfigCheck | None = None
    requirement: str | None = None
    mcp_connect: MCPConnector | None = None
    mcp_instructions: InstructionsFactory | None = None
    warmup_failure_hint: str | None = None

    def __post_init__(self) -> None:
        if self.activation is ToolActivation.CONFIGURED:
            if not self.configurable_name or not self.is_configured or not self.requirement:
                raise ValueError(f"Configured tool {self.id!r} is missing configuration metadata")
        elif self.configurable_name is not None:
            raise ValueError(f"Non-configurable tool {self.id!r} has a configurable name")

        mcp_fields = (
            self.mcp_connect,
            self.mcp_instructions,
            self.warmup_failure_hint,
        )
        if self.backend is ToolBackend.MCP and not all(mcp_fields):
            raise ValueError(f"MCP tool {self.id!r} is missing warmup metadata")
        if self.backend is not ToolBackend.MCP and any(mcp_fields):
            raise ValueError(f"Non-MCP tool {self.id!r} has MCP metadata")


__all__ = [
    "ToolActivation",
    "ToolBackend",
    "ToolId",
    "ToolName",
    "ToolSpec",
]
