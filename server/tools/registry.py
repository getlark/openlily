"""Central index of tool specs exported by implementation modules."""

from __future__ import annotations

from .browser import SPEC as BROWSER_SPEC
from .contracts import ToolActivation, ToolId, ToolName, ToolSpec
from .email import SPEC as EMAIL_SPEC
from .notion import SPEC as NOTION_SPEC
from .session import SPEC as SESSION_SPEC
from .web import EXA_WEB_SPEC, HOSTED_WEB_SPEC
from .x import SPEC as X_SPEC

_TOOL_SPECS = (
    SESSION_SPEC,
    BROWSER_SPEC,
    EMAIL_SPEC,
    NOTION_SPEC,
    X_SPEC,
    HOSTED_WEB_SPEC,
    EXA_WEB_SPEC,
)

TOOL_REGISTRY: dict[ToolId, ToolSpec] = {spec.id: spec for spec in _TOOL_SPECS}

if set(TOOL_REGISTRY) != set(ToolId) or len(TOOL_REGISTRY) != len(_TOOL_SPECS):
    raise RuntimeError("Tool registry must contain exactly one spec for every ToolId")


def get_tool_spec(tool_id: ToolId) -> ToolSpec:
    return TOOL_REGISTRY[tool_id]


def all_tool_specs() -> tuple[ToolSpec, ...]:
    return _TOOL_SPECS


def get_configurable_tool(name: ToolName) -> ToolSpec:
    for spec in all_tool_specs():
        if spec.configurable_name is name:
            return spec
    raise KeyError(name)


def always_on_tools() -> tuple[ToolSpec, ...]:
    return tuple(spec for spec in all_tool_specs() if spec.activation is ToolActivation.ALWAYS)


def is_registered(tool_id: ToolId) -> bool:
    return tool_id in TOOL_REGISTRY


__all__ = [
    "TOOL_REGISTRY",
    "all_tool_specs",
    "always_on_tools",
    "get_configurable_tool",
    "get_tool_spec",
    "is_registered",
]
