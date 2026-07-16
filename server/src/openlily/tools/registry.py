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

if len({spec.id for spec in _TOOL_SPECS}) != len(_TOOL_SPECS):
    raise RuntimeError("Tool registry must contain at most one spec per tool id")
# Every built-in ToolId must have a spec; consumers may add *extra* specs (with
# their own string ids) at runtime via ``register_tool``, so we no longer require
# the registry to match ``ToolId`` exactly -- only that the built-ins are present.
if not set(ToolId).issubset({spec.id for spec in _TOOL_SPECS}):
    raise RuntimeError("Tool registry must contain a spec for every built-in ToolId")

TOOL_REGISTRY: dict[ToolId | str, ToolSpec] = {spec.id: spec for spec in _TOOL_SPECS}


def register_tool(spec: ToolSpec) -> None:
    """Register a custom tool so brains/config can reference it by ``spec.id``.

    Lets a library consumer add a tool without editing this package. The id must
    be unique (re-registering an existing id is rejected). Custom tools use a
    plain-string id; a brain selects it by listing that id in ``BrainSpec.tools``
    or a consumer enables it by id.
    """
    if spec.id in TOOL_REGISTRY:
        raise ValueError(f"A tool with id {spec.id!r} is already registered")
    TOOL_REGISTRY[spec.id] = spec


def get_tool_spec(tool_id: ToolId | str) -> ToolSpec:
    return TOOL_REGISTRY[tool_id]


def all_tool_specs() -> tuple[ToolSpec, ...]:
    return tuple(TOOL_REGISTRY.values())


def get_configurable_tool(name: ToolName) -> ToolSpec:
    for spec in all_tool_specs():
        if spec.configurable_name is name:
            return spec
    raise KeyError(name)


def always_on_tools() -> tuple[ToolSpec, ...]:
    return tuple(spec for spec in all_tool_specs() if spec.activation is ToolActivation.ALWAYS)


def is_registered(tool_id: ToolId | str) -> bool:
    return tool_id in TOOL_REGISTRY


__all__ = [
    "TOOL_REGISTRY",
    "all_tool_specs",
    "always_on_tools",
    "get_configurable_tool",
    "get_tool_spec",
    "is_registered",
    "register_tool",
]
