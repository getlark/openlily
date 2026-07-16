"""Invariants for the central tool registry and brain tool references."""

from __future__ import annotations

from openlily.brains import cartesia_meta, cartesia_openai, openai_realtime, openai_standard
from openlily.tools.contracts import ToolActivation, ToolBackend, ToolName
from openlily.tools.registry import all_tool_specs, is_registered


def test_every_configurable_name_has_exactly_one_spec() -> None:
    configurable = [
        spec.configurable_name
        for spec in all_tool_specs()
        if spec.activation is ToolActivation.CONFIGURED
    ]
    assert set(configurable) == set(ToolName)
    assert len(configurable) == len(set(configurable))


def test_registry_backend_metadata_is_consistent() -> None:
    for spec in all_tool_specs():
        if spec.backend is ToolBackend.MCP:
            assert spec.mcp_connect is not None
            assert spec.mcp_instructions is not None
            assert spec.warmup_failure_hint
        else:
            assert spec.mcp_connect is None
            assert spec.mcp_instructions is None
            assert spec.warmup_failure_hint is None


def test_cloud_brains_reference_registered_tools() -> None:
    specs = (
        openai_standard.SPEC,
        cartesia_openai.SPEC,
        cartesia_meta.SPEC,
        openai_realtime.SPEC,
    )
    for brain in specs:
        assert len(brain.tools) == len(set(brain.tools))
        assert all(is_registered(tool_id) for tool_id in brain.tools)
