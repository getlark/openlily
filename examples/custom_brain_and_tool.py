"""Extend openlily: register a custom brain and tool, then compose a pipeline.

Two things this shows:

1. ``register_brain`` / ``register_tool`` let you add a brain or tool *without*
   editing the openlily package -- build a ``BrainSpec`` / ``ToolSpec`` and
   register it at import time. Custom ids are plain strings, so you're not
   limited to the built-in ``BrainName`` / ``ToolId`` members.

2. You don't have to use the all-in-one factory. Every building block is
   importable, so you can drop just the pieces you want (here, the "working"
   cue and the idle keep-alive) into your own Pipecat pipeline.

This module is illustrative -- the brain's ``build`` is a stub -- so it isn't
meant to run as-is; it's a copy-paste starting point.
"""

from __future__ import annotations

import openlily
from openlily import (
    BrainServices,
    BrainSpec,
    IdleKeepaliveProcessor,
    ToolActivation,
    ToolBackend,
    ToolBundle,
    ToolSpec,
    WorkingSoundProcessor,
)


# --- A custom tool -----------------------------------------------------------
async def _setup_clock_tool() -> ToolBundle:
    async def what_time_is_it(params) -> None:  # a Pipecat direct function
        """Tell the user the current time."""
        from datetime import datetime

        await params.result_callback({"time": datetime.now().strftime("%H:%M")})

    return ToolBundle(
        standard_tools=[what_time_is_it],
        instructions=["You can tell the user the current time."],
    )


CLOCK_SPEC = ToolSpec(
    id="clock",  # a plain-string id; not a built-in ToolId member
    activation=ToolActivation.BRAIN,
    backend=ToolBackend.LOCAL,
    setup=_setup_clock_tool,
)
openlily.register_tool(CLOCK_SPEC)


# --- A custom brain that opts into the tool ----------------------------------
def _build_my_brain(system_instruction: str) -> BrainServices:
    # Construct your STT/LLM/TTS services here (see openlily.brains.* for real
    # examples). Returning ``BrainServices(llm=..., stt=..., tts=...)``.
    raise NotImplementedError("Plug in your own STT/LLM/TTS services.")


MY_BRAIN = BrainSpec(
    name="my_brain",  # a plain-string name; not a built-in BrainName member
    is_realtime=False,
    build=_build_my_brain,
    tools=("clock",),  # references the custom tool by id
)
openlily.register_brain(MY_BRAIN)


# --- Turnkey usage of the custom brain ---------------------------------------
async def run_turnkey(transport) -> None:
    config = openlily.AgentConfig(brain="my_brain")  # resolved from the registry
    agent = await openlily.create_agent(transport, config)
    # ... add agent.worker to a WorkerRunner and run it ...


# --- Compose-your-own pipeline (use only the pieces you want) ----------------
def build_custom_pipeline(transport, stt, llm, tts, user_aggregator, assistant_aggregator):
    """Assemble a Pipecat pipeline by hand, borrowing openlily's processors."""
    from pipecat.pipeline.pipeline import Pipeline

    return Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            IdleKeepaliveProcessor(interval_secs=5.0, max_busy_secs=300.0),
            WorkingSoundProcessor(initial_delay_secs=0.8),
            transport.output(),
            assistant_aggregator,
        ]
    )
