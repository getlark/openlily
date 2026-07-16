"""Deploy openlily to Pipecat Cloud (or run it with the dev runner).

Pipecat Cloud (and the local dev runner) discover a ``bot(runner_args)`` coroutine
in this module and call it once per session. This example wires openlily's
high-level factory into that entry point: pick a brain, optionally enable tools,
and toggle the flourishes (readiness chime, "working" cue) on or off.

Setup:

    pip install openlily
    # set the API keys your brain/tools need, e.g. OPENAI_API_KEY, CARTESIA_API_KEY

Run locally against the browser UI:

    python examples/pipecat_cloud_bot.py --transport webrtc

Deploy: package this file per the Pipecat Cloud docs; no openlily-specific steps.
"""

from __future__ import annotations

from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

import openlily


async def bot(runner_args: RunnerArguments) -> None:
    """Pipecat entry point: build and run one openlily session."""
    config = openlily.AgentConfig(
        # A built-in brain by name, a ``BrainName``, or your own ``BrainSpec``.
        brain="cartesia_openai",
        # Optional configurable tools (each needs its credentials in the env).
        enabled_tools=["email"],
        # Flourishes are opt-out: drop the soft "thinking" cue with
        # ``working_sound=False`` or the startup chime with ``readiness_chime=False``.
        working_sound=True,
        readiness_chime=True,
    )

    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True, audio_out_enabled=True
            ),
            "daily": lambda: TransportParams(
                audio_in_enabled=True, audio_out_enabled=True
            ),
        },
    )

    # Warm slow first-run work (model downloads, MCP servers) once per process.
    await openlily.warmup(config)

    agent = await openlily.create_agent(transport, config)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(agent.worker)
    try:
        await runner.run()
    finally:
        await openlily.close_tool_bundle(agent.tool_bundle)
        await openlily.shutdown_tools()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
