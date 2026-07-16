"""Local command-line entry points for openlily (the terminal voice bot).

This is the thin app layer on top of the library. It is the *only* place that
reads ``.env`` and ``brains.yaml``: it turns them into an :class:`AgentConfig` and
hands that to the factory in :mod:`openlily.agent`. The three run modes match the
original bot:

    openlily                        # --mode local-with-wake-word (default)
    openlily --mode local           # talk via your mic/speakers
    openlily --mode webrtc          # browser debug UI at localhost:7860

(the same modes are available via ``uv run bot.py`` in this directory.)
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from openlily.agent import create_agent, resolve_brain, warmup
from openlily.brains import get_brain
from openlily.brains.config import get_enabled_tools
from openlily.config import DEFAULT_IDLE_TIMEOUT_SECS, AgentConfig
from openlily.tools.bundle import close_tool_bundle
from openlily.tools.runtime import shutdown_tools


def _idle_timeout_secs() -> float:
    """Resolve the idle timeout (seconds) from the environment, else the default."""
    raw = os.getenv("IDLE_TIMEOUT_SECS")
    if raw is None:
        return DEFAULT_IDLE_TIMEOUT_SECS
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            f"Invalid IDLE_TIMEOUT_SECS={raw!r}; using default {DEFAULT_IDLE_TIMEOUT_SECS}s"
        )
        return DEFAULT_IDLE_TIMEOUT_SECS


def build_agent_config() -> AgentConfig:
    """Build the ``AgentConfig`` for a local run from ``brains.yaml`` + env.

    Resolves the brain (``default_brain`` in brains.yaml, else the built-in
    default), the optional tools (brains.yaml ``tools:``), and the idle timeout
    (``IDLE_TIMEOUT_SECS``). Everything else uses the library defaults, which are
    the stock local bot's behavior (working-sound cue on, readiness chime on).
    """
    return AgentConfig(
        brain=get_brain(),
        enabled_tools=get_enabled_tools(),
        idle_timeout_secs=_idle_timeout_secs(),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run the voice bot for a dev-runner session (browser WebRTC UI)."""
    config = build_agent_config()
    brain = resolve_brain(config)
    logger.info(f"Starting bot (brain={brain.name}, realtime={brain.is_realtime})")

    await warmup(config)

    agent = await create_agent(transport, config)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await agent.worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(agent.worker)
    # Run the tool cleanups, if any, in the same task the tools were created in.
    try:
        await runner.run()
    finally:
        await close_tool_bundle(agent.tool_bundle)
        await shutdown_tools()


async def bot(runner_args: RunnerArguments):
    """Dev-runner entry point (browser WebRTC UI)."""
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


async def run_session(*, handle_sigint: bool) -> None:
    """Build and run one local voice session over the mic/speakers.

    Returns when the session ends (idle timeout, client disconnect, or Ctrl+C).
    The bot waits for the user to speak first (no kickoff greeting). Factored out
    so the wake-gated loop can run many sessions in one process.
    """
    from openlily.local import build_local_transport, close_local_transport

    config = build_agent_config()
    brain = resolve_brain(config)
    logger.info(f"Starting session (brain={brain.name}, realtime={brain.is_realtime})")

    transport = build_local_transport()
    # Building the pipeline loads the on-device VAD model and constructs the
    # brain's services - a brief silent stretch on a fresh process. Narrate it so
    # the terminal doesn't look stuck before the readiness chime.
    logger.info("Preparing models and audio...")
    agent = await create_agent(transport, config)

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(agent.worker)
    try:
        await runner.run()
    finally:
        await close_tool_bundle(agent.tool_bundle)
        # Release this session's PyAudio instance so a long-lived process
        # (wake-gated mode) doesn't accumulate PortAudio handles.
        close_local_transport(transport)


async def run_local() -> None:
    """Run the bot over local mic/speakers - the terminal voice CLI."""
    try:
        await warmup(build_agent_config())
        logger.info("Local voice bot ready - start talking. Press Ctrl+C to stop.")
        await run_session(handle_sigint=True)
    finally:
        await shutdown_tools()


def _wake_models() -> list[str]:
    """Resolve the openWakeWord model(s) from $WAKE_MODELS, else the default."""
    raw = os.getenv("WAKE_MODELS")
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            return models
    return ["alexa"]


def run_wake_gated() -> None:
    """Wake-word-gated local mode: stay warm, run a session on each wake word.

    Keeps wake detection and the voice bot in one long-lived process, so the
    expensive imports and model loads are paid once at startup rather than per
    wake word. While idle, an always-on wake-word listener owns the mic; when the
    wake word fires it releases the mic, a local voice session takes over until it
    idles out, then listening resumes. Press Ctrl+C to stop.

    This function is synchronous: the blocking wake-word listen loop runs in the
    main thread (so Ctrl+C interrupts it cleanly). All async work - warmup, every
    session, and shutdown - runs on a single long-lived event loop via
    ``run_until_complete`` (not ``asyncio.run``), so warmed MCP pool connections,
    whose anyio task groups are bound to the loop that created them, stay valid
    and reusable across sessions.
    """
    from openlily.local.wakeword import PyAudioSource, WakeWordEngine, WakeWordListener

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Warm the brain's models up front (before we start listening) so the very
    # first session doesn't stall, and so setup progress is visible before the
    # user is told to say the wake word. Fail-fast: a known-broken setup aborts
    # here with a clear message rather than looping forever in the wake loop.
    loop.run_until_complete(warmup(build_agent_config()))

    models = _wake_models()
    # Constructing the engine imports openwakeword/onnxruntime and loads the
    # models (downloaded once on the first run) - a few silent seconds.
    logger.info("Loading on-device wake-word detection (importing models)...")
    listener = WakeWordListener(WakeWordEngine(models=models), PyAudioSource())

    logger.info(
        f"Wake-gated mode ready - say the wake word ({', '.join(models)}) to start a "
        f"session. Press Ctrl+C to stop."
    )
    try:
        while True:
            label = listener.wait_for_wake()  # blocks (main thread); owns the mic
            logger.info(f"Wake word '{label}' detected; starting session")
            try:
                loop.run_until_complete(run_session(handle_sigint=False))
            except Exception:
                logger.exception("Session error; returning to wake-word listening")
            logger.info("Session ended; resuming wake-word listening")
    except KeyboardInterrupt:
        logger.info("Stopping wake-gated mode")
    finally:
        loop.run_until_complete(shutdown_tools())
        loop.close()


def main() -> None:
    """CLI entry point: parse ``--mode`` and dispatch to a run mode."""
    import argparse

    load_dotenv(override=True)

    # --mode picks how you talk to the bot; everything else is forwarded to
    # Pipecat's dev runner (e.g. --host/--port) in webrtc mode.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=["local-with-wake-word", "local", "webrtc"],
        default="local-with-wake-word",
    )
    args, runner_args = parser.parse_known_args()

    if args.mode == "local-with-wake-word":
        run_wake_gated()
    elif args.mode == "local":
        asyncio.run(run_local())
    else:
        # Hand off to Pipecat's dev runner, which discovers a ``bot`` function in
        # the ``__main__`` module. Inject ours so this works whether launched via
        # ``uv run bot.py`` or the ``openlily`` console script.
        sys.modules["__main__"].bot = bot  # type: ignore[attr-defined]
        sys.argv = [sys.argv[0], *runner_args]
        from pipecat.runner.run import main as runner_main

        runner_main()


__all__ = [
    "bot",
    "build_agent_config",
    "main",
    "run_bot",
    "run_local",
    "run_session",
    "run_wake_gated",
]
