"""openlily - Pipecat Voice Agent.

A simple voice bot: voice in -> LLM -> voice out. The "brain" is swappable
(``brains/``): an all-OpenAI cascade (STT + LLM + TTS) or OpenAI Realtime
(GPT speech-to-speech). Select it via ``default_brain`` in ``brains.yaml``
(copy ``brains.yaml.example``); else ``brains/config.py``'s ``DEFAULT_BRAIN``.

Run it:

    uv run bot.py                              # --mode local: talk via your mic/speakers (the voice CLI)
    uv run bot.py --mode webrtc                # talk via the browser UI at localhost:7860
    uv run bot.py --mode local-with-wake-word  # local, but say the wake word first; one warm process
"""

import sys

# Printed before the heavy imports below (Pipecat, ML runtimes, the brains),
# which take a few seconds warm and tens of seconds on the very first run while
# Python compiles them to bytecode. Without this the terminal looks frozen until
# the first real log line lands. stderr + flush so it shows immediately; guarded
# so importing this module (e.g. in tests) stays quiet.
if __name__ == "__main__":
    print(
        "Starting openlily - loading modules (this takes several seconds, "
        "and up to a minute on the first run while dependencies compile)...",
        file=sys.stderr,
        flush=True,
    )

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotSpeakingFrame,
    UserSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from brains import (
    BrainSpec,
    get_brain,
)
from idle_keepalive import BotBusyFrame, IdleKeepaliveProcessor
from observers import ConversationLogObserver
from prompt import build_system_instruction
from sound import ReadinessChimeFrame, chime_pcm
from tools.bundle import (
    ToolBundle,
    close_tool_bundle,
    register_tool_bundle,
    tools_schema_from_bundle,
)
from tools.runtime import setup_tools, shutdown_tools, warmup_tools
from working_sound import WorkingSoundProcessor

load_dotenv(override=True)


# Whether to play the soft "working" cue (see working_sound.py) during the gap
# between the user finishing and the bot's audio starting. Flip this to False to
# run the pipeline exactly as it did before this feature: the processor is then
# never added.
WORKING_SOUND_ENABLED = True
# Grace period after the user stops (or a tool call starts) before the first
# motif plays; fast turns produce bot audio within this window and stay silent.
WORKING_SOUND_INITIAL_DELAY_SECS = 0.8

# Upper bound on the idle keep-alive heartbeat interval (see idle_keepalive.py).
# The actual interval is the smaller of this and a fraction of the idle timeout,
# so a heartbeat always lands well before the idle timer can expire.
IDLE_KEEPALIVE_MAX_INTERVAL_SECS = 5.0

# Absolute worst-case cap on a single continuous "bot is busy" window (see
# idle_keepalive.py). The keep-alive normally disarms when the bot starts
# speaking; if it never does - e.g. STT returns an empty transcript for
# background noise, or the LLM/TTS stalls - its heartbeats would otherwise reset
# the idle timer forever and the session could never time out. This cap lets the
# idle timer eventually win. Kept generously high so a legitimately long think or
# tool call is never cut off; this is purely a stuck-session backstop.
IDLE_KEEPALIVE_MAX_BUSY_SECS = 300.0


def _idle_keepalive_interval_secs() -> float:
    """Heartbeat cadence kept safely below the idle timeout.

    A heartbeat must reset the idle timer before it expires, so we cap the
    interval at a third of the idle timeout (and at a small absolute ceiling),
    with a 1s floor for absurdly short timeouts.
    """
    idle_timeout = _idle_timeout_secs()
    return max(1.0, min(IDLE_KEEPALIVE_MAX_INTERVAL_SECS, idle_timeout / 3.0))


async def _build_pipeline(
    transport: BaseTransport, brain: BrainSpec
) -> tuple[Pipeline, ToolBundle]:
    """Assemble the pipeline for the selected brain.

    Cascade and realtime share everything except whether STT/TTS are in the
    pipeline - a realtime (speech-to-speech) brain does both internally.

    Returns the merged ``ToolBundle`` too, so the caller can run its cleanups,
    if any, when the session ends.
    """
    # Set up tools before building the LLM: the system prompt is composed from
    # the active tools' descriptions, and the LLM bakes in that prompt at
    # construction. Each brain declares its provider-specific registry IDs;
    # always-on and config-enabled tools are added by the tool runtime.
    tool_bundle = await setup_tools(brain.tools)

    system_instruction = build_system_instruction(tool_bundle.instructions)
    services = brain.build(system_instruction)

    # Now that the LLM exists, wire any LLM-dependent handlers (e.g. MCP tools).
    await register_tool_bundle(tool_bundle, services.llm)

    tools = tools_schema_from_bundle(tool_bundle)
    context = LLMContext(tools=tools) if tools else LLMContext()

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Tuned tighter than the pipecat defaults (confidence=0.7,
            # start_secs=0.2, min_volume=0.6) so short noise bursts don't open a
            # speech segment and get shipped to STT, where they hallucinate
            # (often non-English) transcripts. Higher confidence + a longer
            # start window require sustained, louder speech before we listen;
            # stop_secs stays at the default so turn-end stays responsive.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.8,
                    start_secs=0.3,
                    min_volume=0.5,
                )
            ),
        ),
        # Realtime (speech-to-speech) services need different context-write
        # timing; the aggregator warns if this isn't set for them.
        realtime_service_mode=brain.is_realtime,
    )

    # Idle keep-alive heartbeat, so the bot's silent "thinking" time isn't
    # counted as idle and doesn't trip the session's idle timeout mid-turn (see
    # idle_keepalive.py). Always on - unlike the working-sound cue, this is a
    # correctness fix, not a flourish. BotBusyFrame is registered in the worker's
    # idle_timeout_frames (see _build_worker).
    idle_keepalive = IdleKeepaliveProcessor(
        interval_secs=_idle_keepalive_interval_secs(),
        max_busy_secs=IDLE_KEEPALIVE_MAX_BUSY_SECS,
    )

    # Soft "working" cue, sitting just before transport.output() so it sees the
    # turn/tool/TTS frames it gates on and can push its audio to the output. When
    # disabled it's simply omitted, leaving the pipeline byte-for-byte as before.
    working_sound = (
        [WorkingSoundProcessor(initial_delay_secs=WORKING_SOUND_INITIAL_DELAY_SECS)]
        if WORKING_SOUND_ENABLED
        else []
    )

    if brain.is_realtime:
        elements = [
            transport.input(),
            user_aggregator,
            services.llm,
            idle_keepalive,
            *working_sound,
            transport.output(),
            assistant_aggregator,
        ]
    else:
        elements = [
            transport.input(),
            services.stt,
            user_aggregator,
            services.llm,
            services.tts,
            idle_keepalive,
            *working_sound,
            transport.output(),
            assistant_aggregator,
        ]

    return Pipeline(elements), tool_bundle


# Seconds of silence (no user *or* bot speech) before the session ends itself,
# mirroring the LiveKit agent's ``user_away_timeout``. Pipecat's idle detection
# watches BotSpeakingFrame/UserSpeakingFrame by default, so this is "no one has
# spoken for this long." Override with IDLE_TIMEOUT_SECS.
DEFAULT_IDLE_TIMEOUT_SECS = 30.0


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


def _build_worker(pipeline: Pipeline) -> PipelineWorker:
    idle_timeout = _idle_timeout_secs()

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        # After this much silence, cancel the worker *and* the runner so
        # run()/run_local() return and the process exits cleanly. That exit is
        # what lets the wake word detector (run with --wait) resume listening
        # and re-invoke the bot. cancel_on_idle_timeout/cancel_runner_on_idle_timeout
        # default to True, but we set them explicitly since the exit is the point.
        idle_timeout_secs=idle_timeout,
        # Pipecat's idle detection resets only on these frames. The defaults are
        # BotSpeakingFrame/UserSpeakingFrame ("someone is speaking"); we add
        # BotBusyFrame so the bot's silent thinking/tool time also counts as
        # activity and a long turn can't trip the timeout (see idle_keepalive.py).
        idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame, BotBusyFrame),
        cancel_on_idle_timeout=True,
        cancel_runner_on_idle_timeout=True,
        # Logs user speech, bot speech, and tool-call results to the console.
        # Brain-agnostic, so it covers both the realtime and cascade pipelines.
        observers=[ConversationLogObserver()],
    )

    @worker.event_handler("on_idle_timeout")
    async def _on_idle_timeout(_worker):
        logger.info(
            f"Idle for {idle_timeout}s with no speech; ending session so the process can exit"
        )

    # Play a short "ding" the moment the pipeline is ready to accept audio, so
    # the user gets an audible acknowledgement that the bot is now listening
    # (mirrors the LiveKit client's readiness chime). A ReadinessChimeFrame still
    # doesn't count as bot speech (it's not a TTSAudioRawFrame), so it won't reset
    # the idle timer or trigger interruption logic; the transport resamples it.
    # Unlike a plain OutputAudioRawFrame, though, the local transport's half-duplex
    # gate closes the mic while it plays so the chime's echo never reaches the
    # capture path - on a hardware speakerphone that echo otherwise poisons the
    # fresh echo canceller and swallows the user's first sentence (see sound.py).
    @worker.event_handler("on_pipeline_started")
    async def _on_pipeline_started(worker, _frame):
        pcm, sample_rate = chime_pcm()
        await worker.queue_frame(
            ReadinessChimeFrame(audio=pcm, sample_rate=sample_rate, num_channels=1)
        )
        logger.info("Pipeline ready; played readiness chime")

    return worker


async def _warmup_brain(brain: BrainSpec) -> None:
    """Eagerly download/load the brain's slow first-run models, once per process.

    Brains that pull large models or talk to a local server (e.g. the local
    Whisper/Ollama/Kokoro brain) expose a ``warmup`` that would otherwise run
    lazily mid-conversation - the first utterance stalling on a model download,
    or the first reply on an LLM cold start. Running it here moves that cost to
    startup, where it's visible (and, in wake-gated mode, happens before the
    wake word).

    Fail-fast: ``warmup`` raises on a known-broken setup (e.g. the local model
    server isn't running) with an actionable message, and that propagates to
    abort startup so the user can fix it and relaunch - better than a confusing
    failure once a session is underway.
    """
    if brain.warmup is None:
        return
    logger.info("Warming up models (first run may download; this can take a while)...")
    await brain.warmup()
    logger.info("Warmup complete")


async def _warmup() -> None:
    """Eagerly load brain models and MCP tools once per process (parallel)."""
    import asyncio

    brain = get_brain()
    await asyncio.gather(_warmup_brain(brain), warmup_tools(brain.tools))


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run the voice bot for a dev-runner session (browser WebRTC UI)."""
    brain = get_brain()
    logger.info(f"Starting bot (brain={brain.name}, realtime={brain.is_realtime})")

    await _warmup()

    pipeline, tool_bundle = await _build_pipeline(transport, brain)
    worker = _build_worker(pipeline)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    # Run the tool cleanups, if any, here -- in the same task the tools were
    # created in -- rather than in the disconnect handler.
    try:
        await runner.run()
    finally:
        await close_tool_bundle(tool_bundle)
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
    of ``run_local`` so the wake-gated loop can run many sessions in one process.
    """
    from transport_local import build_local_transport, close_local_transport

    brain = get_brain()
    logger.info(f"Starting session (brain={brain.name}, realtime={brain.is_realtime})")

    transport = build_local_transport()
    # Building the pipeline loads the on-device VAD model and constructs the
    # brain's services - a brief silent stretch on a fresh process. Narrate it so
    # the terminal doesn't look stuck before the readiness chime.
    logger.info("Preparing models and audio...")
    pipeline, tool_bundle = await _build_pipeline(transport, brain)
    worker = _build_worker(pipeline)

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    try:
        await runner.run()
    finally:
        await close_tool_bundle(tool_bundle)
        # Release this session's PyAudio instance so a long-lived process
        # running many sessions (wake-gated mode) doesn't accumulate PortAudio
        # handles. The transport's streams are already closed by worker cleanup.
        close_local_transport(transport)


async def run_local() -> None:
    """Run the bot over local mic/speakers - the terminal voice CLI."""
    try:
        await _warmup()
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
    ``run_until_complete`` (not ``asyncio.run``, which would create and destroy a
    fresh loop each call). One persistent loop is required so the warmed MCP pool
    connections, whose anyio task groups are bound to the loop that created them,
    stay valid and reusable across sessions rather than dying when warmup's loop
    would otherwise close.
    """
    import asyncio

    from wakeword import PyAudioSource, WakeWordEngine, WakeWordListener

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Warm the brain's models up front (before we start listening) so the very
    # first session doesn't stall on a model download / LLM cold start, and so
    # the setup progress is visible before the user is told to say the wake
    # word. Fail-fast: a known-broken setup (e.g. Ollama not running) aborts the
    # process here with a clear message, rather than being swallowed by the
    # per-session error handler in the wake loop below and looping forever.
    loop.run_until_complete(_warmup())

    models = _wake_models()
    # Constructing the engine imports openwakeword/onnxruntime and loads the
    # models (downloaded once on the first run) - a few silent seconds. Narrate it
    # so the terminal doesn't look frozen; the "ready" log below marks completion.
    logger.info("Loading on-device wake-word detection (importing models)...")
    # Threshold and inference framework use WakeWordEngine's defaults (0.5, onnx).
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


if __name__ == "__main__":
    import argparse
    import sys

    # --mode picks how you talk to the bot; everything else is forwarded to
    # Pipecat's dev runner (e.g. --host/--port) in webrtc mode.
    #   local                 mic/speakers voice CLI
    #   local-with-wake-word  like local, but warm process gated by a wake word
    #   webrtc                browser debug UI
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
        import asyncio

        asyncio.run(run_local())
    else:
        sys.argv = [sys.argv[0], *runner_args]
        from pipecat.runner.run import main

        main()
