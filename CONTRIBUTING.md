# Contributing to openlily

Thanks for your interest in openlily. This document covers how the project is laid
out, how to set up a dev environment, and how to extend it (new brains, new tools).

## Dev setup

Everything lives under `server/`.

```bash
cd server
uv sync                 # installs runtime + dev dependencies
brew install portaudio  # macOS: required by PyAudio for local audio
brew install node       # macOS: only if you'll use the browser or notion tool (needs npx)
cp .env.example .env     # then fill in the keys you need
```

Run the bot while developing:

```bash
uv run bot.py --mode local     # talk immediately, no wake word (fastest dev loop)
uv run bot.py --mode webrtc    # browser debug UI at localhost:7860
uv run bot.py                  # default: wake-word gated
```

### Lint and type-check

The `dev` dependency group pins `ruff` and `pyright` (see
[server/pyproject.toml](server/pyproject.toml)). Run both before sending a change:

```bash
uv run ruff check .
uv run ruff format .
uv run pyright
```

Ruff is configured with `line-length = 100` and the `I` (import sorting) and `UP`
(pyupgrade) rule sets.

## Architecture

openlily is an installable package (`src/openlily/`) plus a thin local entry point.

```
openlily/
├── server/                        # the project (package + dev setup)
│   ├── bot.py                     # thin shim over openlily.cli (keeps `uv run bot.py`)
│   ├── src/openlily/              # the installable library
│   │   ├── __init__.py            # public API (AgentConfig, create_agent, register_brain/tool, ...)
│   │   ├── agent.py               # factory: build_pipeline / build_worker / create_agent / warmup
│   │   ├── config.py              # AgentConfig + WorkingSoundConfig
│   │   ├── cli.py                 # local / wake-word / webrtc run modes (reads .env + brains.yaml)
│   │   ├── prompt.py              # durable system instruction + per-session builder
│   │   ├── brains/                # swappable LLM harnesses (see "Brains" below) + register_brain
│   │   ├── tools/                 # agent tools + central registry + register_tool
│   │   ├── observers.py           # console logging of user/bot speech and tool results
│   │   ├── sound.py               # readiness chime + working-sound synthesis
│   │   ├── working_sound.py       # optional "working" cue processor
│   │   ├── idle_keepalive.py      # idle keep-alive heartbeat processor
│   │   ├── env.py                 # small env-var helpers (require_env)
│   │   └── local/                 # local-audio building blocks (not needed for cloud):
│   │       ├── transport.py       #   local mic/speaker transport + WebRTC APM + half-duplex gating
│   │       ├── wakeword/          #   portable, Pipecat-agnostic wake-word detection (openWakeWord)
│   │       └── barge_in.py        #   wake-word barge-in bridge
│   ├── pyproject.toml             # dependencies + build system + `openlily` console script
│   ├── .env.example               # documented environment variables
│   └── brains.yaml.example        # brain selection + per-brain model/voice overrides
├── examples/                      # using openlily as a library (Pipecat Cloud, custom brain/tool)
├── README.md                      # product + usage
├── CONTRIBUTING.md                # this file
└── AGENTS.md                      # guidance for coding agents working in this repo
```

The library core (everything except `cli.py` and `local/`) reads no files or
environment at import time -- all configuration flows through `AgentConfig`. Only
`cli.py` loads `.env` and `brains.yaml` and turns them into an `AgentConfig`.

### Using openlily as a library

Others can `pip install openlily` and build their own Pipecat agent:

```python
import openlily

config = openlily.AgentConfig(brain="cartesia_openai", enabled_tools=["email"])
await openlily.warmup(config)
agent = await openlily.create_agent(my_transport, config)   # add agent.worker to a WorkerRunner
```

Everything is modular: toggle the flourishes (`working_sound=False`,
`readiness_chime=False`), override the prompt/observers/VAD, add a brain or tool
with `openlily.register_brain` / `openlily.register_tool`, or import the individual
processors (`WorkingSoundProcessor`, `IdleKeepaliveProcessor`, `chime_pcm`, ...) and
compose your own pipeline. See [examples/](examples/).

### The pipeline

`openlily/agent.py`'s `build_pipeline` assembles a Pipecat pipeline from the selected brain.
Cascade and realtime share everything except whether STT/TTS are in the pipeline
(a realtime speech-to-speech brain does both internally):

- **Cascade**: `transport.input() → STT → user aggregator → LLM → TTS → transport.output() → assistant aggregator`
- **Realtime**: same, without STT/TTS.

Tools are set up *before* the LLM is built, because the system prompt
([prompt.py](server/src/openlily/prompt.py)) is composed from the active tools'
descriptions and the LLM bakes that prompt in at construction.

## Brains

A *brain* bundles everything that varies between pipelines: how to build the
services, whether STT/TTS are separate, and which tools the LLM gets. The contract
is `BrainSpec` in [server/src/openlily/brains/base.py](server/src/openlily/brains/base.py);
the registry is in [server/src/openlily/brains/\_\_init\_\_.py](server/src/openlily/brains/__init__.py).

To **add a built-in brain** (e.g. a new provider or a local LLM):

1. Create `server/src/openlily/brains/<name>.py` with a
   `build(system_instruction) -> BrainServices` function and a module-level
   `SPEC: BrainSpec`. Look at
   [server/src/openlily/brains/cartesia_openai.py](server/src/openlily/brains/cartesia_openai.py)
   as a template.
2. Add a member to `BrainName` in [server/src/openlily/brains/base.py](server/src/openlily/brains/base.py).
3. Register the `SPEC` in [server/src/openlily/brains/\_\_init\_\_.py](server/src/openlily/brains/__init__.py).
4. (Optional) Add a section in `brains/overrides.py` and `brains.yaml.example` so its
   model/voice are overridable from `brains.yaml`.

Per-brain model/voice overrides flow through `brains.yaml`; each brain reads them via
`get_brain_overrides()` and falls back to a built-in default.

Library **consumers** don't need to edit the package: build a `BrainSpec` (its `name`
may be any string, not just a `BrainName` member) and call `openlily.register_brain(spec)`
at import time, then select it with `AgentConfig(brain="<name>")` or pass the spec directly.

## Tools

Every tool is declared once in the central
[registry](server/src/openlily/tools/registry.py):

- **Per-brain tools** (e.g. hosted or Exa web search): a brain selects these by
  listing registry IDs in `BrainSpec.tools`.
- **Generic tools** (e.g. `tools/browser/`, `tools/email/`): brain-agnostic, layered
  onto every brain centrally. The `session` tool is always on; the optional ones
  (`browser`, `email`, `notion`, `x`) are enabled by name -- via the `tools` list in
  `brains.yaml` for the CLI, or `AgentConfig(enabled_tools=[...])` for the library.
  Enabling one whose credentials are missing is a fail-fast startup error, not a
  silent skip.

A tool provider implements the `ToolProvider` contract in
[server/src/openlily/tools/base.py](server/src/openlily/tools/base.py): `is_configured()`
reports whether its credentials are present, and `create_tools()` returns Pipecat direct
functions whose name, typed signature, and docstring become the LLM tool schema. Direct
provider setup may gracefully return an empty bundle, but a configurable tool explicitly
enabled is validated by the runtime and fails fast.

To **add a built-in generic tool**:

1. Create `server/src/openlily/tools/<name>/` with a provider implementing `ToolProvider`
   and a `setup_<name>_tools() -> ToolBundle` factory, plus a config-presence check.
   [server/src/openlily/tools/email/](server/src/openlily/tools/email/) is a good
   multi-provider example.
2. Add its `ToolId` and optional `ToolName` to
   [server/src/openlily/tools/contracts.py](server/src/openlily/tools/contracts.py), export
   a `ToolSpec` beside the implementation, and add that export to the central index in
   [server/src/openlily/tools/registry.py](server/src/openlily/tools/registry.py). Include
   MCP connector, instructions, and warmup failure metadata in the module's spec when
   applicable. Users can then enable configurable tools by name.

For a per-brain tool, export its `ToolSpec`, index it centrally, and reference
its `ToolId` from the relevant brain's `BrainSpec.tools`; the tool entry itself
does not list compatible brains.

Library **consumers** can add a tool without editing the package: build a `ToolSpec`
(its `id` may be any string) and call `openlily.register_tool(spec)` at import time, then
reference it from a custom brain's `BrainSpec.tools`. See [examples/](examples/).

A `ToolBundle` ([server/src/openlily/tools/bundle.py](server/src/openlily/tools/bundle.py))
carries the tools plus optional prompt snippets (`instructions`), LLM-dependent
`registrations` (e.g. MCP), and `cleanups` run at session end. Bundles merge by
concatenation, so tools compose.

## Working with a coding agent

This project is built on [Pipecat](https://pipecat.ai), which moves fast — a coding
agent's training data is often stale on Pipecat APIs. [AGENTS.md](AGENTS.md) tells an
agent how to look up current Pipecat truth (the Context Hub, deprecation checks)
before writing code. If you use Claude Code, Codex, or Cursor, point your agent at it.

## Testing

A voice bot can't be eyeballed, but you don't need a live call to test it. Pipecat
ships a headless **eval harness** (`pipecat-ai[evals]`) that drives a running bot with
scripted YAML scenarios and asserts on what it does — deterministic checks plus an
optional LLM judge — in fast text mode or full audio mode. See the Pipecat Evals docs
and [AGENTS.md](AGENTS.md) for the workflow. (Eval scenarios aren't checked into this
repo yet — adding a starter suite under `server/evals/` is a welcome contribution.)
