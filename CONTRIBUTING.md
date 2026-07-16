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

```
openlily/
├── server/                   # the bot (all code lives here)
│   ├── bot.py                # entry point: pipeline assembly + run modes (local / wake-word / webrtc dev runner)
│   ├── prompt.py             # durable system instruction + per-session builder (injects active tools + date)
│   ├── brains/               # swappable LLM harnesses (see "Brains" below)
│   ├── tools/                # agent tools: web/ (per-brain), browser/ + email/ (generic), base.py contract
│   ├── wakeword/             # portable, Pipecat-agnostic wake-word detection (openWakeWord)
│   ├── transport_local.py    # local mic/speaker transport + WebRTC APM (AEC/NS/AGC) + half-duplex gating
│   ├── observers.py          # console logging of user/bot speech and tool results
│   ├── sound.py              # readiness chime
│   ├── env.py                # small env-var helpers (require_env)
│   ├── pyproject.toml        # dependencies
│   ├── .env.example          # documented environment variables
│   └── brains.yaml.example   # brain selection + per-brain model/voice overrides
├── README.md                 # product + usage
├── CONTRIBUTING.md           # this file
└── AGENTS.md                 # guidance for coding agents working in this repo
```

### The pipeline

`bot.py`'s `_build_pipeline` assembles a Pipecat pipeline from the selected brain.
Cascade and realtime share everything except whether STT/TTS are in the pipeline
(a realtime speech-to-speech brain does both internally):

- **Cascade**: `transport.input() → STT → user aggregator → LLM → TTS → transport.output() → assistant aggregator`
- **Realtime**: same, without STT/TTS.

Tools are set up *before* the LLM is built, because the system prompt
([prompt.py](server/prompt.py)) is composed from the active tools' descriptions and
the LLM bakes that prompt in at construction.

## Brains

A *brain* bundles everything that varies between pipelines: how to build the
services, whether STT/TTS are separate, and which tools the LLM gets. The contract
is `BrainSpec` in [server/brains/base.py](server/brains/base.py); the registry is in
[server/brains/\_\_init\_\_.py](server/brains/__init__.py).

To **add a brain** (e.g. a new provider or a local LLM):

1. Create `server/brains/<name>.py` with a `build(system_instruction) -> BrainServices`
   function and a module-level `SPEC: BrainSpec`. Look at
   [server/brains/cartesia_openai.py](server/brains/cartesia_openai.py) as a template.
2. Add a member to `BrainName` in [server/brains/base.py](server/brains/base.py).
3. Register the `SPEC` in [server/brains/\_\_init\_\_.py](server/brains/__init__.py).
4. (Optional) Add a section in `brains/overrides.py` and `brains.yaml.example` so its
   model/voice are overridable from `brains.yaml`.

Per-brain model/voice overrides flow through `brains.yaml`; each brain reads them via
`get_brain_overrides()` and falls back to a built-in default.

## Tools

Every tool is declared once in the central
[registry](server/tools/registry.py):

- **Per-brain tools** (e.g. hosted or Exa web search): a brain selects these by
  listing registry IDs in `BrainSpec.tools`.
- **Generic tools** (e.g. `tools/browser/`, `tools/email/`): brain-agnostic, layered
  onto every brain centrally. The `session` tool is always on; the optional ones
  (`browser`, `email`, `notion`, `x`) are enabled by name via the `tools` list in
  `brains.yaml`. Enabling one whose credentials are missing is a fail-fast startup
  error, not a silent skip.

A tool provider implements the `ToolProvider` contract in
[server/tools/base.py](server/tools/base.py): `is_configured()` reports whether its
credentials are present, and `create_tools()` returns Pipecat direct functions whose
name, typed signature, and docstring become the LLM tool schema. Direct provider
setup may gracefully return an empty bundle, but a configurable tool explicitly
enabled in `brains.yaml` is validated by the runtime and fails fast.

To **add a generic tool**:

1. Create `server/tools/<name>/` with a provider implementing `ToolProvider` and a
   `setup_<name>_tools() -> ToolBundle` factory, plus a config-presence check.
   [server/tools/email/](server/tools/email/) is a good multi-provider example.
2. Add its `ToolId` and optional `ToolName` to
   [server/tools/contracts.py](server/tools/contracts.py), export a `ToolSpec`
   beside the implementation, and add that export to the central index in
   [server/tools/registry.py](server/tools/registry.py). Include MCP connector,
   instructions, and warmup failure metadata in the module's spec when
   applicable. Users can then enable configurable tools by adding their name to
   `tools` in `brains.yaml`.

For a per-brain tool, export its `ToolSpec`, index it centrally, and reference
its `ToolId` from the relevant brain's `BrainSpec.tools`; the tool entry itself
does not list compatible brains.

A `ToolBundle` ([server/tools/bundle.py](server/tools/bundle.py)) carries the tools plus
optional prompt snippets (`instructions`), LLM-dependent `registrations` (e.g. MCP),
and `cleanups` run at session end. Bundles merge by concatenation, so tools compose.

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
