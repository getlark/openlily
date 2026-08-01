# AGENTS.md — working on openlily with a coding agent

openlily is an existing voice-assistant app built on **Pipecat**. This file is for
coding agents (Claude Code, Codex, Cursor, …) working *in this repo*. For the
product see [README.md](README.md); for architecture and dev setup see
[CONTRIBUTING.md](CONTRIBUTING.md). The one job of this file is to stop you writing
**stale Pipecat code**.

## Golden rule: don't guess Pipecat APIs — verify them

Pipecat moves fast, so your training data is often wrong about its classes,
imports, and parameters (confidently-wrong old APIs are the #1 failure mode).
Before you type a Pipecat class name, import path, or service parameter from memory,
**look it up against a live source**.

## Set up the Pipecat Context Hub (do this first)

The Context Hub is a local index of Pipecat source, examples, and docs. Prefer it
over your memory.

```bash
# One-time index build (uses the latest package; allow a few minutes)
uvx pipecat-ai-context-hub@latest refresh

# Add the MCP server (use the line for your agent). Loads at NEXT session start.
claude mcp add pipecat-context-hub -- uvx pipecat-ai-context-hub serve   # Claude Code
codex  mcp add pipecat-context-hub -- uvx pipecat-ai-context-hub serve   # Codex
```

Re-run the refresh after bumping the pinned Pipecat version, or periodically.

## How to find current truth

Use the highest rung that works right now:

1. **The `pipecat-context-hub` MCP** (if in your tool list) — returns primary source:
   - `check_deprecation <symbol>` — the reflex check; run it on any symbol you're
     unsure about (e.g. `PipelineTask` → `PipelineWorker`).
   - `search_api` / `get_code_snippet` — exact current signatures and usage.
   - `search_docs` / `get_doc` — learn how a capability works before building it.
   - `search_examples` / `get_example` — a working implementation to start from.
2. **No MCP? Same index from the shell** (only needs `uv`):
   ```bash
   uvx pipecat-ai-context-hub check-deprecation PipelineTask   # <1s reflex check
   uvx pipecat-ai-context-hub search-api "EvalTransportParams"
   uvx pipecat-ai-context-hub search-docs "turn detection"
   uvx pipecat-ai-context-hub status                           # index health
   ```
   Exit 2 means the index isn't built — run the `refresh` above once, then retry.
3. **Installed package source** — the pinned version is on disk and can't be stale:
   ```bash
   python -c "import pipecat, os; print(os.path.dirname(pipecat.__file__))"
   ```
4. **`llms.txt`** — `https://docs.pipecat.ai/llms.txt` (full: `llms-full.txt`), last resort.

## A few Pipecat facts worth keeping straight

- **Terminology**: the runnable unit is a `PipelineWorker` run by a `WorkerRunner`
  (`PipelineTask` is a deprecated alias). "Task" means only an asyncio task.
- **Change a running pipeline by pushing frames, not calling methods** — Pipecat is
  real-time and ordered.
- **Pipeline order matters** and the assistant aggregator goes *after*
  `transport.output()`. See `_build_pipeline` in [server/bot.py](server/bot.py) for
  how this repo wires both the cascade and realtime shapes.
- **The LLM's output is spoken** — no markdown/emoji/bullets. The system prompt in
  [server/prompt.py](server/prompt.py) already enforces this; keep it that way.
- **Tools are plain async functions** whose name, typed signature, and docstring
  become the schema. See [server/tools/base.py](server/tools/base.py).

When in doubt about anything Pipecat, `check_deprecation` / `search_api` first.

## Cursor Cloud specific instructions

The dev environment is already provisioned (the startup update script runs
`uv sync` for the `server/` project with the `local`, `web`, and `email` extras;
`uv` and Python 3.11 — pinned by `server/.python-version` — are baked into the
snapshot, and PortAudio system libs are installed for PyAudio). You do **not**
need to re-install anything to start working. All commands run from `server/`
via `uv run` — see [CONTRIBUTING.md](CONTRIBUTING.md) for the canonical
lint/test/run commands (`uv run ruff check .`, `uv run pyright`, `uv run pytest`,
`uv run bot.py --mode …`). The `local-models` extra is intentionally **not**
installed: its `mlx-whisper` runtime is Apple-Silicon only, so the
`local_whisper_ollama_kokoro` brain cannot run on this Linux VM.

Non-obvious caveats worth knowing here:

- **This VM has no audio device.** The `local` and `local-with-wake-word` run
  modes need a real mic/speakers (PyAudio) and cannot run headless. Use
  `uv run bot.py --mode webrtc` — it serves Pipecat's prebuilt browser client at
  `http://localhost:7860/client/`, where the mic/speakers live in the *browser*,
  not the VM. That is the runnable service to exercise the app here.
- **`cli.py` calls `load_dotenv(override=True)`, so `server/.env` OVERRIDES real
  environment variables** (even a blank `KEY=` clobbers it to empty). Cursor
  injects Secrets as env vars, so do **not** create a `server/.env` that sets any
  key you also provide as a Secret — it will silently override the Secret. Prefer
  no `.env` at all (env-var Secrets flow straight through), or put only
  non-secret tuning in it. `.env` and `brains.yaml` are git-ignored.
- **Provider keys gate a real conversation, not startup.** Keys are read when the
  brain is *built* (on WebRTC client connect), not at server boot. The default
  `cartesia_openai` brain needs `OPENAI_API_KEY` + `CARTESIA_API_KEY`;
  `openai_standard`/`openai_realtime` need only `OPENAI_API_KEY` (set
  `default_brain` in `brains.yaml`). Without valid keys the server still boots and
  the client still reaches `READY`/`READY`; the pipeline then fails at the first
  STT/LLM/TTS call with `HTTP 401` — that 401 confirms the wiring is correct.
- **Driving a turn without a mic:** the automated browser has no real microphone,
  so a *spoken* exchange can't be scripted headlessly. The `/client/` Playground
  has a "Type message" text box that (with a valid `OPENAI_API_KEY`) drives the
  LLM directly — the easiest way to exercise core functionality end to end here.
- **Pre-existing lint/type findings:** the pinned-range tool versions that resolve
  here (ruff `0.15.x`, pyright `1.1.410`) report a few findings in committed code
  (ruff `UP035`/`UP017`; pyright `reportArgumentType` in `agent.py`). These
  predate this setup and are not caused by environment changes.
