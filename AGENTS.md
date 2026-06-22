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
