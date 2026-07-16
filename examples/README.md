# openlily examples

Using openlily as a library in your own Pipecat agent. Install it first:

```bash
pip install openlily        # from this repo: pip install ./server
```

- [pipecat_cloud_bot.py](pipecat_cloud_bot.py) - a turnkey `bot(runner_args)` entry
  point for Pipecat Cloud (or the dev runner): pick a brain, enable tools, and
  toggle the readiness chime / "working" cue via `AgentConfig`.
- [custom_brain_and_tool.py](custom_brain_and_tool.py) - add your own brain and
  tool with `register_brain` / `register_tool` (no need to edit the package), and
  compose your own pipeline from openlily's individual processors.

The API surface lives on the top-level `openlily` package (see `openlily.__all__`):
`AgentConfig`, `create_agent` / `build_pipeline` / `build_worker`, `warmup`,
`register_brain`, `register_tool`, and the composable pieces (`WorkingSoundProcessor`,
`IdleKeepaliveProcessor`, `ConversationLogObserver`, `chime_pcm`, ...).
