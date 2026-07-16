"""openlily local entry point.

Thin shim over :mod:`openlily.cli` so the documented workflow keeps working:

    uv run bot.py                              # --mode local-with-wake-word (default)
    uv run bot.py --mode local                 # talk via your mic/speakers
    uv run bot.py --mode webrtc                # browser UI at localhost:7860

All the real logic lives in the ``openlily`` package (see ``src/openlily/``).
``bot`` is re-exported at module level because Pipecat's dev runner discovers the
``bot(runner_args)`` coroutine in the ``__main__`` module.
"""

import sys

# Printed before the heavy imports below (Pipecat, ML runtimes, the brains),
# which take a few seconds warm and tens of seconds on the very first run. Without
# this the terminal looks frozen until the first real log line lands.
if __name__ == "__main__":
    print(
        "Starting openlily - loading modules (this takes several seconds, "
        "and up to a minute on the first run while dependencies compile)...",
        file=sys.stderr,
        flush=True,
    )

from openlily.cli import bot, main

__all__ = ["bot", "main"]

if __name__ == "__main__":
    main()
