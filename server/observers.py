"""Conversation + tool-result logging via a Pipecat observer.

An observer sees every frame as it's pushed between processors, in both
directions, so a single class can log the whole conversation regardless of the
brain in use:

- the realtime brain pushes the user's transcript *upstream* (from the
  speech-to-speech LLM service) and the bot's text *downstream*;
- the cascade brain pushes the user's transcript downstream from the STT
  service and the bot's text downstream from the LLM service.

This avoids the placement/direction gymnastics an in-pipeline ``FrameProcessor``
would need, and it's why the built-in ``TranscriptionLogObserver`` (which only
logs frames whose source is an ``STTService``) misses the realtime brain's user
transcripts.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService

# Tool-call args/results can be large (e.g. web_search payloads); cap what we
# dump so the logs stay a quick, readable summary rather than a wall of JSON.
_MAX_RESULT_CHARS = 500


def _truncate(text: str, limit: int = _MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} chars total)"


def _dump(value: object) -> str:
    """Render a tool's args/result as compact, truncated JSON for logging."""
    try:
        rendered = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(value)
    return _truncate(rendered)


# When set, tool calls/results are dumped (full, untruncated) as JSONL to a
# per-session file in this directory. Unset/blank disables the dump entirely,
# leaving behavior unchanged. Read once here; the sink no-ops when it's falsy.
_TOOL_CALL_DEBUG_DIR_ENV = "TOOL_CALL_DEBUG_DIR"


class _ToolCallDebugSink:
    """Appends full tool-call/result records to a per-session JSONL file.

    Enabled only when ``TOOL_CALL_DEBUG_DIR`` is set; otherwise every method is
    a no-op so the normal (console-only) path is unchanged. One sink lives on
    each ``ConversationLogObserver`` instance, and since a fresh observer is
    built per session, each session gets its own file. The file is created
    lazily on the first record, so sessions that make no tool calls leave no
    empty files behind. Writes open-append-per-record (tool-call volume is low)
    so there's no handle to close, and any I/O error is logged and swallowed so
    a debug aid can never disrupt the live session.
    """

    def __init__(self) -> None:
        raw = os.getenv(_TOOL_CALL_DEBUG_DIR_ENV, "").strip()
        self._dir = Path(raw) if raw else None
        # Resolved lazily on the first write so an unused session writes nothing.
        self._path: Path | None = None

    @property
    def enabled(self) -> bool:
        return self._dir is not None

    def _resolve_path(self) -> Path | None:
        """Return this session's file path, creating the directory once."""
        if self._dir is None:
            return None
        if self._path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            # A short random suffix disambiguates sessions started within the
            # same second (and across processes) without needing a counter.
            suffix = uuid.uuid4().hex[:8]
            self._dir.mkdir(parents=True, exist_ok=True)
            self._path = self._dir / f"tool-calls-{stamp}-{suffix}.jsonl"
            logger.info(f"Dumping tool calls to {self._path}")
        return self._path

    def write(self, record: dict[str, object]) -> None:
        """Append one record as a JSON line, stamped with the current time."""
        if self._dir is None:
            return
        line = _record_to_json(record)
        try:
            path = self._resolve_path()
            if path is None:
                return
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            logger.warning(f"Failed to write tool-call debug record: {exc}")


def _record_to_json(record: dict[str, object]) -> str:
    """Serialize a debug record to a single JSON line, never raising.

    Full (untruncated) payloads are kept -- unlike the console ``_dump`` -- so
    the file is a faithful capture. ``default=str`` handles most non-JSON types;
    if a value still can't be serialized, we stringify each value individually
    so one bad field can't drop the whole record.
    """
    try:
        return json.dumps(record, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        safe = {key: str(value) for key, value in record.items()}
        return json.dumps(safe, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Matches a leading run of text ending in sentence punctuation that is *followed*
# by whitespace -- i.e. a sentence we know is complete because more text has
# already arrived after it. Requiring the trailing whitespace (rather than
# end-of-buffer) keeps us from flushing a half-finished sentence or splitting on
# things like "3.14" before the next delta lands.
_SENTENCE_RE = re.compile(r"\s*\S.*?[.!?]+(?=\s)", re.S)


class ConversationLogObserver(BaseObserver):
    """Log user speech, bot speech, and tool-call results to the console.

    Dedup strategy: ``on_push_frame`` fires once per processor hop, so a frame
    that traverses several processors would be logged repeatedly. We only act
    when ``data.source`` is the service that *emits* the frame (an ``STTService``
    or ``LLMService``), mirroring the built-in ``TranscriptionLogObserver``'s
    ``isinstance(src, STTService)`` trick. Tool frames are broadcast both ways,
    so for those we additionally require a downstream push.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Buffers the bot's text deltas for the current turn. We flush complete
        # sentences as they arrive (so the logs show progress while the bot is
        # still speaking), then flush whatever is left when the response ends.
        self._bot_buffer = ""
        # Opt-in structured dump of tool calls/results to a per-session file.
        # No-op unless TOOL_CALL_DEBUG_DIR is set, so the default path is unchanged.
        self._tool_debug = _ToolCallDebugSink()

    def _flush_complete_sentences(self) -> None:
        """Log any complete sentences sitting at the front of the buffer."""
        while True:
            match = _SENTENCE_RE.match(self._bot_buffer)
            if not match:
                break
            sentence = match.group(0).strip()
            self._bot_buffer = self._bot_buffer[match.end() :]
            if sentence:
                logger.info(f"BOT: {sentence!r}")

    async def on_push_frame(self, data: FramePushed):
        src = data.source
        frame = data.frame
        direction = data.direction

        # User speech. Source is the STT service (cascade) or the realtime LLM
        # service (realtime), so accept either.
        if isinstance(frame, TranscriptionFrame) and isinstance(src, (STTService, LLMService)):
            text = (frame.text or "").strip()
            if text:
                logger.info(f"USER: {text!r}")
            return

        # Bot speech. Accumulate the LLM's text deltas and log each sentence as
        # soon as it completes, so a long answer shows up incrementally rather
        # than only after the whole turn finishes. Both frames originate at the
        # LLM service.
        if isinstance(src, LLMService):
            if isinstance(frame, LLMTextFrame):
                if frame.text:
                    self._bot_buffer += frame.text
                    self._flush_complete_sentences()
                return
            if isinstance(frame, LLMFullResponseEndFrame):
                # Flush the trailing sentence (which has no whitespace after it
                # to mark completion) plus anything else left in the buffer.
                remainder = self._bot_buffer.strip()
                self._bot_buffer = ""
                if remainder:
                    logger.info(f"BOT: {remainder!r}")
                return

        # Tool calls. These are broadcast upstream and downstream, so restrict
        # to the downstream push from the LLM service to log each one once.
        if isinstance(src, LLMService) and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, FunctionCallInProgressFrame):
                logger.info(f"TOOL CALL {frame.function_name}({_dump(frame.arguments)})")
                self._tool_debug.write(
                    {
                        "ts": _now_iso(),
                        "event": "call",
                        "function_name": frame.function_name,
                        "tool_call_id": frame.tool_call_id,
                        "arguments": frame.arguments,
                        "group_id": frame.group_id,
                    }
                )
                return
            if isinstance(frame, FunctionCallResultFrame):
                logger.info(f"TOOL RESULT {frame.function_name}: {_dump(frame.result)}")
                self._tool_debug.write(
                    {
                        "ts": _now_iso(),
                        "event": "result",
                        "function_name": frame.function_name,
                        "tool_call_id": frame.tool_call_id,
                        "arguments": frame.arguments,
                        "result": frame.result,
                    }
                )
                return
