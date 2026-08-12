"""Unit tests for the TurnContext/TurnRunner seam extracted from
``GatewayRunner._run_agent_inner`` (gateway/turn_context.py + gateway/run.py).

The extraction contract: the closure bodies moved onto ``TurnRunner`` methods
byte-identically (modulo local -> ctx.field rewrites), with every closed-over
local carried as a ``TurnContext`` field. These tests pin the seam's wiring —
shared mutable containers, no-queue early returns — not the progress behavior
itself (that's covered by test_run_progress_topics.py et al.).
"""

import asyncio
import queue as queue_mod
from contextlib import suppress
from types import SimpleNamespace

import pytest

from gateway.turn_context import TurnContext


def _make_runner(ctx):
    from gateway.run import TurnRunner

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return None

    return TurnRunner(_StubGatewayRunner(), ctx)


class TestTurnContext:
    def test_defaults_are_independent_containers(self):
        a, b = TurnContext(), TurnContext()
        a.last_progress_msg[0] = "x"
        a.repeat_count[0] = 3
        a._cleanup_msg_ids.append("1")
        assert b.last_progress_msg == [None]
        assert b.repeat_count == [0]
        assert b._cleanup_msg_ids == []

    def test_shared_containers_visible_to_outer_scope(self):
        # The outer body and the runner share the SAME list objects, so
        # mutation through the ctx is visible to locals captured elsewhere.
        last_progress_msg = [None]
        ctx = TurnContext(last_progress_msg=last_progress_msg)
        ctx.last_progress_msg[0] = "🔍 web_search"
        assert last_progress_msg[0] == "🔍 web_search"


class TestTurnRunner:
    def test_methods_exist_and_bind(self):
        from gateway.run import TurnRunner

        ctx = TurnContext()
        runner = _make_runner(ctx)
        assert callable(runner.progress_callback)
        assert asyncio.iscoroutinefunction(TurnRunner.send_progress_messages)
        assert runner._ctx is ctx

    def test_send_progress_messages_no_queue_returns(self):
        ctx = TurnContext(progress_queue=None)
        runner = _make_runner(ctx)
        assert asyncio.run(runner.send_progress_messages()) is None

    def test_send_progress_messages_no_adapter_returns(self):
        ctx = TurnContext(progress_queue=queue_mod.Queue())
        runner = _make_runner(ctx)  # stub adapter resolver returns None
        assert asyncio.run(runner.send_progress_messages()) is None

    def test_live_status_callback_follows_replacement_adapter(self):
        from gateway.run import TurnRunner

        class StatusAdapter:
            supports_status_text = True

            def __init__(self):
                self.status = []

            def set_status_text(self, chat_id, text):
                self.status.append((chat_id, text))

        first = StatusAdapter()
        replacement = StatusAdapter()
        owner = {"adapter": replacement}
        ctx = TurnContext(
            source=SimpleNamespace(chat_id="chat-1"),
            _run_still_current=lambda: True,
            _live_adapter=lambda: owner["adapter"],
            _live_status_adapter=first,
            _live_status_mode="full",
        )
        runner = TurnRunner(SimpleNamespace(), ctx)

        runner.progress_callback("tool.started", "web_search", args={"query": "x"})

        assert first.status == []
        assert replacement.status
        assert replacement.status[0][0] == "chat-1"

    @pytest.mark.asyncio
    async def test_progress_replacement_opens_fresh_bubble_without_cross_edit(self):
        from gateway.platforms.base import SendResult
        from gateway.run import TurnRunner

        owner = {"adapter": None}

        class EditableAdapter:
            name = "progress-test"
            MAX_MESSAGE_LENGTH = 4000

            def __init__(self, message_id, *, retire_on_send=False):
                self.message_id = message_id
                self.retire_on_send = retire_on_send
                self.sent = []
                self.edits = []

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                self.sent.append(content)
                if self.retire_on_send:
                    owner["adapter"] = replacement
                return SendResult(success=True, message_id=self.message_id)

            async def edit_message(self, chat_id, message_id, content):
                self.edits.append((message_id, content))
                return SendResult(success=True, message_id=message_id)

            async def send_typing(self, chat_id, metadata=None):
                return None

        first = EditableAdapter("progress-1", retire_on_send=True)
        replacement = EditableAdapter("progress-2")
        owner["adapter"] = first
        progress_queue = queue_mod.Queue()
        progress_queue.put("first line")
        ctx = TurnContext(
            source=SimpleNamespace(chat_id="chat-1"),
            progress_queue=progress_queue,
            _run_still_current=lambda: True,
            _live_adapter=lambda: owner["adapter"],
            _live_operation=lambda name: (
                owner["adapter"],
                getattr(owner["adapter"], name, None),
            ),
        )
        runner = TurnRunner(SimpleNamespace(), ctx)
        task = asyncio.create_task(runner.send_progress_messages())
        try:
            for _ in range(50):
                if first.sent:
                    break
                await asyncio.sleep(0.03)
            assert first.sent == ["first line"]
            await asyncio.sleep(1.6)
            progress_queue.put("second line")
            for _ in range(120):
                if replacement.sent:
                    break
                await asyncio.sleep(0.03)
            assert replacement.sent == ["first line\nsecond line"]
            assert first.edits == []
            assert replacement.edits == []
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
