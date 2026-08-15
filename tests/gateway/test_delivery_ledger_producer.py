"""Producer-hook tests: _process_message_background records delivery
obligations around the final send (gateway/platforms/base.py).

Contract: the complete response bundle is recorded before the send await;
each physical operation is marked attempting, then checkpointed after ACK.
Slash commands, ephemeral replies, and empty responses are never recorded.
Pre-send durability failure blocks the physical send; all SQLite work runs
off the event loop.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BackendUnavailableReply,
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, stamp_source_transport_owner


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


class _Adapter(BasePlatformAdapter):  # type: ignore[misc]
    """Minimal concrete adapter driving the real base-class pipeline."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover - unused
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="m1")


def _event(text="hello agent"):
    source = SessionSource(
        platform=Platform.SLACK, chat_id="C1", chat_type="channel"
    )
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.SLACK,
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-42",
    )


def _rows():
    with dl._connect() as conn:
        return conn.execute(
            "SELECT obligation_id, state, content FROM delivery_obligations"
        ).fetchall()


def _blocking_probe():
    """Return a blocking ledger call and an event-loop progress witness."""
    ledger_started = threading.Event()
    event_loop_progressed = threading.Event()
    blocked_event_loop = []

    def _slow_ledger_call(*args, **kwargs):
        ledger_started.set()
        # Generous timeout: a genuinely blocked loop can never set the event
        # (the witness coroutine cannot run), so a longer wait only guards
        # against loaded-CI scheduling flake, not against missing the bug.
        if not event_loop_progressed.wait(timeout=5.0):
            blocked_event_loop.append(True)

    async def _event_loop_witness():
        deadline = asyncio.get_running_loop().time() + 10
        while not ledger_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("ledger call never started")
            await asyncio.sleep(0)
        event_loop_progressed.set()

    return _slow_ledger_call, _event_loop_witness, blocked_event_loop


async def _run(adapter, event, response="final answer"):
    stamp_source_transport_owner(event.source, adapter=adapter)
    adapter._message_handler = AsyncMock(return_value=response)
    session_key = "agent:main:slack:channel:C1"
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)


class TestProducerHook:
    @pytest.mark.asyncio
    async def test_normal_turn_records_and_delivers(self):
        adapter = _Adapter()
        await _run(adapter, _event())

        assert adapter.sent == ["final answer"]
        rows = _rows()
        assert len(rows) == 1
        assert rows[0][1] == "delivered"
        assert rows[0][2] == "final answer"

    @pytest.mark.asyncio
    async def test_send_failure_leaves_failed_row(self):
        adapter = _Adapter()
        adapter.send = AsyncMock(
            return_value=SendResult(success=False, error="chat_not_found")
        )
        await _run(adapter, _event())

        rows = _rows()
        assert len(rows) == 1
        assert rows[0][1] == "failed"

    @pytest.mark.asyncio
    async def test_owner_unavailable_final_is_persisted_pending_for_reconnect(self):
        adapter = _Adapter()
        adapter._final_delivery_adapter = lambda _source: None

        await _run(adapter, _event())

        assert adapter.sent == []
        with dl._connect() as conn:
            state, attempts, owner_pid = conn.execute(
                "SELECT state, attempts, owner_pid FROM delivery_obligations"
            ).fetchone()
        assert (state, attempts, owner_pid) == ("pending", 0, None)

    @pytest.mark.asyncio
    async def test_cancellation_during_initial_send_releases_live_owner(self):
        adapter = _Adapter()
        send_started = asyncio.Event()

        async def blocked_send(*_args, **_kwargs):
            send_started.set()
            await asyncio.Event().wait()

        adapter.send = blocked_send
        task = asyncio.create_task(_run(adapter, _event()))
        await send_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with dl._connect() as conn:
            state, attempts, owner_pid, recovery_claim = conn.execute(
                "SELECT state, attempts, owner_pid, recovery_claim "
                "FROM delivery_obligations"
            ).fetchone()
        assert (state, attempts, owner_pid, recovery_claim) == (
            "attempting",
            0,
            None,
            None,
        )

    @pytest.mark.asyncio
    async def test_backend_notice_is_not_durably_replayed_as_plain_text(self):
        adapter = _Adapter()
        await _run(
            adapter,
            _event(),
            BackendUnavailableReply("backend unavailable"),
        )

        assert adapter.sent == ["backend unavailable"]
        assert _rows() == []


    @pytest.mark.asyncio
    async def test_slow_ledger_record_does_not_block_event_loop(self):
        adapter = _Adapter()
        slow_record, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch(
            "gateway.delivery_ledger.record_obligation",
            side_effect=slow_record,
        ):
            await asyncio.gather(_run(adapter, _event()), event_loop_witness())

        assert blocked_event_loop == []
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_ledger_record_failure_blocks_untracked_physical_send(self):
        adapter = _Adapter()

        with patch(
            "gateway.delivery_ledger.record_obligation",
            side_effect=RuntimeError("state.db unavailable"),
        ):
            await _run(adapter, _event())

        assert adapter.sent == []
        assert _rows() == []

    @pytest.mark.asyncio
    async def test_slow_ledger_update_does_not_block_event_loop(self):
        adapter = _Adapter()
        slow_checkpoint, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch(
            "gateway.delivery_ledger.mark_bundle_operations_completed",
            side_effect=slow_checkpoint,
        ):
            await asyncio.gather(_run(adapter, _event()), event_loop_witness())

        assert blocked_event_loop == []
        assert adapter.sent == ["final answer"]

    @pytest.mark.asyncio
    async def test_crash_between_attempting_and_ack_is_recoverable(self):
        """The core scenario (#58818): process dies mid-send. The row must
        be claimable by a later process and carry the ambiguity marker."""
        adapter = _Adapter()

        async def _dies_mid_send(chat_id, content, reply_to=None, metadata=None):
            raise ConnectionError("gateway killed mid-await")

        adapter.send = _dies_mid_send
        # _send_with_retry raising propagates; the background task catches
        # broadly — drive only through the send block by tolerating the error.
        try:
            await _run(adapter, _event())
        except Exception:
            pass

        rows = _rows()
        assert len(rows) == 1
        # Row is stuck in 'attempting' (or failed if retry wrapper caught it):
        # either way it is non-delivered and recoverable.
        assert rows[0][1] in ("attempting", "failed")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
            )
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is True
