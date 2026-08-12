"""Tests for the gateway delivery-obligation ledger (gateway/delivery_ledger.py).

State machine, dead-owner claiming, attempts cap, stale cutoff, retention,
id stability, and the startup redelivery sweep's contract:
- pending rows redeliver plainly (send never started, no dup risk)
- attempting/failed rows carry the recovered-reply marker (honest
  at-least-once; ambiguity is labeled, never silently resent)
- rows owned by a LIVE process are never claimed
- poison rows abandon at the attempts cap / stale cutoff
"""

import sqlite3
import time
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Isolated state.db per test (autouse HERMES_HOME isolation already
    redirects get_hermes_home; make the redirect explicit and per-test)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _record(oid="ob-1", session_key="agent:main:slack:channel:C1", **kw):
    dl.record_obligation(
        obligation_id=oid,
        session_key=session_key,
        platform=kw.get("platform", "slack"),
        chat_id=kw.get("chat_id", "C1"),
        thread_id=kw.get("thread_id", "171.001"),
        content=kw.get("content", "the final answer"),
    )


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            """SELECT state, attempts, owner_pid, content
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0], "attempts": r[1], "owner_pid": r[2], "content": r[3],
    }


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
        import asyncio

        deadline = asyncio.get_running_loop().time() + 10
        while not ledger_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("ledger call never started")
            await asyncio.sleep(0)
        event_loop_progressed.set()

    return _slow_ledger_call, _event_loop_witness, blocked_event_loop


def _orphan(oid):
    """Make the row look like it belongs to a dead process."""
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )


class TestStateMachine:
    def test_record_starts_pending(self):
        _record()
        assert _row("ob-1")["state"] == "pending"

    def test_existing_schema_migrates_transport_owner_columns(self):
        path = dl._db_path()
        conn = sqlite3.connect(path)
        conn.execute(
            """CREATE TABLE delivery_obligations (
                obligation_id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                platform TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                thread_id TEXT,
                content TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                owner_pid INTEGER,
                owner_started_at INTEGER,
                last_error TEXT
            )"""
        )
        conn.commit()
        conn.close()

        _record()
        with dl._connect() as migrated:
            columns = {
                row[1]
                for row in migrated.execute(
                    "PRAGMA table_info(delivery_obligations)"
                ).fetchall()
            }
            owner = migrated.execute(
                "SELECT transport_platform, transport_profile, "
                "transport_profile_stamped, transport_identity, route_scope_id, "
                "route_user_id, route_chat_type "
                "FROM delivery_obligations WHERE obligation_id='ob-1'"
            ).fetchone()
        assert {
            "transport_platform",
            "transport_profile",
            "transport_profile_stamped",
            "transport_identity",
            "route_scope_id",
            "route_user_id",
            "route_chat_type",
        } <= columns
        assert owner == (None, None, 0, None, None, None, None)


class TestObligationId:
    def test_stable_and_distinct(self):
        a = dl.compute_obligation_id("sk1", "msg1", "hello")
        assert a == dl.compute_obligation_id("sk1", "msg1", "hello")
        # Different thread (baked into session_key) → different id. This is
        # the cron-topic collision class from the earlier outbox attempt.
        assert a != dl.compute_obligation_id("sk1:threadB", "msg1", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg2", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg1", "other")
        assert len(a) == 24

    def test_stamped_physical_platform_is_part_of_identity(self):
        native = dl.compute_obligation_id(
            "sk1",
            "msg1",
            "hello",
            transport_platform="discord",
            transport_profile=None,
            transport_profile_stamped=True,
        )
        relay = dl.compute_obligation_id(
            "sk1",
            "msg1",
            "hello",
            transport_platform="relay",
            transport_profile=None,
            transport_profile_stamped=True,
        )

        assert native != relay

    def test_transport_identity_is_part_of_identity(self):
        first = dl.compute_obligation_id(
            "shared-session",
            "msg1",
            "hello",
            transport_platform="relay",
            transport_profile=None,
            transport_profile_stamped=True,
            transport_identity="discord:app-1",
        )
        second = dl.compute_obligation_id(
            "shared-session",
            "msg1",
            "hello",
            transport_platform="relay",
            transport_profile=None,
            transport_profile_stamped=True,
            transport_identity="discord:app-2",
        )

        assert first != second

    def test_relay_tenant_discriminator_is_part_of_identity(self):
        first = dl.compute_obligation_id(
            "shared-session",
            "msg1",
            "hello",
            transport_platform="relay",
            transport_profile=None,
            transport_profile_stamped=True,
            route_scope_id="guild-1",
            route_user_id="user-1",
        )
        second = dl.compute_obligation_id(
            "shared-session",
            "msg1",
            "hello",
            transport_platform="relay",
            transport_profile=None,
            transport_profile_stamped=True,
            route_scope_id="guild-2",
            route_user_id="user-2",
        )

        assert first != second


class TestSweep:
    def test_live_owner_rows_never_claimed(self):
        _record()  # owner = this (live) process
        assert dl.sweep_recoverable() == []

    def test_dead_owner_pending_claimed_without_marker(self):
        _record()
        _orphan("ob-1")
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is False
        assert claimed[0]["attempts"] == 1
        # Claim re-stamps ownership: a second sweep in the same (live)
        # process must not double-claim.
        assert dl.sweep_recoverable() == []

    def test_claim_cas_includes_observed_process_start_time(self, monkeypatch):
        """PID reuse between liveness check and UPDATE cannot steal a live claim."""
        _record()
        _orphan("ob-1")

        class _RacingConnection:
            def __init__(self, conn):
                self._conn = conn
                self._raced = False

            def execute(self, sql, params=()):
                if "SET owner_pid=?" in sql and not self._raced:
                    self._raced = True
                    self._conn.execute(
                        "UPDATE delivery_obligations "
                        "SET owner_pid=?, owner_started_at=? WHERE obligation_id=?",
                        (999999999, 2, "ob-1"),
                    )
                return self._conn.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        real_transaction = dl._transaction

        @dl.contextmanager
        def racing_transaction():
            with real_transaction() as conn:
                yield _RacingConnection(conn)

        monkeypatch.setattr(dl, "_transaction", racing_transaction)

        assert dl.sweep_recoverable() == []
        with dl._connect() as conn:
            owner_pid, owner_started_at, attempts = conn.execute(
                "SELECT owner_pid, owner_started_at, attempts "
                "FROM delivery_obligations WHERE obligation_id=?",
                ("ob-1",),
            ).fetchone()
        assert (owner_pid, owner_started_at, attempts) == (999999999, 2, 0)


class TestPrune:
    def test_old_delivered_rows_pruned(self):
        _record()
        dl.mark_delivered("ob-1")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET updated_at=? WHERE obligation_id=?",
                (time.time() - dl._RETENTION_SECONDS - 60, "ob-1"),
            )
        dl._prune()
        assert _row("ob-1") is None


class TestLedgerEnabled:
    def test_default_on(self):
        assert dl.ledger_enabled({}) is True
        assert dl.ledger_enabled({"gateway": {}}) is True


class TestGatewayRedeliverySweep:
    """Drive the real GatewayRunner._redeliver_pending_obligations."""

    @staticmethod
    def _runner(adapter=None):
        from gateway.config import Platform
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter} if adapter else {}
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @staticmethod
    def _adapter(success=True):
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=MagicMock(success=success, error="" if success else "nope")
        )
        return adapter

    @pytest.mark.asyncio
    async def test_pending_redelivers_plain_and_clears_resume(self):
        _record()  # pending
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        sent = adapter.send.call_args.kwargs
        assert sent["content"] == "the final answer"  # no marker
        assert sent["metadata"] == {"thread_id": "171.001"}
        assert _row("ob-1")["state"] == "delivered"
        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )

    @pytest.mark.asyncio
    async def test_failed_resume_clear_still_fences_auto_resume(self):
        from datetime import datetime

        from gateway.config import Platform
        from gateway.session import SessionEntry, SessionSource

        _record()
        _orphan("ob-1")
        runner = self._runner()
        setattr(
            runner._async_session_store,
            "clear_resume_pending",
            AsyncMock(side_effect=OSError("disk")),
        )
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="C1",
            chat_type="channel",
        )
        entry = SessionEntry(
            session_key="agent:main:slack:channel:C1",
            session_id="sid",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            origin=source,
            resume_pending=True,
            resume_reason="restart_interrupted",
        )
        store = MagicMock()
        store._lock = threading.Lock()
        store._entries = {entry.session_key: entry}
        store._ensure_loaded_locked = MagicMock()
        runner.session_store = store
        runner._AUTO_RESUME_REASONS = frozenset({"restart_interrupted"})

        assert await runner._redeliver_pending_obligations() == 0
        assert runner._schedule_resume_pending_sessions() == 0

    @pytest.mark.asyncio
    async def test_attempting_redelivers_with_marker(self):
        _record()
        dl.mark_attempting("ob-1")
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        sent = adapter.send.call_args.kwargs
        assert sent["content"].startswith(dl.RECOVERED_MARKER)
        assert sent["content"].endswith("the final answer")

    @pytest.mark.parametrize(
        ("send_success", "ledger_method"),
        [(True, "mark_delivered"), (False, "mark_failed")],
    )
    @pytest.mark.asyncio
    async def test_slow_state_update_does_not_block_event_loop(
        self, send_success, ledger_method
    ):
        import asyncio

        _record()
        _orphan("ob-1")
        runner = self._runner(self._adapter(success=send_success))
        slow_update, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch.object(dl, ledger_method, side_effect=slow_update):
            await asyncio.gather(
                runner._redeliver_pending_obligations(), event_loop_witness()
            )

        assert blocked_event_loop == []


class TestAttemptsOnlySpentOnRealSends:
    """``attempts`` is the redelivery budget — it must buy a send.

    ``self.adapters`` only holds a platform after its ``connect()`` succeeded,
    and the sweep claimed every dead-owner row regardless. A platform that
    failed to connect this boot therefore burned one attempt per boot while
    the caller's ``adapter is None`` branch skipped it without sending — so
    after MAX_ATTEMPTS boots the row abandoned having never been sent once,
    losing exactly the response the ledger exists to guarantee. That failure
    correlates with the crash that created the obligation: the network
    trouble that killed the send tends to still be there on the next boot.
    """

    def test_absent_platform_does_not_burn_attempts(self):
        _record(platform="telegram")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            assert dl.sweep_recoverable(deliverable_platforms={"discord"}) == []

        row = dl.debug_rows()
        assert "abandoned" not in row
        with dl._connect() as conn:
            state, attempts = conn.execute(
                "SELECT state, attempts FROM delivery_obligations "
                "WHERE obligation_id=?", ("ob-1",),
            ).fetchone()
        assert attempts == 0, "an unsendable boot must not spend the budget"
        assert state == "attempting"

    def test_row_still_delivers_once_its_platform_returns(self):
        _record(platform="telegram")
        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            dl.sweep_recoverable(deliverable_platforms={"discord"})

        _orphan("ob-1")
        claimed = dl.sweep_recoverable(deliverable_platforms={"telegram"})
        assert len(claimed) == 1
        assert claimed[0]["attempts"] == 1


class TestUnconnectedPlatformKeepsItsBudget:
    """End-to-end through the real runner: boots where the platform failed to
    connect must not consume the row's redelivery budget."""

    @staticmethod
    def _runner_without_slack():
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}  # slack failed to connect this boot
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @pytest.mark.asyncio
    async def test_row_survives_boots_where_its_platform_is_down(self):
        _record(platform="slack")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 1):
            _orphan("ob-1")
            runner = self._runner_without_slack()
            assert await runner._redeliver_pending_obligations() == 0

        assert _row("ob-1")["state"] != "abandoned", (
            "the obligation was abandoned without a single send being attempted"
        )
        assert _row("ob-1")["attempts"] == 0

