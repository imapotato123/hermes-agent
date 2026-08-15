"""Durable delivery-obligation ledger for gateway final responses.

A final agent response that was generated but not yet confirmed-delivered
to the messaging platform is the one artifact the gateway can lose without
a trace: the turn already burned its tokens, the text exists only in a
Python local, and a crash / planned restart between finalize and platform
ACK drops it silently (#58818, #41696, #63695).

This module records a small durable row per outbound final response in the
shared ``state.db`` (same file and conventions as
``tools.async_delegation`` — WAL, owner pid + process-start-time liveness,
bounded retention). The gateway writes three checkpoints around the send:

    record_obligation()   state='pending'     before any send attempt
    mark_attempting()     state='attempting'  immediately before the await
    mark_delivered() /    state='delivered'   only on SendResult.success
    mark_failed()         state='failed'      on a definitive rejection

On startup, ``sweep_recoverable()`` claims rows whose owning process is
dead and hands them to the gateway for redelivery. Crash semantics are
explicit about ambiguity (the contract review of the earlier
delivery-outbox attempt, #61790, closed it for silently resending
ambiguous sends):

- ``pending``     — the send never started: redeliver plainly, no dup risk.
- ``attempting``  — crashed mid-await: the platform MAY already have the
  message. Redelivered WITH a visible recovered-reply marker so the
  contract is honest at-least-once, never a silent duplicate.
- ``failed``      — definitively rejected once; the restart is a natural
  retry boundary. Also carries the marker.
- ``delivered``   — nothing to do; retention prunes.

Poison rows cannot spin: attempts are capped, stale rows expire, and both
transition to ``abandoned`` (kept briefly for inspection, then pruned).

The pending and attempting checkpoints are an egress gate for final output:
an uncheckpointed physical send is not a durable delivery attempt. Recovery
and diagnostic callers may still handle ledger failures explicitly.
Final response bundles fail closed when their durable row or a pre-send
operation marker cannot be established. Post-send checkpoint failures remain
visible as acknowledgement-ambiguous partial state rather than being erased.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# Redelivery policy knobs (module constants; deliberately not config — the
# ledger itself is gated by ``gateway.delivery_ledger`` and these bounds
# only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

# Visible prefix for redeliveries that might duplicate an already-received
# message (crash mid-send / post-rejection retry). Honest at-least-once.
RECOVERED_MARKER = (
    "♻️ Recovered reply — the gateway restarted during delivery, "
    "so this may be a duplicate:\n\n"
)


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (delivery_ledger)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
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
            recovery_claim TEXT,
            recovery_attempt_charged INTEGER NOT NULL DEFAULT 0,
            transport_platform TEXT,
            transport_profile TEXT,
            transport_profile_stamped INTEGER NOT NULL DEFAULT 0,
            transport_identity TEXT,
            route_scope_id TEXT,
            route_user_id TEXT,
            route_chat_type TEXT,
            operation TEXT NOT NULL DEFAULT 'text',
            payload_json TEXT,
            sequence_no INTEGER NOT NULL DEFAULT 0,
            last_error TEXT
        )"""
    )
    # Existing state.db files predate transport-owner persistence. Additive
    # migration keeps their rows explicitly legacy/unstamped, preserving their
    # historical primary route without weakening newly stamped multiplex rows.
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(delivery_obligations)").fetchall()
    }
    if "transport_profile" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN transport_profile TEXT"
        )
    if "transport_platform" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN transport_platform TEXT"
        )
    if "transport_profile_stamped" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN "
            "transport_profile_stamped INTEGER NOT NULL DEFAULT 0"
        )
    if "transport_identity" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN transport_identity TEXT"
        )
    if "recovery_claim" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN recovery_claim TEXT"
        )
    if "recovery_attempt_charged" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN "
            "recovery_attempt_charged INTEGER NOT NULL DEFAULT 0"
        )
    for column in ("route_scope_id", "route_user_id", "route_chat_type"):
        if column not in columns:
            conn.execute(
                f"ALTER TABLE delivery_obligations ADD COLUMN {column} TEXT"
            )
    if "operation" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN "
            "operation TEXT NOT NULL DEFAULT 'text'"
        )
    if "payload_json" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN payload_json TEXT"
        )
    if "sequence_no" not in columns:
        conn.execute(
            "ALTER TABLE delivery_obligations ADD COLUMN "
            "sequence_no INTEGER NOT NULL DEFAULT 0"
        )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back the
    transaction; they do not close the connection. Using ``with _connect()``
    alone therefore leaks a connection — and its WAL/SHM file descriptors — on
    every call, deferring the close to the garbage collector. On a long-running
    gateway that exhausts ``RLIMIT_NOFILE`` (the cron-ledger sibling of this
    bug was #69567 / PR #69594). ``record_obligation`` runs on every outbound
    final response, so this ledger is the highest-frequency leaker.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        # No such process (or unreadable) — treat unreadable-but-extant
        # processes as alive only if the pid exists.
        try:
            os.kill(pid, 0)  # windows-footgun: ok — EPERM counts as alive below
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(
    session_key: str,
    message_ref: str,
    content: str,
    *,
    transport_platform: Optional[str] = None,
    transport_profile: Optional[str] = None,
    transport_profile_stamped: bool = False,
    transport_identity: Optional[str] = None,
    route_scope_id: Optional[str] = None,
    route_user_id: Optional[str] = None,
    operation: str = "text",
    payload_json: Optional[str] = None,
) -> str:
    """Stable id for one turn, payload, and transport credential owner."""
    payload_body = payload_json if payload_json is not None else content
    is_legacy_text = operation == "text" and payload_json is None
    payload = (
        f"{session_key}|{message_ref}|{content}"
        if is_legacy_text
        else f"{session_key}|{message_ref}|{operation}|{payload_body}"
    )
    if transport_profile_stamped:
        payload = (
            f"{session_key}|{message_ref}|"
            f"stamped:{transport_platform or ''}:"
            f"{transport_profile or ''}:"
            f"{transport_identity or ''}:"
            f"{route_scope_id or ''}:{route_user_id or ''}|"
            + (
                content
                if is_legacy_text
                else f"{operation}|{payload_body}"
            )
        )
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(
    *,
    obligation_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
    transport_platform: Optional[str] = None,
    transport_profile: Optional[str] = None,
    transport_profile_stamped: bool = False,
    transport_identity: Optional[str] = None,
    route_scope_id: Optional[str] = None,
    route_user_id: Optional[str] = None,
    route_chat_type: Optional[str] = None,
    operation: str = "text",
    payload_json: Optional[str] = None,
    sequence_no: int = 0,
) -> None:
    """Record a final response as owed to one transport owner."""
    now = time.time()
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, transport_platform, transport_profile,
                transport_profile_stamped, transport_identity, route_scope_id,
                route_user_id, route_chat_type, operation, payload_json, sequence_no)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obligation_id,
                session_key,
                platform,
                str(chat_id),
                str(thread_id) if thread_id else None,
                content,
                now,
                now,
                pid,
                started,
                transport_platform,
                transport_profile,
                1 if transport_profile_stamped else 0,
                transport_identity,
                route_scope_id,
                route_user_id,
                route_chat_type,
                str(operation or "text"),
                payload_json,
                int(sequence_no),
            ),
        )
    _prune()


def mark_attempting(
    obligation_id: str, *, recovery_claim: Optional[str] = None
) -> bool:
    if recovery_claim is not None:
        pid, started = _owner_stamp()
        with _DB_LOCK, _transaction() as conn:
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='attempting', attempts=attempts+1,
                       recovery_attempt_charged=1, updated_at=?, last_error=NULL
                   WHERE obligation_id=? AND owner_pid=?
                     AND (owner_started_at IS ? OR owner_started_at=?)
                     AND recovery_claim=?
                     AND recovery_attempt_charged=0
                     AND attempts < ?
                     AND state IN ('pending', 'attempting', 'failed')""",
                (
                    time.time(),
                    obligation_id,
                    pid,
                    started,
                    started,
                    recovery_claim,
                    MAX_ATTEMPTS,
                ),
            )
        return bool(cursor.rowcount)
    return _update_state(obligation_id, "attempting")


def mark_delivered(
    obligation_id: str, *, recovery_claim: Optional[str] = None
) -> bool:
    return _update_state(
        obligation_id, "delivered", recovery_claim=recovery_claim
    )


def mark_failed(
    obligation_id: str,
    error: str = "",
    *,
    recovery_claim: Optional[str] = None,
) -> bool:
    return _update_state(
        obligation_id,
        "failed",
        error=error,
        recovery_claim=recovery_claim,
    )


def mark_partial_failed(
    obligation_id: str,
    error: str = "",
    *,
    recovery_claim: Optional[str] = None,
) -> bool:
    """Preserve a partially delivered bundle without automatic whole replay."""
    owner_where, owner_params = _bundle_owner_predicate(recovery_claim)
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state='partial', attempts=?, updated_at=?, last_error=?,
                   recovery_claim=NULL, recovery_attempt_charged=0
               WHERE obligation_id=? AND operation='response_bundle' AND """
            + owner_where,
            [
                MAX_ATTEMPTS,
                time.time(),
                (error or "partial response bundle")[:500],
                obligation_id,
                *owner_params,
            ],
        )
    return bool(cursor.rowcount)


def response_bundle_operation_keys(payload: Dict[str, Any]) -> List[str]:
    """Return stable physical-operation keys for a v1 response bundle."""
    keys: List[str] = []
    text = str(payload.get("text") or "")
    if payload.get("auto_tts") and text:
        segment_count = int(payload.get("auto_tts_segment_count") or 1)
        keys.extend(f"auto_tts:{index}" for index in range(segment_count))
    if text:
        keys.append("text")
    keys.extend(
        f"images:{index}"
        for index, _item in enumerate(payload.get("images") or [])
    )
    force_document = bool(payload.get("force_document_attachments"))
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    image_file_count = 0
    media_count = 0
    local_count = 0
    for item in payload.get("media_files") or []:
        path = str(item[0]) if isinstance(item, (list, tuple)) and item else ""
        is_voice = bool(item[1]) if isinstance(item, (list, tuple)) and len(item) > 1 else False
        if os.path.splitext(path)[1].lower() in image_exts and not is_voice and not force_document:
            image_file_count += 1
        else:
            media_count += 1
    for item in payload.get("local_files") or []:
        path = str(item)
        if os.path.splitext(path)[1].lower() in image_exts and not force_document:
            image_file_count += 1
        else:
            local_count += 1
    keys.extend(
        f"image_files:{index}" for index in range(image_file_count)
    )
    keys.extend(f"media:{index}" for index in range(media_count))
    keys.extend(f"local:{index}" for index in range(local_count))
    return keys


def _bundle_owner_predicate(
    recovery_claim: Optional[str],
) -> tuple[str, List[Any]]:
    """Return the exact live producer/recovery owner predicate for a bundle."""
    pid, started = _owner_stamp()
    predicate = (
        "owner_pid=? AND (owner_started_at IS ? OR owner_started_at=?)"
    )
    params: List[Any] = [pid, started, started]
    if recovery_claim is None:
        predicate += " AND recovery_claim IS NULL"
    else:
        predicate += " AND recovery_claim=?"
        params.append(recovery_claim)
    return predicate, params


def mark_bundle_operation_completed(
    obligation_id: str,
    operation_key: str,
    *,
    recovery_claim: Optional[str] = None,
) -> bool:
    """Durably checkpoint one ACKed physical operation within a bundle."""
    return mark_bundle_operations_completed(
        obligation_id,
        [operation_key],
        recovery_claim=recovery_claim,
    )


def mark_bundle_operations_completed(
    obligation_id: str,
    operation_keys: List[str],
    *,
    recovery_claim: Optional[str] = None,
) -> bool:
    """Atomically checkpoint obligations fulfilled by one physical operation."""
    owner_where, owner_params = _bundle_owner_predicate(recovery_claim)
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT operation, payload_json, state FROM delivery_obligations "
            f"WHERE obligation_id=? AND {owner_where}",
            [obligation_id, *owner_params],
        ).fetchone()
        if not row or row[0] != "response_bundle" or not row[1]:
            return False
        selected_payload_json = str(row[1])
        selected_state = str(row[2])
        try:
            payload = json.loads(selected_payload_json)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        # ``operation_keys`` is a cache, never an authority boundary. Always
        # derive it from the physical plan so stale/malformed rows cannot
        # authorize a nonexistent checkpoint or omit an obligation.
        expected = response_bundle_operation_keys(payload)
        payload["operation_keys"] = expected
        keys = [str(key) for key in operation_keys]
        if not keys or any(key not in expected for key in keys):
            return False
        completed = payload.get("completed_operations")
        if not isinstance(completed, list):
            completed = []
        for key in keys:
            if key not in completed:
                completed.append(key)
        payload["completed_operations"] = completed
        if payload.get("attempting_operation") in keys:
            payload.pop("attempting_operation", None)
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET payload_json=?, state='pending', updated_at=?, last_error=NULL
               WHERE obligation_id=? AND operation='response_bundle'
                 AND payload_json=? AND state=? AND """
            + owner_where,
            [
                json.dumps(payload, sort_keys=True),
                time.time(),
                obligation_id,
                selected_payload_json,
                selected_state,
                *owner_params,
            ],
        )
    return bool(cursor.rowcount)


def mark_bundle_operation_attempting(
    obligation_id: str,
    operation_key: str,
    *,
    recovery_claim: Optional[str] = None,
) -> bool:
    """Durably identify the exact physical operation about to start."""
    owner_where, owner_params = _bundle_owner_predicate(recovery_claim)
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT operation, payload_json, state FROM delivery_obligations "
            f"WHERE obligation_id=? AND {owner_where}",
            [obligation_id, *owner_params],
        ).fetchone()
        if not row or row[0] != "response_bundle" or not row[1]:
            return False
        selected_payload_json = str(row[1])
        selected_state = str(row[2])
        try:
            payload = json.loads(selected_payload_json)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        # ``operation_keys`` is a cache, never an authority boundary. Always
        # derive it from the physical plan so stale/malformed rows cannot
        # authorize a nonexistent checkpoint or omit an obligation.
        expected = response_bundle_operation_keys(payload)
        payload["operation_keys"] = expected
        key = str(operation_key)
        completed = payload.get("completed_operations")
        if not isinstance(completed, list):
            completed = []
        if key not in expected or key in completed:
            return False
        payload["attempting_operation"] = key
        charge_clause = ""
        budget_clause = ""
        budget_params: List[Any] = []
        if recovery_claim is not None:
            charge_clause = (
                ", attempts=CASE WHEN recovery_attempt_charged=0 "
                "THEN attempts+1 ELSE attempts END, "
                "recovery_attempt_charged=1"
            )
            budget_clause = " AND (recovery_attempt_charged=1 OR attempts < ?)"
            budget_params.append(MAX_ATTEMPTS)
        cursor = conn.execute(
            "UPDATE delivery_obligations "
            "SET payload_json=?, state='attempting', updated_at=?, "
            "last_error=NULL"
            + charge_clause
            + " WHERE obligation_id=? AND operation='response_bundle' "
            "AND payload_json=? AND state=? "
            "AND state IN ('pending', 'failed', 'attempting') AND "
            + owner_where
            + budget_clause,
            [
                json.dumps(payload, sort_keys=True),
                time.time(),
                obligation_id,
                selected_payload_json,
                selected_state,
                *owner_params,
                *budget_params,
            ],
        )
    return bool(cursor.rowcount)


def update_bundle_payload(
    obligation_id: str,
    updates: Dict[str, Any],
    *,
    recovery_claim: Optional[str] = None,
) -> bool:
    """Merge durable routing-independent bundle metadata before delivery."""
    owner_where, owner_params = _bundle_owner_predicate(recovery_claim)
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT operation, payload_json, state FROM delivery_obligations "
            f"WHERE obligation_id=? AND {owner_where}",
            [obligation_id, *owner_params],
        ).fetchone()
        if not row or row[0] != "response_bundle" or not row[1]:
            return False
        selected_payload_json = str(row[1])
        selected_state = str(row[2])
        try:
            payload = json.loads(selected_payload_json)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        payload.update(dict(updates))
        payload["operation_keys"] = response_bundle_operation_keys(payload)
        cursor = conn.execute(
            "UPDATE delivery_obligations SET payload_json=?, updated_at=? "
            "WHERE obligation_id=? AND operation='response_bundle' "
            "AND payload_json=? AND state=? AND "
            + owner_where,
            [
                json.dumps(payload, sort_keys=True),
                time.time(),
                obligation_id,
                selected_payload_json,
                selected_state,
                *owner_params,
            ],
        )
    return bool(cursor.rowcount)


def undelivered_session_keys() -> set[str]:
    """Return sessions whose completed answer is still durably owed."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT DISTINCT session_key FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed', 'partial')"""
        ).fetchall()
    return {str(row[0]) for row in rows if row and row[0]}


def release_claim(
    obligation_id: str,
    *,
    consume_attempt: bool = False,
    recovery_claim: Optional[str] = None,
    restore_state: Optional[str] = None,
) -> bool:
    """Release this process's recovery claim.

    Before transport egress, ``consume_attempt=False`` restores a retry charge
    made by this claim, if any. Claim acquisition itself does not spend budget.
    Once egress may have started, ``consume_attempt=True`` clears only the live
    owner stamp so a later sweep preserves both the spent attempt and the row's
    ambiguity state. ``restore_state`` is used only for cancellation during the
    pre-egress attempting checkpoint.

    The opaque ``recovery_claim`` is minted by ``sweep_recoverable()`` and
    prevents a delayed task in this same process from releasing a newer claim.
    """
    if not recovery_claim:
        return False
    if restore_state not in {None, "pending", "attempting", "failed"}:
        return False
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        set_clause = (
            "owner_pid=NULL, owner_started_at=NULL, recovery_claim=NULL, "
            "attempts=CASE "
            "WHEN ? THEN attempts "
            "WHEN recovery_attempt_charged=1 AND attempts > 0 THEN attempts - 1 "
            "ELSE attempts END, recovery_attempt_charged=0"
        )
        params: List[Any] = [consume_attempt]
        if restore_state is not None:
            set_clause += ", state=?"
            params.append(restore_state)
        set_clause += ", updated_at=?"
        params.extend(
            (
                time.time(),
                obligation_id,
                pid,
                started,
                started,
                recovery_claim,
            )
        )
        cursor = conn.execute(
            f"""UPDATE delivery_obligations
               SET {set_clause}
               WHERE obligation_id=? AND owner_pid=?
                 AND (owner_started_at IS ? OR owner_started_at=?)
                 AND recovery_claim=?""",
            params,
        )
    return bool(cursor.rowcount)


def release_original_owner(
    obligation_id: str,
    *,
    expected_state: str,
) -> bool:
    """Release a live first-send row so recovery can claim it immediately.

    Original producer rows have no recovery token. The exact process stamp,
    null-token predicate, and expected state form the CAS boundary, preventing
    a cancelled or stale producer from releasing a newer recovery claim.
    """
    if expected_state not in {"pending", "attempting", "failed"}:
        return False
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET owner_pid=NULL, owner_started_at=NULL, updated_at=?
               WHERE obligation_id=? AND state=? AND owner_pid=?
                 AND (owner_started_at IS ? OR owner_started_at=?)
                 AND recovery_claim IS NULL""",
            (
                time.time(),
                obligation_id,
                expected_state,
                pid,
                started,
                started,
            ),
        )
    return bool(cursor.rowcount)


def _update_state(
    obligation_id: str,
    state: str,
    error: str = "",
    *,
    recovery_claim: Optional[str] = None,
) -> bool:
    with _DB_LOCK, _transaction() as conn:
        params = [state, time.time(), error[:500] if error else None, obligation_id]
        where = "obligation_id=?"
        if recovery_claim is not None:
            pid, started = _owner_stamp()
            where += (
                " AND owner_pid=?"
                " AND (owner_started_at IS ? OR owner_started_at=?)"
                " AND recovery_claim=?"
            )
            params.extend((pid, started, started, recovery_claim))
        set_clause = "state=?, updated_at=?, last_error=?"
        if recovery_claim is not None and state in {"delivered", "failed"}:
            set_clause += ", recovery_claim=NULL, recovery_attempt_charged=0"
        cursor = conn.execute(
            f"""UPDATE delivery_obligations
                SET {set_clause}
                WHERE {where}""",
            params,
        )
    return bool(cursor.rowcount)


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
    deliverable_routes: Optional[set] = None,
    max_claims: Optional[int] = None,
    exclude_obligation_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for
    redelivery.

    Claiming atomically re-stamps the owner to THIS process without spending
    ``attempts``, so a second gateway racing the same sweep cannot double-claim.
    ``mark_attempting`` charges exactly once immediately before physical egress.
    Rows over the attempts cap or older than the stale cutoff transition to
    'abandoned' instead of being returned.

    ``deliverable_platforms`` (platform value strings) restricts claiming to
    platforms the caller can actually send on this boot.  ``attempts`` is the
    redelivery budget, so it must only be spent on a real send: a platform
    that failed to connect would otherwise burn one attempt per boot and hit
    the cap having never been sent once.  Rows for absent platforms are left
    untouched for a later boot; the stale cutoff still bounds them.
    """
    if max_claims is not None and max_claims <= 0:
        return []
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    excluded = exclude_obligation_ids or set()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      transport_platform, transport_profile,
                      transport_profile_stamped, transport_identity,
                      route_scope_id, route_user_id, route_chat_type,
                      operation, payload_json, sequence_no,
                      owner_pid, owner_started_at, updated_at, recovery_claim,
                      recovery_attempt_charged
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')
               ORDER BY created_at ASC, sequence_no ASC, rowid ASC"""
        ).fetchall()
        for (
            oid,
            session_key,
            platform,
            chat_id,
            thread_id,
            content,
            state,
            attempts,
            created_at,
            transport_platform,
            transport_profile,
            transport_profile_stamped,
            transport_identity,
            route_scope_id,
            route_user_id,
            route_chat_type,
            operation,
            payload_json,
            sequence_no,
            owner_pid,
            owner_started_at,
            updated_at,
            previous_recovery_claim,
            recovery_attempt_charged,
        ) in rows:
            if oid in excluded:
                continue
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
                cursor = conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=?, recovery_claim=NULL,
                           recovery_attempt_charged=0
                       WHERE obligation_id=? AND state=? AND attempts=?
                         AND created_at=? AND updated_at=?
                         AND (owner_pid IS ? OR owner_pid=?)
                         AND (owner_started_at IS ? OR owner_started_at=?)
                         AND recovery_claim IS ?
                         AND recovery_attempt_charged=?""",
                    (
                        now,
                        oid,
                        state,
                        attempts,
                        created_at,
                        updated_at,
                        owner_pid,
                        owner_pid,
                        owner_started_at,
                        owner_started_at,
                        previous_recovery_claim,
                        recovery_attempt_charged,
                    ),
                )
                if not cursor.rowcount:
                    logger.debug(
                        "obligation %s changed during stale-abandon decision", oid
                    )
                continue
            if (
                deliverable_platforms is not None
                and platform not in deliverable_platforms
            ):
                # No adapter for this platform this boot — the caller cannot
                # send, so claiming would spend an attempt on a no-op.
                continue
            route = (
                (transport_platform or platform)
                if transport_profile_stamped
                else platform,
                bool(transport_profile_stamped),
                transport_profile if transport_profile_stamped else None,
                transport_identity if transport_profile_stamped else None,
            )
            if deliverable_routes is not None and route not in deliverable_routes:
                # A different profile on the same platform is not a valid
                # credential route. Leave the row untouched for its owner.
                continue
            recovery_claim = secrets.token_hex(16)
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, recovery_claim=?,
                       recovery_attempt_charged=0, updated_at=?
                   WHERE obligation_id=? AND state=? AND attempts=?
                     AND created_at=? AND updated_at=?
                     AND (owner_pid IS ? OR owner_pid=?)
                     AND (owner_started_at IS ? OR owner_started_at=?)
                     AND recovery_claim IS ?
                     AND recovery_attempt_charged=?""",
                (
                    pid,
                    started,
                    recovery_claim,
                    now,
                    oid,
                    state,
                    attempts,
                    created_at,
                    updated_at,
                    owner_pid,
                    owner_pid,
                    owner_started_at,
                    owner_started_at,
                    previous_recovery_claim,
                    recovery_attempt_charged,
                ),
            )
            if cursor.rowcount:
                claimed.append({
                    "obligation_id": oid,
                    "session_key": session_key,
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    "transport_platform": transport_platform,
                    "transport_profile": transport_profile,
                    "transport_profile_stamped": bool(
                        transport_profile_stamped
                    ),
                    "transport_identity": transport_identity,
                    "route_scope_id": route_scope_id,
                    "route_user_id": route_user_id,
                    "route_chat_type": route_chat_type,
                    "operation": operation or "text",
                    "payload_json": payload_json,
                    "sequence_no": int(sequence_no or 0),
                    # pending = send never started, redeliver plainly;
                    # attempting/failed = ambiguous or rejected, carry marker.
                    "needs_marker": state != "pending",
                    "claimed_state": state,
                    "attempts": attempts,
                    "recovery_claim": recovery_claim,
                })
                if max_claims is not None and len(claimed) >= max_claims:
                    break
    return claimed


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    try:
        with _transaction() as conn:
            conn.execute(
                """DELETE FROM delivery_obligations
                   WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""",
                (cutoff,),
            )
            total = conn.execute(
                "SELECT COUNT(*) FROM delivery_obligations"
            ).fetchone()[0]
            excess = max(0, total - _MAX_ROWS)
            if excess:
                cursor = conn.execute(
                    """DELETE FROM delivery_obligations WHERE obligation_id IN (
                         SELECT obligation_id FROM delivery_obligations
                         WHERE state IN ('delivered', 'abandoned')
                         ORDER BY CASE state
                                    WHEN 'delivered' THEN 0
                                    WHEN 'abandoned' THEN 1
                                  END, updated_at ASC
                         LIMIT ?)""",
                    (excess,),
                )
                if cursor.rowcount < excess:
                    logger.warning(
                        "delivery ledger exceeds soft row cap with %d unresolved "
                        "obligation(s); live delivery debt was retained",
                        excess - cursor.rowcount,
                    )
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        gw = config.get("gateway") or {}
        value = gw.get("delivery_ledger", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True


def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0], "session": r[1], "state": r[2], "attempts": r[3],
                "created_at": r[4], "updated_at": r[5], "last_error": r[6],
            }
            for r in rows
        ],
        indent=2,
    )
