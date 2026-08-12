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

Final response bundles fail closed when their durable row or a pre-send
operation marker cannot be established. Post-send checkpoint failures remain
visible as acknowledgement-ambiguous partial state rather than being erased.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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


def mark_attempting(obligation_id: str) -> bool:
    """Transition an existing owed row to attempting; return whether it existed."""
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state='attempting', updated_at=?, last_error=NULL
               WHERE obligation_id=?
                 AND state IN ('pending', 'failed', 'attempting')""",
            (time.time(), obligation_id),
        )
    return bool(cursor.rowcount)


def mark_delivered(obligation_id: str) -> None:
    _update_state(obligation_id, "delivered")


def mark_failed(obligation_id: str, error: str = "") -> None:
    _update_state(obligation_id, "failed", error=error)


def mark_partial_failed(obligation_id: str, error: str = "") -> None:
    """Preserve a partially delivered bundle without automatic whole replay."""
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state='partial', attempts=?, updated_at=?, last_error=?
               WHERE obligation_id=?""",
            (
                MAX_ATTEMPTS,
                time.time(),
                (error or "partial response bundle")[:500],
                obligation_id,
            ),
        )


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


def mark_bundle_operation_completed(obligation_id: str, operation_key: str) -> bool:
    """Durably checkpoint one ACKed physical operation within a bundle."""
    return mark_bundle_operations_completed(obligation_id, [operation_key])


def mark_bundle_operations_completed(
    obligation_id: str, operation_keys: List[str]
) -> bool:
    """Atomically checkpoint obligations fulfilled by one physical operation."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT operation, payload_json FROM delivery_obligations "
            "WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()
        if not row or row[0] != "response_bundle" or not row[1]:
            return False
        try:
            payload = json.loads(str(row[1]))
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
               WHERE obligation_id=?""",
            (json.dumps(payload, sort_keys=True), time.time(), obligation_id),
        )
    return bool(cursor.rowcount)


def mark_bundle_operation_attempting(
    obligation_id: str, operation_key: str
) -> bool:
    """Durably identify the exact physical operation about to start."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT operation, payload_json FROM delivery_obligations "
            "WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()
        if not row or row[0] != "response_bundle" or not row[1]:
            return False
        try:
            payload = json.loads(str(row[1]))
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
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET payload_json=?, state='attempting', updated_at=?, last_error=NULL
               WHERE obligation_id=?
                 AND state IN ('pending', 'failed', 'attempting')""",
            (json.dumps(payload, sort_keys=True), time.time(), obligation_id),
        )
    return bool(cursor.rowcount)


def update_bundle_payload(
    obligation_id: str, updates: Dict[str, Any]
) -> bool:
    """Merge durable routing-independent bundle metadata before delivery."""
    with _DB_LOCK, _transaction() as conn:
        row = conn.execute(
            "SELECT operation, payload_json FROM delivery_obligations "
            "WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()
        if not row or row[0] != "response_bundle" or not row[1]:
            return False
        try:
            payload = json.loads(str(row[1]))
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        payload.update(dict(updates))
        payload["operation_keys"] = response_bundle_operation_keys(payload)
        cursor = conn.execute(
            "UPDATE delivery_obligations SET payload_json=?, updated_at=? "
            "WHERE obligation_id=?",
            (json.dumps(payload, sort_keys=True), time.time(), obligation_id),
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


def release_claim(obligation_id: str) -> bool:
    """Release this process's recovery claim without spending an attempt."""
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET owner_pid=NULL, owner_started_at=NULL,
                   attempts=CASE WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                   updated_at=?
               WHERE obligation_id=? AND owner_pid=?
                 AND (owner_started_at IS ? OR owner_started_at=?)""",
            (time.time(), obligation_id, pid, started, started),
        )
    return bool(cursor.rowcount)


def _update_state(obligation_id: str, state: str, error: str = "") -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=?""",
            (state, time.time(), error[:500] if error else None, obligation_id),
        )


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
    deliverable_routes: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for
    redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments
    ``attempts``, so a second gateway racing the same sweep cannot
    double-claim (the UPDATE is guarded on the previous owner stamp).
    Rows over the attempts cap or older than the stale cutoff transition to
    'abandoned' instead of being returned.

    ``deliverable_platforms`` (platform value strings) restricts claiming to
    platforms the caller can actually send on this boot.  ``attempts`` is the
    redelivery budget, so it must only be spent on a real send: a platform
    that failed to connect would otherwise burn one attempt per boot and hit
    the cap having never been sent once.  Rows for absent platforms are left
    untouched for a later boot; the stale cutoff still bounds them.
    """
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      transport_platform, transport_profile,
                      transport_profile_stamped, transport_identity,
                      route_scope_id, route_user_id, route_chat_type,
                      operation, payload_json, sequence_no,
                      owner_pid, owner_started_at
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
        ) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=? WHERE obligation_id=?""",
                    (now, oid),
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
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1,
                       updated_at=?
                   WHERE obligation_id=?
                     AND (owner_pid IS ? OR owner_pid=?)
                     AND (owner_started_at IS ? OR owner_started_at=?)""",
                (
                    pid,
                    started,
                    now,
                    oid,
                    owner_pid,
                    owner_pid,
                    owner_started_at,
                    owner_started_at,
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
                    "attempts": attempts + 1,
                })
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
                conn.execute(
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
