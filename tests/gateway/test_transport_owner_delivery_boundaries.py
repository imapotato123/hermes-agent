"""Transport-owner authority at runner and durable-delivery boundaries.

Every new user-visible operation must resolve the live adapter that owns the
inbound source. Explicitly stamped owners fail closed; unstamped sources keep
the legacy passed/default-adapter behavior.
"""

import asyncio
import json
import threading

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    EphemeralReply,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _source(*, stamped: bool = True, profile: str | None = "coder") -> SessionSource:
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        thread_id="171.001",
        message_id="m1",
    )
    if stamped:
        source._transport_profile = profile
        source._transport_platform = Platform.SLACK
    return source


def _adapter(name: str):
    return SimpleNamespace(
        name=name,
        platform=Platform.SLACK,
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        send=AsyncMock(return_value=SendResult(success=True, message_id=f"{name}-text")),
        send_multiple_images=AsyncMock(return_value=None),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id=f"{name}-voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id=f"{name}-doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id=f"{name}-video")),
    )


class _RetryBoundaryAdapter(BasePlatformAdapter):
    def __init__(self, name: str):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.SLACK)
        self._name = name
        self.physical_sends = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.physical_sends += 1
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}


def _runner(*, primary=None, coder=None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: primary} if primary is not None else {}
    runner._profile_adapters = (
        {"coder": {Platform.SLACK: coder}} if coder is not None else {}
    )
    runner._active_profile_name = lambda: "main"
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = (
        lambda source, anchor=None: {"thread_id": source.thread_id}
    )
    return runner


@pytest.mark.asyncio
async def test_response_bundle_replays_exact_owner_text_images_and_documents(
    isolated_ledger, tmp_path
):
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")
    payload = json.dumps(
        {"version": 1, "text": "answer",
         "images": [["https://example.test/a.png", "a"]],
         "media_files": [[str(document), False]], "local_files": [],
         "force_document_attachments": False, "auto_tts": False},
        sort_keys=True,
    )
    oid = dl.compute_obligation_id(
        "agent:coder:slack:dm:C1", "m1", "answer",
        transport_platform="slack", transport_profile="coder",
        transport_profile_stamped=True, operation="response_bundle",
        payload_json=payload,
    )
    dl.record_obligation(
        obligation_id=oid, session_key="agent:coder:slack:dm:C1",
        platform="slack", chat_id="C1", thread_id=None, content="answer",
        transport_platform="slack", transport_profile="coder",
        transport_profile_stamped=True, operation="response_bundle",
        payload_json=payload,
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?", (oid,)
        )
        conn.commit()
    live = _adapter("live")
    runner = _runner(coder=live)
    store = MagicMock(clear_resume_pending=AsyncMock(), _store=None)
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 1
    live.send.assert_awaited_once_with(
        chat_id="C1", content="answer", metadata={"thread_id": None}
    )
    live.send_multiple_images.assert_awaited_once()
    live.send_document.assert_awaited_once_with(
        chat_id="C1", file_path=str(document), caption=None,
        metadata={"thread_id": None}
    )


@pytest.mark.asyncio
async def test_recovery_checkpoints_each_prepared_text_post_before_later_failure(
    isolated_ledger,
):
    payload = {
        "version": 1,
        "text": "one two",
        "text_chunks": [
            {"content": "one", "recovered_content": "[recovered] one"},
            {"content": "two", "recovered_content": "[recovered] two"},
        ],
        "completed_operations": [],
    }
    row = {"obligation_id": "text-prefix", "chat_id": "C1"}
    live = _adapter("live")
    live.send_prepared_text_chunk = AsyncMock(
        side_effect=[
            SendResult(success=True, message_id="one"),
            SendResult(success=False, error="second rejected"),
        ]
    )
    runner = _runner(coder=live)

    with patch(
        "gateway.delivery_ledger.mark_bundle_operation_attempting",
        return_value=True,
    ) as attempting, patch(
        "gateway.delivery_ledger.mark_bundle_operations_completed",
        return_value=True,
    ) as completed:
        result = await runner._redeliver_response_bundle(
            row=row,
            payload=payload,
            adapter=live,
            source=_source(),
            metadata={"thread_id": None},
            platform=Platform.SLACK,
            legacy_transport=None,
        )

    assert result.success is False
    assert [call.args[1] for call in attempting.call_args_list] == [
        "text:0",
        "text:1",
    ]
    assert completed.call_args_list[0].args[1] == ["text:0"]
    assert [
        call.kwargs["content"]
        for call in live.send_prepared_text_chunk.await_args_list
    ] == ["one", "two"]


@pytest.mark.asyncio
async def test_recovery_uses_persisted_safe_marker_for_exact_ambiguous_text_post(
    isolated_ledger,
):
    payload = {
        "version": 1,
        "text": "one two",
        "text_chunks": [
            {"content": "one", "recovered_content": "safe-recovered-one"},
            {"content": "two", "recovered_content": "safe-recovered-two"},
        ],
        "completed_operations": ["text:0"],
        "attempting_operation": "text:1",
    }
    live = _adapter("live")
    live.send_prepared_text_chunk = AsyncMock(
        return_value=SendResult(success=True, message_id="two")
    )
    runner = _runner(coder=live)

    with patch(
        "gateway.delivery_ledger.mark_bundle_operation_attempting",
        return_value=True,
    ), patch(
        "gateway.delivery_ledger.mark_bundle_operations_completed",
        return_value=True,
    ):
        result = await runner._redeliver_response_bundle(
            row={"obligation_id": "text-ambiguous", "chat_id": "C1"},
            payload=payload,
            adapter=live,
            source=_source(),
            metadata={"thread_id": None},
            platform=Platform.SLACK,
            legacy_transport=None,
        )

    assert result.success is True
    live.send_prepared_text_chunk.assert_awaited_once()
    assert live.send_prepared_text_chunk.await_args.kwargs["content"] == (
        "safe-recovered-two"
    )


@pytest.mark.asyncio
async def test_failed_bundle_blocks_later_bundle_for_same_session(isolated_ledger):
    live = _adapter("live")
    live.send = AsyncMock(return_value=SendResult(success=False, error="no"))
    runner = _runner(coder=live)
    store = MagicMock(clear_resume_pending=AsyncMock(), _store=None)
    setattr(runner, "session_store", None)
    runner._async_session_store = store
    for index in (1, 2):
        payload = json.dumps(
            {"version": 1, "text": f"answer-{index}"}, sort_keys=True
        )
        oid = dl.compute_obligation_id(
            "same-session", f"m{index}", f"answer-{index}",
            transport_platform="slack", transport_profile="coder",
            transport_profile_stamped=True, operation="response_bundle",
            payload_json=payload,
        )
        dl.record_obligation(
            obligation_id=oid, session_key="same-session", platform="slack",
            chat_id="C1", thread_id=None, content=f"answer-{index}",
            transport_platform="slack", transport_profile="coder",
            transport_profile_stamped=True, operation="response_bundle",
            payload_json=payload, sequence_no=index,
        )
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, "
                "owner_started_at=1 WHERE obligation_id=?", (oid,)
            )
            conn.commit()

    assert await runner._redeliver_pending_obligations() == 0
    assert live.send.await_count == 1


@pytest.mark.asyncio
async def test_recovery_image_batch_failure_is_not_checkpointed(isolated_ledger):
    payload = json.dumps(
        {
            "version": 1,
            "text": "",
            "images": [["https://example.test/a.png", "a"]],
            "operation_keys": ["images"],
            "completed_operations": [],
        },
        sort_keys=True,
    )
    oid = dl.compute_obligation_id(
        "agent:coder:slack:dm:C1", "m1", "",
        transport_platform="slack", transport_profile="coder",
        transport_profile_stamped=True, operation="response_bundle",
        payload_json=payload,
    )
    dl.record_obligation(
        obligation_id=oid, session_key="agent:coder:slack:dm:C1",
        platform="slack", chat_id="C1", thread_id=None, content="",
        transport_platform="slack", transport_profile="coder",
        transport_profile_stamped=True, operation="response_bundle",
        payload_json=payload,
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?", (oid,)
        )
        conn.commit()
    live = _adapter("live")
    live.send_multiple_images = AsyncMock(
        return_value=SendResult(success=False, error="batch rejected")
    )
    runner = _runner(coder=live)
    store = MagicMock(clear_resume_pending=AsyncMock(), _store=None)
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 0
    with dl._connect() as conn:
        state, payload_json = conn.execute(
            "SELECT state, payload_json FROM delivery_obligations "
            "WHERE obligation_id=?", (oid,)
        ).fetchone()
    assert state == "failed"
    assert json.loads(payload_json)["completed_operations"] == []


@pytest.mark.asyncio
async def test_recovery_checkpoints_each_image_before_later_failure(
    isolated_ledger,
):
    payload = {
        "version": 1,
        "text": "",
        "images": [
            ["https://example.test/one.png", "one"],
            ["https://example.test/two.png", "two"],
        ],
        "operation_keys": ["images:0", "images:1"],
        "completed_operations": [],
    }
    row = {"obligation_id": "image-prefix", "chat_id": "C1"}
    live = _adapter("live")
    live.send_multiple_images = AsyncMock(
        side_effect=[None, SendResult(success=False, error="second rejected")]
    )
    runner = _runner(coder=live)

    with patch(
        "gateway.delivery_ledger.mark_bundle_operation_attempting",
        return_value=True,
    ) as attempting, patch(
        "gateway.delivery_ledger.mark_bundle_operations_completed",
        return_value=True,
    ) as completed:
        result = await runner._redeliver_response_bundle(
            row=row,
            payload=payload,
            adapter=live,
            source=_source(),
            metadata={"thread_id": None},
            platform=Platform.SLACK,
            legacy_transport=None,
        )

    assert result.success is False
    assert live.send_multiple_images.await_count == 2
    assert [
        call.kwargs["images"] for call in live.send_multiple_images.await_args_list
    ] == [
        [("https://example.test/one.png", "one")],
        [("https://example.test/two.png", "two")],
    ]
    assert [call.args[1] for call in attempting.call_args_list] == [
        "images:0",
        "images:1",
    ]
    assert completed.call_args_list[0].args[1] == ["images:0"]


@pytest.mark.asyncio
async def test_recovery_missing_tts_capability_does_not_skip_to_text(
    isolated_ledger,
):
    payload = {
        "version": 1,
        "text": "speak this",
        "auto_tts": True,
        "auto_tts_segment_count": 1,
        "operation_keys": ["auto_tts:0", "text"],
        "completed_operations": [],
    }
    row = {
        "obligation_id": "tts-missing",
        "chat_id": "C1",
    }
    live = _adapter("live")
    runner = _runner(coder=live)

    with patch(
        "tools.tts_tool.check_tts_requirements", return_value=False
    ), patch(
        "gateway.delivery_ledger.mark_bundle_operation_attempting",
        return_value=True,
    ), patch(
        "gateway.delivery_ledger.mark_bundle_operations_completed",
        return_value=True,
    ):
        result = await runner._redeliver_response_bundle(
            row=row,
            payload=payload,
            adapter=live,
            source=_source(),
            metadata={"thread_id": None},
            platform=Platform.SLACK,
            legacy_transport=None,
        )

    assert result.success is False
    live.send.assert_not_awaited()
    live.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_tts_cleans_all_generated_segments(
    isolated_ledger, tmp_path, monkeypatch
):
    requested = tmp_path / "requested.mp3"
    one = tmp_path / "one.mp3"
    two = tmp_path / "two.mp3"
    one.write_bytes(b"one")
    two.write_bytes(b"two")
    payload = {
        "version": 1,
        "text": "speak this",
        "auto_tts": True,
        "auto_tts_segment_count": 2,
        "operation_keys": ["auto_tts:0", "auto_tts:1", "text"],
        "completed_operations": [],
    }
    row = {
        "obligation_id": "tts-cleanup",
        "chat_id": "C1",
    }
    live = _adapter("live")
    live.play_tts = AsyncMock(
        return_value=SendResult(success=True, message_id="tts")
    )
    runner = _runner(coder=live)
    monkeypatch.setattr(
        "gateway.run.build_auto_tts_output_path", lambda _platform: str(requested)
    )

    def fake_tts(*, text, output_path):
        return json.dumps(
            {"success": True, "file_paths": [str(one), str(two)]}
        )

    with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=fake_tts
    ), patch(
        "gateway.delivery_ledger.mark_bundle_operation_attempting",
        return_value=True,
    ), patch(
        "gateway.delivery_ledger.mark_bundle_operations_completed",
        return_value=True,
    ):
        result = await runner._redeliver_response_bundle(
            row=row,
            payload=payload,
            adapter=live,
            source=_source(),
            metadata={"thread_id": None},
            platform=Platform.SLACK,
            legacy_transport=None,
        )

    assert result.success is True
    assert live.play_tts.await_count == 2
    assert not requested.exists()
    assert not one.exists()
    assert not two.exists()


@pytest.mark.asyncio
async def test_recovery_cached_keys_cannot_omit_planned_media(
    isolated_ledger, tmp_path
):
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")
    payload = {
        "version": 1,
        "text": "answer",
        "media_files": [[str(document), False]],
        "operation_keys": ["text"],
        "completed_operations": [],
    }
    row = {"obligation_id": "cached-keys", "chat_id": "C1"}
    live = _adapter("live")
    runner = _runner(coder=live)

    with patch(
        "gateway.delivery_ledger.mark_bundle_operation_attempting",
        return_value=True,
    ), patch(
        "gateway.delivery_ledger.mark_bundle_operations_completed",
        return_value=True,
    ):
        result = await runner._redeliver_response_bundle(
            row=row,
            payload=payload,
            adapter=live,
            source=_source(),
            metadata={"thread_id": None},
            platform=Platform.SLACK,
            legacy_transport=None,
        )

    assert result.success is True
    live.send.assert_awaited_once()
    live.send_document.assert_awaited_once_with(
        chat_id="C1",
        file_path=str(document),
        caption=None,
        metadata={"thread_id": None},
    )


@pytest.mark.asyncio
async def test_partial_bundle_resumes_after_last_checkpoint(isolated_ledger):
    live = _adapter("live")
    live.send_document = AsyncMock(side_effect=RuntimeError("document down"))
    runner = _runner(coder=live)
    store = MagicMock(clear_resume_pending=AsyncMock(), _store=None)
    setattr(runner, "session_store", None)
    runner._async_session_store = store
    payload = json.dumps(
        {
            "version": 1,
            "text": "answer",
            "media_files": [["/tmp/report.pdf", False]],
            "operation_keys": ["text", "media:0"],
            "completed_operations": [],
        },
        sort_keys=True,
    )
    oid = dl.compute_obligation_id(
        "same-session", "m1", "answer", transport_platform="slack",
        transport_profile="coder", transport_profile_stamped=True,
        operation="response_bundle", payload_json=payload,
    )
    dl.record_obligation(
        obligation_id=oid, session_key="same-session", platform="slack",
        chat_id="C1", thread_id=None, content="answer",
        transport_platform="slack", transport_profile="coder",
        transport_profile_stamped=True, operation="response_bundle",
        payload_json=payload,
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?", (oid,)
        )
        conn.commit()

    assert await runner._redeliver_pending_obligations() == 0
    assert live.send.await_count == 1
    with dl._connect() as conn:
        state, payload_json = conn.execute(
            "SELECT state, payload_json FROM delivery_obligations "
            "WHERE obligation_id=?", (oid,)
        ).fetchone()
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?", (oid,)
        )
        conn.commit()
    assert state == "failed"
    assert json.loads(payload_json)["completed_operations"] == ["text"]

    live.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="doc")
    )
    assert await runner._redeliver_pending_obligations() == 1
    assert live.send.await_count == 1
    live.send_document.assert_awaited_once()
    with dl._connect() as conn:
        state = conn.execute(
            "SELECT state FROM delivery_obligations WHERE obligation_id=?", (oid,)
        ).fetchone()[0]
    assert state == "delivered"


@pytest.mark.asyncio
async def test_crash_mid_document_labels_only_ambiguous_operation(isolated_ledger):
    live = _adapter("live")
    runner = _runner(coder=live)
    store = MagicMock(clear_resume_pending=AsyncMock(), _store=None)
    setattr(runner, "session_store", None)
    runner._async_session_store = store
    payload = {
        "version": 1,
        "text": "answer",
        "media_files": [["/tmp/report.pdf", False]],
        "operation_keys": ["text", "media:0"],
        "completed_operations": ["text"],
        "attempting_operation": "media:0",
    }
    payload_json = json.dumps(payload, sort_keys=True)
    oid = dl.compute_obligation_id(
        "same-session", "m1", "answer", transport_platform="slack",
        transport_profile="coder", transport_profile_stamped=True,
        operation="response_bundle", payload_json=payload_json,
    )
    dl.record_obligation(
        obligation_id=oid, session_key="same-session", platform="slack",
        chat_id="C1", thread_id=None, content="answer",
        transport_platform="slack", transport_profile="coder",
        transport_profile_stamped=True, operation="response_bundle",
        payload_json=payload_json,
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET state='attempting', "
            "owner_pid=999999999, owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )
        conn.commit()

    assert await runner._redeliver_pending_obligations() == 1
    live.send.assert_not_awaited()
    live.send_document.assert_awaited_once_with(
        chat_id="C1", file_path="/tmp/report.pdf",
        caption=dl.RECOVERED_MARKER.strip(), metadata={"thread_id": None}
    )


def test_registered_owner_uses_physical_transport_platform():
    runner = object.__new__(GatewayRunner)
    relay = SimpleNamespace(
        platform=Platform.RELAY,
        fronts_platform=MagicMock(return_value=True),
        matches_transport_identity=MagicMock(return_value=True),
        prime_routing_source=MagicMock(),
    )
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="C1",
        scope_id="guild-1",
        user_id="user-1",
    )
    source._transport_profile = None
    source._transport_platform = Platform.RELAY
    setattr(source, "_transport_identity", "discord:app-1")

    assert runner._adapter_for_source(source) is relay
    relay.prime_routing_source.assert_called_once_with(source)


def test_relay_registration_keeps_profile_independent_owner_identity():
    runner = object.__new__(GatewayRunner)
    runner._backend_notice_state = None
    runner._active_profile_name = lambda: "main"
    relay = SimpleNamespace(platform=Platform.RELAY)

    runner._share_backend_notice_state(relay)

    assert relay._transport_profile is None


def test_relay_replacement_primes_routing_from_inflight_source():
    runner = object.__new__(GatewayRunner)
    relay = SimpleNamespace(
        platform=Platform.RELAY,
        fronts_platform=MagicMock(return_value=True),
        prime_routing_source=MagicMock(),
    )
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="C1",
        scope_id="guild-1",
        user_id="user-1",
        delivered_via_upstream_relay=True,
    )
    source._transport_profile = None
    source._transport_platform = Platform.RELAY

    assert runner._adapter_for_source(source) is relay
    relay.prime_routing_source.assert_called_once_with(source)


def test_restored_interaction_loses_auth_trust_but_routes_same_relay_identity():
    replacement = SimpleNamespace(
        platform=Platform.RELAY,
        fronts_platform=MagicMock(
            side_effect=lambda platform: getattr(platform, "value", platform)
            == "discord"
        ),
        matches_transport_identity=MagicMock(
            side_effect=lambda identity: identity == "discord:app-1"
        ),
        prime_routing_source=MagicMock(),
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: replacement}
    runner._profile_adapters = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="C1",
        scope_id="guild-1",
        user_id="user-1",
        delivered_via_upstream_relay=True,
    )
    setattr(source, "_transport_profile", None)
    setattr(source, "_transport_platform", Platform.RELAY)
    setattr(source, "_transport_identity", "discord:app-1")

    restored = SessionSource.from_dict(source.to_dict())

    assert restored.delivered_via_upstream_relay is False
    assert runner._adapter_for_source(restored) is replacement
    replacement.prime_routing_source.assert_called_once_with(restored)


def test_relay_replacement_without_platform_advertisement_fails_closed():
    runner = object.__new__(GatewayRunner)
    relay = SimpleNamespace(
        platform=Platform.RELAY,
        prime_routing_source=MagicMock(),
    )
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="C1",
        scope_id="guild-1",
        user_id="user-1",
        delivered_via_upstream_relay=True,
    )
    source._transport_profile = None
    source._transport_platform = Platform.RELAY

    assert runner._adapter_for_source(source) is None
    relay.prime_routing_source.assert_not_called()


def test_relay_replacement_advertisement_error_fails_closed():
    runner = object.__new__(GatewayRunner)
    relay = SimpleNamespace(
        platform=Platform.RELAY,
        fronts_platform=MagicMock(side_effect=RuntimeError("relay unavailable")),
        prime_routing_source=MagicMock(),
    )
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="C1",
        delivered_via_upstream_relay=True,
    )
    source._transport_profile = None
    source._transport_platform = Platform.RELAY

    assert runner._adapter_for_source(source) is None
    relay.prime_routing_source.assert_not_called()


def test_restored_primary_source_routes_to_primary_not_runtime_profile():
    primary = _adapter("primary")
    runtime = _adapter("runtime")
    runner = _runner(primary=primary, coder=runtime)
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        profile="coder",
    )
    source._transport_profile = None
    source._transport_platform = Platform.SLACK
    restored = SessionSource.from_dict(source.to_dict())

    assert runner._adapter_for_source(restored) is primary


def test_restored_missing_owner_fails_closed_not_runtime_fallback():
    primary = _adapter("primary")
    runner = _runner(primary=primary)
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        profile="coder",
    )
    source._transport_profile = "missing"
    source._transport_platform = Platform.SLACK
    restored = SessionSource.from_dict(source.to_dict())

    assert runner._adapter_for_source(restored) is None


@pytest.mark.asyncio
async def test_queued_text_uses_current_stamped_owner_not_captured_adapter():
    stale = _adapter("stale")
    current = _adapter("current")
    runner = _runner(coder=current)

    await runner._deliver_queued_first_response(
        "final answer",
        _source(),
        stale,
        deliver_media=False,
    )

    stale.send.assert_not_awaited()
    current.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_queued_text_missing_stamped_owner_fails_closed():
    stale = _adapter("stale")
    primary = _adapter("primary")
    runner = _runner(primary=primary)

    await runner._deliver_queued_first_response(
        "private answer",
        _source(),
        stale,
        deliver_media=False,
    )

    stale.send.assert_not_awaited()
    primary.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_hand_built_unstamped_source_fails_closed():
    legacy = _adapter("legacy")
    runner = _runner()

    await runner._deliver_queued_first_response(
        "legacy answer",
        _source(stamped=False),
        legacy,
        deliver_media=False,
    )

    legacy.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_deserialized_legacy_source_keeps_compatibility():
    legacy = _adapter("legacy")
    runner = _runner()
    source = SessionSource.from_dict(
        SessionSource(platform=Platform.SLACK, chat_id="C1").to_dict(),
        allow_legacy_unstamped=True,
    )

    await runner._deliver_queued_first_response(
        "legacy answer",
        source,
        legacy,
        deliver_media=False,
    )

    legacy.send.assert_awaited_once()


def test_hand_built_unstamped_runtime_profile_cannot_nominate_credential():
    primary = _adapter("primary")
    runtime = _adapter("runtime")
    runner = _runner(primary=primary, coder=runtime)
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        profile="coder",
    )

    assert runner._adapter_for_source(source) is None


def test_deserialized_legacy_runtime_profile_keeps_exact_compatibility_route():
    primary = _adapter("primary")
    runtime = _adapter("runtime")
    runner = _runner(primary=primary, coder=runtime)
    source = SessionSource.from_dict(
        SessionSource(
            platform=Platform.SLACK,
            chat_id="C1",
            profile="coder",
        ).to_dict(),
        allow_legacy_unstamped=True,
    )

    assert runner._adapter_for_source(source) is runtime


@pytest.mark.asyncio
async def test_post_stream_media_re_resolves_between_physical_sends(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEDIA_OUTPUT_DIR", str(tmp_path))
    voice = tmp_path / "one.mp3"
    document = tmp_path / "two.pdf"
    voice.write_bytes(b"voice")
    document.write_bytes(b"document")

    first = _adapter("first")
    latest = _adapter("latest")
    runner = _runner(coder=first)
    source = _source()
    event = MessageEvent(text="", source=source, message_id="m1")

    async def first_voice(**kwargs):
        runner._profile_adapters["coder"][Platform.SLACK] = latest
        return SendResult(success=True, message_id="first-voice")

    first.send_voice.side_effect = first_voice

    await runner._deliver_media_from_response(
        f"MEDIA:{voice}\nMEDIA:{document}", event, first
    )

    first.send_voice.assert_awaited_once()
    first.send_document.assert_not_awaited()
    latest.send_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_auto_tts_re_resolves_between_generated_files(tmp_path, monkeypatch):
    requested = tmp_path / "requested.mp3"
    one = tmp_path / "one.mp3"
    two = tmp_path / "two.mp3"
    one.write_bytes(b"one")
    two.write_bytes(b"two")

    first = _adapter("first")
    latest = _adapter("latest")
    runner = _runner(coder=first)
    runner._get_guild_id = lambda event: None
    source = _source()
    event = MessageEvent(
        text="voice", source=source, message_id="m1", message_type=MessageType.VOICE
    )

    async def first_voice(**kwargs):
        runner._profile_adapters["coder"][Platform.SLACK] = latest
        return SendResult(success=True, message_id="first-voice")

    first.send_voice.side_effect = first_voice

    def fake_tts(*, text, output_path):
        return '{"success": true, "file_paths": ["%s", "%s"]}' % (one, two)

    monkeypatch.setattr(
        "gateway.run.build_auto_tts_output_path", lambda _platform: str(requested)
    )
    with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
        await runner._send_voice_reply(event, "speak this")

    first.send_voice.assert_awaited_once()
    latest.send_voice.assert_awaited_once()
    assert not one.exists()
    assert not two.exists()


@pytest.mark.asyncio
async def test_runner_cancelled_tts_cleans_late_reported_files(tmp_path, monkeypatch):
    requested = tmp_path / "requested.mp3"
    late = tmp_path / "late.mp3"
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    runner = _runner(coder=_adapter("live"))
    runner._get_guild_id = lambda event: None
    event = MessageEvent(
        text="voice",
        source=_source(),
        message_id="m-cancel",
        message_type=MessageType.VOICE,
    )

    def slow_tts(*, text, output_path):
        started.set()
        release.wait(timeout=5)
        late.write_bytes(b"late")
        finished.set()
        return '{"success": true, "file_path": "%s"}' % late

    monkeypatch.setattr(
        "gateway.run.build_auto_tts_output_path", lambda _platform: str(requested)
    )
    with patch("tools.tts_tool.text_to_speech_tool", side_effect=slow_tts):
        task = asyncio.create_task(runner._send_voice_reply(event, "speak"))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        while not finished.is_set():
            await asyncio.sleep(0)
        for _ in range(100):
            if not late.exists():
                break
            await asyncio.sleep(0.01)

    assert not requested.exists()
    assert not late.exists()


class _ImageFanoutAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(
        self, chat_id, content="", reply_to=None, metadata=None
    ):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.asyncio
async def test_base_image_fanout_re_resolves_between_physical_images():
    stale = _ImageFanoutAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    first = _ImageFanoutAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    latest = _ImageFanoutAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner(coder=first)
    source = _source()
    setattr(stale, "gateway_runner", runner)
    setattr(first, "gateway_runner", runner)
    setattr(latest, "gateway_runner", runner)
    first.send_image = AsyncMock()
    latest.send_image = AsyncMock()

    async def first_image(**kwargs):
        runner._profile_adapters["coder"][Platform.SLACK] = latest
        return SendResult(success=True, message_id="one")

    first.send_image.side_effect = first_image
    await stale.send_multiple_images(
        "C1",
        [("https://example.invalid/one.png", "one"),
         ("https://example.invalid/two.png", "two")],
        source=source,
    )

    first.send_image.assert_awaited_once()
    latest.send_image.assert_awaited_once()


class _NativeChunkAdapter(_ImageFanoutAdapter):
    async def send_multiple_images(
        self,
        chat_id,
        images,
        metadata=None,
        human_delay=0.0,
        *,
        source=None,
    ):
        await self.send_image(
            chat_id=chat_id,
            image_url=images[0][0],
            caption=images[0][1],
            metadata=metadata,
        )
        if await self._handoff_image_batch_if_replaced(
            source=source,
            chat_id=chat_id,
            images=images[1:],
            metadata=metadata,
            human_delay=human_delay,
        ):
            return
        await self.send_image(
            chat_id=chat_id,
            image_url=images[1][0],
            caption=images[1][1],
            metadata=metadata,
        )


@pytest.mark.asyncio
async def test_native_image_chunk_hands_remainder_to_replacement():
    first = _NativeChunkAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    latest = _ImageFanoutAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    latest.send_multiple_images = AsyncMock()
    runner = _runner(coder=first)
    setattr(first, "gateway_runner", runner)
    source = _source()
    first.send_image = AsyncMock()

    async def replace_after_first(**kwargs):
        runner._profile_adapters["coder"][Platform.SLACK] = latest
        return SendResult(success=True, message_id="first")

    first.send_image.side_effect = replace_after_first
    await first.send_multiple_images(
        "C1",
        [("https://example.invalid/one.png", "one"),
         ("https://example.invalid/two.png", "two")],
        source=source,
    )

    first.send_image.assert_awaited_once()
    latest.send_multiple_images.assert_awaited_once()
    assert latest.send_multiple_images.await_args.kwargs["images"] == [
        ("https://example.invalid/two.png", "two")
    ]


class _NativeFallbackAdapter(_ImageFanoutAdapter):
    async def send_multiple_images(
        self,
        chat_id,
        images,
        metadata=None,
        human_delay=0.0,
        *,
        source=None,
    ):
        await super().send_multiple_images(
            chat_id,
            images,
            metadata,
            human_delay,
            source=source,
        )


@pytest.mark.asyncio
async def test_native_fallback_does_not_return_to_retired_generation():
    stale = _NativeFallbackAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    latest = _ImageFanoutAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner(coder=latest)
    setattr(stale, "gateway_runner", runner)
    setattr(latest, "gateway_runner", runner)
    stale.send_image = AsyncMock()
    latest.send_image = AsyncMock(
        return_value=SendResult(success=True, message_id="latest")
    )

    await stale.send_multiple_images(
        "C1",
        [("https://example.invalid/one.png", "one")],
        source=_source(),
    )

    stale.send_image.assert_not_awaited()
    latest.send_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_replacement_batch_never_retries_on_retired_owner():
    stale = _NativeFallbackAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    latest = _ImageFanoutAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner(coder=latest)
    setattr(stale, "gateway_runner", runner)
    setattr(latest, "gateway_runner", runner)
    stale.send_image = AsyncMock()
    latest.send_multiple_images = AsyncMock(side_effect=RuntimeError("send failed"))

    handed_off = await stale._handoff_image_batch_if_replaced(
        source=_source(),
        chat_id="C1",
        images=[("https://example.invalid/one.png", "one")],
        metadata=None,
        human_delay=0.0,
    )

    assert handed_off is True
    latest.send_multiple_images.assert_awaited_once()
    stale.send_image.assert_not_awaited()


class _NoDeleteAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(
        self, chat_id, content="", reply_to=None, metadata=None
    ):
        return SendResult(success=True, message_id="old")

    async def get_chat_info(self, chat_id):
        return {}


class _DeleteAdapter(_NoDeleteAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deleted = []

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))
        return True


@pytest.mark.asyncio
async def test_ephemeral_delete_capability_comes_from_live_sending_owner(monkeypatch):
    stale = _NoDeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    live = _DeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner(coder=live)
    runner._share_backend_notice_state(stale, profile_name="coder")
    runner._share_backend_notice_state(live, profile_name="coder")
    runner._is_user_authorized = lambda source: True
    source = stale.build_source(chat_id="C1", message_id="m1")
    stale.set_message_handler(
        AsyncMock(return_value=EphemeralReply("temporary", ttl_seconds=5))
    )
    stale._keep_typing = lambda *_args, **_kwargs: __import__("asyncio").Event().wait()
    live._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="live-message")
    )

    async def immediate_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", immediate_sleep)
    await stale._process_message_background(
        MessageEvent(text="hello", source=source, message_id="m1"),
        "agent:main:slack:dm:C1",
    )
    for _ in range(5):
        await immediate_sleep(0)

    assert live.deleted == [("C1", "live-message")]


@pytest.mark.asyncio
async def test_busy_command_ephemeral_delete_uses_replacement_sending_owner(monkeypatch):
    stale = _NoDeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    live = _DeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner(coder=live)
    runner._share_backend_notice_state(stale, profile_name="coder")
    runner._share_backend_notice_state(live, profile_name="coder")
    source = stale.build_source(chat_id="C1", message_id="m1")
    stale.set_message_handler(
        AsyncMock(return_value=EphemeralReply("temporary", ttl_seconds=5))
    )
    session_key = "agent:coder:slack:dm:C1"
    stale._active_sessions[session_key] = asyncio.Event()

    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", immediate_sleep)
    await stale._dispatch_active_session_command(
        MessageEvent(text="/stop", source=source, message_id="m1"),
        session_key,
        "stop",
    )
    for _ in range(5):
        await real_sleep(0)

    assert live.deleted == [("C1", "old")]


@pytest.mark.asyncio
async def test_owner_unavailable_final_text_is_left_pending_in_ledger(
    isolated_ledger, monkeypatch
):
    stale = _NoDeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner()
    setattr(stale, "gateway_runner", runner)
    source = stale.build_source(chat_id="C1", message_id="m1")
    setattr(source, "_transport_profile", "coder")
    setattr(source, "_transport_platform", Platform.SLACK)
    stale.set_message_handler(AsyncMock(return_value="private final answer"))
    stale._keep_typing = (
        lambda *_args, **_kwargs: __import__("asyncio").Event().wait()
    )

    async def immediate_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", immediate_sleep)
    await stale._process_message_background(
        MessageEvent(text="hello", source=source, message_id="m1"),
        "agent:coder:slack:dm:C1",
    )

    with dl._connect() as conn:
        row = conn.execute(
            "SELECT state, content, transport_profile, owner_pid "
            "FROM delivery_obligations"
        ).fetchone()
    assert row == ("pending", "private final answer", "coder", None)

    live = _adapter("live")
    setattr(runner, "_profile_adapters", {"coder": {Platform.SLACK: live}})
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 1
    live.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_owner_unavailable_media_only_response_replays_after_reconnect(
    isolated_ledger, monkeypatch, tmp_path
):
    document = tmp_path / "report.pdf"
    document.write_bytes(b"pdf")
    stale = _NoDeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner()
    setattr(stale, "gateway_runner", runner)
    source = stale.build_source(chat_id="C1", message_id="m1")
    setattr(source, "_transport_profile", "coder")
    setattr(source, "_transport_platform", Platform.SLACK)
    stale.set_message_handler(AsyncMock(return_value=f"MEDIA:{document}"))
    stale._keep_typing = (
        lambda *_args, **_kwargs: __import__("asyncio").Event().wait()
    )

    async def immediate_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", immediate_sleep)
    await stale._process_message_background(
        MessageEvent(text="hello", source=source, message_id="m1"),
        "agent:coder:slack:dm:C1",
    )

    with dl._connect() as conn:
        operation, payload_json, state, owner_pid = conn.execute(
            "SELECT operation, payload_json, state, owner_pid "
            "FROM delivery_obligations"
        ).fetchone()
    payload = json.loads(payload_json)
    assert operation == "response_bundle"
    assert payload["text"] == ""
    assert payload["media_files"] == [[str(document), False]]
    assert (state, owner_pid) == ("pending", None)

    live = _adapter("live")
    setattr(runner, "_profile_adapters", {"coder": {Platform.SLACK: live}})
    store = MagicMock(clear_resume_pending=AsyncMock(), _store=None)
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 1
    live.send.assert_not_awaited()
    live.send_document.assert_awaited_once_with(
        chat_id="C1", file_path=str(document), caption=None,
        metadata={"thread_id": None}
    )


@pytest.mark.asyncio
async def test_live_failed_text_blocks_later_images(isolated_ledger, monkeypatch):
    adapter = _NoDeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner(coder=adapter)
    setattr(adapter, "gateway_runner", runner)
    source = adapter.build_source(chat_id="C1", message_id="m1")
    setattr(source, "_transport_profile", "coder")
    setattr(source, "_transport_platform", Platform.SLACK)
    adapter.set_message_handler(
        AsyncMock(return_value="answer\n![a](https://example.test/a.png)")
    )
    adapter._keep_typing = (
        lambda *_args, **_kwargs: __import__("asyncio").Event().wait()
    )
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=False, error="text rejected")
    )
    adapter.send_multiple_images = AsyncMock(return_value=None)

    async def immediate_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", immediate_sleep)
    await adapter._process_message_background(
        MessageEvent(text="hello", source=source, message_id="m1"),
        "agent:coder:slack:dm:C1",
    )

    adapter._send_with_retry.assert_awaited_once()
    adapter.send_multiple_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_image_prefix_is_checkpointed_before_later_failure(
    isolated_ledger, monkeypatch
):
    adapter = _NoDeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.SLACK
    )
    runner = _runner(coder=adapter)
    setattr(adapter, "gateway_runner", runner)
    source = adapter.build_source(chat_id="C1", message_id="m-images")
    setattr(source, "_transport_profile", "coder")
    setattr(source, "_transport_platform", Platform.SLACK)
    adapter.set_message_handler(
        AsyncMock(
            return_value=(
                "![one](https://example.test/one.png)\n"
                "![two](https://example.test/two.png)"
            )
        )
    )
    adapter._keep_typing = (
        lambda *_args, **_kwargs: __import__("asyncio").Event().wait()
    )
    adapter.send_multiple_images = AsyncMock(
        side_effect=[None, SendResult(success=False, error="second rejected")]
    )

    async def immediate_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", immediate_sleep)
    await adapter._process_message_background(
        MessageEvent(text="hello", source=source, message_id="m-images"),
        "agent:coder:slack:dm:C1",
    )

    assert adapter.send_multiple_images.await_count == 2
    with dl._connect() as conn:
        state, payload_json = conn.execute(
            "SELECT state, payload_json FROM delivery_obligations "
            "WHERE operation='response_bundle'"
        ).fetchone()
    persisted = json.loads(payload_json)
    assert state == "failed"
    assert persisted["completed_operations"] == ["images:0"]
    assert persisted["attempting_operation"] == "images:1"


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    return home


def _orphan(oid: str) -> None:
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )


def test_ledger_persists_stamped_transport_owner(isolated_ledger):
    dl.record_obligation(
        obligation_id="owner-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id="171.001",
        content="private answer",
        transport_profile="coder",
        transport_profile_stamped=True,
    )
    _orphan("owner-row")

    rows = dl.sweep_recoverable(
        deliverable_routes={("slack", True, "coder", None)}
    )

    assert rows[0]["transport_profile"] == "coder"
    assert rows[0]["transport_profile_stamped"] is True


@pytest.mark.asyncio
async def test_recovery_missing_secondary_owner_never_uses_primary(isolated_ledger):
    dl.record_obligation(
        obligation_id="secondary-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id="171.001",
        content="secondary private answer",
        transport_profile="coder",
        transport_profile_stamped=True,
    )
    _orphan("secondary-row")

    primary = _adapter("primary")
    runner = _runner(primary=primary)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 0
    primary.send.assert_not_awaited()
    with dl._connect() as conn:
        state, attempts = conn.execute(
            "SELECT state, attempts FROM delivery_obligations "
            "WHERE obligation_id='secondary-row'"
        ).fetchone()
    assert state == "pending"
    assert attempts == 0
    store.clear_resume_pending.assert_awaited_once_with(
        "agent:main:slack:channel:C1"
    )


@pytest.mark.asyncio
async def test_secondary_relay_exact_identity_can_claim_obligation(
    isolated_ledger,
):
    dl.record_obligation(
        obligation_id="secondary-relay-row",
        session_key=(
            "agent:coder:discord:dm:C1:transport=discord%3Aapp-1"
        ),
        platform="discord",
        chat_id="C1",
        thread_id=None,
        content="private answer",
        transport_platform="relay",
        transport_profile="coder",
        transport_profile_stamped=True,
        transport_identity="discord:app-1",
    )
    _orphan("secondary-relay-row")
    relay = _adapter("relay")
    relay.platform = Platform.RELAY
    relay.fronts_platform = MagicMock(return_value=True)
    relay.acknowledged_transport_identities = MagicMock(
        return_value=("discord:app-1",)
    )
    relay.matches_transport_identity = MagicMock(return_value=True)
    relay.prime_routing_source = MagicMock()
    runner = _runner(coder=relay)
    setattr(runner, "_profile_adapters", {"coder": {Platform.RELAY: relay}})
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    runner._async_session_store = store
    setattr(runner, "session_store", None)

    assert await runner._redeliver_pending_obligations() == 1
    relay.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovery_route_disappearing_after_claim_releases_budget(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="vanished-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
        transport_profile="coder",
        transport_profile_stamped=True,
    )
    _orphan("vanished-row")

    primary = _adapter("primary")
    coder = _adapter("coder")
    runner = _runner(primary=primary, coder=coder)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    real_sweep = dl.sweep_recoverable

    def sweep_then_disconnect(*args, **kwargs):
        rows = real_sweep(*args, **kwargs)
        runner._profile_adapters = {}
        return rows

    monkeypatch.setattr(dl, "sweep_recoverable", sweep_then_disconnect)

    assert await runner._redeliver_pending_obligations() == 0
    primary.send.assert_not_awaited()
    coder.send.assert_not_awaited()
    with dl._connect() as conn:
        state, attempts, owner_pid = conn.execute(
            "SELECT state, attempts, owner_pid FROM delivery_obligations "
            "WHERE obligation_id='vanished-row'"
        ).fetchone()
    assert state == "pending"
    assert attempts == 0
    assert owner_pid is None


@pytest.mark.asyncio
async def test_recovery_owner_disappearing_immediately_before_send_releases_budget(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="pre-send-vanished-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
        transport_platform="slack",
        transport_profile="coder",
        transport_profile_stamped=True,
    )
    _orphan("pre-send-vanished-row")
    replacement = _adapter("replacement")
    runner = _runner(coder=replacement)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store
    calls = 0

    def resolve_then_vanish(_source):
        nonlocal calls
        calls += 1
        return replacement if calls == 1 else None

    monkeypatch.setattr(runner, "_adapter_for_source", resolve_then_vanish)

    assert await runner._redeliver_pending_obligations() == 0
    replacement.send.assert_not_awaited()
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT attempts, owner_pid, owner_started_at "
            "FROM delivery_obligations WHERE obligation_id=?",
            ("pre-send-vanished-row",),
        ).fetchone()
    assert row == (0, None, None)


def test_release_claim_rejects_foreign_owner(isolated_ledger):
    dl.record_obligation(
        obligation_id="foreign-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1, attempts=2 WHERE obligation_id='foreign-row'"
        )

    assert dl.release_claim("foreign-row") is False
    with dl._connect() as conn:
        attempts, owner_pid = conn.execute(
            "SELECT attempts, owner_pid FROM delivery_obligations "
            "WHERE obligation_id='foreign-row'"
        ).fetchone()
    assert attempts == 2
    assert owner_pid == 999999999


@pytest.mark.asyncio
async def test_recovery_relay_owner_never_uses_same_platform_native_adapter(
    isolated_ledger,
):
    dl.record_obligation(
        obligation_id="relay-row",
        session_key="agent:main:discord:channel:C1",
        platform="discord",
        chat_id="C1",
        thread_id=None,
        content="relay private answer",
        transport_platform="relay",
        transport_profile=None,
        transport_profile_stamped=True,
        transport_identity="discord:app-1",
        route_scope_id="guild-1",
        route_user_id="user-1",
        route_chat_type="channel",
    )
    _orphan("relay-row")

    native = SimpleNamespace(platform=Platform.DISCORD, send=AsyncMock())
    relay = SimpleNamespace(
        platform=Platform.RELAY,
        _transport_profile=None,
        fronts_platform=MagicMock(return_value=True),
        acknowledged_transport_identities=MagicMock(
            return_value=("discord:app-1",)
        ),
        matches_transport_identity=MagicMock(return_value=True),
        send=AsyncMock(),
        prime_routing_source=MagicMock(),
    )
    relay.send.return_value = SendResult(success=True)
    runner = _runner(primary=native)
    runner.adapters[Platform.RELAY] = relay
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 1
    relay.send.assert_awaited_once()
    assert relay.send.await_args.kwargs["metadata"] == {
        "thread_id": None,
        "_relay_logical_platform": "discord",
        "_relay_transport_identity": "discord:app-1",
        "scope_id": "guild-1",
        "user_id": "user-1",
    }
    primed_source = relay.prime_routing_source.call_args.args[0]
    assert primed_source.platform == Platform.DISCORD
    assert primed_source.scope_id == "guild-1"
    assert primed_source.user_id == "user-1"
    assert primed_source.chat_type == "channel"
    native.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_legacy_unstamped_relay_row_uses_advertised_logical_route(
    isolated_ledger,
):
    dl.record_obligation(
        obligation_id="legacy-relay-row",
        session_key="agent:main:discord:channel:C1",
        platform="discord",
        chat_id="C1",
        thread_id=None,
        content="legacy relay answer",
        transport_profile_stamped=False,
        route_scope_id="guild-legacy",
        route_user_id="user-legacy",
        route_chat_type="channel",
    )
    _orphan("legacy-relay-row")

    relay = SimpleNamespace(
        platform=Platform.RELAY,
        _transport_profile=None,
        fronts_platform=MagicMock(
            side_effect=lambda platform: getattr(platform, "value", platform)
            == "discord"
        ),
        acknowledged_transport_identities=MagicMock(
            return_value=("discord:app-legacy",)
        ),
        send_for_platform=AsyncMock(
            return_value=SendResult(success=True, message_id="legacy-delivered")
        ),
        send=AsyncMock(),
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "main"
    runner.config = GatewayConfig(
        platforms={Platform.RELAY: PlatformConfig(enabled=True)}
    )
    runner._thread_metadata_for_source = (
        GatewayRunner._thread_metadata_for_source.__get__(runner, GatewayRunner)
    )
    runner._thread_metadata_for_target = (
        GatewayRunner._thread_metadata_for_target.__get__(runner, GatewayRunner)
    )
    store = MagicMock()
    store.clear_resume_pending = AsyncMock(return_value=True)
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 1
    relay.send_for_platform.assert_awaited_once_with(
        Platform.DISCORD,
        "C1",
        "legacy relay answer",
        metadata={
            "scope_id": "guild-legacy",
            "user_id": "user-legacy",
        },
    )
    relay.send.assert_not_awaited()
    store.clear_resume_pending.assert_awaited_once_with(
        "agent:main:discord:channel:C1"
    )


@pytest.mark.asyncio
async def test_recovery_relay_advertisement_error_does_not_block_native_row(
    isolated_ledger,
):
    dl.record_obligation(
        obligation_id="native-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="native answer",
        transport_profile_stamped=False,
    )
    _orphan("native-row")
    native = _adapter("native")
    relay = SimpleNamespace(
        platform=Platform.RELAY,
        _transport_profile=None,
        fronts_platform=MagicMock(side_effect=RuntimeError("relay unavailable")),
        acknowledged_transport_identities=MagicMock(return_value=()),
    )
    runner = _runner(primary=native)
    runner.adapters[Platform.RELAY] = relay
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 1
    native.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovery_slack_workspace_scope_reaches_send_metadata(
    isolated_ledger,
):
    dl.record_obligation(
        obligation_id="slack-workspace-row",
        session_key="agent:main:slack:channel:C_SHARED",
        platform="slack",
        chat_id="C_SHARED",
        thread_id="171.001",
        content="workspace private answer",
        transport_platform="slack",
        transport_profile=None,
        transport_profile_stamped=True,
        route_scope_id="T_OTHER",
        route_user_id="U_OTHER",
        route_chat_type="channel",
    )
    _orphan("slack-workspace-row")

    slack = _adapter("primary")
    slack._transport_profile = None
    runner = _runner(primary=slack)
    runner._thread_metadata_for_source = GatewayRunner._thread_metadata_for_source.__get__(
        runner, GatewayRunner
    )
    runner._thread_metadata_for_target = GatewayRunner._thread_metadata_for_target.__get__(
        runner, GatewayRunner
    )
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 1
    assert slack.send.await_args.kwargs["metadata"] == {
        "thread_id": "171.001",
        "slack_team_id": "T_OTHER",
    }


def test_live_delivery_operation_accepts_relay_owner_for_underlying_platform():
    stale = _adapter("native")
    relay = _adapter("relay")
    relay.platform = Platform.RELAY
    relay.fronts_platform = MagicMock(return_value=True)
    relay.matches_transport_identity = MagicMock(return_value=True)
    source = SessionSource(platform=Platform.DISCORD, chat_id="C1")
    source._transport_profile = None
    source._transport_platform = Platform.RELAY
    setattr(source, "_transport_identity", "discord:app-1")
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}

    selected, send = runner._live_delivery_operation(source, stale, "send")

    assert selected is relay
    assert send == relay.send


def test_relay_owner_not_fronting_logical_platform_fails_closed():
    relay = _adapter("relay")
    relay.platform = Platform.RELAY
    relay.fronts_platform = MagicMock(return_value=False)
    source = SessionSource(platform=Platform.DISCORD, chat_id="C1")
    source._transport_profile = None
    source._transport_platform = Platform.RELAY
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}

    assert runner._adapter_for_source(source) is None


def test_restored_relay_owner_with_different_bot_identity_fails_closed():
    relay = _adapter("relay")
    relay.platform = Platform.RELAY
    relay.fronts_platform = MagicMock(return_value=True)
    relay.transport_identity_for_platform = MagicMock(
        return_value="discord:replacement-app"
    )
    source = SessionSource(platform=Platform.DISCORD, chat_id="C1")
    source._transport_profile = None
    source._transport_platform = Platform.RELAY
    source._transport_identity = "discord:original-app"
    restored = SessionSource.from_dict(source.to_dict())
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}

    assert runner._adapter_for_source(restored) is None


def test_restored_relay_owner_with_wrong_identity_platform_fails_closed():
    relay = _adapter("relay")
    relay.platform = Platform.RELAY
    relay.fronts_platform = MagicMock(return_value=True)
    relay.matches_transport_identity = MagicMock(return_value=True)
    source = SessionSource(platform=Platform.DISCORD, chat_id="C1")
    setattr(source, "_transport_profile", None)
    setattr(source, "_transport_platform", Platform.RELAY)
    setattr(source, "_transport_identity", "slack:app-1")
    restored = SessionSource.from_dict(source.to_dict())
    runner = object.__new__(GatewayRunner)
    setattr(runner, "adapters", {Platform.RELAY: relay})
    runner._profile_adapters = {}

    assert runner._adapter_for_source(restored) is None


def test_live_relay_source_with_different_replacement_identity_fails_closed():
    relay = _adapter("relay")
    relay.platform = Platform.RELAY
    relay.fronts_platform = MagicMock(return_value=True)
    relay.matches_transport_identity = MagicMock(return_value=False)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="C1",
        delivered_via_upstream_relay=True,
    )
    setattr(source, "_transport_profile", None)
    setattr(source, "_transport_platform", Platform.RELAY)
    setattr(source, "_transport_identity", "discord:original-app")
    runner = object.__new__(GatewayRunner)
    setattr(runner, "adapters", {Platform.RELAY: relay})
    runner._profile_adapters = {}

    assert runner._adapter_for_source(source) is None


@pytest.mark.asyncio
async def test_send_retry_re_resolves_stamped_owner_between_attempts(monkeypatch):
    stale = _RetryBoundaryAdapter("stale")
    replacement = _RetryBoundaryAdapter("replacement")
    runner = _runner(coder=stale)
    runner._share_backend_notice_state(stale, profile_name="coder")
    runner._share_backend_notice_state(replacement, profile_name="coder")
    source = stale.build_source(chat_id="C1")

    async def stale_failure(**_kwargs):
        runner._profile_adapters["coder"][Platform.SLACK] = replacement
        stale.physical_sends += 1
        return SendResult(success=False, error="network down", retryable=True)

    stale.send = stale_failure
    replacement.send = AsyncMock(return_value=SendResult(success=True))

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", no_sleep)
    result = await stale._send_with_retry(
        chat_id="C1",
        content="private answer",
        max_retries=1,
        base_delay=0,
        source=source,
    )

    assert result.success is True
    assert stale.physical_sends == 1
    replacement.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_logical_platform_releases_claim_without_spending_budget(
    isolated_ledger,
):
    dl.record_obligation(
        obligation_id="future-platform-row",
        session_key="agent:main:future:channel:C1",
        platform="future-platform",
        chat_id="C1",
        thread_id=None,
        content="private answer",
        transport_platform="slack",
        transport_profile=None,
        transport_profile_stamped=True,
    )
    _orphan("future-platform-row")
    primary = _adapter("primary")
    primary._transport_profile = None
    runner = _runner(primary=primary)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 0
    with dl._connect() as conn:
        state, attempts, owner_pid = conn.execute(
            "SELECT state, attempts, owner_pid FROM delivery_obligations "
            "WHERE obligation_id='future-platform-row'"
        ).fetchone()
    assert state == "pending"
    assert attempts == 0
    assert owner_pid is None