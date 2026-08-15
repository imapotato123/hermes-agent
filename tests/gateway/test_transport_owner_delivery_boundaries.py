"""Transport-owner authority at runner and durable-delivery boundaries.

Every new user-visible operation must resolve the live adapter that owns the
inbound source. Explicitly stamped owners fail closed; unstamped sources keep
the legacy passed/default-adapter behavior.
"""

import asyncio
import threading

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    EphemeralReply,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner
from gateway.session import (
    SessionSource,
    source_has_transport_owner,
    stamp_source_transport_owner,
)


def _source(*, stamped: bool = True, profile: str | None = "coder") -> SessionSource:
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        thread_id="171.001",
        message_id="m1",
    )
    if stamped:
        stamp_source_transport_owner(
            source,
            profile=profile,
            platform=Platform.SLACK,
        )
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


class _IngressBoundaryAdapter(_RetryBoundaryAdapter):
    def __init__(
        self,
        name: str,
        *,
        platform: Platform = Platform.SLACK,
        profile: str | None = None,
    ):
        BasePlatformAdapter.__init__(
            self,
            PlatformConfig(enabled=True, token="t"),
            platform,
        )
        self._name = name
        self._transport_profile = profile
        self.handled_events: list[MessageEvent] = []

        async def _handler(event: MessageEvent):
            self.handled_events.append(event)
            return None

        self._message_handler = _handler


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
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.RELAY,
    )
    setattr(source, "_transport_identity", "discord:app-1")

    assert runner._adapter_for_source(source) is relay
    relay.prime_routing_source.assert_called_once_with(source)


def test_nonrelay_physical_platform_mismatch_fails_closed():
    slack = _adapter("slack")
    discord = _adapter("discord")
    discord.platform = Platform.DISCORD
    runner = object.__new__(GatewayRunner)
    runner.adapters = {
        Platform.SLACK: slack,
        Platform.DISCORD: discord,
    }
    runner._profile_adapters = {}
    source = SessionSource(platform=Platform.SLACK, chat_id="C1")
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.DISCORD,
    )

    assert runner._adapter_for_source(source) is None


def test_registered_adapter_weakref_without_owner_capability_fails_closed():
    adapter = _adapter("native")
    runner = _runner(primary=adapter)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1")
    source._transport_adapter_ref = lambda: adapter
    source._transport_platform = Platform.SLACK
    source._transport_profile = None

    assert source_has_transport_owner(source) is False
    assert runner._adapter_for_source(source) is None


@pytest.mark.parametrize("platform", [Platform.HOMEASSISTANT, Platform.WEBHOOK])
def test_ownerless_authenticated_system_platforms_still_fail_closed(platform):
    runner = _runner()
    source = SessionSource(platform=platform, chat_id="system-event")

    assert runner._is_user_authorized(source) is False

    stamp_source_transport_owner(source, profile=None, platform=platform)
    assert runner._is_user_authorized(source) is True


@pytest.mark.asyncio
async def test_native_handle_message_stamps_fresh_matching_source():
    adapter = _IngressBoundaryAdapter("slack", profile="coder")
    source = SessionSource(platform=Platform.SLACK, chat_id="C1")
    event = MessageEvent(text="hello", source=source)

    await BasePlatformAdapter.handle_message(adapter, event)

    pending = list(adapter._session_tasks.values())
    assert len(pending) == 1
    await asyncio.gather(*pending)

    assert source_has_transport_owner(source) is True
    assert source._transport_platform == Platform.SLACK
    assert source._transport_profile == "coder"
    assert adapter.handled_events == [event]


@pytest.mark.asyncio
async def test_native_handle_message_rejects_ownerless_cross_platform_source():
    adapter = _IngressBoundaryAdapter("discord", platform=Platform.DISCORD)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1")
    event = MessageEvent(text="hello", source=source)

    await BasePlatformAdapter.handle_message(adapter, event)

    assert source_has_transport_owner(source) is False
    assert adapter._session_tasks == {}
    assert adapter.handled_events == []


@pytest.mark.asyncio
async def test_background_processing_cannot_mint_owner_for_ownerless_source():
    adapter = _IngressBoundaryAdapter("slack")
    adapter._active_sessions["s1"] = asyncio.Event()
    adapter._release_session_guard = MagicMock()
    source = SessionSource(platform=Platform.SLACK, chat_id="C1")
    event = MessageEvent(text="hello", source=source)

    await BasePlatformAdapter._process_message_background(adapter, event, "s1")

    assert source_has_transport_owner(source) is False
    assert adapter.handled_events == []
    adapter._release_session_guard.assert_called_once_with("s1")


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
        delivered_via_upstream_relay=True,
    )
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.RELAY,
    )
    source._transport_identity = "discord:app-1"

    assert runner._adapter_for_source(source) is relay
    relay.matches_transport_identity.assert_called_with("discord:app-1")
    relay.prime_routing_source.assert_called_once_with(source)


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
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.RELAY,
    )

    assert runner._adapter_for_source(source) is None
    relay.prime_routing_source.assert_not_called()


def test_generic_roundtrip_drops_primary_owner_and_fails_closed():
    primary = _adapter("primary")
    runtime = _adapter("runtime")
    runner = _runner(primary=primary, coder=runtime)
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        profile="coder",
    )
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.SLACK,
    )
    restored = SessionSource.from_dict(source.to_dict())

    assert source_has_transport_owner(restored) is False
    assert runner._adapter_for_source(restored) is None


def test_forged_dictionary_owner_fails_closed_not_runtime_fallback():
    primary = _adapter("primary")
    runner = _runner(primary=primary)
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        profile="coder",
    )
    restored = SessionSource.from_dict(
        {
            **source.to_dict(),
            "transport_owner_stamped": True,
            "transport_profile": "missing",
            "transport_platform": "slack",
        }
    )

    assert source_has_transport_owner(restored) is False
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
async def test_recovery_persists_attempting_before_transport_egress(isolated_ledger):
    dl.record_obligation(
        obligation_id="pending-egress-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
    )
    _orphan("pending-egress-row")

    observed = []

    async def inspect_state_at_egress(*_args, **_kwargs):
        with dl._connect() as conn:
            observed.append(
                conn.execute(
                    "SELECT state, attempts FROM delivery_obligations "
                    "WHERE obligation_id='pending-egress-row'"
                ).fetchone()
            )
        return SendResult(success=True)

    primary = _adapter("primary")
    primary.send.side_effect = inspect_state_at_egress
    runner = _runner(primary=primary)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 1
    assert observed == [("attempting", 1)]


@pytest.mark.asyncio
async def test_cancelled_recovery_checkpoint_refunds_before_transport_egress(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="cancelled-checkpoint-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
    )
    _orphan("cancelled-checkpoint-row")

    checkpoint_started = threading.Event()
    allow_checkpoint = threading.Event()
    checkpoint_finished = threading.Event()
    real_mark_attempting = dl.mark_attempting

    def blocked_mark_attempting(obligation_id, **kwargs):
        checkpoint_started.set()
        try:
            assert allow_checkpoint.wait(timeout=5)
            return real_mark_attempting(obligation_id, **kwargs)
        finally:
            checkpoint_finished.set()

    monkeypatch.setattr(dl, "mark_attempting", blocked_mark_attempting)

    primary = _adapter("primary")
    runner = _runner(primary=primary)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    recovery = asyncio.create_task(runner._redeliver_pending_obligations())
    assert await asyncio.to_thread(checkpoint_started.wait, 5)
    recovery.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await recovery

        with dl._connect() as conn:
            row = conn.execute(
                "SELECT state, attempts, owner_pid, owner_started_at, "
                "recovery_claim FROM delivery_obligations "
                "WHERE obligation_id='cancelled-checkpoint-row'"
            ).fetchone()
        assert row == ("pending", 0, None, None, None)
        primary.send.assert_not_awaited()
    finally:
        allow_checkpoint.set()
        assert await asyncio.to_thread(checkpoint_finished.wait, 5)

    with dl._connect() as conn:
        row = conn.execute(
            "SELECT state, attempts, owner_pid, owner_started_at, "
            "recovery_claim FROM delivery_obligations "
            "WHERE obligation_id='cancelled-checkpoint-row'"
        ).fetchone()
    assert row == ("pending", 0, None, None, None)


@pytest.mark.asyncio
async def test_recovery_checkpoint_error_refunds_before_transport_egress(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="failed-checkpoint-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
    )
    _orphan("failed-checkpoint-row")

    def fail_checkpoint(*_args, **_kwargs):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(dl, "mark_attempting", fail_checkpoint)

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
        row = conn.execute(
            "SELECT state, attempts, owner_pid, owner_started_at, "
            "recovery_claim FROM delivery_obligations "
            "WHERE obligation_id='failed-checkpoint-row'"
        ).fetchone()
    assert row == ("pending", 0, None, None, None)


@pytest.mark.asyncio
async def test_cancelled_recovery_releases_owner_but_preserves_ambiguity(
    isolated_ledger,
):
    dl.record_obligation(
        obligation_id="cancelled-egress-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
    )
    _orphan("cancelled-egress-row")

    entered = asyncio.Event()
    block = asyncio.Event()

    async def block_after_egress(*_args, **_kwargs):
        entered.set()
        await block.wait()
        return SendResult(success=True)

    primary = _adapter("primary")
    primary.send.side_effect = block_after_egress
    runner = _runner(primary=primary)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    recovery = asyncio.create_task(runner._redeliver_pending_obligations())
    await entered.wait()
    recovery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await recovery

    with dl._connect() as conn:
        row = conn.execute(
            "SELECT state, attempts, owner_pid, owner_started_at "
            "FROM delivery_obligations "
            "WHERE obligation_id='cancelled-egress-row'"
        ).fetchone()
    assert row == ("attempting", 1, None, None)

    claimed = dl.sweep_recoverable()
    assert len(claimed) == 1
    assert claimed[0]["needs_marker"] is True
    assert claimed[0]["attempts"] == 2


@pytest.mark.asyncio
async def test_cancelled_recovery_settlement_cannot_strand_or_stale_finalize_claim(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="cancelled-settlement-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
    )
    _orphan("cancelled-settlement-row")

    settlement_started = threading.Event()
    allow_settlement = threading.Event()
    settlement_finished = threading.Event()
    real_mark_delivered = dl.mark_delivered

    def blocked_mark_delivered(obligation_id, **kwargs):
        settlement_started.set()
        try:
            assert allow_settlement.wait(timeout=5)
            return real_mark_delivered(obligation_id, **kwargs)
        finally:
            settlement_finished.set()

    monkeypatch.setattr(dl, "mark_delivered", blocked_mark_delivered)

    primary = _adapter("primary")
    primary.send.return_value = SendResult(success=True)
    runner = _runner(primary=primary)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    recovery = asyncio.create_task(runner._redeliver_pending_obligations())
    assert await asyncio.to_thread(settlement_started.wait, 5)
    recovery.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await recovery

        with dl._connect() as conn:
            row = conn.execute(
                "SELECT state, attempts, owner_pid, owner_started_at "
                "FROM delivery_obligations "
                "WHERE obligation_id='cancelled-settlement-row'"
            ).fetchone()
        assert row == ("attempting", 1, None, None)
    finally:
        allow_settlement.set()
        assert await asyncio.to_thread(settlement_finished.wait, 5)

    with dl._connect() as conn:
        row = conn.execute(
            "SELECT state, attempts, owner_pid, owner_started_at "
            "FROM delivery_obligations "
            "WHERE obligation_id='cancelled-settlement-row'"
        ).fetchone()
    assert row == ("attempting", 1, None, None)


@pytest.mark.asyncio
async def test_recovery_settlement_error_releases_ambiguous_claim(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="failed-settlement-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
    )
    _orphan("failed-settlement-row")

    def fail_settlement(*_args, **_kwargs):
        raise RuntimeError("settlement unavailable")

    monkeypatch.setattr(dl, "mark_delivered", fail_settlement)

    primary = _adapter("primary")
    primary.send.return_value = SendResult(success=True)
    runner = _runner(primary=primary)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store

    assert await runner._redeliver_pending_obligations() == 0
    primary.send.assert_awaited_once()
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT state, attempts, owner_pid, owner_started_at, "
            "recovery_claim FROM delivery_obligations "
            "WHERE obligation_id='failed-settlement-row'"
        ).fetchone()
    assert row == ("attempting", 1, None, None, None)

    claimed = dl.sweep_recoverable()
    assert len(claimed) == 1
    assert claimed[0]["needs_marker"] is True
    assert claimed[0]["attempts"] == 2


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
async def test_recovery_malformed_claimed_transport_platform_fails_closed(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="malformed-owner-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
        transport_platform="slack",
        transport_profile=None,
        transport_profile_stamped=True,
    )
    _orphan("malformed-owner-row")

    primary = _adapter("primary")
    primary._transport_profile = None
    runner = _runner(primary=primary)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store
    real_sweep = dl.sweep_recoverable

    def claim_then_corrupt(*args, **kwargs):
        rows = real_sweep(*args, **kwargs)
        assert len(rows) == 1
        rows[0]["transport_platform"] = "not-a-platform"
        return rows

    monkeypatch.setattr(dl, "sweep_recoverable", claim_then_corrupt)

    assert await runner._redeliver_pending_obligations() == 0
    primary.send.assert_not_awaited()
    with dl._connect() as conn:
        state, attempts, owner_pid = conn.execute(
            "SELECT state, attempts, owner_pid FROM delivery_obligations "
            "WHERE obligation_id='malformed-owner-row'"
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


@pytest.mark.asyncio
async def test_recovery_re_resolves_stamped_owner_after_attempting_checkpoint(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="checkpoint-reconnect-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
        transport_platform="slack",
        transport_profile="coder",
        transport_profile_stamped=True,
    )
    _orphan("checkpoint-reconnect-row")

    stale = _adapter("stale")
    replacement = _adapter("replacement")
    runner = _runner(coder=stale)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store
    real_mark_attempting = dl.mark_attempting

    def checkpoint_then_reconnect(obligation_id, **kwargs):
        transitioned = real_mark_attempting(obligation_id, **kwargs)
        runner._profile_adapters["coder"][Platform.SLACK] = replacement
        return transitioned

    monkeypatch.setattr(dl, "mark_attempting", checkpoint_then_reconnect)

    assert await runner._redeliver_pending_obligations() == 1
    stale.send.assert_not_awaited()
    replacement.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovery_owner_disappearing_during_checkpoint_refunds_claim(
    isolated_ledger, monkeypatch
):
    dl.record_obligation(
        obligation_id="checkpoint-vanished-row",
        session_key="agent:main:slack:channel:C1",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="private answer",
        transport_platform="slack",
        transport_profile="coder",
        transport_profile_stamped=True,
    )
    _orphan("checkpoint-vanished-row")

    stale = _adapter("stale")
    runner = _runner(coder=stale)
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store
    real_mark_attempting = dl.mark_attempting

    def checkpoint_then_disconnect(obligation_id, **kwargs):
        transitioned = real_mark_attempting(obligation_id, **kwargs)
        runner._profile_adapters = {}
        return transitioned

    monkeypatch.setattr(dl, "mark_attempting", checkpoint_then_disconnect)

    assert await runner._redeliver_pending_obligations() == 0
    stale.send.assert_not_awaited()
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT state, attempts, owner_pid, owner_started_at, "
            "recovery_claim FROM delivery_obligations "
            "WHERE obligation_id='checkpoint-vanished-row'"
        ).fetchone()
    assert row == ("pending", 0, None, None, None)


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
    primed_source = relay.prime_routing_source.call_args.args[0]
    assert primed_source.platform == Platform.DISCORD
    assert primed_source.scope_id == "guild-1"
    assert primed_source.user_id == "user-1"
    assert primed_source.chat_type == "channel"
    native.send.assert_not_awaited()


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
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.RELAY,
    )
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
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.RELAY,
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}

    assert runner._adapter_for_source(source) is None


def test_restored_relay_owner_with_different_bot_identity_fails_closed():
    relay = _adapter("relay")
    relay.platform = Platform.RELAY
    relay.fronts_platform = MagicMock(return_value=True)
    relay.matches_transport_identity = MagicMock(return_value=False)
    restored = SessionSource(platform=Platform.DISCORD, chat_id="C1")
    stamp_source_transport_owner(
        restored,
        profile=None,
        platform=Platform.RELAY,
    )
    restored._transport_identity = "discord:original-app"
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.RELAY: relay}
    runner._profile_adapters = {}

    assert runner._adapter_for_source(restored) is None
    relay.matches_transport_identity.assert_called_once_with(
        "discord:original-app"
    )


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
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.RELAY,
    )
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