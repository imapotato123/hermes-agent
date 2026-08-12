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
async def test_queued_unstamped_source_keeps_legacy_adapter_compatibility():
    legacy = _adapter("legacy")
    runner = _runner()

    await runner._deliver_queued_first_response(
        "legacy answer",
        _source(stamped=False),
        legacy,
        deliver_media=False,
    )

    legacy.send.assert_awaited_once()


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
        deliverable_routes={("slack", True, "coder")}
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
    runner.session_store = None
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
