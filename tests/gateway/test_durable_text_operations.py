"""Durable physical text-operation producer and owner-affinity regressions."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner


class _PreparedTextAdapter(BasePlatformAdapter):
    splits_long_messages = True

    def __init__(self, name: str):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.SLACK)
        self._name = name
        self.prepared: list[str] = []
        self.physical: list[tuple[str, str | None]] = []
        self.after_first = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {}

    def prepare_text_chunks(
        self,
        content: str,
        *,
        chat_id: str,
        reply_to: str | None = None,
        metadata=None,
    ) -> list[dict]:
        self.prepared.append(content)
        return [
            {
                "content": "one",
                "recovered_content": "recovered-one",
                "reply_to_original": True,
            },
            {
                "content": "two",
                "recovered_content": "recovered-two",
                "reply_to_original": False,
            },
        ]

    async def send_prepared_text_chunk(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata=None,
    ) -> SendResult:
        self.physical.append((content, reply_to))
        if len(self.physical) == 1 and self.after_first is not None:
            self.after_first()
        return SendResult(success=True, message_id=f"{self._name}-{len(self.physical)}")

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("durable text delivery must not call logical send()")


def _runner(first: _PreparedTextAdapter) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._profile_adapters = {"coder": {Platform.SLACK: first}}
    runner._active_profile_name = lambda: "main"
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = (
        lambda source, anchor=None: {"thread_id": source.thread_id}
    )
    return runner


@pytest.mark.asyncio
async def test_live_bundle_persists_and_checkpoints_each_prepared_text_post(
    isolated_ledger, monkeypatch
):
    first = _PreparedTextAdapter("first")
    latest = _PreparedTextAdapter("latest")
    runner = _runner(first)
    runner._share_backend_notice_state(first, profile_name="coder")
    runner._share_backend_notice_state(latest, profile_name="coder")
    source = first.build_source(chat_id="C1", message_id="m1")
    first.set_message_handler(AsyncMock(return_value="one two"))
    first._keep_typing = lambda *_args, **_kwargs: asyncio.Event().wait()

    def replace_owner() -> None:
        runner._profile_adapters["coder"][Platform.SLACK] = latest

    first.after_first = replace_owner

    async def immediate_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", immediate_sleep)
    await first._process_message_background(
        MessageEvent(text="hello", source=source, message_id="m1"),
        "agent:coder:slack:dm:C1",
    )

    assert first.prepared == ["one two"]
    assert first.physical == [("one", "m1")]
    assert latest.physical == [("two", None)]

    with dl._connect() as conn:
        operation, payload_json, state = conn.execute(
            "SELECT operation, payload_json, state FROM delivery_obligations"
        ).fetchone()
    payload = json.loads(payload_json)
    assert operation == "response_bundle"
    assert state == "delivered"
    assert payload["text_chunks"] == [
        {
            "content": "one",
            "recovered_content": "recovered-one",
            "reply_to_original": True,
        },
        {
            "content": "two",
            "recovered_content": "recovered-two",
            "reply_to_original": False,
        },
    ]
    assert payload["completed_operations"] == ["text:0", "text:1"]
