"""Prepared one-physical-text-post conformance for custom adapters."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_bluebubbles_preserves_paragraph_bubbles_and_posts_one(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

    adapter = BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={"server_url": "http://localhost:1234", "password": "secret"},
        )
    )
    adapter._private_api_enabled = False
    adapter._helper_connected = False
    adapter._resolve_chat_guid = AsyncMock(return_value="iMessage;-;chat")
    adapter._api_post = AsyncMock(
        return_value={"data": {"guid": "msg-1"}}
    )

    plan = adapter.prepare_text_chunks(
        "first paragraph\n\nsecond paragraph",
        chat_id="chat",
    )
    assert [item["content"] for item in plan] == [
        "first paragraph",
        "second paragraph",
    ]

    result = await adapter.send_prepared_text_chunk(
        "chat", plan[0]["content"]
    )
    assert result.success is True
    adapter._api_post.assert_awaited_once()


@pytest.mark.asyncio
async def test_whatsapp_cloud_prepared_post_calls_graph_once(monkeypatch):
    from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

    adapter = WhatsAppCloudAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "phone_number_id": "phone",
                "access_token": "token",
                "reply_prefix": "",
            },
        )
    )
    response = MagicMock(status_code=200)
    response.json.return_value = {"messages": [{"id": "wamid-1"}]}
    adapter._http_client = SimpleNamespace(post=AsyncMock(return_value=response))

    with patch("gateway.platforms.whatsapp_cloud.rich_sent_store.record"):
        plan = adapter.prepare_text_chunks(
            "hello " * 1000,
            chat_id="15551234567",
        )
        assert len(plan) > 1
        result = await adapter.send_prepared_text_chunk(
            "15551234567", plan[0]["content"]
        )

    assert result.success is True
    adapter._http_client.post.assert_awaited_once()
