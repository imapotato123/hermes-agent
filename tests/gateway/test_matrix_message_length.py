"""Tests for Matrix outbound message length configuration (#53026)."""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _make_adapter(**extra):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
            **extra,
        },
    )
    return MatrixAdapter(config)


class TestMatrixMaxMessageLength:
    def test_default_limit_is_16000(self):
        adapter = _make_adapter()
        assert adapter.max_message_length == 16000
        assert adapter._split_threshold == 15900

    def test_extra_override(self):
        adapter = _make_adapter(max_message_length=12000)
        assert adapter.max_message_length == 12000
        assert adapter._split_threshold == 11900

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MATRIX_MAX_MESSAGE_LENGTH", "20000")
        adapter = _make_adapter()
        assert adapter.max_message_length == 20000


def test_prepared_text_plan_splits_long_matrix_content_into_bounded_posts():
    adapter = _make_adapter()

    entries = adapter.prepare_text_chunks(
        "word " * 8000,
        chat_id="!room:example.org",
    )

    assert len(entries) > 1
    assert all(len(entry["content"]) <= adapter.max_message_length for entry in entries)
    assert all(
        len(entry["recovered_content"]) <= adapter.max_message_length
        for entry in entries
    )


@pytest.mark.asyncio
async def test_prepared_text_posts_exactly_once_without_reformatting():
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(return_value="$event")
    adapter.format_message = MagicMock(side_effect=AssertionError("already formatted"))
    adapter.truncate_message = MagicMock(side_effect=AssertionError("already chunked"))

    result = await adapter.send_prepared_text_chunk(
        "!room:example.org",
        "prepared **matrix**",
    )

    assert result.success is True
    adapter._client.send_message_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_prepared_text_does_not_retry_after_ambiguous_e2ee_failure():
    adapter = _make_adapter()
    adapter._encryption = True
    adapter._client = MagicMock()
    adapter._client.crypto = MagicMock()
    adapter._client.crypto.share_keys = AsyncMock()
    adapter._client.send_message_event = AsyncMock(
        side_effect=RuntimeError("ambiguous transport failure")
    )

    result = await adapter.send_prepared_text_chunk(
        "!room:example.org",
        "prepared matrix post",
    )

    assert result.success is False
    adapter._client.send_message_event.assert_awaited_once()
    adapter._client.crypto.share_keys.assert_not_awaited()


