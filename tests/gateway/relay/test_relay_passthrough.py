"""Relay passthrough-over-WS forwarding (Phase 5 §5.1).

Proves the gateway side of §5.1: a connector-forwarded passthrough request
(Discord interaction, Twilio, …) arrives over the SAME outbound /relay WS as
inbound messages (a hosted gateway has no public inbound port), and the relay
adapter handles it — decoding the byte-preserved body and routing a Discord
interaction through the normal agent path (handle_message).

Mirrors test_relay_interrupt.py's wiring discipline (connect() registers the
connector->gateway handlers on the transport).
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.ws_transport import PassthroughForward, _passthrough_from_wire

from tests.gateway.relay.stub_connector import StubConnector


def _desc() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
    )


@pytest.fixture
def adapter():
    stub = StubConnector(_desc())
    stub._identities = [("discord", "appShared")]
    return RelayAdapter(PlatformConfig(), _desc(), transport=stub)


def _interaction_forward(
    payload: dict, *, bot_id: str = "appShared"
) -> PassthroughForward:
    body = json.dumps(payload).encode("utf-8")
    return PassthroughForward(
        platform="discord",
        bot_id=bot_id,
        method="POST",
        path="/interactions/discord/appShared",
        headers=[("content-type", "application/json")],
        body=body,
    )


def test_passthrough_from_wire_byte_preserves_body():
    """The wire frame's base64 body decodes back to the exact bytes (parity with
    the connector's toPassthroughForward)."""
    original = json.dumps({"type": 2, "data": {"name": "ping"}, "guild_id": "g1"}).encode("utf-8")
    wire = {
        "platform": "discord",
        "botId": "appShared",
        "method": "POST",
        "path": "/interactions/discord/appShared",
        "headers": [["content-type", "application/json"]],
        "bodyB64": base64.b64encode(original).decode("ascii"),
    }
    fwd = _passthrough_from_wire(wire)
    assert fwd.platform == "discord"
    assert fwd.bot_id == "appShared"
    assert fwd.body == original
    assert fwd.headers == [("content-type", "application/json")]


@pytest.mark.asyncio
async def test_connect_wires_passthrough_handler_over_ws(adapter):
    """connect() registers the passthrough handler on the transport so a
    connector-delivered passthrough_forward frame reaches the adapter."""
    await adapter.connect()
    stub = adapter._transport
    assert stub._passthrough is not None


@pytest.mark.asyncio
async def test_discord_interaction_routes_through_handle_message(adapter, monkeypatch):
    """A forwarded Discord application-command interaction is decoded and routed
    through the normal agent path (handle_message) with a correct session source."""
    await adapter.connect()
    stub = adapter._transport

    seen = []

    async def fake_handle(event):
        seen.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)

    fwd = _interaction_forward(
        {
            "id": "interaction-1",
            "type": 2,  # APPLICATION_COMMAND
            "channel_id": "chan-9",
            "guild_id": "guild-7",
            "data": {"name": "summarize"},
            "member": {"user": {"id": "user-3", "username": "ben"}},
        }
    )
    await stub.push_passthrough(fwd, buffer_id=None)

    assert len(seen) == 1
    ev = seen[0]
    # APPLICATION_COMMAND interactions are normalized to a leading-slash
    # command (the dispatcher's contract), not the bare registered name.
    assert ev.text == "/summarize"
    assert ev.is_command() is True
    assert ev.get_command() == "summarize"
    assert ev.source.chat_id == "chan-9"
    assert ev.source.scope_id == "guild-7"
    assert ev.source.user_id == "user-3"
    # LOGICAL platform + native-parity chat_type: the session key must match
    # the connector's capability-vault binding (interactionSessionSource →
    # buildSessionKey: platform "discord", chat_type "group") and the relay
    # text lane. Platform.RELAY / "channel" here forked the session and made
    # /sethome file the home channel under platforms.relay (invisible to cron).
    assert ev.source.platform == Platform.DISCORD
    assert ev.source.chat_type == "group"
    # Authenticated upstream-trust marker, parity with the relay text lane
    # (ws_transport._event_from_wire) — /sethome's via_relay guard keys on it.
    assert ev.source.delivered_via_upstream_relay is True
    # Scope captured so the agent's reply re-asserts scope_id for egress.
    assert adapter._scope_by_chat.get("chan-9") == "guild-7"
    # The logical platform is now recorded for egress sender selection too
    # (_capture_scope skips only the generic "relay").
    assert adapter._platform_by_chat.get("chan-9") == "discord"


@pytest.mark.asyncio
async def test_buffered_passthrough_acks_after_consumption(adapter, monkeypatch):
    await adapter.connect()
    stub = adapter._transport
    acked = []
    stub.ack_buffered_inbound = AsyncMock(
        side_effect=lambda value: acked.append(value)
    )

    async def fake_handle(event):
        handoff = getattr(event, "_relay_durable_handoff", None)
        assert handoff is not None
        handoff.set()

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    await stub.push_passthrough(
        _interaction_forward(
            {
                "id": "interaction-buffered",
                "type": 2,
                "channel_id": "chan-9",
                "guild_id": "guild-7",
                "data": {"name": "summarize"},
                "member": {"user": {"id": "user-3"}},
            }
        ),
        buffer_id="buf-pass-1",
    )
    await asyncio.sleep(0)

    assert acked == ["buf-pass-1"]


@pytest.mark.asyncio
async def test_passthrough_exact_bot_identity_survives_roundtrip_and_egress(
    monkeypatch,
):
    stub = StubConnector(_desc())
    stub._identities = [("discord", "appA"), ("discord", "appB")]
    adapter = RelayAdapter(PlatformConfig(), _desc(), transport=stub)
    await adapter.connect()
    seen = []

    async def fake_handle(event):
        seen.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    await stub.push_passthrough(
        _interaction_forward(
            {
                "id": "interaction-b",
                "type": 2,
                "channel_id": "chan-shared",
                "guild_id": "guild-b",
                "data": {"name": "summarize"},
                "member": {"user": {"id": "user-b"}},
            },
            bot_id="appB",
        )
    )

    assert len(seen) == 1
    source = seen[0].source
    assert source._transport_identity == "discord:appB"
    restored = type(source).from_dict(source.to_dict())
    assert restored._transport_identity == "discord:appB"

    result = await adapter.send_for_source(restored, "reply from app B")

    assert result.success is True
    assert stub.sent_bot_ids[-1] == "appB"
    assert "_relay_transport_identity" not in stub.sent[-1]["metadata"]


@pytest.mark.asyncio
async def test_passthrough_unknown_bot_identity_is_dropped(monkeypatch):
    stub = StubConnector(_desc())
    stub._identities = [("discord", "appA")]
    adapter = RelayAdapter(PlatformConfig(), _desc(), transport=stub)
    await adapter.connect()
    seen = []
    monkeypatch.setattr(adapter, "handle_message", lambda event: seen.append(event))

    await stub.push_passthrough(
        _interaction_forward(
            {
                "id": "interaction-unknown",
                "type": 2,
                "channel_id": "chan-shared",
                "guild_id": "guild-b",
                "data": {"name": "summarize"},
                "member": {"user": {"id": "user-b"}},
            },
            bot_id="appB",
        )
    )

    assert seen == []


@pytest.mark.asyncio
async def test_application_command_subcommand_nesting_renders_names_then_values(
    adapter, monkeypatch
):
    """SUB_COMMAND (type 1) appends its name, then recurses into its options."""
    await adapter.connect()
    stub = adapter._transport
    seen = []

    async def fake_handle(event):
        seen.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    fwd = _interaction_forward(
        {
            "id": "i-sub",
            "type": 2,
            "channel_id": "c5",
            "guild_id": "g5",
            "data": {
                "name": "skill",
                "options": [
                    {
                        "name": "run",
                        "type": 1,  # SUB_COMMAND
                        "options": [{"name": "target", "type": 3, "value": "deploy"}],
                    }
                ],
            },
            "member": {"user": {"id": "u5", "username": "ben"}},
        }
    )
    await stub.push_passthrough(fwd)
    assert len(seen) == 1
    ev = seen[0]
    assert ev.text == "/skill run deploy"
    assert ev.is_command() is True
    assert ev.get_command() == "skill"
    assert ev.get_command_args() == "run deploy"


@pytest.mark.asyncio
async def test_dm_interaction_keys_as_discord_dm(adapter, monkeypatch):
    """A guild-less (DM) interaction keys as a Discord DM: logical platform,
    chat_type 'dm', and the authenticated relay marker — the /sethome-in-DM
    shape must file under platforms.discord, never platforms.relay."""
    await adapter.connect()
    stub = adapter._transport
    seen = []

    async def fake_handle(event):
        seen.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    fwd = _interaction_forward(
        {
            "id": "i-dm",
            "type": 2,
            "channel_id": "dm-chan-1",
            "data": {"name": "sethome"},
            "user": {"id": "u9", "username": "ben"},
        }
    )
    await stub.push_passthrough(fwd)
    assert len(seen) == 1
    ev = seen[0]
    assert ev.source.platform == Platform.DISCORD
    assert ev.source.chat_type == "dm"
    assert ev.source.scope_id is None
    assert ev.source.user_id == "u9"
    assert ev.source.delivered_via_upstream_relay is True


