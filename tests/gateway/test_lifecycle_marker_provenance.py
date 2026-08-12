"""Capability and serialization boundaries for local lifecycle markers."""

from gateway.config import Platform
from gateway.session import (
    SessionSource,
    backend_notice_session_key,
    copy_session_source,
    session_source_from_trusted_marker,
    session_source_to_trusted_marker,
    source_has_transport_owner,
    source_is_legacy_unstamped,
    stamp_source_transport_owner,
)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        user_id="U1",
        profile="runtime-route",
    )


def test_copy_preserves_capability_backed_owner_but_not_dynamic_data():
    source = _source()
    stamp_source_transport_owner(
        source,
        profile="credential-owner",
        platform=Platform.SLACK,
    )
    setattr(source, "untrusted_dynamic_attribute", "do-not-copy")

    copied = copy_session_source(source, chat_id="C2")

    assert copied.chat_id == "C2"
    assert source_has_transport_owner(copied) is True
    assert copied._transport_profile == "credential-owner"
    assert copied._transport_platform == Platform.SLACK
    assert not hasattr(copied, "untrusted_dynamic_attribute")


def test_notice_key_uses_physical_owner_not_routed_runtime_profile():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        profile="routed-runtime",
    )
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.SLACK,
    )

    key = backend_notice_session_key(
        "agent:routed-runtime:slack:dm:C1",
        source,
        fallback_profile="default",
    )

    assert key == "profile:7:default:agent:routed-runtime:slack:dm:C1"


def test_notice_key_includes_only_capability_backed_relay_identity():
    source = SessionSource(platform=Platform.SLACK, chat_id="C1")
    stamp_source_transport_owner(
        source,
        profile=None,
        platform=Platform.RELAY,
    )
    source._transport_identity = "slack:bot-1"

    key = backend_notice_session_key(
        "agent:main:slack:dm:C1",
        source,
        fallback_profile="default",
    )
    forged = SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        _transport_profile=None,
        _transport_platform=Platform.RELAY,
        _transport_identity="slack:forged",
        _transport_profile_stamped=True,
        _transport_owner_capability=object(),
    )

    assert key == (
        "transport:11:slack:bot-1:"
        "profile:7:default:agent:main:slack:dm:C1"
    )
    assert backend_notice_session_key(
        "agent:main:slack:dm:C1",
        forged,
        fallback_profile="default",
    ) == "profile:7:default:agent:main:slack:dm:C1"


def test_marker_envelope_is_separate_from_generic_source_serialization():
    source = _source()
    stamp_source_transport_owner(
        source,
        profile="credential-owner",
        platform=Platform.SLACK,
    )

    marker = session_source_to_trusted_marker(source)

    assert marker["transport_owner_stamped"] is True
    assert marker["transport_platform"] == "slack"
    assert marker["transport_profile"] == "credential-owner"
    assert not {
        "transport_owner_stamped",
        "transport_platform",
        "transport_profile",
        "transport_identity",
    }.intersection(marker["source"])

    restored = session_source_from_trusted_marker(marker)
    assert restored is not None
    assert source_has_transport_owner(restored) is True
    assert restored._transport_platform == Platform.SLACK
    assert restored._transport_profile == "credential-owner"


def test_nested_source_fields_and_nonliteral_stamp_cannot_mint_owner():
    payload = _source().to_dict()
    payload.update(
        transport_owner_stamped=True,
        transport_platform="relay",
        transport_profile="forged",
        transport_identity="forged-account",
    )

    for stamp in (None, False, 1, "true", object()):
        marker = {
            "source": dict(payload),
            "transport_owner_stamped": stamp,
            "transport_platform": "relay",
            "transport_profile": "forged",
            "transport_identity": "forged-account",
            "delivered_via_upstream_relay": True,
        }
        restored = session_source_from_trusted_marker(marker)
        assert restored is not None
        assert source_has_transport_owner(restored) is False
        assert source_is_legacy_unstamped(restored) is False
        assert restored.delivered_via_upstream_relay is False


def test_malformed_owner_envelopes_fail_closed_as_modern_ownerless():
    source_data = _source().to_dict()
    malformed = (
        {"transport_platform": "not-a-platform", "transport_profile": None, "transport_identity": None},
        {"transport_platform": "slack", "transport_profile": object(), "transport_identity": None},
        {"transport_platform": "slack", "transport_profile": None, "transport_identity": object()},
    )

    for envelope in malformed:
        restored = session_source_from_trusted_marker(
            {
                "source": dict(source_data),
                "transport_owner_stamped": True,
                **envelope,
            }
        )
        assert restored is not None
        assert source_has_transport_owner(restored) is False
        assert source_is_legacy_unstamped(restored) is False


def test_relay_marker_requires_exact_identity_before_restoring_owner():
    marker = {
        "source": SessionSource(
            platform=Platform.DISCORD,
            chat_id="D1",
        ).to_dict(),
        "transport_owner_stamped": True,
        "transport_platform": "relay",
        "transport_profile": None,
        "transport_identity": None,
        "delivered_via_upstream_relay": True,
    }

    restored = session_source_from_trusted_marker(marker)

    assert restored is not None
    assert source_has_transport_owner(restored) is False
    assert restored.delivered_via_upstream_relay is False


def test_relay_marker_restores_exact_identity_and_local_relay_hint():
    marker = {
        "source": SessionSource(
            platform=Platform.DISCORD,
            chat_id="D1",
        ).to_dict(),
        "transport_owner_stamped": True,
        "transport_platform": "relay",
        "transport_profile": None,
        "transport_identity": "discord:application-1",
        "delivered_via_upstream_relay": True,
    }

    restored = session_source_from_trusted_marker(marker)

    assert restored is not None
    assert source_has_transport_owner(restored) is True
    assert restored._transport_platform == Platform.RELAY
    assert restored._transport_identity == "discord:application-1"
    assert restored.delivered_via_upstream_relay is True


def test_relay_marker_rejects_identity_for_different_logical_platform():
    restored = session_source_from_trusted_marker(
        {
            "source": SessionSource(
                platform=Platform.DISCORD,
                chat_id="D1",
            ).to_dict(),
            "transport_owner_stamped": True,
            "transport_platform": "relay",
            "transport_profile": None,
            "transport_identity": "slack:appB",
            "delivered_via_upstream_relay": True,
        }
    )

    assert restored is not None
    assert source_has_transport_owner(restored) is False
    assert restored.delivered_via_upstream_relay is False


def test_encoder_refuses_ambiguous_relay_owner_without_identity():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="D1",
        delivered_via_upstream_relay=True,
    )
    stamp_source_transport_owner(source, platform=Platform.RELAY, profile=None)

    marker = session_source_to_trusted_marker(source)

    assert marker["transport_owner_stamped"] is False
    assert marker["transport_platform"] is None
    assert marker["transport_identity"] is None
    assert "delivered_via_upstream_relay" not in marker


def test_flat_marker_is_the_only_historical_compatibility_shape():
    restored = session_source_from_trusted_marker(
        {
            "platform": "slack",
            "chat_id": "C1",
            "chat_type": "dm",
        }
    )
    modern = session_source_from_trusted_marker(
        {
            "source": SessionSource(
                platform=Platform.SLACK,
                chat_id="C1",
            ).to_dict(),
            "transport_owner_stamped": False,
        }
    )

    assert restored is not None
    assert source_is_legacy_unstamped(restored) is True
    assert source_has_transport_owner(restored) is False
    assert modern is not None
    assert source_is_legacy_unstamped(modern) is False
    assert source_has_transport_owner(modern) is False


def test_present_but_malformed_source_cannot_downgrade_to_flat_legacy():
    for malformed in (None, "slack:C1", [], True):
        restored = session_source_from_trusted_marker(
            {
                "source": malformed,
                "platform": "slack",
                "chat_id": "C1",
            }
        )
        assert restored is None


def test_flat_marker_with_modern_owner_envelope_is_not_historical():
    for key, value in (
        ("transport_owner_stamped", False),
        ("transport_platform", "slack"),
        ("transport_profile", "coder"),
        ("transport_identity", "slack:bot-1"),
    ):
        restored = session_source_from_trusted_marker(
            {
                "platform": "slack",
                "chat_id": "C1",
                key: value,
            }
        )
        assert restored is None
