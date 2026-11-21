from app.llm.base import KEEPALIVE_EXPIRY_SECONDS, keepalive_limits
from app.llm.openai_compatible import OpenAICompatibleClient
from app.llm.registry import ProviderConfig


def test_limits_outlive_httpx_default():
    assert keepalive_limits().keepalive_expiry == KEEPALIVE_EXPIRY_SECONDS > 5


def test_openai_compatible_pool_keeps_connections_warm():
    cfg = ProviderConfig(provider="openai", api_key="k", model="m", base_url="https://x.test/v1")
    c = OpenAICompatibleClient(cfg)._ensure_client()
    assert c._transport._pool._keepalive_expiry == KEEPALIVE_EXPIRY_SECONDS


import pytest

from app.llm.streaming import parse_answer_payload


def test_truncated_answer_is_salvaged():
    cut = '{"summary": "Use a hash map.", "key_points": ["O(1) lookup", "handles coll'
    data = parse_answer_payload(cut)
    assert data["summary"] == "Use a hash map."
    assert data["key_points"] == ["O(1) lookup", "handles coll"]


def test_truncated_mid_key_keeps_summary():
    data = parse_answer_payload('{"summary": "Short.", "key_po')
    assert data == {"summary": "Short."}


def test_garbage_still_fails():
    with pytest.raises(Exception):
        parse_answer_payload("not json")


def test_max_tokens_in_body(monkeypatch):
    from app.core.config import settings
    from app.llm.base import answer_max_tokens
    monkeypatch.setattr(settings, "llm_answer_max_tokens", 0)
    assert answer_max_tokens() == {}
    monkeypatch.setattr(settings, "llm_answer_max_tokens", 123)
    assert answer_max_tokens() == {"max_tokens": 123}
