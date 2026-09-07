"""The OpenAI-compatible transport: the shared implementation behind OpenAI,
OmniRouter (an OpenAI-compatible endpoint -- see docs/llm-providers.md), and
any self-hosted gateway.

Nothing here touches the network. `httpx.MockTransport` intercepts the actual
HTTP call and hands back a scripted response, which is enough because the
code under test only ever reads a status code, a header, and a JSON body --
the same contract every OpenAI-compatible provider promises.
"""

import asyncio
import json

import httpx
import pytest

from app.llm.base import LLMError, LLMErrorKind
from app.llm.openai_compatible import OpenAICompatibleClient
from app.llm.registry import ProviderConfig

_ANSWER_JSON = (
    '{"summary": "s", "key_points": ["a"], "detailed_answer": "d", '
    '"approach": null, "code": null, "complexity": null, "edge_cases": null, '
    '"sections": null, "warnings": []}'
)


def _client(handler, *, provider="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1"):
    config = ProviderConfig(
        provider=provider, model=model, api_key="TEST_KEY_DO_NOT_LEAK", base_url=base_url,
    )
    client = OpenAICompatibleClient(config)
    client._client = httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))
    return client


def _json_response(status: int, body: dict, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=body, headers=headers or {})


# --------------------------------------------------------------- generation


@pytest.mark.asyncio
async def test_a_complete_answer_is_parsed():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"choices": [{"message": {"content": _ANSWER_JSON}}]})

    answer = await _client(handler).generate_answer("p")
    assert answer.summary == "s"


@pytest.mark.asyncio
async def test_the_request_carries_the_configured_model_and_asks_for_json_mode():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _json_response(200, {"choices": [{"message": {"content": _ANSWER_JSON}}]})

    await _client(handler, model="gpt-5.6-terra").generate_answer("p")
    assert captured["body"]["model"] == "gpt-5.6-terra"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["stream"] is False


@pytest.mark.asyncio
async def test_gpt_5_6_luna_is_just_a_configured_model_identifier():
    """No code path here branches on the model name -- Terra, Luna, and a
    model nobody has shipped yet all take the same route."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _json_response(200, {"choices": [{"message": {"content": _ANSWER_JSON}}]})

    client = _client(handler, model="gpt-5.6-luna")
    await client.generate_answer("p")
    assert captured["body"]["model"] == "gpt-5.6-luna"
    assert client.model_name == "gpt-5.6-luna"


# ------------------------------------------------------------- classification


@pytest.mark.asyncio
async def test_a_401_is_classified_as_auth():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(401, {"error": "invalid key"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.AUTH


@pytest.mark.asyncio
async def test_a_403_is_classified_as_auth():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(403, {"error": "forbidden"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.AUTH


@pytest.mark.asyncio
async def test_a_404_is_classified_as_model_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(404, {"error": "no such model"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.MODEL_UNAVAILABLE


@pytest.mark.asyncio
async def test_a_400_naming_the_model_is_model_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(400, {"error": "model 'nope' does not exist"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.MODEL_UNAVAILABLE


@pytest.mark.asyncio
async def test_an_unrelated_400_is_not_blamed_on_the_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(400, {"error": "messages: too many tokens"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.UNKNOWN


@pytest.mark.asyncio
async def test_a_429_is_classified_as_rate_limit_with_retry_after():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(429, {"error": "slow down"}, headers={"retry-after": "7"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.RATE_LIMIT
    assert caught.value.retry_after_seconds == 7.0


@pytest.mark.asyncio
async def test_a_5xx_is_classified_as_server():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(503, {"error": "overloaded"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.SERVER


@pytest.mark.asyncio
async def test_a_connection_failure_is_classified_as_network():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("could not connect", request=request)

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.NETWORK


@pytest.mark.asyncio
async def test_a_read_timeout_is_classified_as_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_a_non_json_body_is_malformed_not_unknown():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.MALFORMED


@pytest.mark.asyncio
async def test_a_response_missing_choices_is_malformed():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"unexpected": "shape"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.MALFORMED


@pytest.mark.asyncio
async def test_an_empty_completion_is_malformed():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"choices": [{"message": {"content": ""}}]})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    assert caught.value.kind is LLMErrorKind.MALFORMED


@pytest.mark.asyncio
async def test_a_fenced_json_payload_is_still_accepted():
    fenced = f"```json\n{_ANSWER_JSON}\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"choices": [{"message": {"content": fenced}}]})

    assert (await _client(handler).generate_answer("p")).summary == "s"


@pytest.mark.asyncio
async def test_an_error_never_echoes_the_key_or_the_provider_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(401, {"error": "bad key TEST_KEY_DO_NOT_LEAK"})

    with pytest.raises(LLMError) as caught:
        await _client(handler).generate_answer("p")
    rendered = str(caught.value)
    assert "TEST_KEY_DO_NOT_LEAK" not in rendered
    assert "bad key" not in rendered


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed_as_a_provider_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _client(handler).generate_answer("p")


# ------------------------------------------------------------- configuration


@pytest.mark.asyncio
async def test_a_missing_key_is_not_configured():
    config = ProviderConfig(provider="openai", model="m", base_url="https://api.openai.com/v1")
    client = OpenAICompatibleClient(config)
    assert client.configured is False
    with pytest.raises(LLMError) as caught:
        client._ensure_client()
    assert caught.value.kind is LLMErrorKind.NOT_CONFIGURED


def test_openai_gets_its_default_endpoint_without_a_base_url():
    client = OpenAICompatibleClient(ProviderConfig(provider="openai", model="m", api_key="x"))
    assert client._base_url == "https://api.openai.com/v1"


def test_an_omnirouter_style_endpoint_uses_the_configured_base_url():
    """OmniRouter (and any self-hosted gateway) has no fixed endpoint of its
    own -- the whole point of the generic transport is that the base_url is
    just configuration."""
    client = OpenAICompatibleClient(
        ProviderConfig(
            provider="omnirouter", model="some-model", api_key="x",
            base_url="http://localhost:20128/v1",
        )
    )
    assert client._base_url == "http://localhost:20128/v1"
    assert client.provider_name == "omnirouter"


# ------------------------------------------------------- question understanding


@pytest.mark.asyncio
async def test_complete_json_uses_the_override_model_when_given():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _json_response(200, {"choices": [{"message": {"content": "{}"}}]})

    client = _client(handler, model="gpt-5.6-terra")
    await client.complete_json("classify this", model="gpt-5.6-luna", timeout_seconds=1.0)
    assert captured["body"]["model"] == "gpt-5.6-luna"


@pytest.mark.asyncio
async def test_complete_json_falls_back_to_the_client_model_when_no_override_is_given():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _json_response(200, {"choices": [{"message": {"content": "{}"}}]})

    client = _client(handler, model="gpt-5.6-terra")
    await client.complete_json("classify this", timeout_seconds=1.0)
    assert captured["body"]["model"] == "gpt-5.6-terra"


# --------------------------------------------------------------------- streaming


@pytest.mark.asyncio
async def test_streaming_yields_text_deltas_from_sse_lines():
    sse = (
        b'data: {"choices":[{"delta":{"content":"{\\"summ"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"ary\\": \\"hi\\"}"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    client = _client(handler)
    chunks = [chunk async for chunk in client.stream_answer("p")]
    assert "".join(chunks) == '{"summary": "hi"}'


@pytest.mark.asyncio
async def test_streaming_does_not_ask_for_json_mode():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    client = _client(handler)
    async for _ in client.stream_answer("p"):
        pass
    assert "response_format" not in captured["body"]
    assert captured["body"]["stream"] is True


@pytest.mark.asyncio
async def test_a_failure_opening_the_stream_is_classified():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(429, {"error": "slow down"})

    with pytest.raises(LLMError) as caught:
        async for _ in _client(handler).stream_answer("p"):
            pass
    assert caught.value.kind is LLMErrorKind.RATE_LIMIT
