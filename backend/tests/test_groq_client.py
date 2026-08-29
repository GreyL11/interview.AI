"""Groq provider: error classification, messages, streaming, and leakage.

Classification is the part worth testing hardest. Every one of these failures
looks the same to a user ("it didn't answer") but has a completely different
fix -- re-enter a key, pick another model, wait, check the network -- so getting
the mapping wrong turns an actionable error into a shrug.

Nothing here touches the network. Failures are raised by a stub client shaped
like the SDK's, which is enough because the code under test only ever reads an
exception's class name, status code and headers.
"""

import asyncio

import pytest

from app.core.config import settings
from app.llm.base import LLMError, LLMErrorKind
from app.llm.groq_client import GroqClient, GroqConfigError, validate_model


@pytest.fixture
def groq(monkeypatch):
    """A client with a key, so `_ensure_client` is not the thing being tested."""
    monkeypatch.setattr(settings, "groq_api_key", "TEST_KEY_DO_NOT_LEAK")
    monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-120b")
    return GroqClient()


# --------------------------------------------------------------- SDK shapes


class _Response:
    def __init__(self, status_code: int, headers: dict | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


def _sdk_error(name: str, status: int | None = None, headers: dict | None = None,
               message: str = "boom") -> Exception:
    """Build an exception that looks like the Groq SDK's, by class name.

    The real classes are matched by name, not by identity, so a stub with the
    right name exercises exactly the code path a real SDK error would.
    """
    namespace = {}
    if status is not None:
        namespace["status_code"] = status
    exc_type = type(name, (Exception,), namespace)
    exc = exc_type(message)
    if status is not None:
        exc.response = _Response(status, headers)
    return exc


# ----------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("name", "status", "expected"),
    [
        ("AuthenticationError", 401, LLMErrorKind.AUTH),
        ("PermissionDeniedError", 403, LLMErrorKind.AUTH),
        ("NotFoundError", 404, LLMErrorKind.MODEL_UNAVAILABLE),
        ("RateLimitError", 429, LLMErrorKind.RATE_LIMIT),
        ("APITimeoutError", None, LLMErrorKind.TIMEOUT),
        ("APIConnectionError", None, LLMErrorKind.NETWORK),
        ("InternalServerError", 500, LLMErrorKind.SERVER),
        ("APIResponseValidationError", None, LLMErrorKind.MALFORMED),
    ],
)
def test_every_sdk_error_class_maps_to_its_own_kind(groq, name, status, expected):
    assert groq.classify(_sdk_error(name, status)) is expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, LLMErrorKind.AUTH),
        (403, LLMErrorKind.AUTH),
        (404, LLMErrorKind.MODEL_UNAVAILABLE),
        (429, LLMErrorKind.RATE_LIMIT),
        (500, LLMErrorKind.SERVER),
        (503, LLMErrorKind.SERVER),
        (507, LLMErrorKind.SERVER),  # any 5xx, not just the enumerated ones
    ],
)
def test_an_unrecognised_class_still_classifies_by_status(groq, status, expected):
    assert groq.classify(_sdk_error("SomeFutureSdkError", status)) is expected


def test_a_bad_request_naming_the_model_is_treated_as_model_unavailable(groq):
    """Groq's other way of saying "no such model". Classifying it as UNKNOWN
    would send the user hunting for a network problem instead of a model name."""
    exc = _sdk_error("BadRequestError", 400, message="model `nope` does not exist")
    assert groq.classify(exc) is LLMErrorKind.MODEL_UNAVAILABLE


def test_an_unrelated_bad_request_is_not_blamed_on_the_model(groq):
    exc = _sdk_error("BadRequestError", 400, message="messages: too many tokens")
    assert groq.classify(exc) is LLMErrorKind.UNKNOWN


def test_asyncio_timeout_is_a_timeout(groq):
    assert groq.classify(asyncio.TimeoutError()) is LLMErrorKind.TIMEOUT


def test_a_retry_after_header_is_carried_through(groq):
    exc = _sdk_error("RateLimitError", 429, headers={"retry-after": "12"})
    error = groq._as_llm_error(exc)
    assert error.kind is LLMErrorKind.RATE_LIMIT
    assert error.retry_after_seconds == 12.0


def test_a_nonsense_retry_after_is_ignored_rather_than_crashing(groq):
    exc = _sdk_error("RateLimitError", 429, headers={"retry-after": "soon"})
    assert groq._as_llm_error(exc).retry_after_seconds is None


# -------------------------------------------------- deterministic vs retryable


@pytest.mark.parametrize(
    "kind",
    [LLMErrorKind.NOT_CONFIGURED, LLMErrorKind.AUTH, LLMErrorKind.MODEL_UNAVAILABLE],
)
def test_deterministic_failures_are_marked_as_such(kind):
    """Retrying any of these sends the identical request and gets the identical
    answer, so nothing in the app may retry them."""
    assert LLMError("x", kind=kind).is_deterministic is True


@pytest.mark.parametrize(
    "kind",
    [LLMErrorKind.RATE_LIMIT, LLMErrorKind.TIMEOUT, LLMErrorKind.NETWORK, LLMErrorKind.SERVER],
)
def test_transient_failures_are_not_marked_deterministic(kind):
    assert LLMError("x", kind=kind).is_deterministic is False


# ---------------------------------------------------------------- messages


def test_every_kind_produces_an_actionable_sentence():
    """No kind may fall through to a vague "request failed" -- that is the
    message this work existed to eliminate."""
    from app.llm.groq_client import _MESSAGES

    for kind in LLMErrorKind:
        assert kind in _MESSAGES, f"{kind} has no user-facing sentence"
        assert _MESSAGES[kind].strip()


def test_a_missing_model_names_the_model_that_is_missing(groq, monkeypatch):
    monkeypatch.setattr(settings, "groq_model", "some-retired-model")
    error = groq._as_llm_error(_sdk_error("NotFoundError", 404))
    assert "some-retired-model" in str(error)


def test_a_missing_key_says_where_to_put_one(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    client = GroqClient()
    with pytest.raises(LLMError) as caught:
        client._ensure_client()
    assert caught.value.kind is LLMErrorKind.NOT_CONFIGURED
    assert "Settings" in str(caught.value)


def test_an_error_never_echoes_the_provider_text_or_the_key(groq, monkeypatch):
    """A provider's error string can quote the request back, key included."""
    exc = _sdk_error("AuthenticationError", 401, message="bad key TEST_KEY_DO_NOT_LEAK")
    rendered = str(groq._as_llm_error(exc))
    assert "TEST_KEY_DO_NOT_LEAK" not in rendered
    assert "bad key" not in rendered


def test_a_failure_is_logged_with_provider_model_and_classification(groq, caplog):
    import logging
    import time

    with caplog.at_level(logging.INFO):
        groq._fail(_sdk_error("RateLimitError", 429), time.monotonic(), phase="request")

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "provider=groq" in logged
    assert "model=openai/gpt-oss-120b" in logged
    assert "failure=rate_limit" in logged
    assert "duration_ms=" in logged


def test_a_log_line_never_carries_the_key(groq, caplog):
    import logging
    import time

    with caplog.at_level(logging.DEBUG):
        groq.warmup()
        groq._fail(_sdk_error("AuthenticationError", 401), time.monotonic(), phase="request")

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "TEST_KEY_DO_NOT_LEAK" not in logged


# ------------------------------------------------------------ model config


def test_a_usable_model_is_accepted_and_trimmed():
    assert validate_model("  llama-3.3-70b-versatile ") == "llama-3.3-70b-versatile"
    assert validate_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"


@pytest.mark.parametrize("bad", ["", "   ", "two words", "has\ttab", "new\nline"])
def test_an_unusable_model_is_rejected_at_configuration_time(bad):
    with pytest.raises(GroqConfigError):
        validate_model(bad)


def test_the_model_is_read_through_so_a_settings_change_takes_effect(groq, monkeypatch):
    """PUT /settings can change the model at runtime; a cached copy would keep
    sending the old one until restart."""
    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    assert groq.model_name == "llama-3.3-70b-versatile"


# --------------------------------------------------------------- streaming


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeStream:
    """Yields scripted chunks, then optionally raises."""

    def __init__(self, chunks, error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


class _FakeCompletions:
    def __init__(self, result) -> None:
        self._result = result
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeClient:
    def __init__(self, result) -> None:
        self.chat = type("chat", (), {"completions": _FakeCompletions(result)})()


def _wire(groq: GroqClient, result) -> _FakeClient:
    client = _FakeClient(result)
    groq._client = client
    return client


async def _collect(groq: GroqClient, prompt="p") -> str:
    return "".join([chunk async for chunk in groq.stream_answer(prompt)])


@pytest.mark.asyncio
async def test_streaming_yields_text_deltas_and_skips_empty_chunks(groq):
    _wire(groq, _FakeStream([_Chunk('{"summ'), _Chunk(None), _Chunk('ary": "hi"}')]))
    assert await _collect(groq) == '{"summary": "hi"}'


@pytest.mark.asyncio
async def test_streaming_does_not_ask_for_json_mode(groq):
    """Groq's JSON mode buffers the whole object into one chunk, which would
    kill the partial-summary rendering the UI depends on."""
    client = _wire(groq, _FakeStream([_Chunk("x")]))
    await _collect(groq)
    assert "response_format" not in client.chat.completions.calls[0]
    assert client.chat.completions.calls[0]["stream"] is True


@pytest.mark.asyncio
async def test_a_failure_opening_the_stream_is_classified(groq):
    _wire(groq, _sdk_error("RateLimitError", 429))
    with pytest.raises(LLMError) as caught:
        await _collect(groq)
    assert caught.value.kind is LLMErrorKind.RATE_LIMIT


@pytest.mark.asyncio
async def test_a_failure_mid_stream_is_classified(groq):
    """A stream that dies halfway is a different code path from one that never
    opened, and used to reach the user as an unclassified crash."""
    _wire(groq, _FakeStream([_Chunk("partial")], error=_sdk_error("APIConnectionError")))
    with pytest.raises(LLMError) as caught:
        await _collect(groq)
    assert caught.value.kind is LLMErrorKind.NETWORK


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed_as_a_provider_failure(groq):
    """A newer question superseding this one is normal control flow, not an
    error to show the user."""
    _wire(groq, _FakeStream([_Chunk("x")], error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await _collect(groq)


@pytest.mark.asyncio
async def test_a_successful_stream_clears_the_last_error(groq):
    groq.last_error_kind = LLMErrorKind.RATE_LIMIT
    _wire(groq, _FakeStream([_Chunk("x")]))
    await _collect(groq)
    assert groq.last_error_kind is None


@pytest.mark.asyncio
async def test_a_failed_request_records_the_kind_for_settings(groq):
    _wire(groq, _sdk_error("AuthenticationError", 401))
    with pytest.raises(LLMError):
        await _collect(groq)
    assert groq.last_error_kind is LLMErrorKind.AUTH


# --------------------------------------------------------- non-streaming


class _Message:
    def __init__(self, content):
        self.content = content


class _ResponseChoice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_ResponseChoice(content)]


@pytest.mark.asyncio
async def test_a_complete_answer_is_parsed(groq):
    payload = (
        '{"summary": "s", "key_points": ["a"], "detailed_answer": "d", '
        '"approach": null, "code": null, "complexity": null, "edge_cases": null, '
        '"sections": null, "warnings": []}'
    )
    _wire(groq, _Completion(payload))
    answer = await groq.generate_answer("p")
    assert answer.summary == "s"


@pytest.mark.asyncio
async def test_the_non_streaming_path_does_ask_for_json_mode(groq):
    """Here buffering costs nothing, so the stronger guarantee is worth taking."""
    payload = '{"summary": "s", "key_points": [], "detailed_answer": "d", "warnings": []}'
    client = _wire(groq, _Completion(payload))
    await groq.generate_answer("p")
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_an_empty_response_is_malformed_not_unknown(groq):
    _wire(groq, _Completion(""))
    with pytest.raises(LLMError) as caught:
        await groq.generate_answer("p")
    assert caught.value.kind is LLMErrorKind.MALFORMED


@pytest.mark.asyncio
async def test_non_json_output_is_malformed(groq):
    _wire(groq, _Completion("I'm afraid I can't do that."))
    with pytest.raises(LLMError) as caught:
        await groq.generate_answer("p")
    assert caught.value.kind is LLMErrorKind.MALFORMED


@pytest.mark.asyncio
async def test_a_fenced_json_payload_is_still_accepted(groq):
    """Models add markdown fences even when told not to; rejecting those would
    turn a perfectly good answer into an error."""
    payload = (
        '```json\n{"summary": "s", "key_points": [], "detailed_answer": "d", '
        '"warnings": []}\n```'
    )
    _wire(groq, _Completion(payload))
    assert (await groq.generate_answer("p")).summary == "s"


# ------------------------------------------------------------- composition


def test_the_only_provider_is_groq():
    """Regression guard for the Gemini removal: the factory builds one client
    and it is this one."""
    from app.llm.groq_client import build_llm_client

    assert build_llm_client().provider_name == "groq"


def test_construction_never_raises_without_a_key(monkeypatch):
    """The app must still start, transcribe and serve documents with no key;
    only answering should fail."""
    monkeypatch.setattr(settings, "groq_api_key", "")
    assert GroqClient().configured is False
