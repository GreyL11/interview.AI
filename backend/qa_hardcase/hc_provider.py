"""HARD-CASE ADVERSARIAL PASS -- provider routing, provider failure, and
secret leakage (report sections 14, 15, 16).

Entirely mocked: no network, no real SDK. The Groq path is exercised through a
duck-typed fake SDK object injected as the client's cached handle; the
OpenAI-compatible path through httpx.MockTransport, so the real request
building, SSE parsing and error classification all run.

Run: backend/.venv/Scripts/python.exe qa_hardcase/hc_provider.py
"""

import asyncio
import io
import json
import logging

import httpx
from hc_common import Recorder  # noqa: E402  (path bootstrap lives there)

from app.core.config import settings  # noqa: E402
from app.core.deps import api_key_for  # noqa: E402
from app.llm.base import LLMError, LLMErrorKind  # noqa: E402
from app.llm.groq_client import GroqClient  # noqa: E402
from app.llm.messages import message_for  # noqa: E402
from app.llm.openai_compatible import OpenAICompatibleClient  # noqa: E402
from app.llm.registry import (  # noqa: E402
    ProviderConfig,
    ProviderConfigError,
    build_client,
    display_name,
    known_providers,
    validate_base_url,
    validate_model,
)
from app.realtime.question_understanding import (  # noqa: E402
    QuestionUnderstander,
    UnderstandingSource,
)

rec = Recorder("hc_provider")

SECRET = "SUPER_SECRET_TEST_KEY_123456789"
BASE = "https://gateway.example.invalid/v1"


# ---------------------------------------------------------------- fake SDKs

class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class FakeGroqSdk:
    """Duck-typed stand-in for `AsyncGroq`, injected as GroqClient._client."""

    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict] = []

        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer.error is not None:
                    raise outer.error
                return _Response(outer.content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def understanding_payload(**overrides) -> str:
    body = {
        "intent": "conceptual", "relationship": "new_question", "topic": "kafka",
        "domain": "data", "task": "explain", "constraints": [], "requested_output": [],
        "entities": [], "needs_previous_context": False, "needs_previous_answer": False,
        "needs_previous_code": False, "needs_attachments": False, "confidence": 0.9,
    }
    body.update(overrides)
    return json.dumps(body)


ANSWER_JSON = json.dumps({
    "summary": "Kafka partitions data across brokers.",
    "key_points": ["one", "two"],
    "detailed_answer": "detail",
})


def openai_client(
    provider: str, model: str, handler, *, api_key: str = SECRET, base_url: str = BASE
) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(ProviderConfig(
        provider=provider, model=model, api_key=api_key, base_url=base_url,
    ))
    # Inject a mock transport as the cached handle. Everything downstream --
    # body building, raise_for_status, SSE parsing, classification -- is the
    # real implementation.
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return client


def json_handler(payload: dict, status: int = 200, headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, headers=headers or {})
    return handler


# ===================================================================== 14
async def section_14_routing() -> None:
    print("\n--- 14. provider routing (explicit, never inferred) ---")

    CONFIGS = [
        ("groq", "openai/gpt-oss-120b", "", "GroqClient"),
        ("groq", "gpt-5.6-terra", "", "GroqClient"),
        ("openai", "gpt-5.6-terra", "", "OpenAICompatibleClient"),
        ("openai", "gpt-5.6-luna", "", "OpenAICompatibleClient"),
        ("openai_compatible", "arbitrary-model-name", BASE, "OpenAICompatibleClient"),
        ("omnirouter", "gpt-5.6-luna", BASE, "OpenAICompatibleClient"),
        ("omnirouter", "arbitrary-model-name", BASE, "OpenAICompatibleClient"),
    ]
    for provider, model, base_url, expected_class in CONFIGS:
        config = ProviderConfig(
            provider=provider, model=model, api_key=SECRET, base_url=base_url,
        )
        try:
            client = build_client(config)
            actual = type(client).__name__
        except Exception as exc:  # noqa: BLE001
            actual = f"{type(exc).__name__}: {exc}"
        rec.add("14-routing", f"{provider} / {model}", expected_class, actual)
        if actual == expected_class:
            rec.add(
                "14-model-is-config", f"{provider} / {model}",
                model, client.model_name,
            )
            rec.add(
                "14-provider-is-explicit", f"{provider} / {model}",
                provider, client.provider_name,
            )

    # No key-prefix inference: a Groq-shaped key on an OpenAI provider (and
    # vice versa) must not change routing.
    for provider, key, expected_class in [
        ("openai", "gsk_thisLooksLikeAGroqKey", "OpenAICompatibleClient"),
        ("groq", "sk-thisLooksLikeAnOpenAIKey", "GroqClient"),
        ("omnirouter", "sk-proj-anything", "OpenAICompatibleClient"),
        ("openai_compatible", "no-prefix-at-all", "OpenAICompatibleClient"),
    ]:
        config = ProviderConfig(
            provider=provider, model="m1", api_key=key,
            base_url=BASE if provider != "groq" else "",
        )
        client = build_client(config)
        rec.add(
            "14-no-key-inference", f"{provider} + key prefix {key.split('_')[0][:6]!r}",
            expected_class, type(client).__name__,
        )

    # Unknown provider, blank model, missing base_url: configuration errors,
    # never a crash on the first question.
    for label, config, expect in [
        ("unknown provider", ProviderConfig(provider="not_a_provider", model="m"), "ProviderConfigError"),
        ("blank model", ProviderConfig(provider="groq", model="  "), "ProviderConfigError"),
        ("model with whitespace", ProviderConfig(provider="groq", model="a model"), "ProviderConfigError"),
        ("omnirouter, no base_url", ProviderConfig(provider="omnirouter", model="m"), "ProviderConfigError"),
        ("openai_compatible, no base_url", ProviderConfig(provider="openai_compatible", model="m"), "ProviderConfigError"),
    ]:
        try:
            build_client(config)
            actual = "no error"
        except ProviderConfigError:
            actual = "ProviderConfigError"
        except Exception as exc:  # noqa: BLE001
            actual = type(exc).__name__
        rec.add("14-config-validation", label, expect, actual)

    # A base_url that is not a URL at all.
    for value, expect in [
        ("gateway.example.invalid/v1", "ProviderConfigError"),
        ("ftp://gateway/v1", "ProviderConfigError"),
        ("https://gateway/v1", "accepted"),
        ("http://localhost:1234/v1", "accepted"),
    ]:
        try:
            validate_base_url("openai_compatible", value)
            actual = "accepted"
        except ProviderConfigError:
            actual = "ProviderConfigError"
        rec.add("14-base-url-validation", value, expect, actual)

    # Arbitrary model names are accepted -- no stale allow-list.
    for model in ("gpt-5.6-terra", "gpt-5.6-luna", "arbitrary-model-name",
                  "vendor/some_model:v3", "llama-4-405b-instruct"):
        try:
            actual = validate_model(model)
        except ProviderConfigError as exc:
            actual = f"rejected: {exc}"
        rec.add("14-model-openness", model, model, actual)

    rec.note("14-known-providers", "registry", providers=sorted(known_providers()),
             display_names={p: display_name(p) for p in sorted(known_providers())})

    # ---- the SAME pipeline code, on every provider ----
    print("\n--- 14. same Understanding / Answer code, four providers ---")
    for provider, model, base_url in [
        ("openai", "gpt-5.6-terra", "https://api.openai.com/v1"),
        ("openai_compatible", "arbitrary-model-name", BASE),
        ("omnirouter", "gpt-5.6-luna", BASE),
    ]:
        client = openai_client(
            provider, model,
            json_handler({"choices": [{"message": {"content": understanding_payload()}}]}),
            base_url=base_url,
        )
        understander = QuestionUnderstander(client)
        result = await understander.understand("Explain Kafka partitions?", session_id="s", question_id=1)
        rec.add(
            "14-understanding-portable", f"{provider} / {model}",
            "LLM-sourced understanding",
            "LLM-sourced understanding"
            if result.source is UnderstandingSource.LLM else f"fallback:{result.source}",
            topic=result.topic, relationship=result.relationship.value,
        )
        answer_client = openai_client(
            provider, model,
            json_handler({"choices": [{"message": {"content": ANSWER_JSON}}]}),
            base_url=base_url,
        )
        answer = await answer_client.generate_answer("prompt")
        rec.add(
            "14-answer-portable", f"{provider} / {model}",
            "Kafka partitions data across brokers.", answer.summary,
        )

    # Groq, same two stages, same code, fake SDK.
    groq = GroqClient(ProviderConfig(provider="groq", model="openai/gpt-oss-120b", api_key=SECRET))
    groq._client = FakeGroqSdk(content=understanding_payload())
    result = await QuestionUnderstander(groq).understand("Explain Kafka partitions?", session_id="s", question_id=1)
    rec.add(
        "14-understanding-portable", "groq / openai/gpt-oss-120b",
        "LLM-sourced understanding",
        "LLM-sourced understanding"
        if result.source is UnderstandingSource.LLM else f"fallback:{result.source}",
    )
    groq_answer = GroqClient(ProviderConfig(provider="groq", model="m", api_key=SECRET))
    groq_answer._client = FakeGroqSdk(content=ANSWER_JSON)
    answer = await groq_answer.generate_answer("prompt")
    rec.add("14-answer-portable", "groq / m",
            "Kafka partitions data across brokers.", answer.summary)

    # A client with no complete_json degrades Understanding to the
    # deterministic fallback rather than erroring -- the documented contract.
    class NoStructured:
        pass

    result = await QuestionUnderstander(None).understand("Explain Kafka?", session_id="s", question_id=1)
    rec.add("14-understanding-fallback", "no completer configured",
            "deterministic fallback",
            "deterministic fallback"
            if result.source is not UnderstandingSource.LLM else "LLM")

    # api_key_for is name-driven, not branch-driven.
    for provider in ("groq", "openai", "openai_compatible", "omnirouter"):
        field = f"{provider}_api_key"
        original = getattr(settings, field)
        setattr(settings, field, SECRET)
        try:
            rec.add("14-key-by-name", provider, "found", "found"
                    if api_key_for(provider) == SECRET else "missing")
        finally:
            setattr(settings, field, original)
    rec.add("14-key-by-name", "nonexistent provider", "empty",
            "empty" if api_key_for("no_such_provider") == "" else "leaked")


# ===================================================================== 15
async def section_15_failures() -> None:
    print("\n--- 15. provider failure taxonomy ---")

    def status_handler(status: int, body: dict | None = None, headers=None):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json=body or {"error": {
                "message": f"upstream said no, key={SECRET}"}}, headers=headers or {})
        return handler

    def raising_handler(exc: Exception):
        def handler(request: httpx.Request) -> httpx.Response:
            raise exc
        return handler

    CASES = [
        ("authentication failure (401)", status_handler(401), LLMErrorKind.AUTH),
        ("authorization failure (403)", status_handler(403), LLMErrorKind.AUTH),
        ("rate limit (429)", status_handler(429, headers={"retry-after": "7"}), LLMErrorKind.RATE_LIMIT),
        ("invalid model (404)", status_handler(404), LLMErrorKind.MODEL_UNAVAILABLE),
        ("invalid model (400 'model')", status_handler(400, {"error": {"message": "unknown model foo"}}), LLMErrorKind.MODEL_UNAVAILABLE),
        ("provider unavailable (503)", status_handler(503), LLMErrorKind.SERVER),
        ("server error (500)", status_handler(500), LLMErrorKind.SERVER),
        ("timeout", raising_handler(httpx.ReadTimeout("timed out")), LLMErrorKind.TIMEOUT),
        ("network error", raising_handler(httpx.ConnectError("no route")), LLMErrorKind.NETWORK),
    ]
    for label, handler, expected_kind in CASES:
        client = openai_client("openai_compatible", "m", handler)
        try:
            await client.generate_answer("prompt")
            actual, message = "no error", ""
        except LLMError as exc:
            actual, message = exc.kind.value, str(exc)
        rec.add(
            "15-classification", label, expected_kind.value, actual,
            message=message, last_error_kind=(
                client.last_error_kind.value if client.last_error_kind else None
            ),
        )
        rec.add(
            "15-no-secret-in-error", label, "clean",
            "clean" if SECRET not in message else "LEAKED", message=message,
        )
        rec.add(
            "15-normalized-wording", label,
            message_for(expected_kind, display_name("openai_compatible"), "m"),
            message,
        )

    # Malformed body (200, but not the shape this app reads).
    for label, payload in [
        ("malformed JSON body", {"nope": True}),
        ("empty content", {"choices": [{"message": {"content": ""}}]}),
        ("non-JSON content", {"choices": [{"message": {"content": "not json at all"}}]}),
        ("wrong answer schema", {"choices": [{"message": {"content": '{"a":1}'}}]}),
    ]:
        client = openai_client("openai_compatible", "m", json_handler(payload))
        try:
            await client.generate_answer("prompt")
            actual, message = "no error", ""
        except LLMError as exc:
            actual, message = exc.kind.value, str(exc)
        except Exception as exc:  # noqa: BLE001 -- an unclassified escape IS a finding
            actual, message = f"unclassified:{type(exc).__name__}", str(exc)
        rec.add("15-classification", label, LLMErrorKind.MALFORMED.value, actual,
                message=message)
        rec.add("15-no-secret-in-error", label, "clean",
                "clean" if SECRET not in message else "LEAKED")

    # Retry-after is surfaced, not invented.
    client = openai_client("openai_compatible", "m",
                           status_handler(429, headers={"retry-after": "7"}))
    try:
        await client.generate_answer("prompt")
        retry = None
    except LLMError as exc:
        retry = exc.retry_after_seconds
    rec.add("15-retry-after", "429 with Retry-After: 7", "7.0", str(retry))

    # No retry storm: one call in, one request out. The SDK/transport retries
    # nothing a deterministic 4xx.
    hits = {"n": 0}

    def counting(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(401, json={"error": "no"})

    client = openai_client("openai_compatible", "m", counting)
    try:
        await client.generate_answer("prompt")
    except LLMError:
        pass
    rec.add("15-no-retry-storm", "401 -> request count", "1", str(hits["n"]))

    hits["n"] = 0

    def counting_500(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    client = openai_client("openai_compatible", "m", counting_500)
    try:
        await client.generate_answer("prompt")
    except LLMError:
        pass
    rec.note("15-retry-count", "500 -> request count", requests=hits["n"])

    # Cancellation stays cancellation -- never reclassified as a provider error.
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"choices": [{"message": {"content": ANSWER_JSON}}]})

    client = openai_client("openai_compatible", "m", slow)
    task = asyncio.create_task(client.generate_answer("prompt"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
        actual = "completed"
    except asyncio.CancelledError:
        actual = "CancelledError"
    except LLMError as exc:
        actual = f"reclassified as {exc.kind.value}"
    rec.add("15-cancellation", "generate_answer cancelled mid-flight",
            "CancelledError", actual)

    # Same for the streaming path.
    async def slow_stream(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, text="data: {}\n")

    client = openai_client("openai_compatible", "m", slow_stream)

    async def drain():
        async for _ in client.stream_answer("prompt"):
            pass

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
        actual = "completed"
    except asyncio.CancelledError:
        actual = "CancelledError"
    except LLMError as exc:
        actual = f"reclassified as {exc.kind.value}"
    rec.add("15-cancellation", "stream_answer cancelled mid-flight",
            "CancelledError", actual)

    # complete_json must NOT poison last_error_kind (the Settings screen
    # reports the health of answering, not of the classifier).
    client = openai_client("openai_compatible", "m", status_handler(401))
    try:
        await client.complete_json("prompt")
    except Exception:  # noqa: BLE001
        pass
    rec.add("15-understanding-isolation", "complete_json 401 leaves last_error_kind",
            "None", str(client.last_error_kind))

    # An understanding failure degrades to the deterministic fallback rather
    # than failing the turn.
    class Failing:
        async def complete_json(self, prompt, *, model="", timeout_seconds=5.0):
            raise httpx.ConnectError("no route")

    result = await QuestionUnderstander(Failing()).understand("Explain Kafka?", session_id="s", question_id=1)
    rec.add("15-understanding-isolation", "provider down -> understanding fallback",
            "deterministic fallback",
            "deterministic fallback"
            if result.source is not UnderstandingSource.LLM else "LLM")

    # Not configured: no key at all.
    client = OpenAICompatibleClient(ProviderConfig(
        provider="openai_compatible", model="m", api_key="", base_url=BASE))
    try:
        await client.generate_answer("prompt")
        actual, message = "no error", ""
    except LLMError as exc:
        actual, message = exc.kind.value, str(exc)
    rec.add("15-classification", "no API key configured",
            LLMErrorKind.NOT_CONFIGURED.value, actual, message=message)

    # ---- Groq's own exception-name mapping, same taxonomy ----
    print("\n--- 15. groq exception mapping ---")

    def named(name: str, status: int | None = None, headers=None):
        exc = type(name, (Exception,), {})("provider said no, key=" + SECRET)
        if status is not None:
            exc.status_code = status
        if headers:
            exc.headers = headers
        return exc

    GROQ_CASES = [
        ("AuthenticationError", named("AuthenticationError"), LLMErrorKind.AUTH),
        ("PermissionDeniedError", named("PermissionDeniedError"), LLMErrorKind.AUTH),
        ("NotFoundError", named("NotFoundError"), LLMErrorKind.MODEL_UNAVAILABLE),
        ("RateLimitError", named("RateLimitError", headers={"retry-after": "3"}), LLMErrorKind.RATE_LIMIT),
        ("APITimeoutError", named("APITimeoutError"), LLMErrorKind.TIMEOUT),
        ("APIConnectionError", named("APIConnectionError"), LLMErrorKind.NETWORK),
        ("InternalServerError", named("InternalServerError"), LLMErrorKind.SERVER),
        ("APIResponseValidationError", named("APIResponseValidationError"), LLMErrorKind.MALFORMED),
        ("unknown SDK error", named("SomeNewSdkError"), LLMErrorKind.UNKNOWN),
        ("status 503 only", named("SomeWrapper", status=503), LLMErrorKind.SERVER),
    ]
    for label, exc, expected_kind in GROQ_CASES:
        client = GroqClient(ProviderConfig(provider="groq", model="m", api_key=SECRET))
        client._client = FakeGroqSdk(error=exc)
        try:
            await client.generate_answer("prompt")
            actual, message = "no error", ""
        except LLMError as err:
            actual, message = err.kind.value, str(err)
        rec.add("15-groq-classification", label, expected_kind.value, actual,
                message=message)
        rec.add("15-no-secret-in-error", f"groq {label}", "clean",
                "clean" if SECRET not in message else "LEAKED", message=message)

    # Groq cancellation.
    class SlowSdk(FakeGroqSdk):
        def __init__(self):
            super().__init__(content=ANSWER_JSON)
            outer = self

            class _Completions:
                async def create(self, **kwargs):
                    await asyncio.sleep(5)
                    return _Response(outer.content)

            class _Chat:
                completions = _Completions()

            self.chat = _Chat()

    client = GroqClient(ProviderConfig(provider="groq", model="m", api_key=SECRET))
    client._client = SlowSdk()
    task = asyncio.create_task(client.generate_answer("prompt"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
        actual = "completed"
    except asyncio.CancelledError:
        actual = "CancelledError"
    except LLMError as exc:
        actual = f"reclassified as {exc.kind.value}"
    rec.add("15-cancellation", "groq generate_answer cancelled", "CancelledError", actual)


# ===================================================================== 16
async def section_16_secrets() -> None:
    print("\n--- 16. secret leak surfaces ---")

    def clean(label: str, haystack: str, **detail) -> None:
        rec.add(
            "16-secret", label, "clean",
            "clean" if SECRET not in haystack else "LEAKED",
            sample=haystack[:400] if SECRET in haystack else None, **detail,
        )

    config = ProviderConfig(provider="openai_compatible", model="m",
                            api_key=SECRET, base_url=BASE)
    clean("repr(ProviderConfig)", repr(config))
    clean("str(ProviderConfig)", str(config))
    clean("f-string of ProviderConfig", f"{config}")
    clean("repr of a dict holding the config", repr({"cfg": config}))

    original = settings.groq_api_key
    original_openai = settings.openai_compatible_api_key
    settings.groq_api_key = SECRET
    settings.openai_compatible_api_key = SECRET
    try:
        clean("repr(settings)", repr(settings))
        clean("str(settings)", str(settings))
        # model_dump is NOT redacted by design (it feeds internal use), so this
        # is recorded as an observation with the fact stated plainly.
        dumped = json.dumps(settings.model_dump(mode="json"), default=str)
        rec.note("16-serialization", "settings.model_dump(mode='json')",
                 contains_secret=SECRET in dumped)
        rec.note("16-serialization", "settings.model_dump_json()",
                 contains_secret=SECRET in settings.model_dump_json())

        groq = GroqClient(ProviderConfig(provider="groq", model="m", api_key=SECRET))
        clean("repr(GroqClient)", repr(groq))
        oai = OpenAICompatibleClient(config)
        clean("repr(OpenAICompatibleClient)", repr(oai))

        # Every user-facing failure sentence, every provider.
        for kind in LLMErrorKind:
            for provider in sorted(known_providers()):
                clean(f"message_for({kind.value}, {provider})",
                      message_for(kind, display_name(provider), "m"))

        # Logs. The clients log initialisation and every failure; capture the
        # whole app logger tree while a request fails.
        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger("app")
        previous_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            client = openai_client(
                "openai_compatible", "m",
                lambda request: httpx.Response(401, json={
                    "error": {"message": f"bad key {SECRET}"}}),
            )
            try:
                await client.generate_answer("prompt")
            except LLMError:
                pass
            client.warmup()
            groq_failing = GroqClient(ProviderConfig(
                provider="groq", model="m", api_key=SECRET))
            groq_failing._client = FakeGroqSdk(
                error=type("AuthenticationError", (Exception,), {})(f"key {SECRET}"))
            try:
                await groq_failing.generate_answer("prompt")
            except LLMError:
                pass
            # And the metric stream, which goes through the same logger.
            from app.core.metrics import log_metric
            log_metric("hc_probe_metric", provider="openai_compatible", model="m")
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_level)
        clean("app logger output (init + failures + metrics)", buffer.getvalue())

        # Exception tracebacks: the cause chain must not carry the key either.
        import traceback
        client = openai_client(
            "openai_compatible", "m",
            lambda request: httpx.Response(401, json={"error": f"key {SECRET}"}),
        )
        try:
            await client.generate_answer("prompt")
        except LLMError as exc:
            rendered = "".join(traceback.format_exception(exc))
        rec.note(
            "16-traceback", "full traceback of a 401 (includes the httpx cause)",
            contains_secret=SECRET in rendered,
            note="httpx.HTTPStatusError.__str__ carries the response body, so a "
                 "raw traceback dump can carry whatever the provider echoed. "
                 "LLMError's own message is clean (checked above).",
        )

        # HTTP surfaces: settings, health, models.
        from fastapi.testclient import TestClient

        from app.main import app
        with TestClient(app) as http:
            for path in ("/api/settings", "/api/health", "/api/models/status"):
                try:
                    response = http.get(path)
                    clean(f"GET {path}", response.text, status=response.status_code)
                except Exception as exc:  # noqa: BLE001
                    rec.note("16-http", path, error=f"{type(exc).__name__}: {exc}")
            # Writing a key must not echo it back.
            response = http.put("/api/providers/groq/key", json={"api_key": SECRET})
            clean("PUT /api/providers/groq/key", response.text,
                  status=response.status_code)
            response = http.get("/api/settings")
            clean("GET /api/settings after a key write", response.text)
    finally:
        settings.groq_api_key = original
        settings.openai_compatible_api_key = original_openai

    # WebSocket events: drive a real LiveSession whose provider fails with a
    # message echoing the key, and inspect every emitted event.
    from app.realtime.events import EventType
    from app.sessions.schemas import TranscriptSource
    from tests.replay_harness import ReplayHarness

    class LeakyLLM:
        model_name = "m"

        async def generate_answer(self, prompt):
            raise LLMError(f"provider echoed the key {SECRET}", kind=LLMErrorKind.AUTH)

        async def stream_answer(self, prompt):
            raise LLMError(f"provider echoed the key {SECRET}", kind=LLMErrorKind.AUTH)
            yield ""

    h = ReplayHarness(llm=LeakyLLM(), stabilization_ms=1)
    try:
        await h.live.on_transcript(
            "How would you design a rate limiter?", TranscriptSource.LOOPBACK, True)
        await h.settle()
        wire = json.dumps([
            {"type": e.type.value, "turn_id": e.turn_id, "data": e.data}
            for e in h.result.events
        ], default=str)
        rec.note(
            "16-websocket", "events emitted for a turn whose provider echoed the key",
            contains_secret=SECRET in wire,
            error_events=[e.data for e in h.result.of(EventType.ANSWER_ERROR)],
            note="This measures whether the WS layer re-broadcasts a provider "
                 "message verbatim. The provider clients never build such a "
                 "message (see 16-secret above); LeakyLLM does so deliberately.",
        )
    finally:
        await h.close()


async def main() -> None:
    await section_14_routing()
    await section_15_failures()
    await section_16_secrets()
    rec.dump()


if __name__ == "__main__":
    asyncio.run(main())
