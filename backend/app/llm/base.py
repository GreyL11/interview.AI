from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum

from app.schemas.answer import Answer


class LLMClient(ABC):
    """Interface for the reasoning backend. Business logic depends only on
    this, so providers can be swapped (or routed between) without touching
    the orchestrator."""

    #: Short identifier used in metrics and routing decisions. Overridden by
    #: each concrete provider; the router reads it rather than isinstance().
    provider_name: str = "unknown"

    @property
    def model_name(self) -> str:
        """Model actually in use, for metrics. Providers override."""
        return ""

    def warmup(self) -> None:
        """Pay one-off setup cost now instead of on the first question.

        Default is a no-op; providers whose SDK import or client construction
        is expensive override it. Must never raise: a missing key or package
        still has to surface per-answer, not at startup.
        """
        return None

    @abstractmethod
    async def generate_answer(self, prompt: str) -> Answer:
        ...

    async def stream_answer(self, prompt: str) -> AsyncIterator[str]:
        """Yield raw response text as it arrives.

        Default implementation falls back to the non-streaming call and emits
        one chunk, so every client satisfies the streaming contract even if the
        provider cannot stream.
        """
        answer = await self.generate_answer(prompt)
        yield answer.model_dump_json()


class LLMErrorKind(StrEnum):
    """Why a provider call failed, normalised across SDKs.

    Classification happens *inside* each provider, which is the only place
    that understands its own SDK's exception shapes; the router reads this
    enum and never inspects a provider-specific exception.
    """

    RATE_LIMIT = "rate_limit"   # 429 / quota exhausted -- cooldown + fail over
    TRANSIENT = "transient"     # 5xx and friends -- provider may retry, then fail over
    TIMEOUT = "timeout"         # network stall
    AUTH = "auth"               # bad/missing key, permission denied -- never retry
    UNKNOWN = "unknown"


class LLMError(Exception):
    """Raised on provider failure (timeout, API error, or a response that
    doesn't parse into a valid Answer).

    `kind` and `retry_after_seconds` let the router decide between cooling a
    provider down and failing over, without knowing which SDK raised. Both
    are optional so existing `LLMError("message")` call sites keep working.
    """

    def __init__(
        self,
        message: str,
        kind: LLMErrorKind = LLMErrorKind.UNKNOWN,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after_seconds = retry_after_seconds
