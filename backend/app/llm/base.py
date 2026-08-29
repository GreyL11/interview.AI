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
    """Why a provider call failed.

    Classification happens *inside* the provider, which is the only place that
    understands its SDK's exception shapes. Everything above this line reads the
    enum and never inspects a provider-specific exception.

    The split is by *what the user can do about it*, which is why AUTH and
    NOT_CONFIGURED are separate (re-enter a key vs. enter a first key) and why
    NETWORK and SERVER are separate (check the connection vs. wait).
    """

    NOT_CONFIGURED = "not_configured"        # no API key at all
    AUTH = "auth"                            # key present but rejected (401/403)
    MODEL_UNAVAILABLE = "model_unavailable"  # 404 / unknown model for this account
    RATE_LIMIT = "rate_limit"                # 429, quota exhausted
    TIMEOUT = "timeout"                      # no response inside the budget
    NETWORK = "network"                      # could not reach the provider at all
    SERVER = "server"                        # provider-side 5xx
    MALFORMED = "malformed"                  # response did not parse into an Answer
    UNKNOWN = "unknown"


#: Failures that will produce the identical result if the same request is sent
#: again. Retrying one of these only burns the latency budget, so nothing in
#: this app may retry them.
DETERMINISTIC_ERROR_KINDS = frozenset(
    {
        LLMErrorKind.NOT_CONFIGURED,
        LLMErrorKind.AUTH,
        LLMErrorKind.MODEL_UNAVAILABLE,
    }
)


class LLMError(Exception):
    """Raised on provider failure (timeout, API error, or a response that
    doesn't parse into a valid Answer).

    `kind` classifies the failure so callers can react without knowing which
    SDK raised; `retry_after_seconds` carries a provider-supplied Retry-After
    when there was one. Both are optional so plain `LLMError("message")` call
    sites keep working.
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

    @property
    def is_deterministic(self) -> bool:
        """True when retrying this exact request cannot help."""
        return self.kind in DETERMINISTIC_ERROR_KINDS
