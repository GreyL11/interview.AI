"""Latency-aware routing and failover across LLM providers.

Sits behind the same LLMClient interface everything else already depends on,
so LiveSession, prompt building, answer parsing and validation are unchanged
and provider-unaware.

Two rules shape the whole design:

1. **Failover is only safe before the user has seen anything.** Restarting an
   answer from a second provider once text is on screen would splice two
   different answers together. The router therefore withholds provider output
   until the first *useful visible* token, then commits irreversibly to that
   provider. Withholding costs nothing: what's held back is the JSON preamble
   (`{"summary": "`), which produces no visible output anyway -- the first
   ANSWER_DELTA fires at the same moment either way.

2. **A rate limit is a routing signal, not a retry signal.** Retrying a 429
   against the same provider just burns the latency budget before the failover
   that was always going to be needed.

State is a plain in-process dict: this is a single-user desktop app, provider
health is meaningless across restarts, and a store would add I/O to the
critical path for no benefit.
"""

import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric
from app.llm.base import LLMClient, LLMError, LLMErrorKind
from app.llm.streaming import extract_partial_summary
from app.schemas.answer import Answer

logger = get_logger(__name__)

#: Rolling window of recent first-token latencies kept per provider. Small on
#: purpose: enough to smooth out one slow request, short enough to react to a
#: provider that has actually degraded.
_LATENCY_WINDOW = 5


@dataclass
class ProviderState:
    """In-memory health for one provider."""

    name: str
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_failure_at: float | None = None
    last_failure_kind: LLMErrorKind | None = None
    success_count: int = 0
    failure_count: int = 0
    first_token_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=_LATENCY_WINDOW)
    )

    def in_cooldown(self, now: float) -> bool:
        return now < self.cooldown_until

    def median_first_token_ms(self) -> float | None:
        """Median rather than mean: one pathological request shouldn't move
        routing, and with a 5-sample window the median is trivially cheap."""
        if len(self.first_token_ms) < _LATENCY_WINDOW:
            return None  # not enough evidence to route on yet
        ordered = sorted(self.first_token_ms)
        return ordered[len(ordered) // 2]

    def record_success(self, first_token_ms: float | None) -> None:
        self.consecutive_failures = 0
        self.success_count += 1
        if first_token_ms is not None:
            self.first_token_ms.append(first_token_ms)

    def record_failure(self, kind: LLMErrorKind, now: float) -> None:
        self.consecutive_failures += 1
        self.failure_count += 1
        self.last_failure_at = now
        self.last_failure_kind = kind

    def start_cooldown(self, seconds: float, now: float) -> None:
        self.cooldown_until = max(self.cooldown_until, now + seconds)


class RoutingLLMClient(LLMClient):
    """Chooses a provider per request and fails over before first visible output."""

    provider_name = "router"

    def __init__(self, providers: dict[str, LLMClient]) -> None:
        if not providers:
            raise ValueError("RoutingLLMClient needs at least one provider")
        self._providers = providers
        self._state = {name: ProviderState(name) for name in providers}
        self._last_selected: str | None = None

    def warmup(self) -> None:
        for name, provider in self._providers.items():
            try:
                provider.warmup()
            except Exception:
                logger.debug("provider_warmup_failed provider=%s", name)

    # ------------------------------------------------------------ selection

    def provider_status(self) -> list[dict]:
        """Safe, structured health for the Settings screen.

        Deliberately derived from the same ProviderState the routing decisions
        use, so what the user is shown cannot drift from what actually happens.
        Never includes a key or any part of one -- only whether one is present.
        """
        now = time.monotonic()
        order = self._priority_order()
        eligible = self._eligible(now)
        statuses: list[dict] = []
        for index, name in enumerate(order):
            state = self._state[name]
            cooling = state.in_cooldown(now)
            statuses.append(
                {
                    "name": name,
                    "model": self._providers[name].model_name,
                    "configured": True,  # unconfigured providers are never wired up
                    "available": name in eligible,
                    "cooling_down": cooling,
                    "cooldown_remaining_seconds": (
                        round(max(0.0, state.cooldown_until - now), 1) if cooling else None
                    ),
                    "role": "primary" if index == 0 else "backup",
                    "last_failure_kind": (
                        state.last_failure_kind.value if state.last_failure_kind else None
                    ),
                    "median_first_token_ms": state.median_first_token_ms(),
                }
            )
        return statuses

    @property
    def model_name(self) -> str:
        selected = self._last_selected
        return self._providers[selected].model_name if selected else ""

    def _priority_order(self) -> list[str]:
        """Configured priority, filtered to providers actually wired up."""
        configured = [
            name.strip().lower()
            for name in settings.llm_provider_priority.split(",")
            if name.strip()
        ]
        order = [name for name in configured if name in self._providers]
        # Anything configured but unlisted still gets a turn, after the
        # explicit priorities -- a provider with a key should never be
        # unreachable just because it was left out of the priority string.
        order.extend(name for name in self._providers if name not in order)
        return order

    def _eligible(self, now: float) -> list[str]:
        order = self._priority_order()
        available = [n for n in order if not self._state[n].in_cooldown(now)]
        for name in order:
            if name not in available:
                log_metric(
                    "llm_provider_skipped",
                    provider=name,
                    reason="cooldown",
                    cooldown_remaining_ms=int((self._state[name].cooldown_until - now) * 1000),
                )
        # Everything is cooling down: rather than fail outright, fall back to
        # configured order and let the request try anyway. A stale cooldown is
        # a worse outcome than one extra rejected request.
        return available or order

    def _select(self, candidates: list[str]) -> tuple[str, str]:
        """Returns (provider, reason). Priority wins unless latency-aware
        routing is on AND another healthy provider is *decisively* faster on
        enough samples -- a margin, not a coin flip, so routing doesn't
        oscillate between two comparable providers."""
        preferred = candidates[0]
        if not settings.llm_latency_aware_routing or len(candidates) < 2:
            return preferred, "priority"

        preferred_ms = self._state[preferred].median_first_token_ms()
        if preferred_ms is None:
            return preferred, "priority_insufficient_samples"

        best, best_ms = preferred, preferred_ms
        for name in candidates[1:]:
            candidate_ms = self._state[name].median_first_token_ms()
            if candidate_ms is None:
                continue
            # Challenger must beat the incumbent by the whole margin:
            # margin=0.8 means "at least 20% faster", not "faster at all".
            if candidate_ms < best_ms * settings.llm_latency_routing_margin:
                best, best_ms = name, candidate_ms

        if best is preferred:
            return preferred, "priority"
        return best, "faster_recent_first_token"

    # -------------------------------------------------------------- failure

    def _cooldown_seconds(self, error: LLMError) -> float:
        if error.retry_after_seconds is not None:
            return min(error.retry_after_seconds, settings.llm_provider_max_cooldown_seconds)
        if error.kind is LLMErrorKind.AUTH:
            # A bad key will not fix itself in 30 seconds; keep trying the
            # other provider instead of hammering this one every question.
            return settings.llm_provider_auth_cooldown_seconds
        return settings.llm_provider_cooldown_seconds

    def _handle_failure(self, name: str, error: LLMError, now: float) -> None:
        state = self._state[name]
        state.record_failure(error.kind, now)

        cooldown = 0.0
        if error.kind in (LLMErrorKind.RATE_LIMIT, LLMErrorKind.AUTH):
            cooldown = self._cooldown_seconds(error)
        elif state.consecutive_failures >= settings.llm_provider_failure_threshold:
            # Degraded rather than rate-limited: back off briefly so a
            # persistently broken provider stops being tried first.
            cooldown = settings.llm_provider_cooldown_seconds

        if cooldown > 0:
            state.start_cooldown(cooldown, now)
            log_metric(
                "llm_provider_cooldown_started",
                provider=name,
                reason=error.kind.value,
                cooldown_ms=int(cooldown * 1000),
                consecutive_failures=state.consecutive_failures,
            )

    # -------------------------------------------------------------- request

    async def stream_answer(self, prompt: str) -> AsyncIterator[str]:
        now = time.monotonic()
        candidates = self._eligible(now)
        selected, reason = self._select(candidates)
        # Try the chosen provider first, then the rest in priority order.
        attempt_order = [selected] + [n for n in candidates if n != selected]

        last_error: LLMError | None = None
        for index, name in enumerate(attempt_order):
            provider = self._providers[name]
            self._last_selected = name
            started = time.monotonic()

            if index == 0:
                log_metric(
                    "llm_provider_selected",
                    provider=name,
                    model=provider.model_name,
                    reason=reason,
                )
            else:
                log_metric(
                    "llm_provider_fallback",
                    from_provider=attempt_order[index - 1],
                    to_provider=name,
                    model=provider.model_name,
                    reason=last_error.kind.value if last_error else "unknown",
                    ms_before_fallback=elapsed_ms(now, started),
                )

            buffer = ""
            committed = False
            first_token_ms: float | None = None
            try:
                async for chunk in provider.stream_answer(prompt):
                    if committed:
                        yield chunk
                        continue
                    buffer += chunk
                    # Withhold until the first token the user would actually
                    # see. `extract_partial_summary` is the same function
                    # LiveSession uses to decide whether to emit a delta, so
                    # "committed" here means exactly "the UI is about to show
                    # something" -- no second definition of first-useful-token.
                    if extract_partial_summary(buffer):
                        committed = True
                        first_token_ms = elapsed_ms(started, time.monotonic())
                        log_metric(
                            "llm_provider_first_token",
                            provider=name,
                            model=provider.model_name,
                            time_to_first_token_ms=first_token_ms,
                        )
                        yield buffer
            except LLMError as exc:
                last_error = exc
                self._handle_failure(name, exc, time.monotonic())
                log_metric(
                    "llm_provider_request_failed",
                    provider=name,
                    model=provider.model_name,
                    failure_type=exc.kind.value,
                    after_first_token=committed,
                    duration_ms=elapsed_ms(started, time.monotonic()),
                )
                if committed:
                    # Text is already on screen. Switching now would splice two
                    # different answers together; surface the failure instead.
                    raise
                continue

            # Provider finished cleanly.
            if not committed and buffer:
                # Completed without ever producing a visible summary (e.g. an
                # empty or malformed payload). Pass it through so the existing
                # parser/validator produces its normal error rather than
                # silently swallowing the response.
                yield buffer
            self._state[name].record_success(first_token_ms)
            log_metric(
                "llm_provider_stream_completed",
                provider=name,
                model=provider.model_name,
                duration_ms=elapsed_ms(started, time.monotonic()),
                chars=len(buffer),
            )
            return

        raise last_error or LLMError("No LLM provider is configured.", kind=LLMErrorKind.AUTH)

    async def generate_answer(self, prompt: str) -> Answer:
        """Non-streaming path (the one-shot POST /question route). Same
        selection and failover rules, minus the streaming-commit concern."""
        now = time.monotonic()
        candidates = self._eligible(now)
        selected, reason = self._select(candidates)
        attempt_order = [selected] + [n for n in candidates if n != selected]

        last_error: LLMError | None = None
        for index, name in enumerate(attempt_order):
            provider = self._providers[name]
            self._last_selected = name
            started = time.monotonic()
            log_metric(
                "llm_provider_selected" if index == 0 else "llm_provider_fallback",
                provider=name,
                model=provider.model_name,
                reason=reason if index == 0 else (
                    last_error.kind.value if last_error else "unknown"
                ),
            )
            try:
                answer = await provider.generate_answer(prompt)
            except LLMError as exc:
                last_error = exc
                self._handle_failure(name, exc, time.monotonic())
                log_metric(
                    "llm_provider_request_failed",
                    provider=name,
                    model=provider.model_name,
                    failure_type=exc.kind.value,
                    after_first_token=False,
                    duration_ms=elapsed_ms(started, time.monotonic()),
                )
                continue
            self._state[name].record_success(None)
            return answer

        raise last_error or LLMError("No LLM provider is configured.", kind=LLMErrorKind.AUTH)


def build_router() -> LLMClient:
    """Wire up whichever providers are actually configured.

    Each provider is optional and independent: Gemini-only, Groq-only, and
    both are all valid. With neither configured the router still builds (so
    the app starts and typed practice works) and the existing per-answer
    "not configured" LLMError surfaces on the first question, matching the
    pre-router behavior.
    """
    from app.llm.gemini_client import GeminiClient
    from app.llm.groq_client import GroqClient

    providers: dict[str, LLMClient] = {}
    if settings.groq_enabled and settings.groq_api_key:
        providers["groq"] = GroqClient()
    if settings.gemini_enabled and settings.gemini_api_key:
        providers["gemini"] = GeminiClient()

    if not providers:
        # Nothing configured: keep Gemini as the single provider so its
        # existing, well-tested "GEMINI_API_KEY is not configured" message is
        # what the user sees, rather than a new router-specific error.
        providers["gemini"] = GeminiClient()

    logger.info(
        "llm_router_initialised providers=%s priority=%s latency_aware=%s",
        sorted(providers), settings.llm_provider_priority,
        settings.llm_latency_aware_routing,
    )
    return RoutingLLMClient(providers)
