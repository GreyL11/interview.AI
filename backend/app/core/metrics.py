"""Structured latency metrics.

One line per pipeline stage, at INFO, in `metric <event> k=v k=v` form so the
whole speech-end -> first-token path can be reconstructed from a log file with
grep alone. Deliberately not a metrics framework: this is a desktop app with
one user, and a log line is the thing a support bundle already carries.
"""

from typing import Any

from app.core.logging import get_logger

logger = get_logger("app.metrics")


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, str) and " " in value:
        return f'"{value}"'
    return str(value)


def log_metric(event: str, **fields: Any) -> None:
    """Emit one timing event. Fields that are None are omitted, so callers can
    pass optional identifiers unconditionally."""
    rendered = " ".join(
        f"{key}={_format(value)}" for key, value in fields.items() if value is not None
    )
    logger.info("metric %s %s", event, rendered)


def elapsed_ms(started: float, now: float) -> int:
    """Monotonic-clock delta in whole milliseconds."""
    return int((now - started) * 1000)
