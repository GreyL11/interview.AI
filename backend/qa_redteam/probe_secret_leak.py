"""QA RED TEAM -- isolated, test-only script. Does not modify app/ or tests/.

Section 30: secret leak test. Uses a CLEARLY FAKE key (never a real one),
threads it through GroqClient construction/error paths and the metrics
logger, and greps everything captured for the fake key substring.

Run: backend/.venv/Scripts/python.exe qa_redteam/probe_secret_leak.py
"""

import io
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.core.metrics import log_metric  # noqa: E402
from app.llm.base import LLMErrorKind  # noqa: E402
from app.llm.groq_client import GroqClient  # noqa: E402
from app.llm.registry import ProviderConfig

FAKE_KEY = "gsk_QA_REDTEAM_FAKE_KEY_DO_NOT_USE_1234567890abcdef"

results: list[dict] = []


def record(name: str, ok: bool, **detail) -> None:
    results.append({"name": name, "ok": ok, **detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + ("" if ok else f"  DETAIL={json.dumps(detail, default=str)}"))


def main() -> None:
    # Capture every log record app-wide during the probe.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    settings.groq_api_key = FAKE_KEY
    # GroqClient takes its whole configuration as a ProviderConfig now (the
    # provider-neutral registry landed after this probe was written); it no
    # longer reads `settings` itself. The key is still planted in `settings`
    # above because the rest of this probe checks Settings-side surfaces.
    client = GroqClient(ProviderConfig(
        provider="groq", model=settings.llm_model, api_key=FAKE_KEY,
    ))

    try:
        client.warmup()  # constructs the real SDK client with the fake key

        # Exercise every _MESSAGES / classify() path -- none of these must
        # ever quote the key.
        class FakeExc(Exception):
            status_code = 401

        exc = FakeExc("Unauthorized")
        kind = client.classify(exc)
        # The wording moved out of groq_client into the shared
        # `app.llm.messages` when the provider layer became provider-neutral,
        # and `settings.groq_model` became `settings.llm_model`. Both were
        # broken imports here since that refactor -- this probe has not been
        # runnable since, which is why the fix pass repaired it rather than
        # reporting a leak result it never actually produced.
        from app.llm.messages import message_for
        from app.llm.registry import display_name

        message_text = message_for(
            kind, display_name("groq"), model=settings.llm_model
        )

        log_metric(
            "qa_redteam_probe_groq_client_state",
            configured=client.configured,
            model=client.model_name,
            last_error_kind=client.last_error_kind.value if client.last_error_kind else None,
        )

        # Simulate an exception whose str() a careless provider layer might
        # have embedded the key into (the real groq_client.py explicitly
        # never does this -- this checks the actual observed behaviour).
        try:
            raise RuntimeError(f"connect failed for key {FAKE_KEY[:8]}...")
        except RuntimeError as inner:
            logging.getLogger("app.llm.groq_client").exception(
                "simulated_probe_exception"
            )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
        settings.groq_api_key = ""

    captured = buf.getvalue()
    full_key_in_logs = FAKE_KEY in captured
    record(
        "full_fake_key_never_in_captured_logs", not full_key_in_logs,
        captured_chars=len(captured),
    )

    # Message text shown to the user/UI must never contain the key.
    record("error_message_never_contains_key", FAKE_KEY not in message_text, message=message_text)

    # model_dump / repr of the client must not expose it either.
    client_repr = repr(vars(client))
    record("client_repr_never_contains_key", FAKE_KEY not in client_repr, repr=client_repr)

    # Direct check: does settings itself pretty-print safely? (pydantic
    # BaseSettings.__repr__ prints field values by default -- this is the
    # single highest-risk surface for a key leak: any code path that logs
    # `settings` wholesale, or serializes it into an exception message,
    # dumps the raw key.)
    settings.groq_api_key = FAKE_KEY
    settings_repr = repr(settings)
    settings.groq_api_key = ""
    record(
        "settings_repr_never_contains_key", FAKE_KEY not in settings_repr,
        note="Settings is a pydantic BaseSettings; its default __repr__ "
             "prints every field verbatim including groq_api_key. Nothing "
             "in this app was found to log `repr(settings)` or "
             "`settings.model_dump()` wholesale in the reviewed source, but "
             "the object itself carries no redaction -- any future code "
             "path (a debug log, an uncaught exception handler that dumps "
             "locals, a crash reporter) that touches `settings` directly "
             "would leak it. Recording the raw fact, not a specific call "
             "site.",
        settings_repr_excerpt=settings_repr[:200],
    )

    print()
    failed = [r for r in results if not r["ok"]]
    print(f"TOTAL={len(results)} FAIL={len(failed)}")
    Path(__file__).with_name("probe_secret_leak_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
