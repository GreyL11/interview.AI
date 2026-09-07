"""HARD-CASE ADVERSARIAL PASS -- shared scaffolding. Test-only, isolated.

Imports production code and never modifies it. Mirrors tests/conftest.py's
settings hygiene so a developer's .env cannot change a verdict.

Run any hc_*.py with: backend/.venv/Scripts/python.exe qa_hardcase/hc_<x>.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

from app.core.config import Settings, settings  # noqa: E402
from app.core.secrets import InMemorySecretStore, set_secret_store  # noqa: E402

# Same rule as tests/conftest.py: assert against the code's own defaults, not
# whatever is in backend/.env, and never touch the real OS credential store.
_pristine = Settings(_env_file=None)
for _name in type(_pristine).model_fields:
    setattr(settings, _name, getattr(_pristine, _name))
set_secret_store(InMemorySecretStore())


class Recorder:
    def __init__(self, label: str) -> None:
        self.label = label
        self.rows: list[dict] = []

    def add(self, section: str, case: str, expected: str, actual: str, **detail) -> bool:
        ok = expected == actual
        self.rows.append({
            "section": section, "case": case,
            "expected": expected, "actual": actual, "ok": ok, **detail,
        })
        if not ok:
            print(f"  [FAIL] {section} :: {case!r}\n"
                  f"         expected={expected} actual={actual} "
                  f"{json.dumps(detail, default=str)}")
        return ok

    def note(self, section: str, case: str, **detail) -> None:
        """Record an observation with no pass/fail judgement."""
        self.rows.append({"section": section, "case": case, "ok": None, **detail})

    def summary(self) -> dict:
        judged = [r for r in self.rows if r["ok"] is not None]
        failed = [r for r in judged if not r["ok"]]
        by_section: dict[str, dict[str, int]] = {}
        for row in judged:
            bucket = by_section.setdefault(row["section"], {"pass": 0, "fail": 0})
            bucket["pass" if row["ok"] else "fail"] += 1
        return {
            "label": self.label,
            "judged": len(judged),
            "passed": len(judged) - len(failed),
            "failed": len(failed),
            "observations": len(self.rows) - len(judged),
            "by_section": by_section,
        }

    def dump(self) -> None:
        out = Path(__file__).resolve().parent / f"{self.label}_results.json"
        summary = self.summary()
        out.write_text(
            json.dumps({"summary": summary, "rows": self.rows}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n=== {self.label}: {summary['passed']}/{summary['judged']} pass, "
              f"{summary['failed']} fail, {summary['observations']} observations ===")
        for section, counts in sorted(summary["by_section"].items()):
            flag = "  <-- FAILURES" if counts["fail"] else ""
            print(f"    {section:<34} pass={counts['pass']:<4} fail={counts['fail']}{flag}")
        print(f"    written: {out}")
