import json
import os
import subprocess
import sys
import threading
import time

import pytest

from app.__main__ import apply_environment, parse_args
from app.core.lifecycle import announce_ready, parent_is_alive, start_parent_watchdog


# ------------------------------------------------------------------ CLI args


def test_defaults_are_local_only():
    args = parse_args([])
    assert args.host == "127.0.0.1"  # never bind to 0.0.0.0 by default
    assert args.port == 8000
    assert args.token == ""
    assert args.parent_pid == 0


def test_shell_arguments_parse():
    args = parse_args(
        ["--port", "8123", "--token", "abc", "--parent-pid", "4242", "--data-dir", "C:/data"]
    )
    assert args.port == 8123
    assert args.token == "abc"
    assert args.parent_pid == 4242
    assert args.data_dir == "C:/data"


def test_apply_environment_exports_settings(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    apply_environment(parse_args(["--token", "secret", "--data-dir", "C:/data"]))

    assert os.environ["API_TOKEN"] == "secret"
    assert os.environ["DATA_DIR"] == "C:/data"


def test_apply_environment_leaves_unset_values_alone(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    apply_environment(parse_args([]))
    assert "API_TOKEN" not in os.environ


# ------------------------------------------------------------------ readiness


def test_announce_ready_prints_one_json_line(capsys):
    announce_ready(8123, {"host": "127.0.0.1"})

    out = capsys.readouterr().out.strip()
    assert "\n" not in out  # the shell reads exactly one line
    payload = json.loads(out)
    assert payload["ready"] is True
    assert payload["port"] == 8123
    assert payload["pid"] == os.getpid()
    assert payload["host"] == "127.0.0.1"


def test_reported_pid_is_the_process_that_actually_serves():
    """The readiness line reports os.getpid() of the server process.

    This matters because it is not always the PID the shell spawned: on Windows
    a virtualenv python.exe is a launcher stub that runs the real interpreter as
    a grandchild, so the spawned handle points at the stub. The desktop shell
    uses this reported PID as its last-resort kill target; if that guarantee
    broke, dev-mode shutdown would silently orphan the backend.
    """
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        announce_ready(1234)
    assert json.loads(buffer.getvalue())["pid"] == os.getpid()


# ------------------------------------------------------------------- watchdog


def test_current_process_is_alive():
    assert parent_is_alive(os.getpid()) is True


def test_zero_pid_means_no_parent_to_watch():
    assert parent_is_alive(0) is True


def test_dead_process_is_detected():
    """Spawn a real process, kill it, and confirm we notice.

    Uses a real PID rather than a mock: the Windows implementation goes through
    OpenProcess/GetExitCodeProcess, and mocking that would test nothing.
    """
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert parent_is_alive(child.pid) is True
    finally:
        child.kill()
        child.wait(timeout=10)

    # Windows keeps the handle queryable briefly; poll rather than assume.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and parent_is_alive(child.pid):
        time.sleep(0.1)
    assert parent_is_alive(child.pid) is False


def test_watchdog_fires_when_the_parent_dies():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    fired = threading.Event()
    stop = threading.Event()

    start_parent_watchdog(child.pid, on_orphaned=fired.set, interval=0.05, stop_event=stop)
    assert not fired.wait(0.3)  # still alive: must not fire

    child.kill()
    child.wait(timeout=10)

    assert fired.wait(10), "watchdog did not notice the parent exiting"
    stop.set()


def test_watchdog_is_not_started_without_a_parent_pid():
    assert start_parent_watchdog(0, on_orphaned=lambda: None) is None


def test_watchdog_stops_when_asked():
    fired = threading.Event()
    stop = threading.Event()
    thread = start_parent_watchdog(
        os.getpid(), on_orphaned=fired.set, interval=0.05, stop_event=stop
    )
    assert thread is not None
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not fired.is_set()


# ------------------------------------------------------- packaging guardrails


def test_spec_excludes_torch():
    """The whole point of the ONNX/CTranslate2 choice is a torch-free bundle.
    If torch ever creeps back in, the installer grows by more than a gigabyte,
    so the exclusion is asserted rather than trusted."""
    from pathlib import Path

    spec = Path(__file__).resolve().parents[1] / "packaging" / "interview-coach-backend.spec"
    text = spec.read_text(encoding="utf-8")
    assert '"torch"' in text
    assert "collect_data_files(\"faster_whisper\", includes=[\"assets/*\"])" in text


def test_torch_is_not_importable():
    with pytest.raises(ImportError):
        __import__("torch")
