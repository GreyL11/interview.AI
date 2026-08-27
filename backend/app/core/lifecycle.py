import json
import os
import sys
import threading
import time
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

WATCHDOG_INTERVAL_SECONDS = 2.0


def announce_ready(port: int, extra: dict[str, Any] | None = None) -> None:
    """Print the one line the desktop shell waits for.

    The shell reads stdout until this appears rather than sleeping a guessed
    number of seconds: model loading and antivirus scanning make startup time
    unpredictable, and a fixed sleep is either too slow or occasionally wrong.
    """
    payload = {"ready": True, "port": port, "pid": os.getpid()}
    if extra:
        payload.update(extra)
    print(json.dumps(payload), flush=True)


def parent_is_alive(parent_pid: int) -> bool:
    """Is the process that spawned us still running?

    On Windows, killing a parent does not kill its children, so a crashed or
    force-quit shell would otherwise leave this backend running forever, still
    holding the audio device and the database.
    """
    if parent_pid <= 0:
        return True
    if sys.platform == "win32":
        return _windows_process_alive(parent_pid)
    try:
        os.kill(parent_pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _windows_process_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        # A PID can be recycled, but within a session lifetime that is a
        # far smaller risk than leaking an orphaned backend.
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def start_parent_watchdog(
    parent_pid: int,
    on_orphaned,
    interval: float = WATCHDOG_INTERVAL_SECONDS,
    stop_event: threading.Event | None = None,
) -> threading.Thread | None:
    """Exit when the shell that spawned us disappears.

    One of three shutdown layers, alongside the shell's POST /shutdown and the
    Windows Job Object. Each covers a case the others miss: this one handles the
    shell being killed outright.
    """
    if parent_pid <= 0:
        return None

    stop = stop_event or threading.Event()

    def watch() -> None:
        while not stop.wait(interval):
            if not parent_is_alive(parent_pid):
                logger.warning("parent_process_gone pid=%d; shutting down", parent_pid)
                on_orphaned()
                return

    thread = threading.Thread(target=watch, name="parent-watchdog", daemon=True)
    thread.start()
    logger.info("parent_watchdog_started pid=%d", parent_pid)
    return thread


def request_exit(delay: float = 0.25) -> None:
    """Ask the process to stop, giving in-flight work a moment to finish.

    os._exit is used deliberately: uvicorn's graceful path can block on an open
    WebSocket, and an orphaned backend holding the microphone is worse than a
    slightly abrupt exit.
    """

    def stop() -> None:
        time.sleep(delay)
        os._exit(0)

    threading.Thread(target=stop, name="exit", daemon=True).start()
