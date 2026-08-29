"""Entry point used by the desktop shell.

    python -m app --port 8123 --token abc --parent-pid 4242

The shell picks a free port and a random token, spawns this, and waits for the
readiness line on stdout. Running uvicorn directly still works for development.
"""

import argparse
import os
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="call-assistant-backend")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. Loopback only by default; this is a local app.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default="",
                        help="Shared secret the UI must present. Empty disables the check.")
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="Exit if this process disappears. 0 disables the watchdog.")
    parser.add_argument("--data-dir", default="",
                        help="Override DATA_DIR (packaged builds point at LOCALAPPDATA).")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def apply_environment(args: argparse.Namespace) -> None:
    """Push CLI arguments into the environment before app.core.config is
    imported, so pydantic-settings picks them up as if they came from .env."""
    if args.token:
        os.environ["API_TOKEN"] = args.token
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_environment(args)

    # Before *any* import that logs: `app.core.logging` creates the log
    # directory the first time a logger is built, which would make the new data
    # directory look already-in-use and silently skip the migration.
    # Basic logging first, so the migration's own line is not swallowed. "Where
    # did my documents go?" is precisely the support question it answers, and it
    # runs before the app's file handler exists.
    import logging

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from app.core.config import migrate_legacy_data_dir

    migrate_legacy_data_dir()

    # Imported after the environment is set, not before.
    import uvicorn

    from app.core.lifecycle import announce_ready, request_exit, start_parent_watchdog

    start_parent_watchdog(args.parent_pid, on_orphaned=lambda: request_exit(0))

    from app.main import app

    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level=args.log_level, access_log=False
    )
    server = uvicorn.Server(config)

    # Announce only once the socket is actually accepting connections; a
    # readiness line printed before that would race the shell's first request.
    original_startup = server.startup

    async def startup(sockets=None):  # type: ignore[override]
        await original_startup(sockets=sockets)
        announce_ready(args.port, {"host": args.host})

    server.startup = startup  # type: ignore[method-assign]

    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
