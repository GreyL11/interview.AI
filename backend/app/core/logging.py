import logging
import logging.handlers

from app.core.config import settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
#: Small on purpose: this is a diagnostic tail for a desktop app, not an audit
#: trail. Two files keep "reproduce it, then send me the log" workable without
#: letting a long session fill the user's disk.
_MAX_BYTES = 2 * 1024 * 1024
_BACKUPS = 2

_file_handler_installed = False


def _install_file_handler() -> None:
    """Attach a rotating file handler to the root logger, once.

    Best-effort: a packaged app that cannot create its log directory (locked
    down profile, full disk) must still run and still log to stdout, which the
    desktop shell captures anyway.
    """
    global _file_handler_installed
    if _file_handler_installed:
        return
    _file_handler_installed = True  # set first: a failure must not retry per call
    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            settings.logs_dir / "interview-coach.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(_FORMAT))
        logging.getLogger().addHandler(handler)
    except OSError:
        logging.getLogger(__name__).warning(
            "log_file_unavailable dir=%s; continuing with console logging only",
            settings.logs_dir,
        )


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=settings.log_level, format=_FORMAT)
    _install_file_handler()
    return logging.getLogger(name)
