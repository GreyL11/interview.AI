"""Merging persisted secrets into the effective configuration.

Precedence, highest first:

  1. An explicitly supplied environment value (real env var, or `.env`)
  2. A secret persisted in the OS credential store
  3. No key

Environment wins so that a developer running `uvicorn` with a `.env`, or CI
running with injected variables, behaves exactly as before -- a key someone
saved in the desktop app months ago must never silently take over a terminal
session. It also means the packaged app, which ships no `.env`, falls through
to the credential store as intended.
"""

from app.core.config import settings
from app.core.logging import get_logger
from app.core.secrets import (
    SecretStore,
    purge_obsolete_secrets,
    secret_field_names,
    secret_store,
)

logger = get_logger(__name__)


def _apply(name: str, value: str) -> bool:
    """Push a secret into the live configuration, if it has a home there.

    Any `*_api_key` name can be *stored*; only names that are actual Settings
    fields can be applied. Setting an undeclared attribute on the pydantic
    settings object would raise, so this reports rather than assumes.
    """
    if name not in type(settings).model_fields:
        return False
    setattr(settings, name, value)
    return True


def env_supplied(name: str) -> bool:
    """Did normal configuration loading already provide this value?

    A non-empty value at this point can only have come from a real environment
    variable or a `.env` file -- the field defaults are empty strings -- so this
    is the same question as "was it explicitly supplied", without needing to
    reach back into os.environ and re-implement pydantic's own resolution.
    """
    return bool(getattr(settings, name, "") or "")


def load_persisted_secrets(store: SecretStore | None = None) -> dict[str, str]:
    """Fill in any secret the environment did not supply.

    Returns a map of field name to source ("environment" | "credential store" |
    "unset") for logging and for the startup summary. Never returns values.
    """
    store = store if store is not None else secret_store()

    # Probed unconditionally, even when every key came from the environment.
    # Otherwise the packaged app's log would be silent about whether saving a
    # key is possible at all -- which is the first thing to check when a user
    # reports that their key did not survive a restart.
    logger.info("secret_store_available=%s", store.available)

    # Migration: an install upgraded from a build that still had a second
    # provider carries a credential this app will never read again.
    purged = purge_obsolete_secrets(store)
    if purged:
        logger.info("obsolete_secrets_purged names=%s", sorted(purged))

    sources: dict[str, str] = {}

    for name in sorted(secret_field_names()):
        if env_supplied(name):
            sources[name] = "environment"
            continue
        value = store.get(name)
        if value and _apply(name, value):
            sources[name] = "credential store"
        else:
            sources[name] = "unset"

    logger.info(
        "secrets_loaded %s",
        " ".join(f"{name}={source.replace(' ', '_')}" for name, source in sources.items()),
    )
    return sources


def persist_secret(name: str, value: str, store: SecretStore | None = None) -> None:
    """Save a key and apply it to the running process.

    Raises SecretStoreUnavailable if it cannot be saved. The caller decides
    whether to still apply it for the session -- this function does not make
    that choice silently.
    """
    store = store if store is not None else secret_store()
    store.set(name, value)
    _apply(name, value)


def forget_secret(name: str, store: SecretStore | None = None) -> None:
    """Remove a key from the credential store and from the running process.

    The in-memory value is cleared even when the environment originally
    supplied it, so "Remove" visibly does something. The environment value
    returns on the next start, which is the honest behaviour: this app cannot
    unset a variable its parent process set.
    """
    store = store if store is not None else secret_store()
    store.delete(name)
    _apply(name, "")
