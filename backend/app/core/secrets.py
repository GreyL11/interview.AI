"""Where API keys live.

A desktop install has no editable `.env`, so a key typed into Settings has to
go somewhere that survives a restart. That somewhere is the operating system's
own credential store -- Windows Credential Manager, macOS Keychain, Secret
Service on Linux -- reached through `keyring`.

Deliberately *not* an encrypted file: any key this process could use to decrypt
such a file would have to ship alongside it, which is not encryption. If the OS
store is unavailable the store reports itself unavailable and nothing is
written. A key can still be applied for the current session; the API says so
rather than implying it was saved.

The boundary is three functions on purpose. Everything above it deals in
"is a key configured", never in key material.
"""

import re
from typing import Protocol

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Namespace for entries this app owns. Visible to the user in Credential
#: Manager, so it reads as a product name rather than an internal id.
SERVICE_NAME = "Call Assistant"

#: Names this app used before it was renamed. A stored key is filed under the
#: service name that was current when it was saved, so renaming the product
#: would otherwise orphan every existing user's API key -- they would open
#: Settings, see "not configured", and have to find and re-enter it.
#:
#: Read-through-and-adopt rather than a one-shot sweep: `keyring` cannot
#: enumerate a service's entries, so there is nothing to sweep. The first read
#: after an upgrade finds the old entry, copies it forward, and removes it.
LEGACY_SERVICE_NAMES = ("Interview Coach",)

#: What a storable secret may be called.
#:
#: Deliberately a *shape* check rather than an allowlist of known providers:
#: any provider's key can be stored, so adding one is a settings change rather
#: than a code change. It is still constrained, because the name becomes an
#: entry in the user's Windows Credential Manager -- an unbounded string there
#: would let a caller create entries the user cannot recognise or clean up.
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}_api_key$")

#: Entries this app wrote in an earlier version and no longer uses. They are
#: deleted on startup rather than left behind: an abandoned credential the user
#: cannot see in any settings screen, but which still shows up in Windows
#: Credential Manager under this app's name, is the app's mess to clean up.
OBSOLETE_SECRET_KEYS = frozenset({"gemini_api_key"})


def secret_field_names() -> frozenset[str]:
    """Secret names the running configuration actually knows how to apply.

    Derived from Settings rather than hardcoded, so a new `*_api_key` field is
    loaded from the credential store at startup with no further wiring. Storing
    a key is *not* limited to this set -- see `_validate` -- but only these can
    be pushed into the live configuration.
    """
    from app.core.config import Settings

    return frozenset(
        name for name in Settings.model_fields if _NAME_PATTERN.match(name)
    )


class SecretStoreUnavailable(RuntimeError):
    """No credential store on this machine. Raised rather than falling back to
    a file, so a caller can tell the user the truth."""


class SecretStore(Protocol):
    """Minimal contract. Implementations must never log or return key material
    anywhere except from `get`."""

    @property
    def available(self) -> bool: ...

    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None: ...

    def delete(self, name: str) -> None: ...


def _validate(name: str) -> None:
    if not _NAME_PATTERN.match(name or ""):
        raise ValueError(
            f"{name!r} is not a valid secret name; expected something like 'groq_api_key'"
        )


class KeyringSecretStore:
    """OS credential store, via `keyring`.

    Availability is probed once and cached: `keyring` picks a backend at import
    time and it does not change under a running process, and a probe per call
    would put a COM round-trip on the settings path.
    """

    def __init__(self) -> None:
        self._backend = None
        self._probed = False

    @property
    def available(self) -> bool:
        return self._resolve() is not None

    def _resolve(self):
        """Find a usable backend, once.

        keyring normally discovers its backend through package entry points --
        metadata that PyInstaller does not carry into a frozen build. When that
        discovery yields nothing usable, this falls back to importing the
        platform backend directly, which survives freezing because the module
        is a real import the analyser can see.
        """
        if self._probed:
            return self._backend
        self._probed = True

        try:
            import keyring
            from keyring.backends.fail import Keyring as FailKeyring
        except Exception as exc:
            logger.warning("secret_store_unavailable reason=import_failed error=%s", exc)
            return None

        discovered = None
        try:
            discovered = keyring.get_keyring()
        except Exception as exc:
            logger.info("secret_store_discovery_failed error=%s", type(exc).__name__)

        # `fail.Keyring` is the sentinel keyring installs when it finds nothing;
        # it raises on every operation. Treat it as absent rather than
        # discovering that on the user's first save.
        if discovered is not None and not isinstance(discovered, FailKeyring):
            self._backend = discovered
        else:
            self._backend = _platform_backend()

        if self._backend is None:
            logger.warning("secret_store_unavailable reason=no_backend")
        else:
            logger.info("secret_store_ready backend=%s", type(self._backend).__name__)
        return self._backend

    def get(self, name: str) -> str | None:
        _validate(name)
        backend = self._resolve()
        if backend is None:
            return None
        try:
            value = backend.get_password(SERVICE_NAME, name)
        except Exception as exc:
            # Never include the exception's value payload; some backends echo
            # the entry back in their error text.
            logger.warning("secret_read_failed name=%s error=%s", name, type(exc).__name__)
            return None
        if value:
            return value
        return self._adopt_legacy(backend, name)

    def _adopt_legacy(self, backend, name: str) -> str | None:
        """Move a key filed under a previous product name onto the current one.

        Runs only when the current name has no entry, so it costs one extra
        lookup on a genuinely unconfigured install and nothing thereafter.
        """
        for legacy in LEGACY_SERVICE_NAMES:
            try:
                value = backend.get_password(legacy, name)
            except Exception:
                continue
            if not value:
                continue
            try:
                backend.set_password(SERVICE_NAME, name, value)
                backend.delete_password(legacy, name)
                logger.info("secret_migrated name=%s from=%r", name, legacy)
            except Exception as exc:
                # The value is still usable this session even if re-filing
                # failed; the next start will simply try again.
                logger.warning(
                    "secret_migration_incomplete name=%s error=%s", name, type(exc).__name__
                )
            return value
        return None

    def set(self, name: str, value: str) -> None:
        _validate(name)
        backend = self._resolve()
        if backend is None:
            raise SecretStoreUnavailable("no OS credential store is available")
        try:
            backend.set_password(SERVICE_NAME, name, value)
        except Exception as exc:
            logger.warning("secret_write_failed name=%s error=%s", name, type(exc).__name__)
            raise SecretStoreUnavailable("the OS credential store rejected the write") from exc

    def delete(self, name: str) -> None:
        _validate(name)
        backend = self._resolve()
        if backend is None:
            return
        try:
            backend.delete_password(SERVICE_NAME, name)
        except Exception as exc:
            # Deleting something that is not there is success, not an error.
            logger.info("secret_delete_noop name=%s error=%s", name, type(exc).__name__)


class InMemorySecretStore:
    """Deterministic store for tests. Reports itself available so the
    persistence path is exercised without touching the machine's keychain."""

    def __init__(self, available: bool = True) -> None:
        self._values: dict[str, str] = {}
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def get(self, name: str) -> str | None:
        _validate(name)
        return self._values.get(name) if self._available else None

    def set(self, name: str, value: str) -> None:
        _validate(name)
        if not self._available:
            raise SecretStoreUnavailable("no OS credential store is available")
        self._values[name] = value

    def delete(self, name: str) -> None:
        _validate(name)
        self._values.pop(name, None)


def _platform_backend():
    """Import the platform's keyring backend directly, bypassing discovery.

    Only returns a backend that reports itself viable, so this cannot hand back
    something that will raise on first use.
    """
    import sys

    candidates: list[str] = []
    if sys.platform == "win32":
        candidates = ["keyring.backends.Windows:WinVaultKeyring"]
    elif sys.platform == "darwin":
        candidates = ["keyring.backends.macOS:Keyring"]
    else:
        candidates = ["keyring.backends.SecretService:Keyring"]

    for candidate in candidates:
        module_name, class_name = candidate.split(":")
        try:
            import importlib

            backend_class = getattr(importlib.import_module(module_name), class_name)
            # priority raises on an unusable backend; that is how keyring
            # itself decides viability.
            _ = backend_class.priority
            return backend_class()
        except Exception as exc:
            logger.info(
                "secret_backend_unavailable candidate=%s error=%s",
                candidate,
                type(exc).__name__,
            )
    return None


_store: SecretStore = KeyringSecretStore()


def secret_store() -> SecretStore:
    return _store


def purge_obsolete_secrets(store: SecretStore | None = None) -> list[str]:
    """Delete credentials this app no longer uses.

    Runs at startup. Only touches names this app is known to have written
    itself (`OBSOLETE_SECRET_KEYS`) -- never anything a user or another
    application put in the credential store. Returns the names attempted, for
    the startup log.

    Failure is not an error: a machine with no credential store has nothing to
    purge, and an entry that was already gone is the desired end state anyway.
    """
    store = store if store is not None else secret_store()
    purged: list[str] = []
    for name in sorted(OBSOLETE_SECRET_KEYS):
        try:
            store.delete(name)
            purged.append(name)
        except Exception as exc:  # pragma: no cover - store-specific failure
            logger.info(
                "obsolete_secret_purge_failed name=%s error=%s", name, type(exc).__name__
            )
    return purged


def set_secret_store(store: SecretStore) -> None:
    """Swap the store. Tests use this; nothing in the app does."""
    global _store
    _store = store
