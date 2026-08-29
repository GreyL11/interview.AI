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

from typing import Protocol

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Namespace for entries this app owns. Visible to the user in Credential
#: Manager, so it reads as a product name rather than an internal id.
SERVICE_NAME = "Interview Coach"

#: Only these may be persisted. A closed set stops a future caller from using
#: the credential store as a general key-value bucket, and makes it trivial to
#: enumerate what the app is holding on the user's machine.
SECRET_KEYS = frozenset({"groq_api_key", "gemini_api_key"})


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
    if name not in SECRET_KEYS:
        raise ValueError(f"{name} is not a storable secret")


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
            return backend.get_password(SERVICE_NAME, name)
        except Exception as exc:
            # Never include the exception's value payload; some backends echo
            # the entry back in their error text.
            logger.warning("secret_read_failed name=%s error=%s", name, type(exc).__name__)
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


def set_secret_store(store: SecretStore) -> None:
    """Swap the store. Tests use this; nothing in the app does."""
    global _store
    _store = store
