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
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._probe()
        return self._available

    def _probe(self) -> bool:
        try:
            import keyring
            from keyring.backends.fail import Keyring as FailKeyring
        except Exception as exc:
            logger.info("secret_store_unavailable reason=import_failed error=%s", exc)
            return False
        backend = keyring.get_keyring()
        # `fail.Keyring` is what keyring installs when it finds no real backend;
        # it raises on every operation. Treat it as absent rather than
        # discovering that on the user's first save.
        if isinstance(backend, FailKeyring):
            logger.info("secret_store_unavailable reason=no_backend")
            return False
        logger.info("secret_store_ready backend=%s", type(backend).__name__)
        return True

    def get(self, name: str) -> str | None:
        _validate(name)
        if not self.available:
            return None
        import keyring

        try:
            return keyring.get_password(SERVICE_NAME, name)
        except Exception as exc:
            # Never include the exception's value payload; some backends echo
            # the entry back in their error text.
            logger.warning("secret_read_failed name=%s error=%s", name, type(exc).__name__)
            return None

    def set(self, name: str, value: str) -> None:
        _validate(name)
        if not self.available:
            raise SecretStoreUnavailable("no OS credential store is available")
        import keyring

        try:
            keyring.set_password(SERVICE_NAME, name, value)
        except Exception as exc:
            logger.warning("secret_write_failed name=%s error=%s", name, type(exc).__name__)
            raise SecretStoreUnavailable("the OS credential store rejected the write") from exc

    def delete(self, name: str) -> None:
        _validate(name)
        if not self.available:
            return
        import keyring

        try:
            keyring.delete_password(SERVICE_NAME, name)
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


_store: SecretStore = KeyringSecretStore()


def secret_store() -> SecretStore:
    return _store


def set_secret_store(store: SecretStore) -> None:
    """Swap the store. Tests use this; nothing in the app does."""
    global _store
    _store = store
