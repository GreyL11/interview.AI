"""Secret persistence: precedence, leakage, and honest reporting.

Everything here runs against an in-memory secret store. No test touches the
machine's real credential manager, so the suite stays deterministic and leaves
nothing behind on a developer's keychain.
"""

import json
import logging

import pytest

from app.core.config import settings
from app.core.secret_config import (
    env_supplied,
    forget_secret,
    load_persisted_secrets,
    persist_secret,
)
from app.core.secrets import (
    InMemorySecretStore,
    SecretStoreUnavailable,
    secret_store,
    set_secret_store,
)

#: Obvious fakes. Asserted absent from every public payload and captured log.
GROQ_SECRET = "TEST_GROQ_SECRET_DO_NOT_LEAK_123"
#: A provider this app has no code for. Used to prove the credential store
#: accepts any `<name>_api_key`, not just the one provider that ships today.
OTHER_SECRET = "TEST_OTHER_SECRET_DO_NOT_LEAK_456"
OTHER_KEY_NAME = "someprovider_api_key"


@pytest.fixture
def store():
    """Swap in a deterministic store and restore settings afterwards."""
    original_store = secret_store()
    original = settings.model_dump()
    replacement = InMemorySecretStore()
    set_secret_store(replacement)
    yield replacement
    set_secret_store(original_store)
    for key, value in original.items():
        setattr(settings, key, value)


@pytest.fixture
def unavailable_store():
    original_store = secret_store()
    original = settings.model_dump()
    replacement = InMemorySecretStore(available=False)
    set_secret_store(replacement)
    yield replacement
    set_secret_store(original_store)
    for key, value in original.items():
        setattr(settings, key, value)


# ------------------------------------------------------------- precedence


def test_a_persisted_secret_is_loaded_when_the_environment_supplies_nothing(store):
    settings.groq_api_key = ""
    store.set("groq_api_key", GROQ_SECRET)

    sources = load_persisted_secrets()

    assert settings.groq_api_key == GROQ_SECRET
    assert sources["groq_api_key"] == "credential store"


def test_an_environment_value_wins_over_a_persisted_secret(store):
    """A key saved in the desktop app months ago must never quietly take over
    a terminal session that was given its own."""
    settings.groq_api_key = "from-environment"
    store.set("groq_api_key", GROQ_SECRET)

    sources = load_persisted_secrets()

    assert settings.groq_api_key == "from-environment"
    assert sources["groq_api_key"] == "environment"


def test_no_key_anywhere_is_reported_as_unset(store):
    settings.groq_api_key = ""
    sources = load_persisted_secrets()
    assert settings.groq_api_key == ""
    assert sources["groq_api_key"] == "unset"


def test_restart_reloads_what_was_saved(store):
    """Simulates the whole point: save, lose process state, start again."""
    settings.groq_api_key = ""
    persist_secret("groq_api_key", GROQ_SECRET)

    settings.groq_api_key = ""  # the "restart"
    load_persisted_secrets()

    assert settings.groq_api_key == GROQ_SECRET


def test_removing_a_secret_deletes_it_from_the_store(store):
    persist_secret("groq_api_key", GROQ_SECRET)
    forget_secret("groq_api_key")

    assert store.get("groq_api_key") is None
    assert settings.groq_api_key == ""

    settings.groq_api_key = ""
    load_persisted_secrets()
    assert settings.groq_api_key == ""  # stays gone across a restart


def test_env_supplied_reflects_the_effective_configuration(store):
    settings.groq_api_key = ""
    assert env_supplied("groq_api_key") is False
    settings.groq_api_key = "x"
    assert env_supplied("groq_api_key") is True


# ------------------------------------------ storage unavailable is honest


def test_an_unavailable_store_refuses_to_pretend_it_saved(unavailable_store):
    with pytest.raises(SecretStoreUnavailable):
        persist_secret("groq_api_key", GROQ_SECRET)


def test_an_unavailable_store_reads_nothing_rather_than_failing(unavailable_store):
    settings.groq_api_key = ""
    sources = load_persisted_secrets()
    assert sources["groq_api_key"] == "unset"


def test_any_provider_key_name_may_be_stored(store):
    """The store is keyed by shape, not by an allowlist of providers, so a key
    for a provider this build has no code for is still storable and readable."""
    store.set(OTHER_KEY_NAME, OTHER_SECRET)
    assert store.get(OTHER_KEY_NAME) == OTHER_SECRET


def test_a_name_that_is_not_an_api_key_is_rejected(store):
    """Still not a general key-value bucket: the name becomes an entry in the
    user's credential manager, so it has to be recognisably ours."""
    for rejected in ("something_else", "", "Groq_API_KEY", "../etc/passwd", "groq_api_key_"):
        with pytest.raises(ValueError):
            store.set(rejected, "value")


def test_an_unknown_provider_key_is_stored_but_not_applied_to_settings(store):
    """It has nowhere to go in the running configuration, and saying otherwise
    would imply a provider that does not exist had been enabled."""
    persist_secret(OTHER_KEY_NAME, OTHER_SECRET)

    assert store.get(OTHER_KEY_NAME) == OTHER_SECRET
    assert not hasattr(settings, OTHER_KEY_NAME)
    # ...and it is not reported as a loadable secret at startup.
    assert OTHER_KEY_NAME not in load_persisted_secrets()


def test_an_obsolete_gemini_credential_is_purged_on_startup(store):
    """Migration: an install upgraded from a build that still had a second
    provider must not leave an orphaned entry in Credential Manager."""
    store.set("gemini_api_key", "TEST_LEFTOVER_GEMINI_KEY")

    load_persisted_secrets()

    assert store.get("gemini_api_key") is None


# --------------------------------------------------------------- leakage


def test_saving_a_key_returns_status_only(client, store):
    response = client.put("/providers/groq/key", json={"api_key": GROQ_SECRET})

    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is True
    assert body["persisted"] is True
    assert GROQ_SECRET not in json.dumps(body)


def test_settings_never_expose_a_saved_key(client, store):
    client.put("/providers/groq/key", json={"api_key": GROQ_SECRET})
    client.put("/providers/someprovider/key", json={"api_key": OTHER_SECRET})

    serialised = json.dumps(client.get("/settings").json())

    assert GROQ_SECRET not in serialised
    assert OTHER_SECRET not in serialised
    # Not even a fragment: no prefix, suffix, length or hash.
    assert GROQ_SECRET[:8] not in serialised
    assert str(len(GROQ_SECRET)) not in serialised.replace("chunk_size", "")


def test_provider_status_reports_configured_without_key_material(client, store):
    client.put("/providers/groq/key", json={"api_key": GROQ_SECRET})

    groq = next(
        p for p in client.get("/settings").json()["providers"] if p["name"] == "groq"
    )

    assert groq["configured"] is True
    assert GROQ_SECRET not in json.dumps(groq)


def test_replacing_a_key_never_returns_the_previous_one(client, store):
    client.put("/providers/groq/key", json={"api_key": GROQ_SECRET})
    body = client.put("/providers/groq/key", json={"api_key": "TEST_REPLACEMENT_789"}).json()

    assert GROQ_SECRET not in json.dumps(body)
    assert "TEST_REPLACEMENT_789" not in json.dumps(body)
    assert store.get("groq_api_key") == "TEST_REPLACEMENT_789"


def test_logs_never_contain_the_secret(client, store, caplog):
    with caplog.at_level(logging.DEBUG):
        client.put("/providers/groq/key", json={"api_key": GROQ_SECRET})
        client.get("/settings")
        client.delete("/providers/groq/key")

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert GROQ_SECRET not in captured
    assert GROQ_SECRET[:8] not in captured


# ------------------------------------------------- API behaviour and router


def test_removing_a_key_updates_provider_status(client, store):
    client.put("/providers/groq/key", json={"api_key": GROQ_SECRET})
    assert _provider(client, "groq")["configured"] is True

    body = client.delete("/providers/groq/key").json()

    assert body["configured"] is False
    assert body["persisted"] is False
    assert _provider(client, "groq")["configured"] is False


def test_saving_a_key_makes_the_provider_usable_without_a_restart(client, store):
    assert _provider(client, "groq")["configured"] is False

    client.put("/providers/groq/key", json={"api_key": GROQ_SECRET})

    groq = _provider(client, "groq")
    assert groq["configured"] is True
    # `active` proves the cached LLM client was rebuilt rather than left stale
    # holding the old (absent) key.
    assert groq["active"] is True


def test_the_api_admits_when_a_key_could_not_be_saved(client, unavailable_store):
    """Applied for the session, but `persisted` must be False -- and nothing
    may be written to a file to make the claim true."""
    body = client.put("/providers/groq/key", json={"api_key": GROQ_SECRET}).json()

    assert body["configured"] is True
    assert body["persisted"] is False
    assert "session" in body["detail"].lower()


def test_settings_report_whether_secure_storage_exists(client, unavailable_store):
    assert client.get("/settings").json()["secure_storage_available"] is False


def test_a_key_for_an_unknown_provider_is_still_storable(client, store):
    """Keys are stored by name shape, not against an allowlist of providers, so
    a provider this build has no code for can still have its key saved."""
    assert client.put("/providers/nope/key", json={"api_key": "x"}).status_code == 200
    assert store.get("nope_api_key") == "x"


def test_a_malformed_provider_name_is_rejected(client, store):
    """The name becomes an entry in the user's credential manager, so it is
    still constrained -- just by shape rather than by membership."""
    # Case is normalised rather than rejected -- "Groq" and "groq" are the
    # same provider -- so only genuinely malformed names are refused.
    assert client.put("/providers/NOPE/key", json={"api_key": "x"}).status_code == 200
    for bad in ("no pe", "9nope", "a" * 60, "nope-x"):
        assert client.put(f"/providers/{bad}/key", json={"api_key": "x"}).status_code == 404
        assert client.delete(f"/providers/{bad}/key").status_code == 404


def test_an_empty_key_is_rejected(client, store):
    assert client.put("/providers/groq/key", json={"api_key": "   "}).status_code == 400


def test_removing_an_environment_key_says_it_will_return(client, store):
    settings.groq_api_key = "from-environment"
    body = client.delete("/providers/groq/key").json()
    assert "restart" in body["detail"].lower()


def _provider(client, name: str) -> dict:
    return next(p for p in client.get("/settings").json()["providers"] if p["name"] == name)


# ------------------------------------------- frozen-build backend resolution


def test_the_backend_is_resolved_directly_when_discovery_finds_nothing(monkeypatch):
    """PyInstaller does not carry the entry-point metadata keyring uses to find
    its backend, so a frozen build can end up with keyring's `fail` sentinel.
    The store must import the platform backend directly instead of concluding
    that the machine has no credential store."""
    import keyring
    from keyring.backends.fail import Keyring as FailKeyring

    from app.core.secrets import KeyringSecretStore

    monkeypatch.setattr(keyring, "get_keyring", lambda: FailKeyring())

    resolved = KeyringSecretStore()._resolve()

    assert resolved is not None
    assert not isinstance(resolved, FailKeyring)


def test_a_machine_with_no_backend_at_all_reports_unavailable(monkeypatch):
    import app.core.secrets as secrets_module
    import keyring
    from keyring.backends.fail import Keyring as FailKeyring

    monkeypatch.setattr(keyring, "get_keyring", lambda: FailKeyring())
    monkeypatch.setattr(secrets_module, "_platform_backend", lambda: None)

    assert secrets_module.KeyringSecretStore().available is False


# --------------------------------------------- renamed product, kept credentials


def test_a_key_saved_under_the_previous_product_name_is_adopted(monkeypatch):
    """The app was renamed from "Interview Coach" to "Call Assistant".

    Credential Manager files an entry under the service name that was current
    when it was saved, so without this every existing user opens Settings after
    upgrading, sees "not configured", and has to find and re-enter their key.
    """
    from app.core.secrets import LEGACY_SERVICE_NAMES, SERVICE_NAME, KeyringSecretStore

    class FakeBackend:
        def __init__(self):
            self.entries = {(LEGACY_SERVICE_NAMES[0], "groq_api_key"): GROQ_SECRET}

        def get_password(self, service, name):
            return self.entries.get((service, name))

        def set_password(self, service, name, value):
            self.entries[(service, name)] = value

        def delete_password(self, service, name):
            del self.entries[(service, name)]

    store = KeyringSecretStore()
    backend = FakeBackend()
    monkeypatch.setattr(store, "_resolve", lambda: backend)

    assert store.get("groq_api_key") == GROQ_SECRET
    # Moved, not copied: the old entry must not linger in Credential Manager.
    assert (SERVICE_NAME, "groq_api_key") in backend.entries
    assert (LEGACY_SERVICE_NAMES[0], "groq_api_key") not in backend.entries


def test_the_current_name_wins_over_a_stale_legacy_entry(monkeypatch):
    """A key saved since the rename must never be shadowed by an older one."""
    from app.core.secrets import LEGACY_SERVICE_NAMES, SERVICE_NAME, KeyringSecretStore

    class FakeBackend:
        entries = {
            (SERVICE_NAME, "groq_api_key"): "current",
            (LEGACY_SERVICE_NAMES[0], "groq_api_key"): "stale",
        }

        def get_password(self, service, name):
            return self.entries.get((service, name))

        def set_password(self, service, name, value):
            self.entries[(service, name)] = value

        def delete_password(self, service, name):
            self.entries.pop((service, name), None)

    store = KeyringSecretStore()
    monkeypatch.setattr(store, "_resolve", lambda: FakeBackend())
    assert store.get("groq_api_key") == "current"
