import json

import pytest

from app.core.config import settings


@pytest.fixture
def settings_client(client, monkeypatch):
    # Snapshot and restore: these endpoints mutate process-wide settings.
    original = settings.model_dump()
    yield client
    for key, value in original.items():
        setattr(settings, key, value)


def test_get_settings_never_returns_the_key(settings_client, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "super-secret-value")

    body = settings_client.get("/settings").json()

    assert body["gemini_key_configured"] is True
    assert "super-secret-value" not in str(body)
    assert "gemini_api_key" not in body


def test_get_settings_reports_unconfigured_key(settings_client, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert settings_client.get("/settings").json()["gemini_key_configured"] is False


def test_get_settings_exposes_the_setup_fields(settings_client):
    body = settings_client.get("/settings").json()
    for field in (
        "gemini_model", "embedding_model", "stt_model", "stt_device",
        "chunk_size", "chunk_overlap", "rag_top_k", "rag_min_similarity",
        "data_dir", "audio_capture_mic", "audio_capture_loopback", "audio_available",
    ):
        assert field in body


def test_update_settings_applies_changes(settings_client):
    body = settings_client.put("/settings", json={"rag_top_k": 9}).json()
    assert body["rag_top_k"] == 9
    assert settings.rag_top_k == 9


def test_keys_cannot_be_set_through_the_generic_settings_update(settings_client):
    """Keys moved to PUT /providers/{name}/key, which is the only path that
    also persists them. Accepting them here too would give two ways to set a
    key with different persistence semantics."""
    settings_client.put("/settings", json={"gemini_api_key": "SHOULD_BE_IGNORED_123"})
    assert settings.gemini_api_key != "SHOULD_BE_IGNORED_123"


def test_partial_update_leaves_other_fields_alone(settings_client):
    before = settings_client.get("/settings").json()["chunk_size"]
    settings_client.put("/settings", json={"rag_top_k": 3})
    assert settings_client.get("/settings").json()["chunk_size"] == before


def test_update_audio_channels(settings_client):
    body = settings_client.put("/settings", json={"audio_capture_loopback": False}).json()
    assert body["audio_capture_loopback"] is False


def test_audio_devices_endpoint_never_raises(settings_client):
    response = settings_client.get("/audio/devices")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_audio_channels_documents_the_routing_rule(settings_client):
    body = settings_client.get("/audio/channels").json()
    assert set(body["channels"]) == {"MIC", "LOOPBACK"}
    assert "loopback" in body["note"].lower()


def test_models_status_lists_both_models(settings_client):
    body = settings_client.get("/models/status").json()
    kinds = {m["kind"] for m in body}
    assert kinds == {"embedding", "stt"}
    for entry in body:
        assert isinstance(entry["downloaded"], bool)
        assert entry["path"]


def test_settings_never_returns_key_material(client):
    """The API is write-only for keys. A screenshot of Settings must not be
    able to leak one, so not even a masked prefix is returned."""
    body = client.get("/settings").json()
    serialised = json.dumps(body)
    assert "api_key" not in serialised
    for provider in body["providers"]:
        assert set(provider) & {"key", "api_key", "secret"} == set()


def test_provider_status_lists_both_providers_in_priority_order(client):
    body = client.get("/settings").json()
    names = [p["name"] for p in body["providers"]]
    assert names == [n.strip() for n in body["provider_priority"].split(",")]
    assert set(names) == {"groq", "gemini"}


def test_an_unconfigured_provider_reports_unconfigured_not_missing(client):
    """It must still appear, so Settings can explain *why* it is unusable
    rather than silently omitting it."""
    body = client.get("/settings").json()
    gemini = next(p for p in body["providers"] if p["name"] == "gemini")
    assert gemini["configured"] is False
    assert gemini["available"] is False


def test_settings_report_that_changes_do_not_persist(client):
    """Mutations apply to the running process only. The UI relies on this flag
    to avoid implying a restart will keep them."""
    assert client.get("/settings").json()["settings_persist"] is False
