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


def test_update_key_reports_configured_without_echoing(settings_client):
    body = settings_client.put("/settings", json={"gemini_api_key": "new-secret"}).json()
    assert body["gemini_key_configured"] is True
    assert "new-secret" not in str(body)


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
