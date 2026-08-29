"""Model download lifecycle.

The UI used to be given a boolean, which cannot tell "never started" from
"downloading right now" from "failed halfway" — so a first run looked identical
to a broken one for the several minutes a 250MB download takes.

The rule these tests protect: **a model is never reported ready until it has
actually loaded.** Files on disk are not readiness; a truncated download leaves
files behind too.
"""

import pytest

from app.core.config import settings
from app.model_status import (
    DOWNLOADED,
    DOWNLOADING,
    FAILED,
    LOADING,
    NOT_DOWNLOADED,
    READY,
    ModelTracker,
    model_dir,
)


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    # An empty data dir, so "on disk" starts false and every transition is the
    # tracker's doing rather than the developer's real model cache.
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return ModelTracker()


def state_of(tracker: ModelTracker, kind: str) -> str:
    return next(entry for entry in tracker.snapshot() if entry["kind"] == kind)["state"]


def entry_of(tracker: ModelTracker, kind: str) -> dict:
    return next(entry for entry in tracker.snapshot() if entry["kind"] == kind)


# ----------------------------------------------------------------- the states


def test_both_models_start_not_downloaded(tracker):
    assert state_of(tracker, "stt") == NOT_DOWNLOADED
    assert state_of(tracker, "embedding") == NOT_DOWNLOADED


def test_the_happy_path_runs_through_every_state(tracker):
    assert state_of(tracker, "stt") == NOT_DOWNLOADED

    tracker.downloading("stt")
    assert state_of(tracker, "stt") == DOWNLOADING

    tracker.loading("stt")
    assert state_of(tracker, "stt") == LOADING

    tracker.ready("stt")
    assert state_of(tracker, "stt") == READY


def test_downloading_is_not_reported_as_downloaded(tracker):
    """The distinction the old boolean could not make."""
    tracker.downloading("stt")
    assert entry_of(tracker, "stt")["downloaded"] is False


def test_a_failure_carries_the_sentence_the_user_should_read(tracker):
    message = "Could not download the speech model. The first run needs internet access."
    tracker.failed("stt", message)

    entry = entry_of(tracker, "stt")
    assert entry["state"] == FAILED
    assert entry["detail"] == message
    assert entry["downloaded"] is False


def test_a_failure_survives_being_read_more_than_once(tracker):
    """The Settings screen polls; a state that cleared itself on read would
    flicker back to "not downloaded" and look like it was starting again."""
    tracker.failed("embedding", "boom")
    assert state_of(tracker, "embedding") == FAILED
    assert state_of(tracker, "embedding") == FAILED


def test_a_retry_clears_a_previous_failure(tracker):
    """Otherwise a successful retry still reads as failed."""
    tracker.failed("stt", "boom")
    tracker.reset("stt")
    assert state_of(tracker, "stt") == NOT_DOWNLOADED
    assert entry_of(tracker, "stt")["detail"] is None


def test_the_legacy_boolean_can_never_contradict_the_state(tracker):
    """`downloaded` is derived, so an older client and a newer one cannot be
    told different things."""
    on_disk = {DOWNLOADED, LOADING, READY}
    for transition, expected in (
        (lambda: tracker.reset("stt"), NOT_DOWNLOADED),
        (lambda: tracker.downloading("stt"), DOWNLOADING),
        (lambda: tracker.loading("stt"), LOADING),
        (lambda: tracker.ready("stt"), READY),
        (lambda: tracker.failed("stt", "x"), FAILED),
    ):
        transition()
        entry = entry_of(tracker, "stt")
        assert entry["state"] == expected
        assert entry["downloaded"] == (entry["state"] in on_disk)


# ------------------------------------------------------------- disk agreement


def _write_model(directory, name="model.bin"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"weights")


def test_a_model_downloaded_by_a_previous_run_is_recognised(tracker):
    """State is in-process and does not survive a restart, so a fresh tracker
    has to re-derive "on disk" rather than reporting a model the user already
    downloaded as missing."""
    _write_model(model_dir("stt"))
    assert state_of(tracker, "stt") == DOWNLOADED
    assert entry_of(tracker, "stt")["downloaded"] is True


def test_a_model_deleted_behind_the_app_is_noticed(tracker):
    """The user can clear the folder to force a re-download; the app must not
    keep claiming the files are there."""
    directory = model_dir("embedding")
    _write_model(directory, "model.onnx")
    assert state_of(tracker, "embedding") == DOWNLOADED

    (directory / "model.onnx").unlink()
    assert state_of(tracker, "embedding") == NOT_DOWNLOADED


def test_an_interrupted_download_does_not_count_as_downloaded(tracker):
    """Hugging Face writes to `<hash>.incomplete` and renames on success, so a
    partial download must not satisfy the on-disk check -- otherwise a killed
    first run leaves the app permanently claiming a model it cannot load."""
    directory = model_dir("stt")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "abc123.incomplete").write_bytes(b"half a model")

    assert state_of(tracker, "stt") == NOT_DOWNLOADED


def test_an_in_flight_state_is_not_overwritten_by_the_disk_check(tracker):
    """A download in progress has files appearing under it. Re-deriving from
    disk mid-download would flip the UI back and forth."""
    _write_model(model_dir("stt"))
    tracker.downloading("stt")
    assert state_of(tracker, "stt") == DOWNLOADING


def test_reported_paths_are_under_the_configured_data_dir(tracker, tmp_path):
    """Models must never land in the install directory or the PyInstaller
    extraction directory: one is read-only for a per-user install, the other is
    deleted when the process exits."""
    for entry in tracker.snapshot():
        assert str(tmp_path) in entry["path"]


def test_the_two_models_have_separate_locations(tracker):
    assert model_dir("stt") != model_dir("embedding")


def test_the_two_models_track_independently(tracker):
    tracker.failed("stt", "boom")
    tracker.ready("embedding")
    assert state_of(tracker, "stt") == FAILED
    assert state_of(tracker, "embedding") == READY


def test_an_unknown_kind_is_ignored_rather_than_crashing(tracker):
    tracker.ready("not-a-model")
    assert {entry["kind"] for entry in tracker.snapshot()} == {"stt", "embedding"}


def test_the_reported_name_follows_the_configured_model(tracker, monkeypatch):
    monkeypatch.setattr(settings, "stt_model", "tiny.en")
    assert entry_of(tracker, "stt")["name"] == "tiny.en"


# ------------------------------------------------------------------ threading


def test_transitions_are_safe_from_several_threads(tracker):
    """Loaders run on `asyncio.to_thread` workers while `/models/status` is
    answered on the event loop."""
    import threading

    def hammer() -> None:
        for _ in range(200):
            tracker.downloading("stt")
            tracker.ready("stt")
            tracker.snapshot()

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state_of(tracker, "stt") in {DOWNLOADING, READY}
