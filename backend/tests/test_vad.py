import numpy as np

from app.stt.vad import FRAME_MS, EnergyVad, SegmentEvent, Segmenter


def push_all(segmenter: Segmenter, probabilities: list[float]) -> list[SegmentEvent]:
    return [segmenter.push(p) for p in probabilities]


def test_frame_is_32ms():
    assert FRAME_MS == 32


def test_silence_produces_no_events():
    events = push_all(Segmenter(), [0.0] * 50)
    assert set(events) == {SegmentEvent.NONE}


def test_speech_start_requires_consecutive_frames():
    segmenter = Segmenter(start_frames=3)
    # Alternating frames never accumulate a run, so no utterance opens.
    events = push_all(segmenter, [0.9, 0.0] * 10)
    assert SegmentEvent.SPEECH_START not in events


def test_speech_start_then_end():
    silence_frames = 700 // FRAME_MS + 1
    segmenter = Segmenter(start_frames=2, silence_ms=700, min_utterance_ms=0)

    events = push_all(segmenter, [0.9] * 20 + [0.0] * silence_frames)

    assert events[1] == SegmentEvent.SPEECH_START
    assert events[-1] == SegmentEvent.SPEECH_END
    assert events.count(SegmentEvent.SPEECH_END) == 1


def test_brief_pause_does_not_end_the_utterance():
    """A mid-sentence breath must not cut the question in half."""
    segmenter = Segmenter(start_frames=2, silence_ms=700, min_utterance_ms=0)
    short_pause = [0.0] * (300 // FRAME_MS)

    events = push_all(segmenter, [0.9] * 10 + short_pause + [0.9] * 10)

    assert SegmentEvent.SPEECH_END not in events
    assert segmenter.in_speech


def test_utterance_shorter_than_minimum_is_discarded():
    """A click can clear the start threshold; it must not reach the model."""
    segmenter = Segmenter(start_frames=2, silence_ms=100, min_utterance_ms=1000)
    events = push_all(segmenter, [0.9] * 3 + [0.0] * 10)

    assert SegmentEvent.SPEECH_START in events
    assert SegmentEvent.SPEECH_END not in events
    assert not segmenter.in_speech


def test_max_duration_forces_an_end():
    """Someone who never pauses must still produce a final transcript."""
    segmenter = Segmenter(start_frames=2, silence_ms=10_000, max_utterance_ms=320)
    events = push_all(segmenter, [0.9] * 40)

    assert SegmentEvent.SPEECH_END in events
    assert not segmenter.in_speech


def test_start_frames_count_toward_duration():
    segmenter = Segmenter(start_frames=3, silence_ms=10_000, max_utterance_ms=100_000)
    push_all(segmenter, [0.9] * 3)
    assert segmenter.duration_ms == 3 * FRAME_MS


def test_reset_clears_state():
    segmenter = Segmenter(start_frames=2)
    push_all(segmenter, [0.9] * 5)
    assert segmenter.in_speech

    segmenter.reset()
    assert not segmenter.in_speech
    assert segmenter.duration_ms == 0


def test_threshold_is_respected():
    segmenter = Segmenter(threshold=0.8, start_frames=2)
    assert SegmentEvent.SPEECH_START not in push_all(segmenter, [0.7] * 10)


def test_consecutive_utterances_are_separated():
    silence = [0.0] * (700 // FRAME_MS + 1)
    segmenter = Segmenter(start_frames=2, silence_ms=700, min_utterance_ms=0)

    events = push_all(segmenter, [0.9] * 10 + silence + [0.9] * 10 + silence)

    assert events.count(SegmentEvent.SPEECH_START) == 2
    assert events.count(SegmentEvent.SPEECH_END) == 2


def test_energy_vad_separates_loud_from_silent():
    vad = EnergyVad(floor=0.01)
    rng = np.random.default_rng(0)
    loud = (rng.standard_normal(512) * 0.3).astype(np.float32)
    quiet = np.zeros(512, dtype=np.float32)

    assert vad.probability(loud) > 0.5
    assert vad.probability(quiet) == 0.0
