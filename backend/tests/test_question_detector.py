import pytest

from app.realtime.events import RejectionReason
from app.realtime.question_detector import QuestionDetector
from app.schemas.classification import Category


@pytest.mark.parametrize(
    "text,category",
    [
        ("How would you handle duplicate records in a data pipeline?", Category.SCENARIO),
        ("Write a function to reverse a linked list", Category.CODING),
        ("Tell me about a time you had a team conflict", Category.BEHAVIORAL),
        ("How would you design a URL shortener?", Category.SYSTEM_DESIGN),
    ],
)
def test_accepts_real_questions(text, category):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.classification.category == category


def test_accepts_imperative_coding_prompt():
    """Not phrased as a question, but it is the task being set."""
    detection = QuestionDetector().inspect("Write a function to reverse a linked list")
    assert detection.accepted
    assert detection.classification.requires_code


@pytest.mark.parametrize("filler", ["yeah", "mm-hmm", "right", "ok sure"])
def test_rejects_short_acknowledgements(filler):
    detection = QuestionDetector().inspect(filler)
    assert not detection.accepted
    assert detection.reason in (RejectionReason.TOO_SHORT, RejectionReason.NOT_A_QUESTION)


def test_rejects_statements():
    detection = QuestionDetector().inspect("thanks that makes a lot of sense to me")
    assert not detection.accepted
    assert detection.reason == RejectionReason.NOT_A_QUESTION


# --------------------------------------------------- conversational preamble
# The live-session bug: an interviewer acknowledges the previous answer, then
# asks the real question, and Whisper supplies no question mark.

REPORTED = (
    "Just tell me how much you write yourself in Python. As you said, four. "
    "Very good. So tell me what is the difference between shallow copy and deep copy."
)


def test_reported_transcript_is_accepted():
    detection = QuestionDetector().inspect(REPORTED)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}"


def test_reported_transcript_passes_the_clean_prompt_to_coaching():
    detection = QuestionDetector().inspect(REPORTED)
    assert detection.text == (
        "So tell me what is the difference between shallow copy and deep copy?"
    )
    # Filler must not reach the model.
    assert "Very good" not in detection.text
    assert "As you said" not in detection.text


def test_reported_transcript_is_classified_not_unknown():
    detection = QuestionDetector().inspect(REPORTED)
    assert detection.classification is not None
    assert detection.classification.category != Category.UNKNOWN


@pytest.mark.parametrize(
    "text",
    [
        "Very good. So tell me what is the difference between shallow copy and deep copy.",
        "Okay, can you explain Python decorators?",
        "Great. Describe the challenges you faced.",
        "Very good. Now tell me how you implemented authentication.",
        "That's interesting. Walk me through the architecture.",
        "How did you implement this pipeline",
        "Can you give me an example",
        "Tell me the difference between shallow copy and deep copy",
        "Explain your project",
        "Walk me through your architecture",
    ],
)
def test_accepts_prompts_end_to_end(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Very good",
        "Okay",
        "That's correct",
        "Thank you",
        "Interesting",
        "Great answer",
        "Let's continue",
    ],
)
def test_rejects_acknowledgements_end_to_end(text):
    detection = QuestionDetector().inspect(text)
    assert not detection.accepted, f"wrongly accepted: {text!r}"


def test_detail_records_which_layer_fired():
    assert QuestionDetector().inspect("Tell me about caching").detail == "interview_prompt"
    assert QuestionDetector().inspect("Why did you choose that").detail == "interrogative"
    assert QuestionDetector().inspect("Is this thread safe?").detail == "punctuation"
    assert QuestionDetector().inspect("Very good").detail == "no_question_pattern"


def test_coalesces_a_rapid_correction():
    detector = QuestionDetector(coalesce_ms=1000)
    first = detector.inspect("How would you scale this service?", now=100.0)
    assert first.accepted

    second = detector.inspect("Actually assume ten thousand QPS", now=100.4)
    assert second.accepted
    assert second.supersedes
    assert "scale this service" in second.text
    assert "ten thousand QPS" in second.text


def test_does_not_coalesce_after_the_window():
    detector = QuestionDetector(coalesce_ms=1000)
    detector.inspect("How would you scale this service?", now=100.0)
    later = detector.inspect("What is a database index anyway?", now=105.0)

    assert later.accepted
    assert not later.supersedes
    assert "scale this service" not in later.text


def test_reset_clears_coalescing_state():
    detector = QuestionDetector(coalesce_ms=1000)
    detector.inspect("How would you scale this service?", now=100.0)
    detector.reset()
    following = detector.inspect("What is a database index anyway?", now=100.2)
    assert not following.supersedes


def test_min_words_is_configurable():
    assert not QuestionDetector(min_words=10).inspect("What is a database index?").accepted


def test_low_confidence_is_rejected():
    detector = QuestionDetector(min_confidence=0.99)
    detection = detector.inspect("What is a database index?")
    assert not detection.accepted
    assert detection.reason == RejectionReason.LOW_CONFIDENCE


# ------------------------------------------------------- preceding context
# Reported bug: "By using this study, just write a character count program."
# isn't itself phrased as a question, so it was rejected and discarded. The
# very next utterance, "How many times each character is repeated?", was then
# answered as a standalone fragment with no idea what "it" refers to.


def test_a_rejected_setup_utterance_is_prepended_to_the_next_question():
    """Context is attached to what the LLM sees (effective_text), not to what
    the UI displays (text) -- the panel should show the clean question the
    interviewer actually asked, not a merged wall of text."""
    detector = QuestionDetector()
    setup = detector.inspect(
        "By using this study, just write a character count program.", now=100.0
    )
    assert not setup.accepted

    question = detector.inspect("How many times each character is repeated?", now=101.0)
    assert question.accepted
    assert question.text == "How many times each character is repeated?"
    assert "character count program" not in question.text
    assert "character count program" in question.effective_text
    assert "How many times each character is repeated" in question.effective_text


def test_context_older_than_the_window_is_not_used():
    detector = QuestionDetector(context_window_ms=1000)
    detector.inspect(
        "By using this study, just write a character count program.", now=100.0
    )

    question = detector.inspect("How many times each character is repeated?", now=105.0)
    assert question.accepted
    assert "character count program" not in question.effective_text


def test_context_is_cleared_once_consumed():
    """The setup fragment answers exactly one question; it must not also
    attach itself to whatever is asked right after."""
    detector = QuestionDetector()
    detector.inspect(
        "By using this study, just write a character count program.", now=100.0
    )
    detector.inspect("How many times each character is repeated?", now=101.0)

    unrelated = detector.inspect("What is a database index?", now=102.0)
    assert unrelated.accepted
    assert "character count program" not in unrelated.effective_text


def test_a_previous_accepted_question_is_never_used_as_context():
    """Only rejected fragments become context. An accepted question is a
    complete, separate turn and must never be glued onto the next one, even
    when it lands just outside the correction window."""
    detector = QuestionDetector(coalesce_ms=50)
    first = detector.inspect("How would you design a URL shortener?", now=100.0)
    assert first.accepted

    second = detector.inspect("What is a database index?", now=100.5)
    assert second.accepted
    assert not second.supersedes
    assert "URL shortener" not in second.text


def test_buffer_context_false_ignores_and_skips_the_context_buffer():
    """The flag session.py uses for typed/candidate speech: no read, no write."""
    detector = QuestionDetector()
    detector.inspect(
        "By using this study, just write a character count program.",
        now=100.0,
        buffer_context=False,
    )

    question = detector.inspect(
        "How many times each character is repeated?", now=101.0, buffer_context=False
    )
    assert question.accepted
    assert "character count program" not in question.effective_text


# --------------------------------------------------------------- follow-ups
# A bare "Why?" is below question_min_words and would normally be rejected
# outright -- but right after a real question, it plainly means something.
# SessionMemory already threads the previous Q&A into every LLM prompt, so
# the detector's only job is deciding whether it's worth asking at all.


@pytest.mark.parametrize("followup", ["Why?", "How?", "Why", "How"])
def test_a_short_followup_is_accepted_after_a_recent_question(followup):
    detector = QuestionDetector()
    first = detector.inspect("What is a hash map?", now=100.0)
    assert first.accepted

    second = detector.inspect(followup, now=110.0)
    assert second.accepted
    assert second.detail == "follow_up"


def test_a_short_followup_is_rejected_with_no_recent_question():
    detector = QuestionDetector()
    detection = detector.inspect("Why?", now=100.0)
    assert not detection.accepted
    assert detection.reason == RejectionReason.TOO_SHORT


def test_a_short_followup_expires_outside_the_followup_window():
    detector = QuestionDetector(followup_window_ms=5000)
    detector.inspect("What is a hash map?", now=100.0)

    detection = detector.inspect("Why?", now=110.0)  # 10s later, well past 5s
    assert not detection.accepted
    assert detection.reason == RejectionReason.TOO_SHORT


def test_a_short_acknowledgement_is_not_treated_as_a_followup():
    """The follow-up bypass only lifts the word-count floor; a short filler
    still has to pass the same sentence classifier as everything else."""
    detector = QuestionDetector()
    detector.inspect("What is a hash map?", now=100.0)

    detection = detector.inspect("Okay.", now=101.0)
    assert not detection.accepted


def test_manual_short_followup_is_not_accepted():
    """buffer_context=False (typed/candidate speech) gets no follow-up bypass
    either -- session.py's contract for that flag covers both mechanisms."""
    detector = QuestionDetector()
    detector.inspect("What is a hash map?", now=100.0, buffer_context=True)

    detection = detector.inspect("Why?", now=101.0, buffer_context=False)
    assert not detection.accepted


def test_a_short_followup_does_not_pollute_the_context_buffer():
    """A follow-up too thin to carry meaning shouldn't be remembered as setup
    for some later, unrelated question."""
    detector = QuestionDetector()
    detector.inspect("What is a hash map?", now=100.0)
    detector.inspect("Why?", now=101.0)

    unrelated = detector.inspect("How would you design a URL shortener?", now=102.0)
    assert unrelated.accepted
    assert unrelated.effective_text == unrelated.text


# ------------------------------------------------------------------- noise


def test_garbage_fragments_are_not_remembered_as_context():
    """Stray punctuation / broken STT output shouldn't attach itself to a
    later question just because it arrived recently."""
    detector = QuestionDetector()
    detector.inspect("?? -- ...", now=100.0)

    question = detector.inspect("How would you design a URL shortener?", now=100.5)
    assert question.accepted
    assert question.effective_text == question.text


# --------------------------------------------------------------- stability
# "Can you explain what happens when" is grammatically a question (starts
# with "can you") but trails off mid-clause. Firing on it wastes a Gemini
# call on a fragment the interviewer was still in the middle of saying.


@pytest.mark.parametrize(
    "text",
    [
        "Can you explain what happens when",
        "Tell me about a project you built and",
        "What is the difference between a list and a",
        "How would you scale this if",
    ],
)
def test_a_dangling_clause_is_flagged_unstable(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.stable is False


@pytest.mark.parametrize(
    "text",
    [
        "Explain caching?",
        "Write a program to reverse a linked list",
        "What is the difference between a list and a tuple?",
        "Is this thread safe?",
    ],
)
def test_a_complete_question_is_flagged_stable(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.stable is True


def test_a_followup_is_always_stable_even_ending_in_a_dangling_word():
    """'When?' ends on a word that would flag a longer clause as unstable,
    but as a short, self-contained follow-up it must not be delayed."""
    detector = QuestionDetector()
    detector.inspect("What is a hash map?", now=100.0)

    detection = detector.inspect("When?", now=101.0)
    assert detection.accepted
    assert detection.stable is True


def test_coalesced_continuation_of_a_dangling_clause_is_stable():
    """The existing correction-coalesce window already merges a quick
    continuation into one detection -- once merged, the *combined* text is
    re-evaluated fresh and is no longer dangling."""
    detector = QuestionDetector(coalesce_ms=1000)
    first = detector.inspect("Can you explain what happens when", now=100.0)
    assert first.accepted and first.stable is False

    second = detector.inspect("a transaction fails?", now=100.4)
    assert second.accepted
    assert second.stable is True
    assert second.supersedes
    assert "transaction fails" in second.text


# ---------------------------------------------------------- coding problems
# A coding problem's setup clause ("Given an array, find two numbers") has no
# terminal "?" and is exactly the imperative-task shape -- and is exactly the
# phrasing most likely to be split by a natural pause before its closing
# condition ("...whose sum equals a target value") arrives as its own
# VAD-bounded utterance, well outside the short correction window.


def test_multi_sentence_coding_setup_merges_across_a_realistic_pause():
    detector = QuestionDetector(coalesce_ms=1000, context_window_ms=4000)
    first = detector.inspect(
        "Given an array of integers, I want you to find two numbers", now=100.0
    )
    assert first.accepted
    assert first.detail == "imperative_task"

    # 2.5s later: well past coalesce_ms, comfortably inside context_window_ms --
    # realistic for a second utterance needing its own VAD silence-close.
    second = detector.inspect("whose sum equals a target value", now=102.5)
    assert second.accepted
    assert second.supersedes
    assert "find two numbers" in second.text
    assert "sum equals a target value" in second.text


def test_a_complete_imperative_coding_prompt_still_uses_the_short_window():
    """The extended window only applies while the *last* accept was an
    imperative-task fragment. A complete "?" question reverts to the normal,
    short correction window immediately."""
    detector = QuestionDetector(coalesce_ms=200, context_window_ms=4000)
    detector.inspect("Given an array of integers, I want you to find two numbers", now=100.0)
    merged = detector.inspect("whose sum equals a target value?", now=100.3)
    assert merged.accepted and merged.detail == "punctuation"

    # A later, unrelated question arriving after the short window (but still
    # inside the long one) must NOT merge into the now-complete question.
    unrelated = detector.inspect("What is a database index?", now=104.0)
    assert unrelated.accepted
    assert "two numbers" not in unrelated.text


def test_constraint_added_immediately_merges_into_the_coding_question():
    detector = QuestionDetector()
    detector.inspect("Find duplicate elements in an array.", now=100.0)

    constrained = detector.inspect("But you cannot use extra space.", now=100.3)
    assert constrained.accepted
    assert "duplicate elements" in constrained.text
    assert "cannot use extra space" in constrained.text


def test_correction_to_a_different_coding_problem_excludes_the_old_one():
    detector = QuestionDetector()
    detector.inspect("Write a function to reverse a string.", now=100.0)

    corrected = detector.inspect("No, actually reverse a linked list.", now=100.3)
    assert corrected.accepted
    assert "linked list" in corrected.text
    assert "reverse a string" not in corrected.text


def test_topic_change_via_lets_do_excludes_the_old_coding_problem():
    detector = QuestionDetector()
    detector.inspect(
        "Find the longest substring without repeating characters.", now=100.0
    )

    changed = detector.inspect(
        "Actually, let's do longest palindromic substring instead.", now=100.3
    )
    assert changed.accepted
    assert "palindromic" in changed.text
    assert "repeating characters" not in changed.text


def test_a_complete_coding_prompt_is_not_delayed():
    """"Write a function to reverse a linked list" is a complete ask despite
    being imperative-task-triggered -- it must fire immediately, not wait."""
    detection = QuestionDetector().inspect("Write a function to reverse a linked list")
    assert detection.accepted
    assert detection.stable is True


def test_short_coding_followups_are_accepted_normally():
    """These are all >= min_words and end in '?', so they need no special
    coding-specific handling -- confirming the existing generic path already
    covers them is the point of this test."""
    for text in (
        "Can we optimize it?", "Without sorting?", "Without extra space?",
        "What's the complexity?", "What happens if the array is empty?",
    ):
        detection = QuestionDetector().inspect(text)
        assert detection.accepted, text


def test_acknowledgements_are_not_accepted_even_with_recent_coding_context():
    detector = QuestionDetector()
    detector.inspect("Write a function to reverse a linked list.", now=100.0)

    for filler, now in (("Okay.", 100.2), ("Yeah.", 100.4), ("Right.", 100.6)):
        detection = detector.inspect(filler, now=now)
        assert not detection.accepted, filler
