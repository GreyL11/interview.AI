import pytest

from app.realtime.prompt_detector import (
    REASON_IMPERATIVE_TASK,
    REASON_INTERROGATIVE,
    REASON_INTERVIEW_PROMPT,
    REASON_PUNCTUATION,
    extract_interview_prompt,
    is_interview_prompt,
)

# The transcript from the live session that exposed the bug: two prompts, a
# numeric aside, and an acknowledgement, with no question mark anywhere.
REPORTED = (
    "Just tell me how much you write yourself in Python. As you said, four. "
    "Very good. So tell me what is the difference between shallow copy and deep copy."
)


# ------------------------------------------------------------------ regression


def test_reported_transcript_is_detected():
    assert is_interview_prompt(REPORTED)


def test_reported_transcript_extracts_the_last_prompt():
    match = extract_interview_prompt(REPORTED)
    assert match is not None
    assert match.prompt == (
        "So tell me what is the difference between shallow copy and deep copy?"
    )
    assert match.reason == REASON_INTERVIEW_PROMPT
    # The earlier prompt and the filler are dropped.
    assert "As you said" not in match.prompt
    assert "Very good" not in match.prompt


# ------------------------------------------------------------------- positives


@pytest.mark.parametrize(
    "text",
    [
        "What is the difference between shallow copy and deep copy?",
        "What is the difference between shallow copy and deep copy",
        "Tell me the difference between shallow copy and deep copy",
        "Explain your project",
        "Walk me through your architecture",
        "Very good. So tell me what is the difference between shallow copy and deep copy.",
        "Okay, can you explain Python decorators?",
        "Great. Describe the challenges you faced.",
        "How did you implement this pipeline",
        "Can you give me an example",
    ],
)
def test_required_positives(text):
    assert is_interview_prompt(text), f"should be detected: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Can you explain decorators",
        "How does Python garbage collection work",
        "Why would you use a generator",
        "When would you use multiprocessing instead of multithreading",
        "Where would you store this data",
        "Which approach would you pick",
        "Who owned that service",
        "Is it thread safe",
        "Are you familiar with asyncio",
        "Did you write the tests yourself",
        "Have you used Kubernetes in production",
        "Do you prefer SQL or NoSQL",
    ],
)
def test_interrogatives_without_question_marks(text):
    assert is_interview_prompt(text), f"should be detected: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Tell me about yourself",
        "Tell me the difference between X and Y",
        "Explain your project",
        "Explain how this architecture works",
        "Walk me through your experience",
        "Walk me through your project",
        "Describe your current role",
        "Describe your architecture",
        "Compare ETL and ELT",
        "Give me an example",
        "Talk about your previous project",
        "What challenges did you face",
        "Take me through your resume",
        "Elaborate on that design",
        "Help me understand your caching layer",
        "Share an example of a failure",
    ],
)
def test_imperative_interview_prompts(text):
    assert is_interview_prompt(text), f"should be detected: {text!r}"


def test_coding_task_prompts_still_work():
    """Regression guard: coding prompts are imperatives too and must not be
    lost when tightening the acknowledgement filter."""
    for text in [
        "Write a function to reverse a linked list",
        "Implement a rate limiter",
        "Given an array of integers find the two that sum to a target",
        "Design a URL shortener",
    ]:
        assert is_interview_prompt(text), f"should be detected: {text!r}"


def test_lets_do_signals_a_new_prompt():
    """"Let's do X" is how an interviewer commonly replaces a problem mid-
    sentence without using a recognized imperative verb (e.g. "let's do
    longest palindromic substring" has no "find"/"write"/"reverse")."""
    for text in ["let's do longest palindromic substring", "let's try a different approach"]:
        assert is_interview_prompt(text), f"should be detected: {text!r}"


# ------------------------------------------------------------------- negatives


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
def test_required_negatives(text):
    assert not is_interview_prompt(text), f"should NOT be detected: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "Alright",
        "All right",
        "Perfect",
        "Excellent",
        "That is interesting",
        "I understand",
        "I see",
        "Got it",
        "Makes sense",
        "Thanks very much",
        "Sounds good",
        "Fair enough",
        "Moving on",
        "Well done",
        "Good answer",
        "No problem",
        "Yeah",
        "Mm-hmm",
    ],
)
def test_more_acknowledgements(text):
    assert not is_interview_prompt(text), f"should NOT be detected: {text!r}"


def test_statement_containing_an_interrogative_word_does_not_fire():
    """"how" mid-sentence is not a question. This is why interrogatives are
    only trusted at a clause start."""
    assert not is_interview_prompt("I understand how you feel about that")
    assert not is_interview_prompt("That is exactly how we did it")


def test_statement_containing_a_task_verb_does_not_fire():
    """The reported transcript contains "you write yourself"; a bare verb
    anywhere must not trigger."""
    assert not is_interview_prompt("I write most of my code in Python these days")
    assert not is_interview_prompt("We build our services on top of that")


def test_tag_question_acknowledgement_does_not_fire():
    assert not is_interview_prompt("Right?")
    assert not is_interview_prompt("Okay?")


def test_empty_and_whitespace():
    assert not is_interview_prompt("")
    assert not is_interview_prompt("   ")
    assert extract_interview_prompt("") is None


# ----------------------------------------------------------------------- mixed


@pytest.mark.parametrize(
    "text,expected_fragment",
    [
        (
            "Very good. What is the difference between shallow and deep copy?",
            "What is the difference between shallow and deep copy?",
        ),
        ("Okay. Explain your project.", "Explain your project?"),
        (
            "That's interesting. Walk me through the architecture.",
            "Walk me through the architecture?",
        ),
        (
            "Very good. Now tell me how you implemented authentication.",
            "Now tell me how you implemented authentication?",
        ),
    ],
)
def test_acknowledgement_plus_question_triggers_and_extracts(text, expected_fragment):
    match = extract_interview_prompt(text)
    assert match is not None, f"should be detected: {text!r}"
    assert match.prompt == expected_fragment


def test_acknowledgement_prefix_without_sentence_punctuation():
    """Whisper often returns a whole utterance with no full stops at all."""
    match = extract_interview_prompt(
        "okay great so tell me what is the difference between a list and a tuple"
    )
    assert match is not None
    assert "difference between a list and a tuple" in match.prompt


def test_interrogative_after_a_discourse_connective():
    match = extract_interview_prompt("Alright, what is a Python closure")
    assert match is not None
    assert match.reason in (REASON_INTERROGATIVE, REASON_PUNCTUATION)


# ------------------------------------------------------------------ extraction


def test_last_prompt_wins_when_several_are_present():
    match = extract_interview_prompt(
        "Tell me about your role. Good. Now explain how you handled scaling."
    )
    assert match is not None
    assert match.prompt.startswith("Now explain how you handled scaling")
    assert "your role" not in match.prompt


def test_trailing_qualifier_is_kept():
    """A correction after the question must not be discarded — it changes the
    answer."""
    match = extract_interview_prompt(
        "How would you scale this service? Actually assume ten thousand QPS"
    )
    assert match is not None
    assert "scale this service" in match.prompt
    assert "ten thousand QPS" in match.prompt


def test_question_mark_is_added_when_missing():
    match = extract_interview_prompt("Explain your project")
    assert match is not None
    assert match.prompt.endswith("?")


def test_existing_question_mark_is_not_doubled():
    match = extract_interview_prompt("Explain your project?")
    assert match is not None
    assert match.prompt.endswith("?")
    assert not match.prompt.endswith("??")


def test_reason_is_reported():
    assert extract_interview_prompt("Is this thread safe?").reason == REASON_PUNCTUATION
    assert extract_interview_prompt("Tell me about caching").reason == REASON_INTERVIEW_PROMPT
    assert extract_interview_prompt("Why did you choose that").reason == REASON_INTERROGATIVE
    assert (
        extract_interview_prompt("Write a function to reverse a list").reason
        == REASON_IMPERATIVE_TASK
    )


def test_sentence_is_reported_for_diagnosis():
    match = extract_interview_prompt("Very good. Tell me about caching.")
    assert match is not None
    assert match.sentence == "Tell me about caching."


def test_detection_is_deterministic():
    first = extract_interview_prompt(REPORTED)
    second = extract_interview_prompt(REPORTED)
    assert first == second
