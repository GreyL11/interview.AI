import pytest

from app.intelligence.classifier import classify
from app.schemas.classification import Category, Domain


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How would you handle duplicate records in a data pipeline?", Category.SCENARIO),
        ("Write a function to reverse a linked list", Category.CODING),
        ("Write a query to find the top 5 customers by revenue", Category.SQL),
        ("Tell me about a time you had a team conflict", Category.BEHAVIORAL),
        ("How would you design a URL shortener?", Category.SYSTEM_DESIGN),
        ("Why is this failing with a stack trace on startup?", Category.DEBUGGING),
        ("And what about at higher scale?", Category.FOLLOW_UP),
        ("What is a database index?", Category.TECHNICAL_KNOWLEDGE),
        ("thanks that makes sense", Category.UNKNOWN),
    ],
)
def test_category_detection(question, expected):
    assert classify(question).category == expected


def test_non_question_flagged():
    result = classify("thanks that makes sense")
    assert result.is_question is False
    assert result.category == Category.UNKNOWN


def test_domain_detection():
    assert classify("How do you dedupe rows in an ETL pipeline?").domain == Domain.DATA_ENGINEERING


def test_flags_for_personal_category():
    result = classify("Tell me about a time you had a team conflict")
    assert result.requires_personal_context is True
    assert result.requires_rag is True
    assert result.requires_code is False


def test_flags_for_coding_category():
    result = classify("Write a function to reverse a linked list")
    assert result.requires_code is True
    assert result.requires_reasoning is True
    assert result.requires_rag is False


@pytest.mark.parametrize(
    "question",
    [
        "What's wrong with this code?",
        "Why does this implementation fail?",
        "How would you fix the race condition?",
        "This isn't working, why not?",
    ],
)
def test_verbal_debugging_phrasings_are_classified_as_debugging(question):
    """These are natural spoken phrasings for a debugging follow-up on code
    the interviewer just described -- none of them contain the literal
    "bug"/"crash"/"stack trace" keywords the original regex required."""
    assert classify(question).category == Category.DEBUGGING


def test_a_positive_why_does_this_work_question_is_not_misclassified_as_debugging():
    """The debugging regex targets fail/broken specifically so a positive
    "why does X work" mechanism question -- common and unrelated to
    debugging -- is not swept in by a generic "why does...work" match."""
    result = classify("Why does event-driven architecture work well for this?")
    assert result.category != Category.DEBUGGING


# --------------------------------------------------------------------- SQL
# Measured gaps: each of these was TECHNICAL_KNOWLEDGE or UNKNOWN before the
# _SQL pattern was widened. "Find the second highest salary." was the worst --
# UNKNOWN means is_question=False, so it was dropped without an answer at all.


@pytest.mark.parametrize(
    "question",
    [
        "How would you find the second highest salary?",
        "Find the second highest salary.",
        "Find customers who never placed an order.",
        "What's the difference between WHERE and HAVING?",
        "When would you use a window function?",
        "How would you rank employees by salary?",
        "Write SQL to calculate a running total.",
        "How do you join these two tables?",
        "Can you rewrite that with row_number?",
    ],
)
def test_sql_phrasings_are_classified_as_sql(question):
    assert classify(question).category == Category.SQL


def test_a_bare_find_query_is_a_question_not_unknown():
    """UNKNOWN carries is_question=False, which drops the utterance before it
    ever reaches the LLM -- the failure mode this pattern exists to prevent."""
    result = classify("Find the second highest salary.")
    assert result.is_question is True
    assert result.requires_code is True


@pytest.mark.parametrize(
    "question",
    [
        "How does database indexing work?",
        "What is a database index?",
        "What is a hash table?",
        "How do you query an API?",
        "How would you scale the database layer?",
    ],
)
def test_database_topic_questions_are_not_classified_as_sql(question):
    """SQL means "produce or reason about a query". A question that merely
    mentions a table/index/database/query is technical knowledge; the DATABASE
    signal belongs on the Domain axis, which these still carry."""
    assert classify(question).category != Category.SQL


# -------------------------------------------------------------- behavioral


@pytest.mark.parametrize(
    "question",
    [
        "Tell me about a challenging project.",
        "Describe a time you disagreed with your manager.",
        "Tell me about a failure.",
        "What's the biggest challenge you've faced?",
        "Have you ever had a conflict with a teammate?",
        "How did you handle a difficult stakeholder?",
        "Give me an example of when something went wrong.",
        "What would you say was your biggest achievement?",
        "Tell me about a time you took ownership.",
        "How have you influenced a team without authority?",
        "Describe a situation where you led a project.",
        "Tell me about a difficult problem you solved.",
        "Tell me about a difficult technical problem you solved.",
    ],
)
def test_behavioral_phrasings_are_classified_as_behavioral(question):
    assert classify(question).category == Category.BEHAVIORAL


@pytest.mark.parametrize(
    "question",
    [
        "What is the biggest challenge in distributed systems?",
        "Describe a situation where a cache would help.",
        "How would you solve a performance problem in production?",
        "How does a team of microservices communicate?",
    ],
)
def test_technical_questions_are_not_classified_as_behavioral(question):
    """challenge/problem/team/situation are ordinary technical vocabulary. The
    pattern requires a personal-past marker ("...you"), not just the noun."""
    assert classify(question).category != Category.BEHAVIORAL


def test_a_personal_narrative_outranks_a_topic_keyword():
    """"tell me about a time you..." is a request for the candidate's story.
    The "bug" keyword must not pull it into DEBUGGING."""
    assert classify("Tell me about a time you fixed a production bug.").category == (
        Category.BEHAVIORAL
    )


def test_behavioral_followups_stay_narrow_rather_than_becoming_behavioral():
    """A follow-up must not re-trigger the full STAR schema -- "What did you
    learn?" should answer the learning, not generate a fresh story. Keeping
    these off Category.BEHAVIORAL is what makes that happen (see
    _SCHEMA_BY_CATEGORY), so it is asserted rather than left implicit."""
    for followup in (
        "What did you learn from that?",
        "What would you do differently?",
        "Why did you choose that approach?",
        "What was the outcome?",
    ):
        assert classify(followup).category != Category.BEHAVIORAL, followup
