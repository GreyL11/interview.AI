from app.llm.prompts import (
    BEHAVIORAL_SCHEMA_HINT,
    CODING_SCHEMA_HINT,
    DEBUGGING_SCHEMA_HINT,
    GENERIC_SCHEMA_HINT,
    SQL_SCHEMA_HINT,
    SYSTEM_DESIGN_SCHEMA_HINT,
    build_prompt,
)
from app.schemas.classification import Category


def _schema_used(prompt: str) -> str:
    return prompt.split("Respond using exactly this JSON shape:\n", 1)[1].split("\n\n", 1)[0]


def test_coding_questions_get_the_coding_schema():
    prompt = build_prompt("Reverse a linked list", Category.CODING, [], [])
    assert _schema_used(prompt) == CODING_SCHEMA_HINT


def test_sql_questions_get_the_sql_schema():
    prompt = build_prompt("Find duplicate emails", Category.SQL, [], [])
    assert _schema_used(prompt) == SQL_SCHEMA_HINT


def test_debugging_questions_get_the_debugging_schema():
    prompt = build_prompt("This endpoint is slow, why?", Category.DEBUGGING, [], [])
    assert _schema_used(prompt) == DEBUGGING_SCHEMA_HINT


def test_system_design_and_architecture_share_the_system_design_schema():
    for category in (Category.SYSTEM_DESIGN, Category.ARCHITECTURE):
        prompt = build_prompt("Design a URL shortener", category, [], [])
        assert _schema_used(prompt) == SYSTEM_DESIGN_SCHEMA_HINT


def test_behavioral_questions_get_the_star_schema():
    prompt = build_prompt("Tell me about a conflict", Category.BEHAVIORAL, [], [])
    assert _schema_used(prompt) == BEHAVIORAL_SCHEMA_HINT


def test_unmapped_categories_fall_back_to_generic():
    for category in (
        Category.TECHNICAL_KNOWLEDGE,
        Category.SCENARIO,
        Category.PERSONAL_EXPERIENCE,
        Category.RESUME,
        Category.PROJECT,
        Category.FOLLOW_UP,
        Category.UNKNOWN,
    ):
        prompt = build_prompt("What is a hash map?", category, [], [])
        assert _schema_used(prompt) == GENERIC_SCHEMA_HINT


def test_current_question_is_clearly_marked():
    prompt = build_prompt(
        "What about eviction?", Category.TECHNICAL_KNOWLEDGE, [],
        ["Q: What is a hash map?", "A: A key-value lookup structure."],
    )
    assert "CURRENT INTERVIEWER QUESTION" in prompt
    assert prompt.rstrip().endswith("What about eviction?")


def test_background_context_is_labeled_as_background_only():
    prompt = build_prompt(
        "What about eviction?", Category.TECHNICAL_KNOWLEDGE, [],
        ["Q: What is a hash map?", "A: A key-value lookup structure."],
    )
    assert "INTERVIEW CONTEXT" in prompt
    # The background section must appear before the current question, so the
    # instruction ordering matches what the rules describe.
    assert prompt.index("INTERVIEW CONTEXT") < prompt.index("CURRENT INTERVIEWER QUESTION")


def test_no_background_section_when_nothing_to_include():
    prompt = build_prompt("What is a hash map?", Category.TECHNICAL_KNOWLEDGE, [], [])
    assert "INTERVIEW CONTEXT" not in prompt
