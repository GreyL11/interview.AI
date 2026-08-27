import pytest

from app.llm.streaming import extract_partial_summary, parse_answer_payload


def test_returns_none_before_the_key_arrives():
    assert extract_partial_summary('{"cat') is None
    assert extract_partial_summary('{"key_points": ["a"]}') is None


def test_extracts_summary_as_it_streams():
    assert extract_partial_summary('{"summary": "Make the pipe') == "Make the pipe"


def test_extracts_completed_summary():
    assert extract_partial_summary('{"summary": "Done.", "key_points": []}') == "Done."


def test_grows_monotonically_across_chunks():
    payload = '{"summary": "Make the pipeline idempotent.", "key_points": []}'
    seen = [
        extract_partial_summary(payload[:i])
        for i in range(1, len(payload) + 1)
    ]
    values = [s for s in seen if s is not None]
    assert values
    for earlier, later in zip(values, values[1:]):
        assert later.startswith(earlier) or earlier.startswith(later)


def test_handles_escaped_quotes():
    assert extract_partial_summary(r'{"summary": "He said \"hi\" once"}') == 'He said "hi" once'


def test_handles_escaped_newline():
    assert extract_partial_summary(r'{"summary": "line one\nline two"}') == "line one\nline two"


def test_waits_when_an_escape_is_split_across_chunks():
    # One trailing backslash: the character it escapes has not arrived yet, so
    # it must be held back rather than emitted as a literal.
    partial = '{"summary": "ends with' + "\\"
    assert extract_partial_summary(partial) == "ends with"


def test_completed_backslash_escape_is_emitted():
    # Two backslashes are a finished escape for one literal backslash.
    assert extract_partial_summary(r'{"summary": "a\\b"}') == "a\\b"


def test_empty_summary_value():
    assert extract_partial_summary('{"summary": ""}') == ""


def test_parse_payload_plain_json():
    assert parse_answer_payload('{"summary": "x"}')["summary"] == "x"


def test_parse_payload_strips_markdown_fences():
    fenced = '```json\n{"summary": "x"}\n```'
    assert parse_answer_payload(fenced)["summary"] == "x"


def test_parse_payload_strips_bare_fences():
    assert parse_answer_payload('```\n{"summary": "x"}\n```')["summary"] == "x"


def test_parse_payload_rejects_garbage():
    with pytest.raises(Exception):
        parse_answer_payload("not json at all")
