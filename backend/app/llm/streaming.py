import json
import re

_SUMMARY_KEY = re.compile(r'"summary"\s*:\s*"')


def extract_partial_summary(buffer: str) -> str | None:
    """Pull the `summary` value out of a JSON document that is still arriving.

    Gemini streams the structured answer as raw JSON text, so nothing is
    parseable until the final token. This reads just far enough to show the
    summary as it is typed, and returns None until the key appears. Deliberately
    a scanner rather than a tolerant JSON parser: it only ever needs one string
    value, and a scanner cannot be fooled into half-building the rest.
    """
    match = _SUMMARY_KEY.search(buffer)
    if match is None:
        return None

    out: list[str] = []
    index = match.end()
    while index < len(buffer):
        char = buffer[index]
        if char == "\\":
            if index + 1 >= len(buffer):
                break  # escape split across chunks; wait for more
            escape = buffer[index + 1]
            out.append(_UNESCAPE.get(escape, escape))
            index += 2
            continue
        if char == '"':
            break
        out.append(char)
        index += 1

    return "".join(out)


_UNESCAPE = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def parse_answer_payload(text: str) -> dict:
    """Parse the completed response, tolerating markdown fences some models add
    even when asked for raw JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    return json.loads(cleaned)
