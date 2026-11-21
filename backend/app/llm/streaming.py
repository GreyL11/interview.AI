import json
import re

_SUMMARY_KEY = re.compile(r'"summary"\s*:\s*"')


def extract_partial_summary(buffer: str) -> str | None:
    """Pull the `summary` value out of a JSON document that is still arriving.

    The provider streams the structured answer as raw JSON text, so nothing is
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
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # A max_tokens cut-off (or dropped stream) leaves valid JSON minus its
        # tail. Show the user what arrived rather than failing the answer.
        repaired = _close_truncated(cleaned)
        if repaired is not None:
            return repaired
        summary = extract_partial_summary(cleaned)
        if summary:
            return {"summary": summary}
        raise


def _close_truncated(text: str) -> dict | None:
    """Close the open string/brackets of a cut-off JSON object; trailing
    half-written members are dropped back to the last complete one."""
    stack: list[str] = []
    in_str = esc = False
    for ch in text:
        if in_str:
            esc = (not esc) if ch == "\\" else False
            if ch == '"' and not esc:
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    tail = ('"' if in_str else "") + "".join(reversed(stack))
    candidate = text + tail
    # Progressively drop the trailing member (cut at the last comma) until it parses.
    for _ in range(8):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) and data.get("summary") else None
        except json.JSONDecodeError:
            cut = text.rfind(",")
            if cut <= 0:
                return None
            text = text[:cut]
            candidate = text + _closers(text)
    return None


def _closers(text: str) -> str:
    stack: list[str] = []
    in_str = esc = False
    for ch in text:
        if in_str:
            esc = (not esc) if ch == "\\" else False
            if ch == '"' and not esc:
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return ('"' if in_str else "") + "".join(reversed(stack))
