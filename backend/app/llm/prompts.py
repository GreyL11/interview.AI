from app.schemas.classification import Category

SYSTEM_INSTRUCTION = """You are an interview coaching assistant helping a candidate practice for job interviews.

Rules:
1. Answer clearly and concisely, in a practical, interview-ready way.
2. Never invent personal experience. If no personal context is supplied, speak in the
   conditional ("I would...", "My approach would be..."), not the past tense
   ("I did...", "I built...").
3. Put the single most useful sentence in "summary" -- write it as if it is the only
   line the candidate has time to read before speaking. Everything else supports it.
4. For technical questions: explain the approach and mention trade-offs when relevant.
5. For scenario questions: diagnose the problem, describe how you'd investigate,
   propose a solution, and explain how you'd validate it.
6. For coding questions: explain your approach first, then give optimized code,
   then state time/space complexity, then list edge cases. If multiple approaches
   are explicitly requested (e.g. "brute-force first, then optimize"), summarize
   the brute-force approach briefly in the approach/explanation, but "code" holds
   only the final, optimized, working solution unless the brute-force code itself
   was explicitly asked for.
7. For debugging questions: state the likely cause, how you'd diagnose it, the fix,
   and why the fix works.
8. For SQL questions: explain the approach, give the query, explain it, then note
   performance considerations (indexes, scan cost) if relevant.
9. For system design questions: cover requirements, high-level architecture,
   components, data flow, trade-offs, and scaling considerations.
10. For behavioral questions: use Situation, Task, Action, Result.
11. There is exactly one question to answer: the one under "CURRENT INTERVIEWER
    QUESTION" below. Any earlier conversation is background only -- if it looks like
    a different topic, ignore it and answer the current question; only draw on it
    when the current question is clearly a follow-up (e.g. "what about...", "why",
    a bare pronoun referring back).
12. A narrow follow-up gets a narrow answer, not a restatement of everything already
    said. "What's the time complexity?" gets the complexity, not the solution again.
    "Can we optimize it?" focuses on the optimization and how it compares to the
    previous approach. "Without extra space?" / "without sorting?" re-solves under
    that new constraint. Use the previous conversation to know what "it"/"this" refers
    to, but do not repeat code or explanation the candidate already has.
13. Respond with ONLY a single JSON object matching the requested schema. No markdown
    fences, no commentary outside the JSON.
"""

GENERIC_SCHEMA_HINT = """{
  "summary": "one or two sentence answer",
  "key_points": ["short bullet", "short bullet"],
  "detailed_answer": "full explanation"
}"""

CODING_SCHEMA_HINT = """{
  "summary": "one or two sentence answer",
  "approach": ["step 1", "step 2"],
  "code": "full code solution",
  "complexity": {"time": "O(n)", "space": "O(n)"},
  "edge_cases": ["edge case 1", "edge case 2"]
}"""

SQL_SCHEMA_HINT = """{
  "summary": "one or two sentence answer",
  "key_points": ["approach step 1", "approach step 2"],
  "code": "the SQL query",
  "sections": [
    {"heading": "Explanation", "content": "what the query does and why"},
    {"heading": "Performance", "content": "indexes / scan cost considerations, if relevant"}
  ]
}"""

DEBUGGING_SCHEMA_HINT = """{
  "summary": "one or two sentence answer",
  "sections": [
    {"heading": "Likely Cause", "content": "..."},
    {"heading": "Diagnosis", "content": "how you would confirm it"},
    {"heading": "Fix", "content": "..."},
    {"heading": "Why It Works", "content": "..."}
  ],
  "code": "fix snippet, only if a code change is the fix, else omit"
}"""

SYSTEM_DESIGN_SCHEMA_HINT = """{
  "summary": "one or two sentence answer",
  "sections": [
    {"heading": "Requirements", "content": "functional and non-functional"},
    {"heading": "High-Level Architecture", "content": "..."},
    {"heading": "Components", "content": "..."},
    {"heading": "Data Flow", "content": "..."},
    {"heading": "Trade-offs", "content": "..."},
    {"heading": "Scaling Considerations", "content": "..."}
  ]
}"""

BEHAVIORAL_SCHEMA_HINT = """{
  "summary": "one or two sentence answer",
  "sections": [
    {"heading": "Situation", "content": "..."},
    {"heading": "Task", "content": "..."},
    {"heading": "Action", "content": "..."},
    {"heading": "Result", "content": "..."}
  ]
}"""

# Deliberately not every Category: most (TECHNICAL_KNOWLEDGE, SCENARIO,
# PERSONAL_EXPERIENCE, RESUME, PROJECT, FOLLOW_UP, UNKNOWN) read fine as
# summary/key_points/detailed_answer and get no special-cased structure.
_SCHEMA_BY_CATEGORY: dict[Category, str] = {
    Category.CODING: CODING_SCHEMA_HINT,
    Category.SQL: SQL_SCHEMA_HINT,
    Category.DEBUGGING: DEBUGGING_SCHEMA_HINT,
    Category.SYSTEM_DESIGN: SYSTEM_DESIGN_SCHEMA_HINT,
    Category.ARCHITECTURE: SYSTEM_DESIGN_SCHEMA_HINT,
    Category.BEHAVIORAL: BEHAVIORAL_SCHEMA_HINT,
}


def build_prompt(
    question: str,
    category: Category,
    context: list[str],
    conversation_history: list[str],
) -> str:
    parts = [SYSTEM_INSTRUCTION]

    if context or conversation_history:
        background = ["INTERVIEW CONTEXT (background only -- see rule 11 above):"]
        if context:
            background.append(
                "Relevant personal/knowledge-base context:\n"
                + "\n".join(f"- {c}" for c in context)
            )
        if conversation_history:
            background.append(
                "Previous Q&A this session (oldest first):\n" + "\n".join(conversation_history)
            )
        parts.append("\n\n".join(background))

    schema = _SCHEMA_BY_CATEGORY.get(category, GENERIC_SCHEMA_HINT)
    parts.append(f"Respond using exactly this JSON shape:\n{schema}")
    parts.append(f"CURRENT INTERVIEWER QUESTION ({category.value}): {question}")

    return "\n\n".join(parts)
