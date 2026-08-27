from app.schemas.classification import Category

SYSTEM_INSTRUCTION = """You are an interview coaching assistant helping a candidate practice for job interviews.

Rules:
1. Answer clearly and concisely, in a practical, interview-ready way.
2. Never invent personal experience. If no personal context is supplied, speak in the
   conditional ("I would...", "My approach would be..."), not the past tense
   ("I did...", "I built...").
3. For technical questions: explain the approach and mention trade-offs when relevant.
4. For scenario questions: diagnose the problem, describe how you'd investigate,
   propose a solution, and explain how you'd validate it.
5. For coding questions: explain your approach first, then give optimized code,
   then state time/space complexity, then list edge cases.
6. Respond with ONLY a single JSON object matching the requested schema. No markdown
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


def build_prompt(
    question: str,
    category: Category,
    context: list[str],
    conversation_history: list[str],
) -> str:
    parts = [SYSTEM_INSTRUCTION]

    if context:
        parts.append("Relevant personal/knowledge-base context:\n" + "\n".join(f"- {c}" for c in context))

    if conversation_history:
        parts.append("Recent conversation (oldest first):\n" + "\n".join(conversation_history))

    schema = CODING_SCHEMA_HINT if category == Category.CODING else GENERIC_SCHEMA_HINT
    parts.append(f"Respond using exactly this JSON shape:\n{schema}")
    parts.append(f"Interview question ({category.value}): {question}")

    return "\n\n".join(parts)
