import re

from app.schemas.answer import Answer
from app.schemas.classification import Classification

_FIRST_PERSON_PAST = re.compile(
    r"\bI\s+(built|led|shipped|implemented|wrote|deployed|designed|worked on|created|fixed|owned)\b",
    re.IGNORECASE,
)


class AnswerValidationError(Exception):
    """Raised when the LLM's answer is structurally unusable."""


def validate(
    answer: Answer,
    classification: Classification,
    context_found: bool = False,
) -> Answer:
    """Validate an answer and attach warnings.

    The fabrication check keys off `context_found` — whether retrieval actually
    returned grounding — rather than `classification.requires_personal_context`,
    which only says personal context was *expected*. A behavioural question
    against an empty knowledge base expects personal context and finds none;
    that is exactly when an invented "I led the migration" is most dangerous and
    least likely to be caught.
    """
    if not answer.summary.strip():
        raise AnswerValidationError("Answer summary is empty")

    warnings = list(answer.warnings)

    if not context_found:
        combined = " ".join([
            answer.summary,
            answer.detailed_answer,
            *answer.key_points,
            *(s.content for s in answer.sections or []),
        ])
        if _FIRST_PERSON_PAST.search(combined):
            warnings.append(
                "Answer claims first-person past experience but no personal context was "
                "retrieved; treat specific claims as illustrative, not verified fact."
            )

    return answer.model_copy(update={"warnings": warnings})
