import re

from app.schemas.answer import Answer
from app.schemas.classification import Classification

_FIRST_PERSON_PAST = re.compile(
    r"\bI\s+(built|led|shipped|implemented|wrote|deployed|designed|worked on|created|fixed|owned)\b",
    re.IGNORECASE,
)


class AnswerValidationError(Exception):
    """Raised when the LLM's answer is structurally unusable."""


def validate(answer: Answer, classification: Classification) -> Answer:
    if not answer.summary.strip():
        raise AnswerValidationError("Answer summary is empty")

    warnings = list(answer.warnings)

    if not classification.requires_personal_context:
        combined = " ".join([answer.summary, answer.detailed_answer, *answer.key_points])
        if _FIRST_PERSON_PAST.search(combined):
            warnings.append(
                "Answer claims first-person past experience but no personal context was retrieved; "
                "treat specific claims as illustrative, not verified fact."
            )

    return answer.model_copy(update={"warnings": warnings})
