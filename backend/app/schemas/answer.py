from pydantic import BaseModel

from app.schemas.classification import Classification


class Complexity(BaseModel):
    time: str
    space: str


class AnswerSection(BaseModel):
    """One labeled block of a structured answer (e.g. "Likely Cause",
    "Situation"). Generic on purpose: debugging, SQL, system design, and
    behavioral answers all reduce to "an ordered list of headed sections",
    so one shape covers every mode without a bespoke field per category."""

    heading: str
    content: str


class Answer(BaseModel):
    """Generic interview answer. Mode-specific fields are optional and
    populated only when the route calls for that structure."""

    summary: str
    key_points: list[str] = []
    detailed_answer: str = ""

    # Coding-only fields
    approach: list[str] | None = None
    code: str | None = None
    complexity: Complexity | None = None
    edge_cases: list[str] | None = None

    # Debugging / SQL / system design / behavioral: e.g. Likely Cause,
    # Diagnosis, Fix; or Situation, Task, Action, Result.
    sections: list[AnswerSection] | None = None

    warnings: list[str] = []


class StructuredResponse(BaseModel):
    question: str
    classification: Classification
    answer: Answer
