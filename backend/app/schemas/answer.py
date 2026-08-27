from pydantic import BaseModel

from app.schemas.classification import Classification


class Complexity(BaseModel):
    time: str
    space: str


class Answer(BaseModel):
    """Generic interview answer. Coding-specific fields are optional and
    populated only when the route is CODING."""

    summary: str
    key_points: list[str] = []
    detailed_answer: str = ""

    # Coding-only fields
    approach: list[str] | None = None
    code: str | None = None
    complexity: Complexity | None = None
    edge_cases: list[str] | None = None

    warnings: list[str] = []


class StructuredResponse(BaseModel):
    question: str
    classification: Classification
    answer: Answer
