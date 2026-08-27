from enum import StrEnum

from app.schemas.classification import Category


class Route(StrEnum):
    RAG = "RAG"                # retriever -> LLM
    REASONING = "REASONING"    # LLM only
    CODING = "CODING"          # coding prompt -> LLM
    SQL = "SQL"                # SQL prompt -> LLM
    FOLLOW_UP = "FOLLOW_UP"    # session memory + previous context -> LLM


_ROUTES: dict[Category, Route] = {
    Category.PERSONAL_EXPERIENCE: Route.RAG,
    Category.RESUME: Route.RAG,
    Category.PROJECT: Route.RAG,
    Category.BEHAVIORAL: Route.RAG,
    Category.TECHNICAL_KNOWLEDGE: Route.REASONING,
    Category.SYSTEM_DESIGN: Route.REASONING,
    Category.SCENARIO: Route.REASONING,
    Category.DEBUGGING: Route.REASONING,
    Category.ARCHITECTURE: Route.REASONING,
    Category.CODING: Route.CODING,
    Category.SQL: Route.SQL,
    Category.FOLLOW_UP: Route.FOLLOW_UP,
    Category.UNKNOWN: Route.REASONING,
}


def route_for(category: Category) -> Route:
    return _ROUTES[category]
