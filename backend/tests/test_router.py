import pytest

from app.intelligence.router import Route, route_for
from app.schemas.classification import Category


@pytest.mark.parametrize(
    "category,expected",
    [
        (Category.PERSONAL_EXPERIENCE, Route.RAG),
        (Category.RESUME, Route.RAG),
        (Category.PROJECT, Route.RAG),
        (Category.BEHAVIORAL, Route.RAG),
        (Category.TECHNICAL_KNOWLEDGE, Route.REASONING),
        (Category.SYSTEM_DESIGN, Route.REASONING),
        (Category.SCENARIO, Route.REASONING),
        (Category.DEBUGGING, Route.REASONING),
        (Category.ARCHITECTURE, Route.REASONING),
        (Category.CODING, Route.CODING),
        (Category.SQL, Route.SQL),
        (Category.FOLLOW_UP, Route.FOLLOW_UP),
        (Category.UNKNOWN, Route.REASONING),
    ],
)
def test_routing(category, expected):
    assert route_for(category) == expected


def test_every_category_is_routable():
    for category in Category:
        assert route_for(category) is not None
