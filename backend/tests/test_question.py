import pytest


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}])
def test_invalid_question_rejected(client, payload):
    assert client.post("/question", json=payload).status_code == 422


def test_valid_question_returns_structured_response(client):
    response = client.post(
        "/question",
        json={"question": "How would you handle duplicate records in a data pipeline?"},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["question"].startswith("How would you handle")
    assert body["classification"]["category"] == "SCENARIO"
    assert body["classification"]["domain"] == "DATA_ENGINEERING"
    assert body["answer"]["summary"]
    assert body["answer"]["key_points"]
