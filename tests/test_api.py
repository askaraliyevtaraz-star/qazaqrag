from fastapi.testclient import (
    TestClient,
)


def test_health(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_ready(
    client: TestClient,
) -> None:
    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["ready"] is True
    assert data["qdrant_points"] == 12


def test_query(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={"question": ("How many students?")},
    )

    assert response.status_code == 200

    data = response.json()

    assert "[S1]" in data["answer"]

    assert data["sources"][0]["source"] == "clubs_kk.md"

    assert data["retrieval_method"] == ("dense+bm25+rrf+reranker")


def test_query_validation(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={"question": ""},
    )

    assert response.status_code == 422
