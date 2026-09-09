import pytest
from fastapi.testclient import (
    TestClient,
)

from app.main import create_app
from tests.fakes import (
    FakeRAGService,
)


@pytest.fixture
def client():
    app = create_app(service_override=(FakeRAGService()))

    with TestClient(app) as client:
        yield client
