import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_db():
    """Mock the database session for API tests."""
    session = AsyncMock()
    return session


@pytest.fixture
async def client():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_create_simulation(client):
    response = await client.post("/api/simulations", json={
        "name": "Test Sim",
        "initial_capital": 10000.0,
        "config": {"symbols": ["AAPL"], "interval": "1d"},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Sim"
    assert data["initial_capital"] == 10000.0
    assert data["status"] == "created"


@pytest.mark.asyncio
async def test_list_simulations(client):
    # Create one first
    await client.post("/api/simulations", json={
        "name": "Test", "initial_capital": 5000.0,
    })
    response = await client.get("/api/simulations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
