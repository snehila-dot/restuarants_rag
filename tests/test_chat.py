"""Tests for chat API endpoint."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, patch

from app.models.restaurant import Restaurant


async def test_chat_endpoint_success(
    client: AsyncClient,
    sample_restaurants: list[Restaurant]
) -> None:
    """Test successful chat endpoint response."""
    # Mock LLM response to avoid actual API calls
    with patch("app.services.llm.client.chat.completions.create") as mock_llm:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="I found some Italian restaurants for you."))]
        mock_llm.return_value = mock_response
        
        response = await client.post(
            "/api/chat",
            json={"message": "Italian restaurant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "restaurants" in data
        assert "language" in data
        assert isinstance(data["restaurants"], list)


async def test_chat_endpoint_validation_error(client: AsyncClient) -> None:
    """Test chat endpoint with invalid input."""
    response = await client.post(
        "/api/chat",
        json={"message": ""}  # Empty message should fail validation
    )
    
    assert response.status_code == 422  # Validation error


async def test_chat_endpoint_language_detection(
    client: AsyncClient,
    sample_restaurants: list[Restaurant]
) -> None:
    """Test language detection in chat endpoint."""
    with patch("app.services.llm.client.chat.completions.create") as mock_llm:
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Test response"))]
        mock_llm.return_value = mock_response
        
        response = await client.post(
            "/api/chat",
            json={"message": "Ich suche ein Restaurant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "de"
