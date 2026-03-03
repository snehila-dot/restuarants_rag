"""Tests for chat API endpoint."""

from unittest.mock import AsyncMock, patch

from httpx import AsyncClient

from app.models.restaurant import Restaurant
from app.services.query_parser import ParsedQuery


def _mock_parsed_query(**overrides: object) -> ParsedQuery:
    """Helper to create a mock ParsedQuery with defaults."""
    defaults: dict[str, object] = {
        "cuisine_types": [],
        "excluded_cuisines": [],
        "price_ranges": [],
        "excluded_price_ranges": [],
        "features": [],
        "dish_keywords": [],
        "location_name": None,
        "mood": None,
        "group_size": None,
        "time_preference": None,
        "sort_by": None,
        "language": "en",
    }
    defaults.update(overrides)
    return ParsedQuery(**defaults)  # type: ignore[arg-type]


async def test_chat_endpoint_success(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Test successful chat endpoint response."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch("app.services.llm.client.chat.completions.create") as mock_llm,
    ):
        mock_parser.return_value = _mock_parsed_query(cuisine_types=["italian"])

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="I found some Italian restaurants for you."
                )
            )
        ]
        mock_llm.return_value = mock_response

        response = await client.post(
            "/api/chat", json={"message": "Italian restaurant"}
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
        json={"message": ""},  # Empty message should fail validation
    )

    assert response.status_code == 422  # Validation error


async def test_chat_endpoint_language_detection(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Test language detection in chat endpoint."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch("app.services.llm.client.chat.completions.create") as mock_llm,
    ):
        mock_parser.return_value = _mock_parsed_query(language="de")

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="Test response"))
        ]
        mock_llm.return_value = mock_response

        response = await client.post(
            "/api/chat", json={"message": "Ich suche ein Restaurant"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "de"
