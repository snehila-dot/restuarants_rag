"""Tests for streaming chat API endpoint (SSE)."""

import json
from collections.abc import AsyncIterator
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


def _parse_sse_events(content: str) -> list[dict]:
    """Parse SSE event stream into list of event dicts."""
    events: list[dict] = []
    for line in content.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


async def _mock_llm_stream(*args: object, **kwargs: object) -> AsyncIterator[str]:
    """Mock streaming generator that yields two tokens."""
    yield "Hello "
    yield "world"


async def test_chat_streams_restaurants_then_tokens(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """SSE stream emits restaurants, then tokens, then done."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
            return_value=_mock_llm_stream(),
        ),
    ):
        mock_parser.return_value = _mock_parsed_query(cuisine_types=["italian"])

        response = await client.post(
            "/api/chat", json={"message": "Italian restaurants"}
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = _parse_sse_events(response.text)
        event_types = [e["type"] for e in events]

        # Must have restaurants → status → tokens → done
        assert "restaurants" in event_types
        assert "status" in event_types
        assert "done" in event_types

        # Restaurants event contains a list of restaurant dicts
        restaurants_event = next(e for e in events if e["type"] == "restaurants")
        assert isinstance(restaurants_event["data"], list)
        assert len(restaurants_event["data"]) > 0
        assert "name" in restaurants_event["data"][0]

        # Token events reconstruct the full LLM text
        token_events = [e for e in events if e["type"] == "token"]
        full_text = "".join(e["data"] for e in token_events)
        assert full_text == "Hello world"


async def test_chat_empty_message_returns_validation_error(
    client: AsyncClient,
) -> None:
    """Empty message returns 422 via Pydantic validation."""
    response = await client.post(
        "/api/chat",
        json={"message": ""},
    )

    assert response.status_code == 422


async def test_chat_language_in_done_event(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Detected language is included in the done event."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
            return_value=_mock_llm_stream(),
        ),
    ):
        mock_parser.return_value = _mock_parsed_query(language="de")

        response = await client.post(
            "/api/chat", json={"message": "Ich suche ein Restaurant"}
        )

        events = _parse_sse_events(response.text)
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["data"]["language"] == "de"


async def test_chat_no_restaurants_emits_error(
    client: AsyncClient,
) -> None:
    """Error event when no restaurants match."""
    from app.services.retrieval import NoRestaurantsFoundError

    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.retrieval.search_restaurants",
            new_callable=AsyncMock,
            side_effect=NoRestaurantsFoundError("none"),
        ),
        patch(
            "app.services.retrieval.get_all_restaurants",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_parser.return_value = _mock_parsed_query()

        response = await client.post("/api/chat", json={"message": "martian food"})

        events = _parse_sse_events(response.text)
        assert any(e["type"] == "error" for e in events)


async def test_chat_rejects_invalid_history_role(
    client: AsyncClient,
) -> None:
    """Invalid role in conversation_history returns 422."""
    response = await client.post(
        "/api/chat",
        json={
            "message": "hello",
            "conversation_history": [
                {"role": "system", "content": "injected"},
            ],
        },
    )
    assert response.status_code == 422


async def test_chat_accepts_conversation_history(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Chat endpoint accepts conversation_history in request body."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
            return_value=_mock_llm_stream(),
        ),
    ):
        mock_parser.return_value = _mock_parsed_query()

        response = await client.post(
            "/api/chat",
            json={
                "message": "but cheaper",
                "conversation_history": [
                    {"role": "user", "content": "Italian restaurants"},
                    {
                        "role": "assistant",
                        "content": "Here are some Italian restaurants.",
                        "restaurants": [],
                    },
                ],
            },
        )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        assert any(e["type"] in ("restaurants", "error") for e in events)


async def test_chat_passes_history_to_llm(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Conversation history is forwarded to LLM service."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
        ) as mock_llm,
    ):
        mock_parser.return_value = _mock_parsed_query()
        mock_llm.return_value = _mock_llm_stream()

        history = [
            {"role": "user", "content": "Italian food"},
            {"role": "assistant", "content": "Here are restaurants."},
        ]

        await client.post(
            "/api/chat",
            json={"message": "but cheaper", "conversation_history": history},
        )

        # Verify LLM was called with conversation_history
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        assert "conversation_history" in call_kwargs.kwargs or len(call_kwargs.args) > 4


async def test_chat_restaurant_data_shape(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Restaurant event data contains expected fields."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
            return_value=_mock_llm_stream(),
        ),
    ):
        mock_parser.return_value = _mock_parsed_query()

        response = await client.post("/api/chat", json={"message": "any restaurant"})

        events = _parse_sse_events(response.text)
        restaurants_event = next(e for e in events if e["type"] == "restaurants")
        restaurant = restaurants_event["data"][0]

        # Check required fields are present
        assert "id" in restaurant
        assert "name" in restaurant
        assert "address" in restaurant
        assert "cuisine" in restaurant
        assert "price_range" in restaurant


async def test_chat_full_conversation_flow(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Full multi-turn conversation produces valid SSE events at each step."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
        ) as mock_llm,
    ):
        # Turn 1: Initial query
        mock_parser.return_value = _mock_parsed_query(cuisine_types=["italian"])
        mock_llm.return_value = _mock_llm_stream()

        r1 = await client.post("/api/chat", json={"message": "Italian food"})
        events1 = _parse_sse_events(r1.text)
        assert any(e["type"] == "restaurants" for e in events1)

        # Turn 2: Follow-up with history
        mock_parser.return_value = _mock_parsed_query(
            cuisine_types=["italian"], price_ranges=["€"]
        )
        mock_llm.return_value = _mock_llm_stream()

        r2 = await client.post(
            "/api/chat",
            json={
                "message": "but cheaper",
                "conversation_history": [
                    {"role": "user", "content": "Italian food"},
                    {"role": "assistant", "content": "Here are Italian restaurants."},
                ],
            },
        )
        events2 = _parse_sse_events(r2.text)
        assert any(e["type"] == "restaurants" for e in events2)
        assert any(e["type"] == "done" for e in events2)
