"""Tests for LLM service."""

from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm import generate_response_stream


async def test_generate_response_stream_includes_history() -> None:
    """Conversation history is included in LLM messages."""
    history = [
        {"role": "user", "content": "Italian restaurants"},
        {"role": "assistant", "content": "Here are some Italian restaurants."},
    ]

    captured_messages: list[dict] = []

    async def mock_create(**kwargs: object) -> MagicMock:
        captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="ok"))]

        async def single_chunk():
            yield mock_chunk

        return single_chunk()

    with patch("app.services.llm.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)

        tokens = []
        async for token in generate_response_stream(
            user_message="but cheaper",
            restaurants=[],
            language="en",
            conversation_history=history,
        ):
            tokens.append(token)

    # Should have: system, user1, assistant1, user2
    assert len(captured_messages) >= 4
    assert captured_messages[0]["role"] == "system"
    assert captured_messages[1]["role"] == "user"
    assert captured_messages[1]["content"] == "Italian restaurants"
    assert captured_messages[2]["role"] == "assistant"
    assert captured_messages[-1]["role"] == "user"
    assert captured_messages[-1]["content"] == "but cheaper"


async def test_generate_response_stream_works_without_history() -> None:
    """Stream works with empty history (backward compat)."""

    async def mock_create(**kwargs: object) -> MagicMock:
        messages = kwargs.get("messages", [])
        assert len(messages) == 2  # system + user only

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="hi"))]

        async def single_chunk():
            yield mock_chunk

        return single_chunk()

    with patch("app.services.llm.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)

        tokens = []
        async for token in generate_response_stream(
            user_message="hello",
            restaurants=[],
            language="en",
        ):
            tokens.append(token)

    assert tokens == ["hi"]
