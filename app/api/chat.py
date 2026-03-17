"""Chat API endpoint with Server-Sent Events streaming."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.limiter import limiter
from app.schemas.restaurant import ChatRequest, ConversationMessage, RestaurantResponse
from app.services import llm, query_parser, retrieval

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _sse_event(event_type: str, data: object) -> str:
    """Format a Server-Sent Event.

    Args:
        event_type: Event category (restaurants, token, done, error, status).
        data: JSON-serialisable payload.

    Returns:
        SSE-formatted string ``"data: {...}\\n\\n"``.
    """
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _chat_stream(
    message: str,
    language: str | None,
    conversation_history: list[ConversationMessage],
    session: AsyncSession,
) -> AsyncIterator[str]:
    """Async generator that yields SSE events for the chat response.

    Phase 1: Parse query, fetch restaurants, emit restaurant data.
    Phase 2: Stream LLM response tokens.

    Args:
        message: User's chat message.
        language: Explicit language preference or ``None``.
        session: Database session.

    Yields:
        SSE-formatted event strings.
    """
    try:
        # --- Phase 1: Parse + DB lookup ---
        filters = await query_parser.parse_query(message)
        detected_language = language or filters.language

        try:
            restaurants = await retrieval.search_restaurants(session, filters)
        except retrieval.NoRestaurantsFoundError:
            if filters.has_location:
                filters.location_radius_m = 1500
                try:
                    restaurants = await retrieval.search_restaurants(session, filters)
                except retrieval.NoRestaurantsFoundError:
                    restaurants = []
            else:
                restaurants = await retrieval.get_all_restaurants(session, limit=3)

        if not restaurants:
            yield _sse_event("error", "No restaurants found matching your criteria.")
            return

        # Emit restaurant cards immediately
        restaurant_responses = [
            RestaurantResponse.model_validate(r).model_dump(mode="json")
            for r in restaurants
        ]
        yield _sse_event("restaurants", restaurant_responses)

        # --- Phase 2: Stream LLM response ---
        yield _sse_event("status", "generating")

        location_hint = (
            f" near {filters.location_name}" if filters.location_name else ""
        )

        async for token in llm.generate_response_stream(
            user_message=message,
            restaurants=restaurants,
            language=detected_language,
            location_hint=location_hint,
        ):
            yield _sse_event("token", token)

        yield _sse_event("done", {"language": detected_language})

    except Exception as e:
        logger.error("Error in chat stream: %s", e, exc_info=True)
        yield _sse_event(
            "error",
            "An error occurred while processing your request.",
        )


@router.post("/chat")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> StreamingResponse:
    """Stream chat response as Server-Sent Events.

    The client receives:

    - ``restaurants`` event with matched restaurant data
    - ``status`` event when LLM generation starts
    - ``token`` events with LLM response chunks
    - ``done`` event when complete
    - ``error`` event on failure

    Args:
        request: Starlette request (used by rate limiter).
        body: Validated chat request with message and optional history.
        session: Database session (injected dependency).

    Returns:
        ``StreamingResponse`` with ``text/event-stream`` content type.
    """
    return StreamingResponse(
        _chat_stream(body.message, body.language, body.conversation_history, session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
