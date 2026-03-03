"""Chat API endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.limiter import limiter
from app.schemas.restaurant import ChatRequest, ChatResponse, RestaurantResponse
from app.services import llm, query_parser, retrieval

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> ChatResponse:
    """
    Process a chat message and return restaurant recommendations.

    Args:
        request: Starlette request (used by rate limiter)
        body: ChatRequest with user's message and optional language
        session: Database session (injected dependency)

    Returns:
        ChatResponse with generated message and relevant restaurants

    Raises:
        HTTPException: On errors during processing
    """
    try:
        # Parse user query to extract filters (async LLM extraction)
        filters = await query_parser.parse_query(body.message)

        # Detect language — prefer explicit request, then parser result
        language = body.language or filters.language

        # Retrieve matching restaurants
        try:
            restaurants = await retrieval.search_restaurants(session, filters)
        except retrieval.NoRestaurantsFoundError:
            if filters.has_location:
                # Widen radius and retry once before giving up
                filters.location_radius_m = 1500
                try:
                    restaurants = await retrieval.search_restaurants(session, filters)
                except retrieval.NoRestaurantsFoundError:
                    restaurants = []
            else:
                # No location filter — fall back to top-rated
                restaurants = await retrieval.get_all_restaurants(session, limit=3)

            if not restaurants:
                raise HTTPException(
                    status_code=404,
                    detail="No restaurants found matching your criteria.",
                ) from None

        # Generate LLM response
        location_hint = (
            f" near {filters.location_name}" if filters.location_name else ""
        )
        response_message = await llm.generate_response(
            user_message=body.message,
            restaurants=restaurants,
            language=language,
            location_hint=location_hint,
        )

        # Convert to response schemas
        restaurant_responses = [
            RestaurantResponse.model_validate(r) for r in restaurants
        ]

        return ChatResponse(
            message=response_message,
            restaurants=restaurant_responses,
            language=language,
        )

    except retrieval.NoRestaurantsFoundError as e:
        logger.warning("No restaurants found: %s", e)
        raise HTTPException(
            status_code=404, detail=str(e)
        ) from None

    except Exception as e:
        logger.error("Error processing chat request: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again.",
        ) from e
