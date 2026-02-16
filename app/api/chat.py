"""Chat API endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas.restaurant import ChatRequest, ChatResponse, RestaurantResponse
from app.services import llm, query_parser, retrieval

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    """
    Process a chat message and return restaurant recommendations.

    Args:
        request: ChatRequest with user's message and optional language
        session: Database session (injected dependency)

    Returns:
        ChatResponse with generated message and relevant restaurants

    Raises:
        HTTPException: On errors during processing
    """
    try:
        # Detect language if not provided
        language = request.language or query_parser.detect_language(request.message)

        # Parse user query to extract filters
        filters = query_parser.parse_query(request.message)

        # Retrieve matching restaurants
        try:
            restaurants = await retrieval.search_restaurants(session, filters)
        except retrieval.NoRestaurantsFoundError:
            # If no matches with filters, return top-rated restaurants
            restaurants = await retrieval.get_all_restaurants(session, limit=3)

            if not restaurants:
                raise HTTPException(
                    status_code=404,
                    detail="No restaurants found. Please seed the database first.",
                )

        # Generate LLM response
        response_message = await llm.generate_response(
            user_message=request.message, restaurants=restaurants, language=language
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
        logger.warning(f"No restaurants found: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again.",
        )
