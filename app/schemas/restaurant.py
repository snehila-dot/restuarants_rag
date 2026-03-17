import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class MenuItemResponse(BaseModel):
    """Response schema for a single menu item."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    price: float | None = None
    price_text: str | None = None
    category: str | None = None
    description: str | None = None


class RestaurantResponse(BaseModel):
    """Response schema for restaurant data."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    address: str
    phone: str | None = None
    website: str | None = None
    cuisine: list[str]
    price_range: str
    price_range_text: str | None = Field(
        default=None, description="Actual price range (e.g. 'EUR 8–25')"
    )
    opening_hours: dict[str, str] | None = None
    features: list[str]
    rating: float | None = None
    review_count: int
    summary: str | None = None
    menu_items: list[MenuItemResponse] = Field(
        default_factory=list,
        description="Structured menu items from the menu_items table",
    )
    menu_url: str | None = Field(
        default=None, description="Direct URL to the menu page"
    )
    facebook_url: str | None = Field(default=None, description="Facebook page URL")
    instagram_url: str | None = Field(default=None, description="Instagram profile URL")
    google_place_id: str | None = Field(
        default=None, description="Google Places API identifier"
    )
    latitude: float | None = None
    longitude: float | None = None


class ConversationMessage(BaseModel):
    """A single message in conversation history."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=2000)
    restaurants: list[RestaurantResponse] | None = Field(
        default=None,
        description="Restaurants shown in this assistant message (if any)",
    )


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(
        ..., min_length=1, max_length=1000, description="User's question"
    )
    language: str | None = Field(
        default=None,
        pattern="^(de|en)$",
        description="Response language (de=German, en=English)",
    )
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=10,
        description="Previous messages for conversation context (max 5 pairs)",
    )


class MessageResponse(BaseModel):
    """Individual message in chat response."""

    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    message: str = Field(..., description="Assistant's response message")
    restaurants: list[RestaurantResponse] = Field(
        default_factory=list, description="Relevant restaurants found"
    )
    language: str = Field(default="en", description="Response language")
