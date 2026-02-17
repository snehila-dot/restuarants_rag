import uuid
from datetime import datetime
from typing import Optional

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
    phone: Optional[str] = None
    website: Optional[str] = None
    cuisine: list[str]
    price_range: str
    opening_hours: Optional[dict[str, str]] = None
    features: list[str]
    rating: Optional[float] = None
    review_count: int
    summary: Optional[str] = None
    menu_items: list[MenuItemResponse] = Field(
        default_factory=list,
        description="Structured menu items from the menu_items table",
    )
    menu_url: Optional[str] = Field(
        default=None, description="Direct URL to the menu page"
    )
    facebook_url: Optional[str] = Field(
        default=None, description="Facebook page URL"
    )
    instagram_url: Optional[str] = Field(
        default=None, description="Instagram profile URL"
    )
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(
        ..., min_length=1, max_length=1000, description="User's question"
    )
    language: Optional[str] = Field(
        default=None,
        pattern="^(de|en)$",
        description="Response language (de=German, en=English)",
    )


class MessageResponse(BaseModel):
    """Individual message in chat response."""

    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    message: str = Field(..., description="Assistant's response message")
    restaurants: list[RestaurantResponse] = Field(
        default_factory=list, description="Relevant restaurants found"
    )
    language: str = Field(default="en", description="Response language")
