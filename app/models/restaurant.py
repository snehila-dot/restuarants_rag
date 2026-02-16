from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from app.models.menu_item import MenuItem


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Restaurant(Base):
    """Restaurant model representing a restaurant in Graz."""

    __tablename__ = "restaurants"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    address: Mapped[str] = mapped_column(String(300), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(50))
    website: Mapped[Optional[str]] = mapped_column(String(500))

    # Cuisine types as JSON array (e.g., ["Italian", "Pizza"])
    cuisine: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Price range: "€", "€€", "€€€", "€€€€"
    price_range: Mapped[str] = mapped_column(String(10), nullable=False, default="€€")

    # Opening hours as JSON (e.g., {"monday": "10:00-22:00", ...})
    opening_hours: Mapped[Optional[dict[str, str]]] = mapped_column(JSON)

    # Features/attributes (e.g., ["vegan_options", "outdoor_seating", "wheelchair_accessible"])
    features: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Rating (0.0 - 5.0)
    rating: Mapped[Optional[float]] = mapped_column()

    # Number of reviews
    review_count: Mapped[int] = mapped_column(default=0)

    # Short description/summary
    summary: Mapped[Optional[str]] = mapped_column(Text)

    # Menu items — stored in a separate table via relationship
    menu_items: Mapped[list[MenuItem]] = relationship(
        back_populates="restaurant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Direct URL to the restaurant's menu page (if found)
    menu_url: Mapped[Optional[str]] = mapped_column(String(500))

    # Geographic coordinates
    latitude: Mapped[Optional[float]] = mapped_column()
    longitude: Mapped[Optional[float]] = mapped_column()

    # Data sources (e.g., ["google_maps", "website"])
    data_sources: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow
    )
    last_verified: Mapped[Optional[datetime]] = mapped_column()

    def __repr__(self) -> str:
        return f"<Restaurant(name='{self.name}', cuisine={self.cuisine})>"
