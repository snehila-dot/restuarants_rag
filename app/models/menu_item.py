from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.restaurant import Base

if TYPE_CHECKING:
    from app.models.restaurant import Restaurant


class MenuItem(Base):
    """A single dish/drink on a restaurant's menu."""

    __tablename__ = "menu_items"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    restaurant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("restaurants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    price: Mapped[float | None] = mapped_column(Float)
    price_text: Mapped[str | None] = mapped_column(String(50))
    category: Mapped[str | None] = mapped_column(String(50))
    description: Mapped[str | None] = mapped_column(String(500))

    # Back-reference to the parent restaurant
    restaurant: Mapped[Restaurant] = relationship(back_populates="menu_items")

    def __repr__(self) -> str:
        return f"<MenuItem(name='{self.name}', price={self.price}, restaurant_id={self.restaurant_id})>"
