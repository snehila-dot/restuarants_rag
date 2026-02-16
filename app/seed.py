"""Database seeder for restaurant data."""

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, select

from app.database import AsyncSessionLocal, engine
from app.models.menu_item import MenuItem
from app.models.restaurant import Base, Restaurant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Regex to pull a numeric value out of European-style price strings
# Matches: €12.50, € 12,50, 12.50€, 12,90 €, EUR 9.50, etc.
_PRICE_NUM_RE = re.compile(r"\d+[.,]\d{1,2}|\d+")


def _parse_price(price_text: str | None) -> float | None:
    """Convert a price string like '€14,90' to a float (14.9)."""
    if not price_text:
        return None
    m = _PRICE_NUM_RE.search(price_text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


async def seed_from_json(file_path: str) -> None:
    """
    Seed database from JSON file.

    Reads restaurant data (including nested ``menu_items`` dicts)
    and inserts ``Restaurant`` + ``MenuItem`` rows.

    Args:
        file_path: Path to JSON file with restaurant data
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    restaurants_data = data.get("restaurants", [])
    if not restaurants_data:
        logger.warning("No restaurants found in JSON file")
        return

    async with AsyncSessionLocal() as session:
        # Clear all existing data before re-import (cascade deletes menu items)
        result = await session.execute(select(Restaurant))
        existing = result.scalars().all()
        if existing:
            logger.info(
                "Clearing %d existing entries before re-import…",
                len(existing),
            )
            await session.execute(delete(MenuItem))
            await session.execute(delete(Restaurant))
            await session.flush()

        logger.info("Seeding %d restaurants from JSON…", len(restaurants_data))

        total_menu_items = 0
        for entry in restaurants_data:
            restaurant_id = uuid.uuid4()
            restaurant = Restaurant(
                id=restaurant_id,
                name=entry["name"],
                address=entry["address"],
                phone=entry.get("phone"),
                website=entry.get("website"),
                cuisine=entry.get("cuisine", []),
                price_range=entry.get("price_range", "€€"),
                rating=entry.get("rating"),
                review_count=entry.get("review_count", 0),
                features=entry.get("features", []),
                summary=entry.get("summary"),
                menu_url=entry.get("menu_url"),
                latitude=entry.get("latitude"),
                longitude=entry.get("longitude"),
                opening_hours=entry.get("opening_hours"),
                data_sources=entry.get("data_sources", []),
                last_verified=datetime.utcnow(),
            )
            session.add(restaurant)

            # Insert menu items into the separate table
            for item_dict in entry.get("menu_items", []):
                item_name = item_dict.get("name", "").strip()
                if not item_name:
                    continue
                price_text = item_dict.get("price", "")
                menu_item = MenuItem(
                    id=uuid.uuid4(),
                    restaurant_id=restaurant_id,
                    name=item_name[:200],
                    price=_parse_price(price_text),
                    price_text=price_text[:50] if price_text else None,
                    category=item_dict.get("category", "Other")[:50],
                    description=item_dict.get("description", "")[:500] or None,
                )
                session.add(menu_item)
                total_menu_items += 1

        await session.commit()
        logger.info(
            "Seeded %d restaurants with %d menu items",
            len(restaurants_data),
            total_menu_items,
        )


async def main() -> None:
    """Main seeding function."""
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed from scraped JSON
    json_file = Path("data/restaurants.json")
    if json_file.exists():
        await seed_from_json(str(json_file))
    else:
        logger.warning("No data/restaurants.json found. Run 'python -m scraper' first.")


if __name__ == "__main__":
    asyncio.run(main())
