"""Retrieval service for querying and ranking restaurants from the database."""

import math

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.menu_item import MenuItem
from app.models.restaurant import Restaurant
from app.services.query_parser import QueryFilters

# Approximate metres-per-degree at Graz latitude (~47°N)
_M_PER_DEG_LAT = 111_000.0
_M_PER_DEG_LNG = 111_000.0 * math.cos(math.radians(47.07))


class NoRestaurantsFoundError(Exception):
    """Raised when no restaurants match the given filters."""

    pass


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Fast approximate distance in metres between two points near Graz."""
    dlat = (lat2 - lat1) * _M_PER_DEG_LAT
    dlng = (lng2 - lng1) * _M_PER_DEG_LNG
    return math.sqrt(dlat * dlat + dlng * dlng)


async def search_restaurants(
    session: AsyncSession, filters: QueryFilters
) -> list[Restaurant]:
    """
    Search for restaurants matching the given filters.

    Args:
        session: Database session
        filters: Structured query filters

    Returns:
        List of matching Restaurant objects, ranked by relevance

    Raises:
        NoRestaurantsFoundError: If no restaurants match the criteria
    """
    # Start with base query — eager-load menu items
    query = select(Restaurant).options(selectinload(Restaurant.menu_items))

    # Apply price range filters
    if filters.price_ranges:
        query = query.where(Restaurant.price_range.in_(filters.price_ranges))

    # Apply dish keyword filter via SQL join on menu_items table
    if filters.dish_keywords:
        like_clauses = [MenuItem.name.ilike(f"%{kw}%") for kw in filters.dish_keywords]
        query = query.join(Restaurant.menu_items).where(or_(*like_clauses)).distinct()

    # Execute query
    result = await session.execute(query)
    restaurants = list(result.scalars().unique().all())

    # Post-process filtering for JSON fields (SQLite limitation)
    if filters.cuisine_types:
        restaurants = [
            r
            for r in restaurants
            if any(
                cuisine.lower() in [c.lower() for c in r.cuisine]
                for cuisine in filters.cuisine_types
            )
        ]

    if filters.features:
        restaurants = [
            r
            for r in restaurants
            if all(feature in r.features for feature in filters.features)
        ]

    # Geo-distance filtering — keep only restaurants within radius
    if filters.has_location:
        assert filters.location_lat is not None  # for type checker
        assert filters.location_lng is not None
        radius = filters.location_radius_m
        scored: list[tuple[float, Restaurant]] = []
        for r in restaurants:
            if r.latitude is None or r.longitude is None:
                continue
            dist = _haversine_m(
                filters.location_lat,
                filters.location_lng,
                r.latitude,
                r.longitude,
            )
            if dist <= radius:
                scored.append((dist, r))
        # Sort by distance (closest first)
        scored.sort(key=lambda t: t[0])
        restaurants = [r for _, r in scored]
    else:
        # No location → rank by rating
        restaurants.sort(key=lambda r: (r.rating or 0.0, r.review_count), reverse=True)

    # Limit results
    restaurants = restaurants[: settings.max_results]

    if not restaurants:
        raise NoRestaurantsFoundError("No restaurants found matching your criteria")

    return restaurants


async def get_all_restaurants(
    session: AsyncSession, limit: int = 10
) -> list[Restaurant]:
    """
    Get all restaurants, sorted by rating.

    Args:
        session: Database session
        limit: Maximum number of restaurants to return

    Returns:
        List of Restaurant objects
    """
    query = (
        select(Restaurant)
        .options(selectinload(Restaurant.menu_items))
        .order_by(Restaurant.rating.desc().nulls_last(), Restaurant.review_count.desc())
        .limit(limit)
    )

    result = await session.execute(query)
    restaurants = list(result.scalars().all())

    return restaurants
