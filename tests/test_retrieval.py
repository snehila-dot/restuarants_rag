"""Tests for retrieval service."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.restaurant import Restaurant
from app.services.query_parser import QueryFilters, parse_query
from app.services.retrieval import NoRestaurantsFoundError, search_restaurants


async def test_search_restaurants_by_cuisine(
    session: AsyncSession,
    sample_restaurants: list[Restaurant]
) -> None:
    """Test searching restaurants by cuisine."""
    filters = parse_query("Italian restaurant")
    results = await search_restaurants(session, filters)
    
    assert len(results) > 0
    assert any("Italian" in r.cuisine for r in results)


async def test_search_restaurants_by_price(
    session: AsyncSession,
    sample_restaurants: list[Restaurant]
) -> None:
    """Test searching restaurants by price range."""
    filters = QueryFilters()
    filters.price_ranges = ["€"]
    
    results = await search_restaurants(session, filters)
    assert len(results) > 0
    assert all(r.price_range == "€" for r in results)


async def test_search_restaurants_by_features(
    session: AsyncSession,
    sample_restaurants: list[Restaurant]
) -> None:
    """Test searching restaurants by features."""
    filters = QueryFilters()
    filters.features = ["vegan_options"]
    
    results = await search_restaurants(session, filters)
    assert len(results) > 0
    assert all("vegan_options" in r.features for r in results)


async def test_search_restaurants_no_results(session: AsyncSession) -> None:
    """Test handling of no results."""
    filters = QueryFilters()
    filters.cuisine_types = ["NonexistentCuisine"]
    
    with pytest.raises(NoRestaurantsFoundError):
        await search_restaurants(session, filters)
