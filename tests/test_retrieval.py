"""Tests for retrieval service."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.restaurant import Restaurant
from app.services.query_parser import Mood, QueryFilters, SortPreference
from app.services.retrieval import NoRestaurantsFoundError, search_restaurants


async def test_search_restaurants_by_cuisine(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
) -> None:
    """Test searching restaurants by cuisine."""
    filters = QueryFilters()
    filters.cuisine_types = ["Italian"]
    results = await search_restaurants(session, filters)

    assert len(results) > 0
    assert any("Italian" in r.cuisine for r in results)


async def test_search_restaurants_by_price(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
) -> None:
    """Test searching restaurants by price range."""
    filters = QueryFilters()
    filters.price_ranges = ["€"]

    results = await search_restaurants(session, filters)
    assert len(results) > 0
    assert all(r.price_range == "€" for r in results)


async def test_search_restaurants_by_features(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
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


# ---------------------------------------------------------------------------
# Task 4: New filter tests
# ---------------------------------------------------------------------------


async def test_search_excludes_cuisines(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
) -> None:
    """Excluded cuisines are filtered out."""
    filters = QueryFilters()
    filters.excluded_cuisines = ["Italian"]
    results = await search_restaurants(session, filters)
    for r in results:
        assert "Italian" not in r.cuisine


async def test_search_excludes_price_ranges(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
) -> None:
    """Excluded price ranges are filtered out."""
    filters = QueryFilters()
    filters.excluded_price_ranges = ["€€€", "€€€€"]
    results = await search_restaurants(session, filters)
    for r in results:
        assert r.price_range not in ("€€€", "€€€€")


async def test_search_mood_boost_date_night(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
) -> None:
    """Date night mood boosts restaurants with wine/outdoor_seating."""
    filters = QueryFilters()
    filters.mood = Mood.DATE_NIGHT
    results = await search_restaurants(session, filters)
    assert len(results) > 0
    top = results[0]
    assert any(
        f in top.features for f in ("serves_wine", "outdoor_seating", "reservations")
    )


async def test_search_time_filter_excludes_closed(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
) -> None:
    """Time filter excludes restaurants closed on that day."""
    filters = QueryFilters()
    filters.time_preference = "sunday"
    results = await search_restaurants(session, filters)
    for r in results:
        if r.opening_hours:
            sunday = r.opening_hours.get("sunday", "")
            assert sunday.lower() != "closed"


async def test_search_sort_by_price_asc(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
) -> None:
    """Sort by price ascending puts cheap restaurants first."""
    filters = QueryFilters()
    filters.sort_by = SortPreference.PRICE_ASC
    results = await search_restaurants(session, filters)
    price_order = {"€": 1, "€€": 2, "€€€": 3, "€€€€": 4}
    prices = [price_order.get(r.price_range, 99) for r in results]
    assert prices == sorted(prices)
