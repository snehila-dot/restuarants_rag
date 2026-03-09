"""Tests for app.seed module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.models.menu_item import MenuItem
from app.models.restaurant import Restaurant
from app.seed import _merge_cuisines, _parse_price, seed_from_json

# ---------------------------------------------------------------------------
# _parse_price
# ---------------------------------------------------------------------------


class TestParsePrice:
    """Tests for the _parse_price helper."""

    def test_standard_euro(self) -> None:
        assert _parse_price("€14,90") == 14.9

    def test_euro_with_space(self) -> None:
        assert _parse_price("€ 12,50") == 12.5

    def test_euro_suffix(self) -> None:
        assert _parse_price("12.50€") == 12.5

    def test_euro_suffix_space(self) -> None:
        assert _parse_price("9,90 €") == 9.9

    def test_eur_prefix(self) -> None:
        assert _parse_price("EUR 9.50") == 9.5

    def test_integer_price(self) -> None:
        assert _parse_price("€10") == 10.0

    def test_none_input(self) -> None:
        assert _parse_price(None) is None

    def test_empty_string(self) -> None:
        assert _parse_price("") is None

    def test_no_number(self) -> None:
        assert _parse_price("free") is None

    def test_comma_decimal(self) -> None:
        assert _parse_price("14,50") == 14.5

    def test_dot_decimal(self) -> None:
        assert _parse_price("14.50") == 14.5


# ---------------------------------------------------------------------------
# _merge_cuisines
# ---------------------------------------------------------------------------


class TestMergeCuisines:
    """Tests for the _merge_cuisines helper."""

    def test_no_overlap(self) -> None:
        result = _merge_cuisines(["Italian"], ["Asian"])
        assert result == ["Italian", "Asian"]

    def test_duplicate_case_insensitive(self) -> None:
        result = _merge_cuisines(["Italian"], ["italian"])
        assert result == ["Italian"]

    def test_osm_takes_priority(self) -> None:
        result = _merge_cuisines(["PIZZA"], ["pizza", "Burger"])
        assert result == ["PIZZA", "Burger"]

    def test_both_empty(self) -> None:
        assert _merge_cuisines([], []) == []

    def test_osm_only(self) -> None:
        assert _merge_cuisines(["Italian", "Pizza"], []) == ["Italian", "Pizza"]

    def test_google_only(self) -> None:
        assert _merge_cuisines([], ["Asian"]) == ["Asian"]

    def test_preserves_order(self) -> None:
        result = _merge_cuisines(["C", "A"], ["B"])
        assert result == ["C", "A", "B"]


# ---------------------------------------------------------------------------
# seed_from_json  (integration — uses real async DB session from conftest)
# ---------------------------------------------------------------------------


@pytest.fixture
def restaurant_json(tmp_path: Path) -> Path:
    """Create a minimal restaurants.json for seeding."""
    data = {
        "restaurants": [
            {
                "name": "Seed Test Bistro",
                "address": "Testgasse 1, 8010 Graz",
                "phone": "+43 316 123456",
                "cuisine": ["Austrian"],
                "google_cuisine": ["European"],
                "price_range": "€€",
                "rating": 4.2,
                "review_count": 55,
                "features": ["outdoor_seating"],
                "summary": "A seeded test restaurant.",
                "menu_items": [
                    {"name": "Wiener Schnitzel", "price": "€14,90", "category": "Main"},
                    {"name": "Apfelstrudel", "price": "€6,50", "category": "Dessert"},
                ],
            },
            {
                "name": "Seed Test Sushi",
                "address": "Testgasse 2, 8010 Graz",
                "cuisine": ["Japanese"],
                "price_range": "€€€",
                "menu_items": [],
            },
        ],
    }
    path = tmp_path / "restaurants.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def empty_json(tmp_path: Path) -> Path:
    """Create a restaurants.json with no restaurants."""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"restaurants": []}), encoding="utf-8")
    return path


async def test_seed_from_json_inserts_restaurants(
    session: None,  # noqa: ARG001 — triggers DB setup
    restaurant_json: Path,
) -> None:
    """seed_from_json should insert restaurants and menu items."""
    # Patch AsyncSessionLocal to use the test engine's session factory
    from tests.conftest import TestSessionLocal

    with patch("app.seed.AsyncSessionLocal", TestSessionLocal):
        await seed_from_json(str(restaurant_json))

    async with TestSessionLocal() as s:
        result = await s.execute(select(Restaurant))
        restaurants = result.scalars().all()

    assert len(restaurants) == 2

    names = {r.name for r in restaurants}
    assert "Seed Test Bistro" in names
    assert "Seed Test Sushi" in names

    # Check cuisine merge worked (Austrian + European)
    bistro = next(r for r in restaurants if r.name == "Seed Test Bistro")
    assert "Austrian" in bistro.cuisine
    assert "European" in bistro.cuisine


async def test_seed_from_json_inserts_menu_items(
    session: None,  # noqa: ARG001
    restaurant_json: Path,
) -> None:
    """seed_from_json should insert menu items linked to restaurants."""
    from tests.conftest import TestSessionLocal

    with patch("app.seed.AsyncSessionLocal", TestSessionLocal):
        await seed_from_json(str(restaurant_json))

    async with TestSessionLocal() as s:
        result = await s.execute(select(MenuItem))
        items = result.scalars().all()

    assert len(items) == 2
    item_names = {i.name for i in items}
    assert "Wiener Schnitzel" in item_names
    assert "Apfelstrudel" in item_names

    schnitzel = next(i for i in items if i.name == "Wiener Schnitzel")
    assert schnitzel.price == 14.9
    assert schnitzel.category == "Main"


async def test_seed_from_json_skips_empty_menu_item_names(
    session: None,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    """Menu items with blank names should be skipped."""
    data = {
        "restaurants": [
            {
                "name": "Blank Menu Test",
                "address": "Test 1, Graz",
                "cuisine": [],
                "menu_items": [
                    {"name": "", "price": "€5"},
                    {"name": "  ", "price": "€6"},
                    {"name": "Valid Item", "price": "€7"},
                ],
            },
        ],
    }
    path = tmp_path / "restaurants.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    from tests.conftest import TestSessionLocal

    with patch("app.seed.AsyncSessionLocal", TestSessionLocal):
        await seed_from_json(str(path))

    async with TestSessionLocal() as s:
        result = await s.execute(select(MenuItem))
        items = result.scalars().all()

    assert len(items) == 1
    assert items[0].name == "Valid Item"


async def test_seed_from_json_nonexistent_file() -> None:
    """seed_from_json should return early for missing files."""
    await seed_from_json("/nonexistent/path/restaurants.json")
    # No exception — just logs an error and returns


async def test_seed_from_json_empty_data(
    session: None,  # noqa: ARG001
    empty_json: Path,
) -> None:
    """seed_from_json with no restaurants should insert nothing."""
    from tests.conftest import TestSessionLocal

    with patch("app.seed.AsyncSessionLocal", TestSessionLocal):
        await seed_from_json(str(empty_json))

    async with TestSessionLocal() as s:
        result = await s.execute(select(Restaurant))
        restaurants = result.scalars().all()

    assert len(restaurants) == 0


async def test_seed_from_json_replaces_existing(
    session: None,  # noqa: ARG001
    restaurant_json: Path,
) -> None:
    """Running seed_from_json twice should clear and re-insert."""
    from tests.conftest import TestSessionLocal

    with patch("app.seed.AsyncSessionLocal", TestSessionLocal):
        await seed_from_json(str(restaurant_json))
        # Seed again — should replace, not duplicate
        await seed_from_json(str(restaurant_json))

    async with TestSessionLocal() as s:
        result = await s.execute(select(Restaurant))
        restaurants = result.scalars().all()

    assert len(restaurants) == 2
