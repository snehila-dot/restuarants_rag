"""Tests for query parser service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.query_parser import (
    Mood,
    ParsedQuery,
    QueryFilters,
    SortPreference,
    _empty_parsed_query,
    _keyword_fallback,
    _llm_extract,
    _resolve_location,
    detect_language,
    parse_query,
)

# ---------------------------------------------------------------------------
# Legacy tests (detect_language kept for backward compat)
# ---------------------------------------------------------------------------


def test_detect_language_english() -> None:
    """Test English language detection."""
    assert detect_language("Find me a vegan restaurant") == "en"
    assert detect_language("cheap Italian food") == "en"


# ---------------------------------------------------------------------------
# Task 1: Schema tests
# ---------------------------------------------------------------------------


def test_parsed_query_minimal() -> None:
    """ParsedQuery accepts all-empty fields."""
    pq = ParsedQuery(
        cuisine_types=[],
        excluded_cuisines=[],
        price_ranges=[],
        excluded_price_ranges=[],
        features=[],
        dish_keywords=[],
        location_name=None,
        mood=None,
        group_size=None,
        time_preference=None,
        sort_by=None,
        language="en",
    )
    assert pq.language == "en"
    assert pq.cuisine_types == []


def test_parsed_query_full() -> None:
    """ParsedQuery accepts all fields populated."""
    pq = ParsedQuery(
        cuisine_types=["italian"],
        excluded_cuisines=["asian"],
        price_ranges=["€€"],
        excluded_price_ranges=["€€€€"],
        features=["outdoor_seating"],
        dish_keywords=["pizza"],
        location_name="hauptplatz",
        mood=Mood.DATE_NIGHT,
        group_size=4,
        time_preference="sunday evening",
        sort_by=SortPreference.RATING,
        language="de",
    )
    assert pq.mood == Mood.DATE_NIGHT
    assert pq.sort_by == SortPreference.RATING


def test_mood_enum_values() -> None:
    """Mood enum has all expected values."""
    assert Mood.DATE_NIGHT == "date_night"
    assert Mood.BUSINESS == "business"
    assert Mood.CASUAL == "casual"
    assert Mood.FAMILY == "family"
    assert Mood.CELEBRATION == "celebration"


def test_sort_preference_enum_values() -> None:
    """SortPreference enum has all expected values."""
    assert SortPreference.RATING == "rating"
    assert SortPreference.DISTANCE == "distance"
    assert SortPreference.PRICE_ASC == "price_asc"
    assert SortPreference.PRICE_DESC == "price_desc"


def test_query_filters_new_fields() -> None:
    """QueryFilters has new fields with correct defaults."""
    qf = QueryFilters()
    assert qf.excluded_cuisines == []
    assert qf.excluded_price_ranges == []
    assert qf.mood is None
    assert qf.group_size is None
    assert qf.time_preference is None
    assert qf.sort_by is None
    assert qf.language == "en"


# ---------------------------------------------------------------------------
# Task 2: LLM extraction tests
# ---------------------------------------------------------------------------


def test_empty_parsed_query() -> None:
    """_empty_parsed_query returns all-empty ParsedQuery."""
    pq = _empty_parsed_query()
    assert pq.cuisine_types == []
    assert pq.excluded_cuisines == []
    assert pq.location_name is None
    assert pq.mood is None
    assert pq.language == "en"


@pytest.mark.asyncio
async def test_llm_extract_simple_cuisine() -> None:
    """LLM extraction returns ParsedQuery for simple cuisine query."""
    mock_parsed = ParsedQuery(
        cuisine_types=["italian"],
        excluded_cuisines=[],
        price_ranges=[],
        excluded_price_ranges=[],
        features=[],
        dish_keywords=["pizza"],
        location_name=None,
        mood=None,
        group_size=None,
        time_preference=None,
        sort_by=None,
        language="en",
    )

    mock_message = MagicMock()
    mock_message.parsed = mock_parsed
    mock_message.refusal = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("app.services.query_parser._parser_client") as mock_client:
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        result = await _llm_extract("I want Italian pizza")

    assert result.cuisine_types == ["italian"]
    assert result.dish_keywords == ["pizza"]
    assert result.language == "en"


@pytest.mark.asyncio
async def test_llm_extract_refusal_returns_empty() -> None:
    """LLM refusal returns empty ParsedQuery."""
    mock_message = MagicMock()
    mock_message.parsed = None
    mock_message.refusal = "I cannot process this request."

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("app.services.query_parser._parser_client") as mock_client:
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        result = await _llm_extract("something inappropriate")

    assert result.cuisine_types == []
    assert result.mood is None
    assert result.language == "en"


@pytest.mark.asyncio
async def test_llm_extract_null_parsed_returns_empty() -> None:
    """LLM returning null parsed (no refusal) returns empty ParsedQuery."""
    mock_message = MagicMock()
    mock_message.parsed = None
    mock_message.refusal = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("app.services.query_parser._parser_client") as mock_client:
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        result = await _llm_extract("...")

    assert result.cuisine_types == []
    assert result.language == "en"


# ---------------------------------------------------------------------------
# Task 3: Location resolution, keyword fallback, async parse_query
# ---------------------------------------------------------------------------


def test_resolve_location_known() -> None:
    """Known Graz location resolves to coordinates."""
    name, lat, lng = _resolve_location("hauptplatz")
    assert name == "hauptplatz"
    assert lat is not None
    assert lng is not None


def test_resolve_location_substring() -> None:
    """Location name containing a known place resolves via substring match."""
    name, lat, lng = _resolve_location("near hauptplatz area")
    assert name == "hauptplatz"
    assert lat is not None


def test_resolve_location_unknown() -> None:
    """Unknown location returns None tuple."""
    name, lat, lng = _resolve_location("nonexistent_place_xyz")
    assert name is None
    assert lat is None
    assert lng is None


def test_keyword_fallback_cuisine() -> None:
    """Keyword fallback extracts basic cuisine types."""
    pq = _keyword_fallback("I want Italian food")
    assert "italian" in pq.cuisine_types


def test_keyword_fallback_price() -> None:
    """Keyword fallback extracts price ranges."""
    pq = _keyword_fallback("cheap restaurant")
    assert "€" in pq.price_ranges


def test_keyword_fallback_language_de() -> None:
    """Keyword fallback detects German."""
    pq = _keyword_fallback("Ich suche ein Restaurant in der Innenstadt")
    assert pq.language == "de"


@pytest.mark.asyncio
async def test_parse_query_uses_llm() -> None:
    """parse_query calls LLM and resolves location."""
    mock_parsed = ParsedQuery(
        cuisine_types=["italian"],
        excluded_cuisines=[],
        price_ranges=["€€"],
        excluded_price_ranges=[],
        features=["outdoor_seating"],
        dish_keywords=[],
        location_name="hauptplatz",
        mood=Mood.DATE_NIGHT,
        group_size=2,
        time_preference=None,
        sort_by=None,
        language="en",
    )

    with patch(
        "app.services.query_parser._llm_extract", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.return_value = mock_parsed
        filters = await parse_query("romantic Italian dinner near Hauptplatz")

    assert filters.cuisine_types == ["italian"]
    assert filters.location_name == "hauptplatz"
    assert filters.location_lat is not None
    assert filters.mood == Mood.DATE_NIGHT


@pytest.mark.asyncio
async def test_parse_query_falls_back_on_llm_error() -> None:
    """parse_query falls back to keywords when LLM fails."""
    with patch(
        "app.services.query_parser._llm_extract", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.side_effect = Exception("API timeout")
        filters = await parse_query("cheap Italian food")

    assert "italian" in filters.cuisine_types
    assert "€" in filters.price_ranges


@pytest.mark.asyncio
async def test_parse_query_with_history_resolves_refinement() -> None:
    """Follow-up 'but cheaper' with history produces lower price filter."""
    history = [
        {"role": "user", "content": "Italian restaurants"},
        {
            "role": "assistant",
            "content": "Here are some Italian restaurants in the €€ range.",
            "restaurants": [{"name": "Test Place", "cuisine": ["Italian"], "price_range": "€€"}],
        },
    ]

    mock_parsed = ParsedQuery(
        cuisine_types=["italian"],
        excluded_cuisines=[],
        price_ranges=["€"],
        excluded_price_ranges=[],
        features=[],
        dish_keywords=[],
        location_name=None,
        mood=None,
        group_size=None,
        time_preference=None,
        sort_by=None,
        language="en",
    )

    with patch(
        "app.services.query_parser._llm_extract", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.return_value = mock_parsed
        filters = await parse_query("but cheaper", conversation_history=history)

    # The mock returns what the LLM would produce given history context
    assert "italian" in filters.cuisine_types
    assert "€" in filters.price_ranges
    # Verify _llm_extract was called with the message (history is in the prompt)
    mock_extract.assert_called_once()


@pytest.mark.asyncio
async def test_parse_query_without_history_still_works() -> None:
    """parse_query with no history works exactly as before."""
    mock_parsed = ParsedQuery(
        cuisine_types=["italian"],
        excluded_cuisines=[],
        price_ranges=[],
        excluded_price_ranges=[],
        features=[],
        dish_keywords=[],
        location_name=None,
        mood=None,
        group_size=None,
        time_preference=None,
        sort_by=None,
        language="en",
    )

    with patch(
        "app.services.query_parser._llm_extract", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.return_value = mock_parsed
        filters = await parse_query("Italian food")

    assert filters.cuisine_types == ["italian"]


@pytest.mark.asyncio
async def test_parse_query_group_size_adds_feature() -> None:
    """Group size >= 5 adds good_for_groups feature."""
    mock_parsed = ParsedQuery(
        cuisine_types=[],
        excluded_cuisines=[],
        price_ranges=[],
        excluded_price_ranges=[],
        features=[],
        dish_keywords=[],
        location_name=None,
        mood=None,
        group_size=8,
        time_preference=None,
        sort_by=None,
        language="en",
    )

    with patch(
        "app.services.query_parser._llm_extract", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.return_value = mock_parsed
        filters = await parse_query("dinner for 8")

    assert "good_for_groups" in filters.features
