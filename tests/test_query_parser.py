"""Tests for query parser service."""


from app.services.query_parser import (
    Mood,
    ParsedQuery,
    QueryFilters,
    SortPreference,
    detect_language,
    parse_query,
)


def test_detect_language_english() -> None:
    """Test English language detection."""
    assert detect_language("Find me a vegan restaurant") == "en"
    assert detect_language("cheap Italian food") == "en"


def test_detect_language_german() -> None:
    """Test German language detection."""
    assert detect_language("Ich suche ein Restaurant") == "de"
    assert detect_language("günstige italienische Küche") == "de"


def test_parse_query_cuisine() -> None:
    """Test cuisine extraction."""
    filters = parse_query("I want Italian food")
    assert "italian" in filters.cuisine_types

    filters = parse_query("vegan restaurant")
    assert "vegan" in filters.cuisine_types


def test_parse_query_price() -> None:
    """Test price range extraction."""
    filters = parse_query("cheap restaurant")
    assert "€" in filters.price_ranges

    filters = parse_query("expensive fine dining")
    assert "€€€" in filters.price_ranges


def test_parse_query_features() -> None:
    """Test feature extraction."""
    filters = parse_query("restaurant with outdoor seating")
    assert "outdoor_seating" in filters.features

    filters = parse_query("vegan options available")
    assert "vegan_options" in filters.features


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
