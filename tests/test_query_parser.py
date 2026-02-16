"""Tests for query parser service."""

from app.services.query_parser import detect_language, parse_query


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
