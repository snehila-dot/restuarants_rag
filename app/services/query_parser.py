"""Query parser service for extracting user intent and filters from natural language."""

import re
from typing import Optional


class QueryFilters:
    """Structured filters extracted from user query."""

    def __init__(self) -> None:
        self.cuisine_types: list[str] = []
        self.price_ranges: list[str] = []
        self.features: list[str] = []
        self.dish_keywords: list[str] = []
        self.location: Optional[str] = None
        self.query_text: str = ""


def detect_language(text: str) -> str:
    """
    Detect if the text is German or English.

    Args:
        text: Input text to analyze

    Returns:
        'de' for German, 'en' for English
    """
    # Common German words and patterns
    german_indicators = [
        r"\b(ich|du|der|die|das|und|oder|für|mit|von|auf|über)\b",
        r"\b(restaurant|café|wo|gibt|suche|möchte|gerne)\b",
        r"\b(günstig|teuer|billig)\b",
    ]

    text_lower = text.lower()
    german_matches = sum(
        len(re.findall(pattern, text_lower, re.IGNORECASE))
        for pattern in german_indicators
    )

    # Default to English unless we have strong German signals
    return "de" if german_matches >= 2 else "en"


def parse_query(message: str) -> QueryFilters:
    """
    Parse user query to extract structured filters.

    Args:
        message: User's natural language query

    Returns:
        QueryFilters object with extracted information
    """
    filters = QueryFilters()
    filters.query_text = message
    message_lower = message.lower()

    # Cuisine type detection (English and German)
    cuisine_patterns = {
        "italian": ["italian", "italienisch", "pizza", "pasta"],
        "asian": [
            "asian",
            "asiatisch",
            "chinese",
            "chinesisch",
            "thai",
            "japanese",
            "japanisch",
            "sushi",
        ],
        "austrian": ["austrian", "österreichisch", "schnitzel", "traditional"],
        "indian": ["indian", "indisch", "curry"],
        "mexican": ["mexican", "mexikanisch", "taco", "burrito"],
        "mediterranean": ["mediterranean", "mediterran", "greek", "griechisch"],
        "vegan": ["vegan"],
        "vegetarian": ["vegetarian", "vegetarisch"],
        "burger": ["burger", "burgers"],
        "cafe": ["café", "cafe", "coffee", "kaffee"],
    }

    for cuisine, keywords in cuisine_patterns.items():
        if any(keyword in message_lower for keyword in keywords):
            filters.cuisine_types.append(cuisine)

    # Price range detection
    price_patterns = {
        "€": ["cheap", "günstig", "billig", "budget", "affordable", "inexpensive"],
        "€€": ["moderate", "mittel", "reasonable"],
        "€€€": ["expensive", "teuer", "upscale", "fine dining", "gehobene küche"],
        "€€€€": ["luxury", "luxus", "very expensive", "sehr teuer"],
    }

    for price_range, keywords in price_patterns.items():
        if any(keyword in message_lower for keyword in keywords):
            filters.price_ranges.append(price_range)

    # Feature detection
    feature_patterns = {
        "vegan_options": ["vegan"],
        "vegetarian_options": ["vegetarian", "vegetarisch"],
        "outdoor_seating": [
            "outdoor",
            "terrace",
            "terrasse",
            "garden",
            "garten",
            "outside",
        ],
        "wheelchair_accessible": [
            "wheelchair",
            "accessible",
            "rollstuhl",
            "barrierefrei",
        ],
        "delivery": ["delivery", "lieferung", "takeaway", "abholen"],
        "reservations": ["reservation", "reservierung", "booking", "book"],
        "wifi": ["wifi", "wlan", "internet"],
        "parking": ["parking", "parkplatz"],
    }

    for feature, keywords in feature_patterns.items():
        if any(keyword in message_lower for keyword in keywords):
            filters.features.append(feature)

    # Dish / menu item keyword detection (common dishes in Graz restaurants)
    dish_patterns = [
        "schnitzel", "tafelspitz", "gulasch", "goulash", "käsespätzle",
        "ramen", "sushi", "sashimi", "maki", "nigiri", "tempura",
        "tonkatsu", "gyoza", "udon", "bento",
        "pizza", "pasta", "risotto", "lasagna", "gnocchi", "tiramisu",
        "curry", "tikka", "naan", "biryani", "tandoori", "dal",
        "taco", "burrito", "enchilada", "quesadilla", "guacamole",
        "kebab", "döner", "falafel", "hummus", "shawarma",
        "burger", "steak", "ribs", "wings",
        "pho", "pad thai", "spring roll", "dim sum", "wonton",
        "crêpe", "croissant", "quiche",
        "bowl", "wrap", "sandwich", "salad",
    ]

    for dish in dish_patterns:
        if dish in message_lower:
            filters.dish_keywords.append(dish)

    # Location detection (mainly for filtering Graz-specific queries)
    if any(word in message_lower for word in ["graz", "center", "zentrum", "downtown"]):
        filters.location = "graz"

    return filters
