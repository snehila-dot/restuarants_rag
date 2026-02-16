"""Query parser service for extracting user intent and filters from natural language."""

import re

# ---------------------------------------------------------------------------
# Graz location database — (latitude, longitude) for known places
# Coordinates sourced from OpenStreetMap.
# ---------------------------------------------------------------------------

_GRAZ_LOCATIONS: dict[str, tuple[float, float]] = {
    # Squares & landmarks
    "jakominiplatz": (47.0668, 15.4424),
    "hauptplatz": (47.0707, 15.4382),
    "lendplatz": (47.0740, 15.4310),
    "griesplatz": (47.0654, 15.4380),
    "mariahilferplatz": (47.0715, 15.4303),
    "karmeliterplatz": (47.0743, 15.4380),
    "tummelplatz": (47.0727, 15.4402),
    "franziskanerplatz": (47.0710, 15.4375),
    "freiheitsplatz": (47.0726, 15.4373),
    "südtirolerplatz": (47.0648, 15.4330),
    "suedtirolerplatz": (47.0648, 15.4330),
    "andreas-hofer-platz": (47.0697, 15.4355),
    "europaplatz": (47.0707, 15.4172),
    "hasnerplatz": (47.0790, 15.4430),
    "dietrichsteinplatz": (47.0780, 15.4460),
    "mehlplatz": (47.0712, 15.4397),
    "glockenspielplatz": (47.0712, 15.4397),
    "kaiser-josef-platz": (47.0725, 15.4435),
    "schillerplatz": (47.0682, 15.4595),
    # Landmarks
    "schlossberg": (47.0757, 15.4370),
    "schloßberg": (47.0757, 15.4370),
    "uhrturm": (47.0757, 15.4370),
    "stadtpark": (47.0735, 15.4453),
    "oper": (47.0711, 15.4430),
    "opernhaus": (47.0711, 15.4430),
    "kunsthaus": (47.0710, 15.4335),
    "murinsel": (47.0710, 15.4337),
    "hauptbahnhof": (47.0707, 15.4172),
    "bahnhof": (47.0707, 15.4172),
    "uni graz": (47.0773, 15.4497),
    "universität": (47.0773, 15.4497),
    "tu graz": (47.0696, 15.4505),
    "messe graz": (47.0584, 15.4571),
    "stadion": (47.0470, 15.4555),
    "merkur arena": (47.0470, 15.4555),
    # Streets
    "herrengasse": (47.0717, 15.4377),
    "sporgasse": (47.0722, 15.4385),
    "schmiedgasse": (47.0700, 15.4398),
    "annenstraße": (47.0700, 15.4330),
    "annenstrasse": (47.0700, 15.4330),
    "münzgrabenstraße": (47.0650, 15.4510),
    "keplerstraße": (47.0690, 15.4332),
    "keplerstrasse": (47.0690, 15.4332),
    "zinzendorfgasse": (47.0757, 15.4483),
    "glacisstraße": (47.0740, 15.4470),
    "glacisstrasse": (47.0740, 15.4470),
    "elisabethstraße": (47.0735, 15.4500),
    "grieskai": (47.0680, 15.4345),
    "lendkai": (47.0720, 15.4320),
    "kaiser-franz-josef-kai": (47.0695, 15.4360),
    "neutorgasse": (47.0669, 15.4378),
    # Districts
    "innere stadt": (47.0710, 15.4380),
    "innenstadt": (47.0710, 15.4380),
    "zentrum": (47.0710, 15.4380),
    "center": (47.0710, 15.4380),
    "altstadt": (47.0710, 15.4380),
    "geidorf": (47.0822, 15.4433),
    "st. leonhard": (47.0730, 15.4520),
    "st leonhard": (47.0730, 15.4520),
    "jakomini": (47.0660, 15.4460),
    "lend": (47.0740, 15.4290),
    "gries": (47.0620, 15.4340),
    "st. peter": (47.0556, 15.4716),
    "st peter": (47.0556, 15.4716),
    "waltendorf": (47.0758, 15.4630),
    "mariatrost": (47.1023, 15.4813),
    "andritz": (47.1053, 15.4172),
    "gösting": (47.0962, 15.4064),
    "goesting": (47.0962, 15.4064),
    "eggenberg": (47.0691, 15.4052),
    "wetzelsdorf": (47.0596, 15.4068),
    "straßgang": (47.0381, 15.4155),
    "strassgang": (47.0381, 15.4155),
    "puntigam": (47.0345, 15.4360),
    "liebenau": (47.0434, 15.4589),
    "ries": (47.0950, 15.4730),
    "sankt peter": (47.0556, 15.4716),
}

# Default search radius in metres
_DEFAULT_RADIUS_M = 800


class QueryFilters:
    """Structured filters extracted from user query."""

    def __init__(self) -> None:
        self.cuisine_types: list[str] = []
        self.price_ranges: list[str] = []
        self.features: list[str] = []
        self.dish_keywords: list[str] = []
        self.location_name: str | None = None
        self.location_lat: float | None = None
        self.location_lng: float | None = None
        self.location_radius_m: float = _DEFAULT_RADIUS_M
        self.query_text: str = ""

    @property
    def has_location(self) -> bool:
        return self.location_lat is not None and self.location_lng is not None


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


def _detect_location(
    message_lower: str,
) -> tuple[str | None, float | None, float | None]:
    """Match the longest known Graz location in the message.

    Returns ``(name, lat, lng)`` or ``(None, None, None)``.
    Longest-match-first avoids "lend" matching inside "glockenspielplatz".
    """
    # Sort by key length descending so longer names match first
    for name in sorted(_GRAZ_LOCATIONS, key=len, reverse=True):
        if name in message_lower:
            lat, lng = _GRAZ_LOCATIONS[name]
            return name, lat, lng
    return None, None, None


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
        "schnitzel",
        "tafelspitz",
        "gulasch",
        "goulash",
        "käsespätzle",
        "ramen",
        "sushi",
        "sashimi",
        "maki",
        "nigiri",
        "tempura",
        "tonkatsu",
        "gyoza",
        "udon",
        "bento",
        "pizza",
        "pasta",
        "risotto",
        "lasagna",
        "gnocchi",
        "tiramisu",
        "curry",
        "tikka",
        "naan",
        "biryani",
        "tandoori",
        "dal",
        "taco",
        "burrito",
        "enchilada",
        "quesadilla",
        "guacamole",
        "kebab",
        "döner",
        "falafel",
        "hummus",
        "shawarma",
        "burger",
        "steak",
        "ribs",
        "wings",
        "pho",
        "pad thai",
        "spring roll",
        "dim sum",
        "wonton",
        "crêpe",
        "croissant",
        "quiche",
        "bowl",
        "wrap",
        "sandwich",
        "salad",
    ]

    for dish in dish_patterns:
        if dish in message_lower:
            filters.dish_keywords.append(dish)

    # Location detection — resolve to coordinates
    loc_name, loc_lat, loc_lng = _detect_location(message_lower)
    if loc_name is not None:
        filters.location_name = loc_name
        filters.location_lat = loc_lat
        filters.location_lng = loc_lng

    return filters
