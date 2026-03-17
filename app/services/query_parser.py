"""Query parser service for extracting user intent and filters from natural language."""

import logging
import re
from enum import StrEnum

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)


class SortPreference(StrEnum):
    """User's explicit sorting preference."""

    RATING = "rating"
    DISTANCE = "distance"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"


class Mood(StrEnum):
    """Occasion or atmosphere the user is looking for."""

    DATE_NIGHT = "date_night"
    BUSINESS = "business"
    CASUAL = "casual"
    FAMILY = "family"
    CELEBRATION = "celebration"


class ParsedQuery(BaseModel):
    """LLM-extracted structured query from user message."""

    cuisine_types: list[str]
    excluded_cuisines: list[str]
    price_ranges: list[str]
    excluded_price_ranges: list[str]
    features: list[str]
    dish_keywords: list[str]
    location_name: str | None
    mood: Mood | None
    group_size: int | None
    time_preference: str | None
    sort_by: SortPreference | None
    language: str


# Parser-specific OpenAI client (separate from response generation client)
_parser_client = AsyncOpenAI(api_key=settings.openai_api_key)

PARSER_SYSTEM_PROMPT = """\
You extract structured restaurant search filters from user messages about \
restaurants in Graz, Austria. Extract ONLY what the user explicitly states \
or clearly implies. When uncertain, leave fields empty/null rather than guessing.

FIELD INSTRUCTIONS:

cuisine_types: Normalized cuisine categories. Map variations to canonical names:
  "pizza/pasta" → "italian", "sushi/ramen/japanese" → "asian",
  "schnitzel/wiener" → "austrian", "kebab/döner" → "turkish",
  "curry/naan/tikka" → "indian", "taco/burrito" → "mexican",
  "gyros/souvlaki" → "mediterranean", "burger" → "burger",
  "café/coffee" → "cafe", "vegan" → "vegan", "vegetarian" → "vegetarian"
  Only include cuisines the user WANTS.

excluded_cuisines: Cuisines the user explicitly does NOT want.
  "no Italian" → ["italian"], "anything but Asian" → ["asian"]

price_ranges: Map to symbols: "€" (cheap/budget), "€€" (moderate/mid-range), \
"€€€" (upscale/expensive), "€€€€" (luxury).

excluded_price_ranges: Prices the user explicitly avoids.
  "nothing too expensive" → ["€€€", "€€€€"], "not cheap" → ["€"]

features: ONLY use values from this list:
  vegan_options, vegetarian_options, outdoor_seating, wheelchair_accessible, \
  delivery, reservations, wifi, parking, serves_breakfast, serves_brunch, \
  serves_lunch, serves_dinner, serves_beer, serves_wine, serves_cocktails, \
  serves_coffee, dogs_allowed, good_for_children, good_for_groups, \
  sports_viewing, live_music, children_menu

dish_keywords: Specific dish names mentioned (schnitzel, curry, burger, etc.)

location_name: A Graz location if mentioned. Normalize to lowercase:
  "near the clock tower" → "uhrturm", "main square" → "hauptplatz", \
  "train station" → "hauptbahnhof", "old town" → "altstadt", \
  "Lendplatz area" → "lendplatz", "university" → "uni graz"

mood: Occasion/atmosphere. Use null if not expressed.
  "romantic dinner" → "date_night", "work lunch" → "business", \
  "just grabbing a bite" → "casual", "birthday party" → "celebration", \
  "with the kids" → "family"

group_size: Number of people if mentioned. "just me" → 1, "for four" → 4. \
  Null if not mentioned.

time_preference: When the user wants to eat, as natural language. \
  "open right now" → "open now", "sunday evening", "late night", \
  "for lunch tomorrow" → "lunch". Null if not mentioned.

sort_by: Explicit sorting preference. Null if not mentioned.
  "best rated" → "rating", "closest" → "distance", \
  "cheapest" → "price_asc", "most expensive" → "price_desc"

language: "de" if the message is in German, "en" otherwise.\
"""


def _empty_parsed_query() -> ParsedQuery:
    """Return a ParsedQuery with all fields empty/null."""
    return ParsedQuery(
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


def _format_history_for_parser(
    conversation_history: list[dict[str, object]],
) -> str:
    """Build a conversation context summary for the parser LLM.

    Args:
        conversation_history: List of message dicts with role, content,
            and optional restaurants.

    Returns:
        Formatted context string, or empty string if no history.
    """
    if not conversation_history:
        return ""

    lines: list[str] = [
        "\nCONVERSATION CONTEXT (use to resolve references like "
        '"cheaper", "more like that", "which one", etc.):'
    ]
    for msg in conversation_history:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))[:200]  # Truncate long messages
        if role == "user":
            lines.append(f"- User: {content}")
        elif role == "assistant":
            restaurants = msg.get("restaurants") or []
            if restaurants and isinstance(restaurants, list):
                names = [
                    str(r.get("name", "?")) if isinstance(r, dict) else "?"
                    for r in restaurants[:5]
                ]
                lines.append(f"- Assistant: {content[:100]}...")
                lines.append(f"  (Showed: {', '.join(names)})")
            else:
                lines.append(f"- Assistant: {content}")

    lines.append(
        "The user's NEW message follows. This is a REFINEMENT of the "
        "previous search. CARRY FORWARD all constraints from the conversation "
        "(cuisine, location, features, etc.) and ONLY change the specific "
        "aspect the user mentions. For example, 'but cheaper' means keep the "
        "same cuisine and features, just lower the price range."
    )
    return "\n".join(lines)


async def _llm_extract(
    message: str,
    conversation_history: list[dict[str, object]] | None = None,
) -> ParsedQuery:
    """Extract structured query filters from a user message using the LLM.

    Args:
        message: User's natural language query
        conversation_history: Previous messages for resolving follow-ups

    Returns:
        ParsedQuery with extracted filters

    Raises:
        Exception: If LLM API call fails (caller should handle fallback)
    """
    try:
        system_content = PARSER_SYSTEM_PROMPT
        if conversation_history:
            system_content += _format_history_for_parser(conversation_history)

        response = await _parser_client.beta.chat.completions.parse(
            model=settings.parser_model,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        "Extract restaurant search filters from this"
                        f' message: "{message}"'
                    ),
                },
            ],
            response_format=ParsedQuery,
            temperature=0.0,
            timeout=5.0,
        )

        # Handle refusal (safety filter)
        if response.choices[0].message.refusal:
            logger.warning(
                "Parser LLM refused request: %s",
                response.choices[0].message.refusal,
            )
            return _empty_parsed_query()

        parsed = response.choices[0].message.parsed
        if parsed is None:
            logger.warning("Parser LLM returned null parsed result")
            return _empty_parsed_query()

        return parsed

    except (OpenAIError, ValidationError, KeyError, IndexError):
        logger.exception("LLM extraction failed")
        raise


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

# Valid feature values (must match DB schema / prompt instructions)
_VALID_FEATURES: set[str] = {
    "vegan_options",
    "vegetarian_options",
    "outdoor_seating",
    "wheelchair_accessible",
    "delivery",
    "reservations",
    "wifi",
    "parking",
    "serves_breakfast",
    "serves_brunch",
    "serves_lunch",
    "serves_dinner",
    "serves_beer",
    "serves_wine",
    "serves_cocktails",
    "serves_coffee",
    "dogs_allowed",
    "good_for_children",
    "good_for_groups",
    "sports_viewing",
    "live_music",
    "children_menu",
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
        self.excluded_cuisines: list[str] = []
        self.excluded_price_ranges: list[str] = []
        self.mood: Mood | None = None
        self.group_size: int | None = None
        self.time_preference: str | None = None
        self.sort_by: SortPreference | None = None
        self.language: str = "en"

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


def _resolve_location(
    location_name: str,
) -> tuple[str | None, float | None, float | None]:
    """Resolve a location name to coordinates using the Graz location database.

    Tries exact match first, then substring match in the dictionary keys.

    Returns:
        (name, lat, lng) or (None, None, None) if not found.
    """
    name_lower = location_name.lower().strip()

    # Exact match
    if name_lower in _GRAZ_LOCATIONS:
        lat, lng = _GRAZ_LOCATIONS[name_lower]
        return name_lower, lat, lng

    # Substring match — longest first to avoid "lend" matching inside longer names
    for key in sorted(_GRAZ_LOCATIONS, key=len, reverse=True):
        if key in name_lower:
            lat, lng = _GRAZ_LOCATIONS[key]
            return key, lat, lng

    return None, None, None


def _keyword_fallback(message: str) -> ParsedQuery:
    """Minimal keyword-based parser used when LLM API is unavailable.

    Extracts basic cuisine, price, and language. Does NOT support
    negation, mood, time, group size, or sorting.
    """
    message_lower = message.lower()

    cuisine_map: dict[str, list[str]] = {
        "italian": ["italian", "italienisch", "pizza", "pasta"],
        "asian": ["asian", "asiatisch", "chinese", "thai", "japanese", "sushi"],
        "austrian": ["austrian", "österreichisch", "schnitzel"],
        "indian": ["indian", "indisch", "curry"],
        "mexican": ["mexican", "mexikanisch", "taco", "burrito"],
        "vegan": ["vegan"],
        "vegetarian": ["vegetarian", "vegetarisch"],
        "burger": ["burger"],
        "cafe": ["café", "cafe", "coffee", "kaffee"],
    }
    cuisine_types = [
        cuisine
        for cuisine, keywords in cuisine_map.items()
        if any(kw in message_lower for kw in keywords)
    ]

    price_map: dict[str, list[str]] = {
        "€": ["cheap", "günstig", "billig", "budget"],
        "€€": ["moderate", "mittel"],
        "€€€": ["expensive", "teuer", "fine dining"],
    }
    price_ranges = [
        price
        for price, keywords in price_map.items()
        if any(kw in message_lower for kw in keywords)
    ]

    german_words = ["ich", "suche", "ein", "der", "die", "das", "und", "für", "mit"]
    german_count = sum(1 for w in german_words if f" {w} " in f" {message_lower} ")
    language = "de" if german_count >= 2 else "en"

    return ParsedQuery(
        cuisine_types=cuisine_types,
        excluded_cuisines=[],
        price_ranges=price_ranges,
        excluded_price_ranges=[],
        features=[],
        dish_keywords=[],
        location_name=None,
        mood=None,
        group_size=None,
        time_preference=None,
        sort_by=None,
        language=language,
    )


async def parse_query(
    message: str,
    conversation_history: list[dict[str, object]] | None = None,
) -> QueryFilters:
    """Parse user query to extract structured filters.

    Uses LLM structured extraction (gpt-4o-mini) with keyword fallback
    on API failure.

    Args:
        message: User's natural language query
        conversation_history: Previous messages for resolving follow-ups

    Returns:
        QueryFilters with extracted and enriched filters
    """
    try:
        parsed = await _llm_extract(message, conversation_history)
    except Exception as e:
        # Broad catch is intentional: _llm_extract may fail for any reason
        # (API errors, validation, unexpected bugs). We always fall back to
        # keyword parsing rather than crashing the chat endpoint.
        logger.warning(
            "LLM extraction failed (%s), using keyword fallback", type(e).__name__
        )
        parsed = _keyword_fallback(message)

    filters = QueryFilters()
    filters.query_text = message
    filters.cuisine_types = parsed.cuisine_types
    filters.excluded_cuisines = parsed.excluded_cuisines
    filters.price_ranges = parsed.price_ranges
    filters.excluded_price_ranges = parsed.excluded_price_ranges
    # Filter out hallucinated feature names from LLM extraction
    valid_features = [f for f in parsed.features if f in _VALID_FEATURES]
    invalid_features = [f for f in parsed.features if f not in _VALID_FEATURES]
    if invalid_features:
        logger.warning("Dropped invalid features from LLM: %s", invalid_features)
    filters.features = valid_features
    filters.dish_keywords = parsed.dish_keywords
    filters.mood = parsed.mood
    filters.group_size = parsed.group_size
    filters.time_preference = parsed.time_preference
    filters.sort_by = parsed.sort_by
    filters.language = parsed.language

    # Resolve location name to coordinates
    if parsed.location_name:
        loc_name, loc_lat, loc_lng = _resolve_location(parsed.location_name)
        if loc_name is not None:
            filters.location_name = loc_name
            filters.location_lat = loc_lat
            filters.location_lng = loc_lng

    # Group size >= 5 adds good_for_groups to features
    if parsed.group_size is not None and parsed.group_size >= 5:
        if "good_for_groups" not in filters.features:
            filters.features.append("good_for_groups")

    return filters
