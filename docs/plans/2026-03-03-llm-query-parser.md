# LLM Query Parser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the keyword-based query parser with OpenAI structured outputs (gpt-4o-mini) for dramatically better natural language understanding, adding negation, mood, time, group size, and sort filters.

**Architecture:** User message → gpt-4o-mini structured extraction → ParsedQuery Pydantic model → location resolution + mood mapping → enriched QueryFilters → retrieval service. Fallback to minimal keyword parser on API failure.

**Tech Stack:** OpenAI Python SDK (`client.beta.chat.completions.parse`), Pydantic v2 (`BaseModel`), existing FastAPI + SQLAlchemy stack. No new dependencies.

**Design Doc:** `docs/plans/2026-03-03-llm-query-parser-design.md`

---

### Task 1: Config + Schema Foundation

**Files:**
- Modify: `app/config.py` (add `parser_model` field)
- Modify: `app/services/query_parser.py` (add enums, `ParsedQuery` model, expand `QueryFilters`)
- Test: `tests/test_query_parser.py` (add schema tests)

**Step 1: Write failing tests for new schema**

Add to `tests/test_query_parser.py`:

```python
"""Tests for ParsedQuery schema and expanded QueryFilters."""

import pytest
from app.services.query_parser import Mood, SortPreference, ParsedQuery, QueryFilters


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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_query_parser.py -v -x`
Expected: FAIL with `ImportError` — `Mood`, `SortPreference`, `ParsedQuery` don't exist yet.

**Step 3: Add parser_model to config**

In `app/config.py`, add after `llm_temperature`:

```python
parser_model: str = Field(
    default="gpt-4o-mini",
    description="LLM model for query parsing (cheaper/faster than response model)",
)
```

**Step 4: Add enums, ParsedQuery, and expand QueryFilters**

In `app/services/query_parser.py`, add after the imports (before `_GRAZ_LOCATIONS`):

```python
from enum import Enum
from pydantic import BaseModel


class SortPreference(str, Enum):
    """User's explicit sorting preference."""

    RATING = "rating"
    DISTANCE = "distance"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"


class Mood(str, Enum):
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
```

Expand the existing `QueryFilters.__init__` to add:

```python
self.excluded_cuisines: list[str] = []
self.excluded_price_ranges: list[str] = []
self.mood: Mood | None = None
self.group_size: int | None = None
self.time_preference: str | None = None
self.sort_by: SortPreference | None = None
self.language: str = "en"
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_query_parser.py -v`
Expected: All new schema tests PASS. Existing tests still pass (no behavior change yet).

**Step 6: Commit**

```bash
git add app/config.py app/services/query_parser.py tests/test_query_parser.py
git commit -m "feat: add ParsedQuery schema, enums, and expanded QueryFilters for LLM parser"
```

---

### Task 2: LLM Extraction Function

**Files:**
- Modify: `app/services/query_parser.py` (add `_llm_extract`, system prompt)
- Test: `tests/test_query_parser.py` (add extraction tests with mocked OpenAI)

**Step 1: Write failing tests for LLM extraction**

Add to `tests/test_query_parser.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.query_parser import _llm_extract, ParsedQuery


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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_query_parser.py::test_llm_extract_simple_cuisine -v`
Expected: FAIL with `ImportError` — `_llm_extract` doesn't exist yet.

**Step 3: Implement LLM extraction**

In `app/services/query_parser.py`, add after the `ParsedQuery` class:

```python
import logging
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

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


async def _llm_extract(message: str) -> ParsedQuery:
    """
    Extract structured query filters from a user message using the LLM.

    Args:
        message: User's natural language query

    Returns:
        ParsedQuery with extracted filters
    """
    try:
        response = await _parser_client.beta.chat.completions.parse(
            model=settings.parser_model,
            messages=[
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f'Extract restaurant search filters from this message: "{message}"',
                },
            ],
            response_format=ParsedQuery,
            temperature=0.0,
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

    except Exception:
        logger.exception("LLM extraction failed")
        raise
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_query_parser.py -v -k "llm_extract"`
Expected: Both `test_llm_extract_simple_cuisine` and `test_llm_extract_refusal_returns_empty` PASS.

**Step 5: Commit**

```bash
git add app/services/query_parser.py tests/test_query_parser.py
git commit -m "feat: add LLM extraction function with OpenAI structured outputs"
```

---

### Task 3: Location Resolution + Keyword Fallback + Async parse_query

**Files:**
- Modify: `app/services/query_parser.py` (refactor `parse_query` to async, add `_keyword_fallback`)
- Test: `tests/test_query_parser.py` (update existing tests, add fallback + location tests)

**Step 1: Write failing tests**

Add to `tests/test_query_parser.py`:

```python
from app.services.query_parser import (
    _keyword_fallback,
    _resolve_location,
    parse_query,
)


def test_resolve_location_known() -> None:
    """Known Graz location resolves to coordinates."""
    name, lat, lng = _resolve_location("hauptplatz")
    assert name == "hauptplatz"
    assert lat is not None
    assert lng is not None


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

    with patch("app.services.query_parser._llm_extract", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = mock_parsed
        filters = await parse_query("romantic Italian dinner near Hauptplatz")

    assert filters.cuisine_types == ["italian"]
    assert filters.location_name == "hauptplatz"
    assert filters.location_lat is not None  # Resolved from _GRAZ_LOCATIONS
    assert filters.mood == Mood.DATE_NIGHT


@pytest.mark.asyncio
async def test_parse_query_falls_back_on_llm_error() -> None:
    """parse_query falls back to keywords when LLM fails."""
    with patch("app.services.query_parser._llm_extract", new_callable=AsyncMock) as mock_extract:
        mock_extract.side_effect = Exception("API timeout")
        filters = await parse_query("cheap Italian food")

    assert "italian" in filters.cuisine_types
    assert "€" in filters.price_ranges
```

**Step 2: Run to verify failures**

Run: `pytest tests/test_query_parser.py -v -k "resolve_location or keyword_fallback or parse_query_uses or parse_query_falls" -x`
Expected: FAIL — `_resolve_location`, `_keyword_fallback` don't exist, `parse_query` is still sync.

**Step 3: Implement location resolution**

In `app/services/query_parser.py`, rename existing `_detect_location` to `_resolve_location` and keep its logic:

```python
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

    # Substring match (e.g. "near hauptplatz" → "hauptplatz")
    for key in sorted(_GRAZ_LOCATIONS, key=len, reverse=True):
        if key in name_lower:
            lat, lng = _GRAZ_LOCATIONS[key]
            return key, lat, lng

    return None, None, None
```

**Step 4: Implement keyword fallback**

In `app/services/query_parser.py`, add `_keyword_fallback` — a stripped-down version of the current `parse_query` that returns a `ParsedQuery`:

```python
def _keyword_fallback(message: str) -> ParsedQuery:
    """Minimal keyword-based parser used when LLM API is unavailable.

    Extracts basic cuisine, price, and language. Does NOT support
    negation, mood, time, group size, or sorting.
    """
    message_lower = message.lower()

    # Cuisine detection (simplified)
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

    # Price detection (simplified)
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

    # Language detection (simplified)
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
```

**Step 5: Rewrite parse_query as async**

Replace the existing `parse_query` function entirely:

```python
async def parse_query(message: str) -> QueryFilters:
    """
    Parse user query to extract structured filters.

    Uses LLM structured extraction (gpt-4o-mini) with keyword fallback
    on API failure.

    Args:
        message: User's natural language query

    Returns:
        QueryFilters with extracted and enriched filters
    """
    # Extract via LLM, fall back to keywords
    try:
        parsed = await _llm_extract(message)
    except Exception:
        logger.warning("LLM extraction failed, using keyword fallback")
        parsed = _keyword_fallback(message)

    # Build QueryFilters from parsed result
    filters = QueryFilters()
    filters.query_text = message
    filters.cuisine_types = parsed.cuisine_types
    filters.excluded_cuisines = parsed.excluded_cuisines
    filters.price_ranges = parsed.price_ranges
    filters.excluded_price_ranges = parsed.excluded_price_ranges
    filters.features = list(parsed.features)  # Copy to avoid mutation
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
```

**Step 6: Delete old keyword code**

Remove the old `detect_language`, `_detect_location`, and synchronous `parse_query` functions. Remove the old pattern dictionaries (`cuisine_patterns`, `price_patterns`, `feature_patterns`, `dish_patterns`). Keep `_GRAZ_LOCATIONS` and `_DEFAULT_RADIUS_M`.

**Step 7: Fix existing tests**

The existing tests in `test_query_parser.py` call `parse_query` synchronously and import `detect_language`. Update them:

- Remove `test_detect_language_english` and `test_detect_language_german` (language detection is now part of LLM extraction; tested via `test_keyword_fallback_language_de`)
- Update `test_parse_query_cuisine`, `test_parse_query_price`, `test_parse_query_features` to be async and mock the LLM, OR replace them with the new `test_parse_query_uses_llm` and `test_parse_query_falls_back_on_llm_error` tests

Replace the old tests with:

```python
# Remove these old imports:
# from app.services.query_parser import detect_language, parse_query

# Replace with:
# (already imported above in the new test additions)
```

**Step 8: Run full test suite**

Run: `pytest tests/test_query_parser.py -v`
Expected: All tests PASS.

**Step 9: Commit**

```bash
git add app/services/query_parser.py tests/test_query_parser.py
git commit -m "feat: rewrite parse_query as async LLM extraction with keyword fallback"
```

---

### Task 4: Retrieval Extensions (Negation, Mood, Time, Sort)

**Files:**
- Modify: `app/services/retrieval.py` (add negation, mood boost, time filter, sort override)
- Modify: `tests/conftest.py` (add richer sample restaurants for new filter tests)
- Test: `tests/test_retrieval.py` (add new filter tests)

**Step 1: Expand sample restaurants in conftest**

Add to `tests/conftest.py` in the `sample_restaurants` fixture (after the existing two restaurants):

```python
Restaurant(
    id=uuid.uuid4(),
    name="Test Austrian Traditional",
    address="Herrengasse 10, Graz",
    cuisine=["Austrian"],
    price_range="€€€",
    rating=4.3,
    review_count=200,
    features=["reservations", "serves_wine", "outdoor_seating"],
    summary="Traditional Austrian fine dining",
    opening_hours={"monday": "11:00-22:00", "tuesday": "11:00-22:00",
                   "wednesday": "11:00-22:00", "thursday": "11:00-22:00",
                   "friday": "11:00-23:00", "saturday": "17:00-23:00",
                   "sunday": "closed"},
    latitude=47.0717,
    longitude=15.4377,
    data_sources=["test"],
    last_verified=datetime.utcnow(),
),
Restaurant(
    id=uuid.uuid4(),
    name="Test Family Pizza",
    address="Lendplatz 5, Graz",
    cuisine=["Italian", "Pizza"],
    price_range="€",
    rating=4.0,
    review_count=80,
    features=["good_for_children", "children_menu", "good_for_groups", "delivery"],
    summary="Family-friendly pizzeria",
    opening_hours={"monday": "10:00-22:00", "tuesday": "10:00-22:00",
                   "wednesday": "10:00-22:00", "thursday": "10:00-22:00",
                   "friday": "10:00-23:00", "saturday": "10:00-23:00",
                   "sunday": "12:00-21:00"},
    latitude=47.0740,
    longitude=15.4310,
    data_sources=["test"],
    last_verified=datetime.utcnow(),
),
```

**Step 2: Write failing retrieval tests**

Add to `tests/test_retrieval.py`:

```python
from app.services.query_parser import Mood, SortPreference


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
    # Austrian Traditional has serves_wine + outdoor_seating — should rank high
    assert len(results) > 0
    top = results[0]
    assert any(
        f in top.features
        for f in ("serves_wine", "outdoor_seating", "reservations")
    )


async def test_search_time_filter_excludes_closed(
    session: AsyncSession,
    sample_restaurants: list[Restaurant],
) -> None:
    """Time filter excludes restaurants closed on that day."""
    filters = QueryFilters()
    filters.time_preference = "sunday"
    results = await search_restaurants(session, filters)
    # Austrian Traditional is closed sunday — should be excluded
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
```

**Step 3: Run to verify failures**

Run: `pytest tests/test_retrieval.py -v -k "excludes or mood or time_filter or sort_by" -x`
Expected: FAIL — retrieval doesn't handle these filters yet.

**Step 4: Implement retrieval extensions**

In `app/services/retrieval.py`, add imports and mood mapping:

```python
from app.services.query_parser import Mood, SortPreference

# Mood → preferred features mapping (soft ranking boosts)
_MOOD_FEATURES: dict[Mood, list[str]] = {
    Mood.DATE_NIGHT: ["serves_wine", "outdoor_seating", "reservations"],
    Mood.BUSINESS: ["reservations", "wifi"],
    Mood.FAMILY: ["good_for_children", "children_menu"],
    Mood.CASUAL: [],
    Mood.CELEBRATION: ["good_for_groups", "reservations", "serves_cocktails"],
}

# Price range ordering for sort
_PRICE_ORDER: dict[str, int] = {"€": 1, "€€": 2, "€€€": 3, "€€€€": 4}
```

In `search_restaurants`, add after the existing post-processing filters (after the features filter block, before the geo-distance section):

```python
# Negation: exclude restaurants matching excluded cuisines
if filters.excluded_cuisines:
    restaurants = [
        r for r in restaurants
        if not any(
            exc.lower() in [c.lower() for c in r.cuisine]
            for exc in filters.excluded_cuisines
        )
    ]

# Negation: exclude restaurants matching excluded price ranges
if filters.excluded_price_ranges:
    restaurants = [
        r for r in restaurants
        if r.price_range not in filters.excluded_price_ranges
    ]

# Time preference: exclude restaurants known to be closed
if filters.time_preference:
    time_lower = filters.time_preference.lower()
    day_names = [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ]
    target_days = [d for d in day_names if d in time_lower]
    if target_days:
        restaurants = [
            r for r in restaurants
            if not r.opening_hours  # Keep restaurants without hours data
            or not any(
                r.opening_hours.get(day, "").lower() == "closed"
                for day in target_days
            )
        ]
```

Replace the sorting section (the `else` branch and the geo-distance sort) with:

```python
# Sorting
if filters.has_location:
    # ... (keep existing geo-distance code unchanged)
    pass
elif filters.sort_by == SortPreference.PRICE_ASC:
    restaurants.sort(key=lambda r: _PRICE_ORDER.get(r.price_range, 99))
elif filters.sort_by == SortPreference.PRICE_DESC:
    restaurants.sort(key=lambda r: _PRICE_ORDER.get(r.price_range, 0), reverse=True)
elif filters.sort_by == SortPreference.RATING:
    restaurants.sort(key=lambda r: (r.rating or 0.0, r.review_count), reverse=True)
else:
    # Default: rating-based, with mood boost if applicable
    if filters.mood and filters.mood in _MOOD_FEATURES:
        preferred = _MOOD_FEATURES[filters.mood]

        def _mood_score(r: Restaurant) -> tuple[int, float, int]:
            feature_hits = sum(1 for f in preferred if f in r.features)
            return (feature_hits, r.rating or 0.0, r.review_count)

        restaurants.sort(key=_mood_score, reverse=True)
    else:
        restaurants.sort(
            key=lambda r: (r.rating or 0.0, r.review_count), reverse=True
        )
```

Note: Keep the existing geo-distance block intact — it already handles `filters.has_location`. The new sorting logic goes in the `else` branch.

**Step 5: Run tests**

Run: `pytest tests/test_retrieval.py -v`
Expected: All tests PASS (old and new).

**Step 6: Commit**

```bash
git add app/services/retrieval.py tests/test_retrieval.py tests/conftest.py
git commit -m "feat: add negation, mood boost, time filter, and sort to retrieval service"
```

---

### Task 5: Chat Endpoint Integration

**Files:**
- Modify: `app/api/chat.py` (make parse_query call async, use language from filters)
- Test: `tests/test_chat.py` (update mocks)

**Step 1: Update chat.py**

In `app/api/chat.py`, change:

```python
# OLD:
language = request.language or query_parser.detect_language(request.message)
filters = query_parser.parse_query(request.message)

# NEW:
filters = await query_parser.parse_query(request.message)
language = request.language or filters.language
```

**Step 2: Update chat tests**

In `tests/test_chat.py`, add a mock for the parser LLM call alongside the existing response LLM mock:

```python
from app.services.query_parser import ParsedQuery


def _mock_parsed_query(**overrides) -> ParsedQuery:
    """Helper to create a mock ParsedQuery with defaults."""
    defaults = dict(
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
    defaults.update(overrides)
    return ParsedQuery(**defaults)
```

Update each test to mock both the parser and the response LLM. For example, `test_chat_endpoint_success`:

```python
async def test_chat_endpoint_success(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Test successful chat endpoint response."""
    with (
        patch("app.services.query_parser._llm_extract", new_callable=AsyncMock) as mock_parser,
        patch("app.services.llm.client.chat.completions.create") as mock_llm,
    ):
        mock_parser.return_value = _mock_parsed_query(cuisine_types=["italian"])

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="I found some Italian restaurants."))
        ]
        mock_llm.return_value = mock_response

        response = await client.post("/api/chat", json={"message": "Italian restaurant"})

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "restaurants" in data
        assert isinstance(data["restaurants"], list)
```

Update `test_chat_endpoint_language_detection` similarly — mock the parser to return `language="de"`:

```python
mock_parser.return_value = _mock_parsed_query(language="de")
```

**Step 3: Run chat tests**

Run: `pytest tests/test_chat.py -v`
Expected: All PASS.

**Step 4: Commit**

```bash
git add app/api/chat.py tests/test_chat.py
git commit -m "feat: wire async LLM parser into chat endpoint"
```

---

### Task 6: Final Validation

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS.

**Step 2: Lint**

Run: `ruff check . && ruff format --check .`
Expected: Clean (no errors). If there are warnings, fix them.

**Step 3: Type check**

Run: `mypy app/`
Expected: Clean. Note: `client.beta.chat.completions.parse` may need a `# type: ignore[attr-defined]` if mypy doesn't recognize the beta namespace. This is acceptable — it's an OpenAI SDK limitation, not a type error.

**Step 4: Verify existing behavior preserved**

Run: `pytest tests/test_retrieval.py::test_search_restaurants_by_cuisine tests/test_retrieval.py::test_search_restaurants_by_price tests/test_retrieval.py::test_search_restaurants_by_features tests/test_retrieval.py::test_search_restaurants_no_results -v`
Expected: All original retrieval tests still PASS.

**Step 5: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "chore: lint and type fixes for LLM query parser"
```
