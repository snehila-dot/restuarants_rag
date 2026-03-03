# LLM-Based Query Parser Design

**Date:** 2026-03-03
**Status:** Approved
**Approach:** OpenAI Structured Outputs (gpt-4o-mini)

## Problem

The current `query_parser.py` uses keyword matching to extract structured filters from user messages. This approach is brittle: it misses natural phrasing ("somewhere cozy for date night"), can't handle negation ("not Italian"), and doesn't support contextual filters like mood, time awareness, or group size.

## Solution

Replace keyword matching with a single OpenAI `gpt-4o-mini` API call using structured outputs (`response_format` with JSON schema). The LLM extracts an expanded `ParsedQuery` Pydantic model directly from natural language.

**Why gpt-4o-mini:** Fast (~100-200ms), cheap (~$0.15/1M input tokens), reliable for simple schemas. Cost at 10K queries/month: ~$0.60.

**Why native structured outputs (not instructor):** OpenAI guarantees 100% schema compliance. No extra dependency needed. The schema is simple enough that instructor's auto-retry adds no value.

## Schema: `ParsedQuery`

```python
from enum import Enum
from pydantic import BaseModel

class SortPreference(str, Enum):
    RATING = "rating"
    DISTANCE = "distance"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"

class Mood(str, Enum):
    DATE_NIGHT = "date_night"
    BUSINESS = "business"
    CASUAL = "casual"
    FAMILY = "family"
    CELEBRATION = "celebration"

class ParsedQuery(BaseModel):
    cuisine_types: list[str]           # ["italian", "asian"]
    excluded_cuisines: list[str]       # ["japanese"] from "not sushi"
    price_ranges: list[str]            # ["euro", "euro_euro"]
    excluded_price_ranges: list[str]   # ["euro_euro_euro_euro"] from "nothing fancy"
    features: list[str]                # ["outdoor_seating", "vegan_options"]
    dish_keywords: list[str]           # ["schnitzel", "ramen"]
    location_name: str | None          # "hauptplatz", "near uhrturm"
    mood: Mood | None                  # Occasion/atmosphere
    group_size: int | None             # Number of people
    time_preference: str | None        # "open now", "sunday evening"
    sort_by: SortPreference | None     # Explicit sorting preference
    language: str                      # "en" or "de"
```

### Design Decisions

- **Negation as separate fields** (`excluded_cuisines`, `excluded_price_ranges`) rather than complex include/exclude structures.
- **Mood as enum** constrained to 5 values the retrieval layer can map to feature/price combinations.
- **`location_name` stays as string** — resolved to coordinates post-extraction via the existing `_GRAZ_LOCATIONS` dictionary.
- **`time_preference` stays as string** — retrieval layer parses it against `opening_hours` JSON.
- **No Pydantic `default` values** — OpenAI structured outputs don't support them.

## System Prompt (Extraction)

The system prompt instructs gpt-4o-mini to:

1. Extract ONLY what the user explicitly states or clearly implies
2. Map cuisine variations to canonical names (e.g., "sushi" -> "asian")
3. Use a closed vocabulary for features (only values that exist in the DB)
4. Normalize Graz location names to lowercase dictionary keys
5. Detect language from the message
6. Leave fields empty/null when uncertain (never guess)

Key extraction rules:
- Negation: "no Italian" -> `excluded_cuisines: ["italian"]`
- Mood: "romantic dinner" -> `mood: "date_night"`
- Location: "near the clock tower" -> `location_name: "uhrturm"`
- Group: "dinner for 8" -> `group_size: 8`
- Sort: "best rated" -> `sort_by: "rating"`

## Processing Flow

```
user_message
  -> LLM structured extraction (gpt-4o-mini, ~100-200ms)
  -> ParsedQuery (validated Pydantic model)
  -> resolve location_name to coordinates via _GRAZ_LOCATIONS
  -> resolve mood to feature/price boosts
  -> return enriched QueryFilters
  -> retrieval service applies filters to DB
  -> response LLM generates answer (gpt-4, existing)
```

## Integration Changes

### Files Modified

| File | Change |
|------|--------|
| `app/config.py` | Add `parser_model` setting (default: "gpt-4o-mini") |
| `app/services/query_parser.py` | Full rewrite: LLM extraction replaces keywords. Becomes async. |
| `app/services/retrieval.py` | Extend: negation filters, mood boosts, time filtering, sort override |
| `app/api/chat.py` | Minor: `parse_query()` becomes `await parse_query()`, language from ParsedQuery |

### Files NOT Modified

- `app/services/llm.py` — response generation untouched
- `app/models/restaurant.py` — no schema changes
- `app/templates/` — no UI changes
- `app/seed.py` — no data changes

### What Stays from Current Parser

- `_GRAZ_LOCATIONS` dictionary — used for coordinate resolution post-extraction
- `QueryFilters` class structure — expanded with new fields

### What's Deleted

- All keyword pattern dictionaries (cuisine, price, feature, dish)
- `detect_language()` function
- `parse_query()` regex logic

## Retrieval Changes

### New Filter Logic

| Filter | Implementation |
|--------|---------------|
| Negation | Post-filter: remove restaurants matching `excluded_cuisines` / `excluded_price_ranges` |
| Mood | Soft boost in ranking (not hard filter). Map mood -> preferred features + min price |
| Group size | `>= 5` -> require `good_for_groups` feature |
| Time preference | Parse against `opening_hours` JSON. Missing hours = don't exclude (benefit of doubt) |
| Sort preference | Override default sort order |

### Mood -> Feature Mapping (Soft Boosts)

```python
MOOD_FEATURES = {
    "date_night":   {"prefer": ["serves_wine", "outdoor_seating", "reservations"], "min_price": "euro_euro"},
    "business":     {"prefer": ["reservations", "wifi"], "min_price": "euro_euro"},
    "family":       {"prefer": ["good_for_children", "children_menu"], "min_price": None},
    "casual":       {"prefer": [], "min_price": None},
    "celebration":  {"prefer": ["good_for_groups", "reservations", "serves_cocktails"], "min_price": "euro_euro"},
}
```

These are **ranking boosts**, not hard filters. A "date night" query won't exclude restaurants that lack outdoor seating — it just ranks those with it higher.

## Error Handling

| Scenario | Handling |
|----------|---------|
| LLM API timeout/failure | Fall back to minimal keyword parser (`_keyword_fallback()`) |
| LLM returns refusal | Treat as no filters -> return top-rated defaults |
| Empty extraction | Valid: "surprise me" -> top-rated restaurants |
| Unknown location name | Ignore location filter, proceed without geo-filtering |
| Missing opening hours | Don't exclude from time-filtered results |

A stripped-down keyword fallback (~30 lines) is kept for API outages only.

## Testing Strategy

Tests mock the OpenAI call and return pre-defined ParsedQuery JSON. We test our code (schema validation, location resolution, retrieval logic), not LLM extraction quality.

### Unit Tests

| Test | Validates |
|------|-----------|
| `test_parse_simple_cuisine` | "Italian restaurant" -> `cuisine_types=["italian"]` |
| `test_parse_negation` | "not sushi" -> `excluded_cuisines=["asian"]` |
| `test_parse_mood` | "romantic dinner" -> `mood="date_night"` |
| `test_parse_location` | "near Hauptplatz" -> `location_name="hauptplatz"` |
| `test_parse_group` | "dinner for 8" -> `group_size=8` |
| `test_parse_time` | "open on Sunday" -> `time_preference="sunday"` |
| `test_parse_sort` | "best rated" -> `sort_by="rating"` |
| `test_parse_complex` | Combined filters from complex query |
| `test_parse_empty` | "surprise me" -> empty filters, no crash |
| `test_parse_german` | German input -> correct extraction + `language="de"` |
| `test_fallback_on_api_error` | API failure -> keyword fallback works |
| `test_retrieval_negation` | Excluded cuisines filtered out |
| `test_retrieval_mood_boost` | Date night ranks wine restaurants higher |
| `test_retrieval_time_filter` | "open sunday" excludes closed restaurants |
