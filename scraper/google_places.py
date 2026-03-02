"""Google Places API (New) enrichment module.

Enriches existing restaurant data with structured fields from Google's
Places API, replacing the brittle Playwright-based Google Maps scraper.

Usage::

    python -m scraper --google-places

Requires ``GOOGLE_PLACES_API_KEY`` environment variable.

Fields enriched (only fills gaps — never overwrites existing data):
- rating, review_count, price_range
- website, phone, opening_hours
- features (outdoor_seating, vegetarian_options, delivery, takeout, etc.)
- summary (editorial summary from Google)
- google_place_id (for future re-syncs)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_API_BASE = "https://places.googleapis.com/v1/places:searchText"

# Request all restaurant-relevant fields across pricing tiers.
# Essentials + Enterprise + Atmosphere in one call — ~$25/1k requests,
# well within the $200/month free credit for ~500 Graz restaurants.
_FIELD_MASK = ",".join(
    [
        # Essentials
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.types",
        "places.primaryType",
        # Pro
        "places.businessStatus",
        "places.googleMapsUri",
        # Enterprise
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.priceRange",
        "places.websiteUri",
        "places.internationalPhoneNumber",
        "places.regularOpeningHours",
        # Atmosphere — dining features
        "places.editorialSummary",
        "places.reviews",
        "places.outdoorSeating",
        "places.servesVegetarianFood",
        "places.delivery",
        "places.takeout",
        "places.reservable",
        "places.dineIn",
        "places.curbsidePickup",
        # Atmosphere — meal types
        "places.servesBreakfast",
        "places.servesBrunch",
        "places.servesLunch",
        "places.servesDinner",
        # Atmosphere — drinks
        "places.servesBeer",
        "places.servesWine",
        "places.servesCocktails",
        "places.servesCoffee",
        "places.servesDessert",
        # Atmosphere — audience & amenities
        "places.allowsDogs",
        "places.goodForChildren",
        "places.goodForGroups",
        "places.goodForWatchingSports",
        "places.liveMusic",
        "places.menuForChildren",
        "places.restroom",
        "places.parkingOptions",
        "places.paymentOptions",
        "places.accessibilityOptions",
    ]
)

# Google priceLevel enum → our €-based scale
_PRICE_LEVEL_MAP: dict[str, str] = {
    "PRICE_LEVEL_FREE": "€",
    "PRICE_LEVEL_INEXPENSIVE": "€",
    "PRICE_LEVEL_MODERATE": "€€",
    "PRICE_LEVEL_EXPENSIVE": "€€€",
    "PRICE_LEVEL_VERY_EXPENSIVE": "€€€€",
}

# Google day index (0=Sunday, 1=Monday, ..., 6=Saturday) → our day names
_DAY_NAMES = [
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
]

_DELAY_BETWEEN_REQUESTS = 0.2  # seconds — API is fast, no Playwright-level delays
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1, 3, 10]  # seconds between retries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_opening_hours(hours_data: dict[str, Any]) -> dict[str, str] | None:
    """Convert Google Places opening hours periods to our format.

    Google format::

        {"periods": [{"open": {"day": 1, "hour": 10, "minute": 0},
                      "close": {"day": 1, "hour": 22, "minute": 0}}]}

    Our format::

        {"monday": "10:00-22:00", ...}
    """
    periods = hours_data.get("periods", [])
    if not periods:
        return None

    day_hours: dict[str, list[str]] = {}

    for period in periods:
        open_info = period.get("open", {})
        close_info = period.get("close")

        day_idx = open_info.get("day")
        if day_idx is None or not (0 <= day_idx <= 6):
            continue

        day_name = _DAY_NAMES[day_idx]
        open_time = f"{open_info.get('hour', 0):02d}:{open_info.get('minute', 0):02d}"

        if close_info:
            close_time = (
                f"{close_info.get('hour', 0):02d}:{close_info.get('minute', 0):02d}"
            )
            # Midnight closing → represent as 24:00
            if close_time == "00:00":
                close_time = "24:00"
            time_range = f"{open_time}-{close_time}"
        else:
            # No close time = open 24 hours
            time_range = "00:00-24:00"

        day_hours.setdefault(day_name, []).append(time_range)

    if not day_hours:
        return None

    # Join multiple periods for the same day (e.g. lunch + dinner)
    return {day: ", ".join(ranges) for day, ranges in day_hours.items()}


def _parse_reviews(place: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract top reviews into a compact storable format.

    Google returns up to 5 "most relevant" reviews per place.
    We keep: author, rating, text, relative publish time.
    """
    raw_reviews: list[dict[str, Any]] = place.get("reviews", [])
    if not raw_reviews:
        return []

    parsed: list[dict[str, Any]] = []
    for rev in raw_reviews:
        text = (rev.get("text") or {}).get("text", "").strip()
        if not text:
            continue
        parsed.append(
            {
                "author": (rev.get("authorAttribution") or {}).get(
                    "displayName", "Anonymous"
                ),
                "rating": rev.get("rating"),
                "text": text,
                "time": rev.get("relativePublishTimeDescription", ""),
                "language": (rev.get("text") or {}).get("languageCode", ""),
            }
        )
    return parsed


def _extract_features(place: dict[str, Any]) -> list[str]:
    """Extract feature flags from Google Places boolean fields."""
    feature_map: dict[str, str] = {
        # Dining
        "outdoorSeating": "outdoor_seating",
        "servesVegetarianFood": "vegetarian_options",
        "delivery": "delivery",
        "takeout": "takeaway",
        "reservable": "reservations",
        "dineIn": "dine_in",
        "curbsidePickup": "curbside_pickup",
        # Meal types
        "servesBreakfast": "serves_breakfast",
        "servesBrunch": "serves_brunch",
        "servesLunch": "serves_lunch",
        "servesDinner": "serves_dinner",
        # Drinks
        "servesBeer": "serves_beer",
        "servesWine": "serves_wine",
        "servesCocktails": "serves_cocktails",
        "servesCoffee": "serves_coffee",
        "servesDessert": "serves_dessert",
        # Audience & amenities
        "allowsDogs": "dogs_allowed",
        "goodForChildren": "good_for_children",
        "goodForGroups": "good_for_groups",
        "goodForWatchingSports": "sports_viewing",
        "liveMusic": "live_music",
        "menuForChildren": "children_menu",
        "restroom": "restroom",
    }

    features: list[str] = []
    for google_key, our_key in feature_map.items():
        if place.get(google_key) is True:
            features.append(our_key)

    # Wheelchair accessibility from nested object
    access = place.get("accessibilityOptions", {})
    if isinstance(access, dict) and access.get("wheelchairAccessibleEntrance") is True:
        features.append("wheelchair_accessible")

    # Parking from nested object
    parking = place.get("parkingOptions", {})
    if isinstance(parking, dict):
        if any(parking.get(k) is True for k in parking):
            features.append("parking")

    # Payment options from nested object
    payments = place.get("paymentOptions", {})
    if isinstance(payments, dict):
        if payments.get("acceptsCashOnly") is True:
            features.append("cash_only")
        if payments.get("acceptsNfc") is True:
            features.append("contactless_payment")

    return features


def _infer_cuisine(primary_type: str) -> list[str] | None:
    """Map Google's primaryType to cuisine labels.

    Only returns a value for types that clearly indicate a cuisine.
    Generic types like ``"restaurant"`` return ``None``.
    """
    mapping: dict[str, list[str]] = {
        "italian_restaurant": ["Italian"],
        "pizza_restaurant": ["Italian", "Pizza"],
        "chinese_restaurant": ["Chinese"],
        "japanese_restaurant": ["Japanese"],
        "thai_restaurant": ["Thai"],
        "indian_restaurant": ["Indian"],
        "mexican_restaurant": ["Mexican"],
        "korean_restaurant": ["Korean"],
        "vietnamese_restaurant": ["Vietnamese"],
        "turkish_restaurant": ["Turkish"],
        "greek_restaurant": ["Greek"],
        "lebanese_restaurant": ["Lebanese"],
        "french_restaurant": ["French"],
        "spanish_restaurant": ["Spanish"],
        "american_restaurant": ["American"],
        "brazilian_restaurant": ["Brazilian"],
        "indonesian_restaurant": ["Indonesian"],
        "middle_eastern_restaurant": ["Middle Eastern"],
        "mediterranean_restaurant": ["Mediterranean"],
        "asian_restaurant": ["Asian"],
        "seafood_restaurant": ["Seafood"],
        "steak_house": ["Steak"],
        "sushi_restaurant": ["Japanese", "Sushi"],
        "ramen_restaurant": ["Japanese", "Ramen"],
        "barbecue_restaurant": ["Barbecue"],
        "vegan_restaurant": ["Vegan"],
        "vegetarian_restaurant": ["Vegetarian"],
        "hamburger_restaurant": ["Burger"],
        "sandwich_shop": ["Sandwiches"],
        "ice_cream_shop": ["Ice Cream"],
        "bakery": ["Bakery"],
        "cafe": ["Cafe"],
        "coffee_shop": ["Cafe"],
        "bar": ["Bar"],
        "pub": ["Bar", "Pub"],
        "wine_bar": ["Wine Bar"],
        "brunch_restaurant": ["Brunch"],
        "breakfast_restaurant": ["Breakfast"],
    }
    return mapping.get(primary_type)


def _names_match(our_name: str, google_name: str) -> bool:
    """Check if restaurant names are a reasonable match.

    Uses case-insensitive substring matching to handle minor differences
    like "Café Central" vs "Cafe Central Graz".
    """
    ours = our_name.lower().strip()
    theirs = google_name.lower().strip()

    if not ours or not theirs:
        return False

    # Exact match
    if ours == theirs:
        return True

    # One contains the other
    if ours in theirs or theirs in ours:
        return True

    # Significant-word overlap (handles "Der Steirer" vs "Steirer")
    our_words = {w for w in ours.split() if len(w) > 2}
    their_words = {w for w in theirs.split() if len(w) > 2}
    if our_words and their_words:
        overlap = our_words & their_words
        ratio = len(overlap) / max(len(our_words), len(their_words))
        if overlap and ratio >= 0.5:
            return True

    return False


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------


async def _search_place(
    client: httpx.AsyncClient,
    api_key: str,
    name: str,
    lat: float | None = None,
    lng: float | None = None,
) -> dict[str, Any] | None:
    """Search for a single restaurant via Text Search and return place data.

    Returns the top result if the name matches, ``None`` otherwise.
    """
    body: dict[str, Any] = {
        "textQuery": f"{name} Graz restaurant",
        "maxResultCount": 3,
    }

    # Bias towards known coordinates if available
    if lat is not None and lng is not None:
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 500.0,
            }
        }
    else:
        # Default to Graz city centre
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": 47.0707, "longitude": 15.4395},
                "radius": 10000.0,
            }
        }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": _FIELD_MASK,
    }

    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.post(_API_BASE, json=body, headers=headers)

            if resp.status_code == 429:
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                logger.warning(
                    "Rate limited — retrying in %ds (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 403:
                logger.error(
                    "Google Places API: 403 Forbidden — check API key and billing"
                )
                return None

            resp.raise_for_status()
            data = resp.json()
            places: list[dict[str, Any]] = data.get("places", [])

            if not places:
                return None

            # Find best match by name
            for place in places:
                display_name = place.get("displayName", {}).get("text", "")
                if _names_match(name, display_name):
                    return place

            # No name match — skip to avoid wrong enrichment
            logger.debug(
                "No name match for '%s' (got: %s)",
                name,
                [p.get("displayName", {}).get("text", "") for p in places],
            )
            return None

        except httpx.HTTPStatusError as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BACKOFF[attempt]
                logger.warning(
                    "HTTP %d for '%s' — retrying in %ds",
                    exc.response.status_code,
                    name,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            logger.error("HTTP error for '%s': %s", name, exc)
            return None

        except httpx.RequestError as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BACKOFF[attempt]
                logger.warning(
                    "Request failed for '%s': %s — retrying in %ds",
                    name,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            logger.error("Request error for '%s': %s", name, exc)
            return None

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def enrich_restaurants(
    restaurants: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    max_failures: int = 10,
) -> list[dict[str, Any]]:
    """Enrich restaurant dicts with data from Google Places API (New).

    For each restaurant, searches Google Places by name + location,
    then merges enrichment data into gaps (never overwrites existing values).

    Args:
        restaurants: List of parsed restaurant dicts.
        api_key: Google Places API key. Falls back to
            ``GOOGLE_PLACES_API_KEY`` environment variable.
        max_failures: Abort after this many consecutive failures.

    Returns:
        The same list with enriched data merged in.
    """
    key = api_key or os.environ.get("GOOGLE_PLACES_API_KEY", "")
    if not key:
        logger.error(
            "GOOGLE_PLACES_API_KEY not set — skipping Google Places enrichment. "
            "Set it in .env or export it before running."
        )
        return restaurants

    total = len(restaurants)
    logger.info("Google Places API enrichment: %d restaurants to process", total)

    enriched_count = 0
    consecutive_failures = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for idx, restaurant in enumerate(restaurants):
            name = restaurant.get("name", "unknown")
            logger.info("[%d/%d] %s", idx + 1, total, name)

            place = await _search_place(
                client,
                key,
                name,
                lat=restaurant.get("latitude"),
                lng=restaurant.get("longitude"),
            )

            if not place:
                consecutive_failures += 1
                logger.debug("  No match for '%s'", name)
                if consecutive_failures >= max_failures:
                    logger.error(
                        "Aborting: %d consecutive failures — check API key / quota",
                        max_failures,
                    )
                    break
                await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)
                continue

            consecutive_failures = 0
            changes: list[str] = []

            # --- Google Place ID (always store for future re-syncs) ---
            place_id = place.get("id")
            if place_id:
                restaurant["google_place_id"] = place_id
                changes.append(f"place_id={place_id[:20]}…")

            # --- Rating ---
            if place.get("rating") and not restaurant.get("rating"):
                restaurant["rating"] = place["rating"]
                changes.append(f"rating={place['rating']}")

            # --- Review count ---
            user_rating_count = place.get("userRatingCount")
            if user_rating_count and restaurant.get("review_count", 0) == 0:
                restaurant["review_count"] = user_rating_count
                changes.append(f"review_count={user_rating_count}")

            # --- Price range (actual EUR amounts from Google) ---
            google_price_range = place.get("priceRange", {})
            start_price = google_price_range.get("startPrice", {})
            end_price = google_price_range.get("endPrice", {})
            if start_price.get("units") and end_price.get("units"):
                currency = start_price.get("currencyCode", "EUR")
                low = start_price["units"]
                high = end_price["units"]
                restaurant["price_range_text"] = f"{currency} {low}–{high}"
                changes.append(f"price_range_text={currency} {low}–{high}")

            # --- Price level (€/€€/€€€/€€€€ from Google) ---
            price_level = place.get("priceLevel")
            if price_level and restaurant.get("price_range") in (None, "€€"):
                mapped = _PRICE_LEVEL_MAP.get(price_level)
                if mapped:
                    restaurant["price_range"] = mapped
                    changes.append(f"price_range={mapped}")

            # --- Website ---
            if place.get("websiteUri") and not restaurant.get("website"):
                restaurant["website"] = place["websiteUri"]
                changes.append("website")

            # --- Phone ---
            if place.get("internationalPhoneNumber") and not restaurant.get("phone"):
                restaurant["phone"] = place["internationalPhoneNumber"]
                changes.append("phone")

            # --- Opening hours ---
            if place.get("regularOpeningHours") and not restaurant.get("opening_hours"):
                parsed = _parse_opening_hours(place["regularOpeningHours"])
                if parsed:
                    restaurant["opening_hours"] = parsed
                    changes.append("opening_hours")

            # --- Summary (editorial) ---
            editorial = place.get("editorialSummary", {})
            if isinstance(editorial, dict) and editorial.get("text"):
                if not restaurant.get("summary"):
                    restaurant["summary"] = editorial["text"]
                    changes.append("summary")

            # --- Reviews (top 5 from Google) ---
            reviews = _parse_reviews(place)
            if reviews and not restaurant.get("google_reviews"):
                restaurant["google_reviews"] = reviews
                changes.append(f"reviews={len(reviews)}")

            # --- Features (merge, don't replace) ---
            google_features = _extract_features(place)
            existing_features = set(restaurant.get("features", []))
            new_features = [f for f in google_features if f not in existing_features]
            if new_features:
                restaurant.setdefault("features", []).extend(new_features)
                changes.append(f"features+={new_features}")

            # --- Business status ---
            biz_status = place.get("businessStatus")
            if biz_status and biz_status != "OPERATIONAL":
                restaurant["business_status"] = biz_status
                changes.append(f"status={biz_status}")

            # --- Google cuisine (stored separately from OSM cuisine) ---
            primary_type = place.get("primaryType", "")
            google_types = place.get("types", [])
            if primary_type:
                restaurant["google_primary_type"] = primary_type
                inferred = _infer_cuisine(primary_type)
                if inferred:
                    restaurant["google_cuisine"] = inferred
                    changes.append(f"google_cuisine={inferred}")
            if google_types:
                restaurant["google_types"] = google_types

            # --- Google metadata ---
            display_name = place.get("displayName", {}).get("text")
            if display_name:
                restaurant["google_name"] = display_name

            google_addr = place.get("formattedAddress")
            if google_addr:
                restaurant["google_address"] = google_addr

            maps_uri = place.get("googleMapsUri")
            if maps_uri:
                restaurant["google_maps_url"] = maps_uri

            # --- Mark data source ---
            sources: list[str] = restaurant.setdefault("data_sources", [])
            if "google_places" not in sources:
                sources.append("google_places")

            enriched_count += 1
            if changes:
                logger.info("  ✓ %s", ", ".join(changes))
            else:
                logger.info("  ✓ matched (no new data)")

            await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)

    logger.info(
        "Google Places enrichment done: %d/%d restaurants enriched",
        enriched_count,
        total,
    )
    return restaurants
