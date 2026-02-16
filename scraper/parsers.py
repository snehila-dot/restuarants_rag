"""Transform raw OSM data into the application's Restaurant schema."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Maps 2-letter OSM day abbreviations → full English day names
_DAY_MAP: dict[str, str] = {
    "Mo": "monday",
    "Tu": "tuesday",
    "We": "wednesday",
    "Th": "thursday",
    "Fr": "friday",
    "Sa": "saturday",
    "Su": "sunday",
}

_DAY_ORDER = list(_DAY_MAP.values())

# Regex for a simple day-range token like "Mo-Fr" or single day "Sa"
_DAY_RANGE_RE = re.compile(
    r"^(Mo|Tu|We|Th|Fr|Sa|Su)(?:-(Mo|Tu|We|Th|Fr|Sa|Su))?$"
)


# --- Address -------------------------------------------------------------------


def parse_address(tags: dict[str, str]) -> str:
    """Build a human-readable address from ``addr:*`` tags."""
    street = tags.get("addr:street", "")
    number = tags.get("addr:housenumber", "")
    postcode = tags.get("addr:postcode", "")
    city = tags.get("addr:city", "Graz")

    street_part = f"{street} {number}".strip()
    city_part = f"{postcode} {city}".strip()

    if street_part and city_part:
        return f"{street_part}, {city_part}"
    if street_part:
        return f"{street_part}, Graz"
    if city_part:
        return city_part
    return "Graz, Austria"


# --- Cuisine -------------------------------------------------------------------


def parse_cuisine(tags: dict[str, str]) -> list[str]:
    """Parse the ``cuisine`` tag into a title-cased list.

    Falls back to inferring from ``amenity`` type when no cuisine tag exists.
    """
    raw = tags.get("cuisine", "")
    if raw:
        return [c.strip().replace("_", " ").title() for c in raw.split(";") if c.strip()]

    amenity = tags.get("amenity", "")
    fallback: dict[str, list[str]] = {
        "cafe": ["Cafe"],
        "fast_food": ["Fast Food"],
        "bar": ["Bar"],
        "pub": ["Bar"],
        "biergarten": ["Austrian", "Beer Garden"],
    }
    return fallback.get(amenity, ["Restaurant"])


# --- Features ------------------------------------------------------------------

_FEATURE_TAG_MAP: list[tuple[str, str, set[str]]] = [
    ("outdoor_seating", "outdoor_seating", {"yes"}),
    ("wheelchair", "wheelchair_accessible", {"yes", "limited"}),
    ("diet:vegan", "vegan_options", {"yes", "only"}),
    ("diet:vegetarian", "vegetarian_options", {"yes", "only"}),
    ("internet_access", "wifi", {"wlan", "yes"}),
    ("delivery", "delivery", {"yes"}),
    ("takeaway", "takeaway", {"yes"}),
    ("reservation", "reservations", {"yes"}),
]


def parse_features(tags: dict[str, str]) -> list[str]:
    """Extract boolean feature flags from OSM tags."""
    features: list[str] = []
    for tag_key, feature_name, positive_values in _FEATURE_TAG_MAP:
        if tags.get(tag_key, "").lower() in positive_values:
            features.append(feature_name)
    return features


# --- Opening hours -------------------------------------------------------------


def _expand_day_range(token: str) -> list[str]:
    """Expand ``'Mo-Fr'`` → ``['monday', …, 'friday']`` or ``'Sa'`` → ``['saturday']``."""
    m = _DAY_RANGE_RE.match(token)
    if not m:
        return []
    start_abbr, end_abbr = m.group(1), m.group(2)
    start_day = _DAY_MAP[start_abbr]
    if end_abbr is None:
        return [start_day]
    end_day = _DAY_MAP[end_abbr]
    start_idx = _DAY_ORDER.index(start_day)
    end_idx = _DAY_ORDER.index(end_day)
    if end_idx >= start_idx:
        return _DAY_ORDER[start_idx : end_idx + 1]
    # Wrap around (e.g. Fr-Mo)
    return _DAY_ORDER[start_idx:] + _DAY_ORDER[: end_idx + 1]


def parse_opening_hours(raw: str | None) -> dict[str, str] | None:
    """Best-effort parse of OSM ``opening_hours`` into per-day strings.

    Returns ``None`` when the format is too complex to parse reliably.
    """
    if not raw:
        return None

    result: dict[str, str] = {}
    try:
        # Split on ';' for independent rules
        rules = [r.strip() for r in raw.split(";") if r.strip()]
        for rule in rules:
            # Handle "24/7"
            if rule == "24/7":
                for day in _DAY_ORDER:
                    result[day] = "00:00-24:00"
                continue

            # Handle "off" rules like "Su off"
            if " off" in rule:
                day_part = rule.replace(" off", "").strip()
                for day in _expand_day_range(day_part):
                    result[day] = "closed"
                continue

            # Expect format: "Mo-Fr 10:00-22:00" or "Sa 11:00-23:00"
            parts = rule.split(None, 1)
            if len(parts) != 2:
                continue
            day_part, time_part = parts
            # day_part can be comma-separated: "Mo,We,Fr 10:00-22:00"
            day_tokens = [d.strip() for d in day_part.split(",")]
            for token in day_tokens:
                for day in _expand_day_range(token):
                    result[day] = time_part.strip()
    except Exception:
        logger.debug("Failed to parse opening_hours: %r", raw)
        return None

    return result if result else None


# --- Price range ---------------------------------------------------------------


def parse_price_range(tags: dict[str, str]) -> str:
    """Infer price range from amenity type (OSM has no reliable price tag)."""
    amenity = tags.get("amenity", "")
    mapping: dict[str, str] = {
        "fast_food": "€",
        "cafe": "€€",
        "bar": "€€",
        "pub": "€€",
        "biergarten": "€€",
        "restaurant": "€€",
    }
    return mapping.get(amenity, "€€")


# --- Contact -------------------------------------------------------------------


def parse_phone(tags: dict[str, str]) -> str | None:
    """Extract phone number from ``phone`` or ``contact:phone``."""
    return tags.get("phone") or tags.get("contact:phone") or None


def parse_website(tags: dict[str, str]) -> str | None:
    """Extract website from ``website`` or ``contact:website``."""
    return tags.get("website") or tags.get("contact:website") or None


# --- Composite -----------------------------------------------------------------


def raw_to_restaurant(element: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw Overpass element dict into our Restaurant schema dict."""
    tags: dict[str, str] = element.get("tags", {})
    name = tags.get("name", "")

    return {
        "name": name,
        "address": parse_address(tags),
        "phone": parse_phone(tags),
        "website": parse_website(tags),
        "cuisine": parse_cuisine(tags),
        "price_range": parse_price_range(tags),
        "rating": None,
        "review_count": 0,
        "features": parse_features(tags),
        "opening_hours": parse_opening_hours(tags.get("opening_hours")),
        "summary": None,
        "menu_items": [],
        "menu_url": None,
        "latitude": element.get("latitude"),
        "longitude": element.get("longitude"),
        "data_sources": ["openstreetmap"],
    }
