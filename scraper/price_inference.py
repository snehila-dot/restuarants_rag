"""Infer restaurant price levels from menu prices and cuisine heuristics.

Assigns a ``price_range`` value (``"€"``, ``"€€"``, ``"€€€"``, or
``"€€€€"``) to each restaurant using two strategies:

1. **Menu-price median** (primary) — parse prices from extracted menu
   items, exclude drinks, compute the median, and map to a tier.
2. **Cuisine heuristic** (fallback) — when fewer than 3 priced
   non-drink items exist, fall back to a cuisine-based lookup table.

Thresholds (from project spec)::

    median <  10  →  €
    median <  20  →  €€
    median <  35  →  €€€
    median >= 35  →  €€€€

Triggered automatically during the scraper pipeline (always runs if
restaurant data is available).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex to extract a numeric price from strings like "€12,50", "12.90€",
# "EUR 14", etc.  Captures the numeric part only.
_PRICE_NUM_RE = re.compile(
    r"€\s*(\d+(?:[.,]\d{1,2})?)"
    r"|"
    r"(\d+(?:[.,]\d{1,2})?)\s*€"
    r"|"
    r"EUR\s*(\d+(?:[.,]\d{1,2})?)",
    re.IGNORECASE,
)

# Minimum non-drink menu items with valid prices to use menu-based
# inference instead of the heuristic.
_MIN_MENU_ITEMS = 3

# Cuisine → price tier mapping.  Keys are lowercased for matching.
_CHEAP_CUISINES: frozenset[str] = frozenset(
    {
        "fast food",
        "kebab",
        "döner",
        "doner",
        "ice cream",
        "bakery",
        "turkish",
    }
)

_MODERATE_CUISINES: frozenset[str] = frozenset(
    {
        "restaurant",
        "cafe",
        "burger",
        "pizza",
        "pasta",
        "asian",
        "chinese",
        "thai",
        "vietnamese",
        "indian",
        "middle eastern",
        "mexican",
        "greek",
        "austrian",
        "korean",
        "american",
        "vegetarian",
        "vegan",
    }
)

_UPSCALE_CUISINES: frozenset[str] = frozenset(
    {
        "steakhouse",
        "seafood",
        "mediterranean",
        "japanese",
        "italian",
    }
)

_LUXURY_CUISINES: frozenset[str] = frozenset(
    {
        "fine dining",
    }
)


def _parse_price(price_str: str) -> float | None:
    """Extract a numeric float from a price string.

    Handles ``€12``, ``€ 12.50``, ``€12,50``, ``12,90€``,
    ``12.50€``, ``EUR 12``, ``EUR 12.50``.

    Returns ``None`` if parsing fails.
    """
    m = _PRICE_NUM_RE.search(price_str)
    if not m:
        return None
    # One of the three groups will have matched
    raw = m.group(1) or m.group(2) or m.group(3)
    if raw is None:
        return None
    # European comma → dot
    raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _median(values: list[float]) -> float:
    """Return the median of a non-empty sorted list of floats."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _price_tier(median_price: float) -> str:
    """Map a median menu price to a price-range tier."""
    if median_price < 10:
        return "€"
    if median_price < 20:
        return "€€"
    if median_price < 35:
        return "€€€"
    return "€€€€"


def _infer_from_menu(menu_items: list[dict[str, str]]) -> str | None:
    """Infer price range from menu item prices.

    Excludes drinks and requires at least ``_MIN_MENU_ITEMS`` valid
    prices.  Returns ``None`` when data is insufficient.
    """
    prices: list[float] = []
    for item in menu_items:
        # Exclude drinks — they don't represent meal cost
        category = (item.get("category") or "").strip()
        if category.lower() == "drink":
            continue
        price_str = item.get("price", "")
        if not price_str:
            continue
        val = _parse_price(price_str)
        if val is not None and val > 0:
            prices.append(val)

    if len(prices) < _MIN_MENU_ITEMS:
        return None

    return _price_tier(_median(prices))


def _infer_from_cuisine(cuisines: list[str]) -> str:
    """Infer price range from cuisine labels using a tiered heuristic.

    When multiple cuisines map to different tiers, the most common
    tier wins.  Ties are broken in favour of the higher tier.

    Falls back to ``"€€"`` if no cuisine data is available.
    """
    if not cuisines:
        return "€€"

    # Map: tier → count
    tier_counts: dict[str, int] = {"€": 0, "€€": 0, "€€€": 0, "€€€€": 0}

    for c in cuisines:
        low = c.lower()
        if low in _LUXURY_CUISINES:
            tier_counts["€€€€"] += 1
        elif low in _UPSCALE_CUISINES:
            tier_counts["€€€"] += 1
        elif low in _CHEAP_CUISINES:
            tier_counts["€"] += 1
        elif low in _MODERATE_CUISINES:
            tier_counts["€€"] += 1
        # else: unmapped cuisine — doesn't contribute

    # Find tier(s) with highest count (ignoring zeros)
    max_count = max(tier_counts.values())
    if max_count == 0:
        return "€€"

    # Among tied tiers, pick the most expensive one
    # Order from most to least expensive for tie-breaking
    for tier in ("€€€€", "€€€", "€€", "€"):
        if tier_counts[tier] == max_count:
            return tier

    return "€€"


def infer_price_levels(
    restaurants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign ``price_range`` to every restaurant in the list.

    Uses menu-item prices when available (≥ 3 non-drink priced items),
    otherwise falls back to a cuisine-based heuristic.

    Modifies the list **in-place** and returns it.

    Args:
        restaurants: Restaurant dicts to process.

    Returns:
        The same list with ``price_range`` set on each restaurant.
    """
    total = len(restaurants)
    from_menu = 0
    from_heuristic = 0
    from_default = 0
    distribution: dict[str, int] = {"€": 0, "€€": 0, "€€€": 0, "€€€€": 0}

    logger.info("Inferring price levels for %d restaurants …", total)

    for restaurant in restaurants:
        menu_items = restaurant.get("menu_items", [])
        cuisines = restaurant.get("cuisine", [])

        # Primary: menu-price median
        tier = _infer_from_menu(menu_items)
        if tier is not None:
            from_menu += 1
        else:
            # Fallback: cuisine heuristic
            tier = _infer_from_cuisine(cuisines)
            if cuisines:
                from_heuristic += 1
            else:
                from_default += 1

        restaurant["price_range"] = tier
        distribution[tier] = distribution.get(tier, 0) + 1

    logger.info(
        "Price inference complete: %d from menu prices, "
        "%d from category heuristic, %d defaulted to €€",
        from_menu,
        from_heuristic,
        from_default,
    )
    logger.info(
        "Price distribution: € %d, €€ %d, €€€ %d, €€€€ %d",
        distribution["€"],
        distribution["€€"],
        distribution["€€€"],
        distribution["€€€€"],
    )
    return restaurants
