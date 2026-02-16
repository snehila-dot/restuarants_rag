"""Discover missing restaurant websites via DuckDuckGo search.

For restaurants scraped from OpenStreetMap that lack a ``website`` tag,
this module searches DuckDuckGo for the restaurant name + "Graz" and
returns the most likely official website URL.

Triggered via ``python -m scraper --discover-websites``.

Requires: ``pip install duckduckgo-search``
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urlparse

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# Delay between searches to avoid rate-limiting
_DELAY_BETWEEN_SEARCHES = 3.0

# Domains that are aggregators, not official restaurant websites
_SKIP_DOMAINS = frozenset(
    {
        "google.com",
        "google.at",
        "maps.google.com",
        "facebook.com",
        "instagram.com",
        "tripadvisor.com",
        "tripadvisor.at",
        "tripadvisor.de",
        "yelp.com",
        "yelp.at",
        "lieferando.at",
        "lieferando.de",
        "mjam.net",
        "wolt.com",
        "uber.com",
        "ubereats.com",
        "foursquare.com",
        "wikipedia.org",
        "de.wikipedia.org",
        "en.wikipedia.org",
        "youtube.com",
        "tiktok.com",
        "twitter.com",
        "x.com",
        "linkedin.com",
        "pinterest.com",
        "reddit.com",
        "thefork.com",
        "quandoo.at",
        "quandoo.com",
        "herold.at",
        "falter.at",
        "graz.at",
        "stadtbekannt.at",
        "gastrofinder.at",
        "restaurant.info",
        "speisekarte.de",
        "restaurantguru.com",
    }
)

# Patterns that indicate a URL is likely an official restaurant website
_GOOD_SIGNALS = re.compile(
    r"(speisekarte|menu|reservier|kontakt|about|impressum|öffnungszeiten)",
    re.IGNORECASE,
)


def _is_aggregator(url: str) -> bool:
    """Check if a URL belongs to a known aggregator/platform."""
    try:
        domain = urlparse(url).netloc.lower()
        # Strip www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        # Check exact match and parent domain
        return domain in _SKIP_DOMAINS or any(
            domain.endswith(f".{skip}") for skip in _SKIP_DOMAINS
        )
    except Exception:
        return False


def _pick_best_url(results: list[dict[str, str]], restaurant_name: str) -> str | None:
    """Pick the most likely official website from search results.

    Strategy:
    1. Skip known aggregator domains (Google, Tripadvisor, etc.)
    2. Prefer results where the restaurant name appears in the URL or title
    3. Return the first non-aggregator result as fallback
    """
    name_lower = restaurant_name.lower().strip()
    # Create simple name tokens for matching (remove common suffixes)
    name_tokens = [
        t
        for t in re.split(r"[\s\-/&'+.,()]+", name_lower)
        if t and len(t) > 2 and t not in {"the", "das", "der", "die", "und", "and", "bar", "cafe"}
    ]

    candidates: list[tuple[int, str]] = []

    for result in results:
        url = result.get("href", "")
        title = result.get("title", "").lower()

        if not url or _is_aggregator(url):
            continue

        # Score: higher = better match
        score = 0
        domain = urlparse(url).netloc.lower().replace("www.", "")

        # Restaurant name tokens appear in domain
        for token in name_tokens:
            if token in domain:
                score += 3

        # Restaurant name tokens appear in title
        for token in name_tokens:
            if token in title:
                score += 1

        # "Graz" in title is a good sign
        if "graz" in title:
            score += 1

        candidates.append((score, url))

    if not candidates:
        return None

    # Sort by score descending, pick best
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def discover_website(restaurant_name: str, address: str = "") -> str | None:
    """Search DuckDuckGo for a restaurant's official website.

    Args:
        restaurant_name: Name of the restaurant.
        address: Optional address for disambiguation.

    Returns:
        URL string of the most likely official website, or None.
    """
    # Build search query
    query = f'"{restaurant_name}" Graz Restaurant'
    if address and "graz" not in address.lower():
        query += f" {address}"

    try:
        results = DDGS().text(query, region="at-de", max_results=8)
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for '%s': %s", restaurant_name, exc)
        return None

    if not results:
        return None

    return _pick_best_url(results, restaurant_name)


async def discover_missing_websites(
    restaurants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find websites for restaurants that don't have one.

    Searches DuckDuckGo for each restaurant without a ``website`` field
    and fills it in if a likely official website is found.

    A delay is inserted between searches to respect rate limits.

    Args:
        restaurants: List of restaurant dicts (modified in place).

    Returns:
        The same list with discovered websites filled in.
    """
    missing = [r for r in restaurants if not r.get("website")]
    if not missing:
        logger.info("All restaurants already have websites — skipping discovery")
        return restaurants

    logger.info(
        "Discovering websites for %d/%d restaurants without one…",
        len(missing),
        len(restaurants),
    )

    found_count = 0
    for idx, restaurant in enumerate(missing, start=1):
        name = restaurant.get("name", "unknown")
        address = restaurant.get("address", "")

        logger.info("Searching %d/%d: %s", idx, len(missing), name)

        # DuckDuckGo search is synchronous — run in executor to not block
        url = await asyncio.get_event_loop().run_in_executor(
            None, discover_website, name, address
        )

        if url:
            restaurant["website"] = url
            sources = restaurant.get("data_sources", [])
            if "web_search" not in sources:
                sources.append("web_search")
                restaurant["data_sources"] = sources
            found_count += 1
            logger.info("  → Found: %s", url)
        else:
            logger.debug("  → No website found for %s", name)

        # Polite delay between searches
        await asyncio.sleep(_DELAY_BETWEEN_SEARCHES)

    logger.info(
        "Website discovery complete: found %d/%d missing websites",
        found_count,
        len(missing),
    )
    return restaurants
