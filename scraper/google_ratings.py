"""Enrich restaurants with Google Maps ratings and review counts.

Uses Playwright to navigate to each restaurant's Google Maps page
and extract the rating (float) and review count (int) from the
detail panel.

Triggered via ``python -m scraper --ratings`` or called from the
pipeline after website enrichment.

.. note::

   Google Maps serves pages in German locale for Austrian IP
   addresses regardless of the ``locale`` setting.  Selectors
   therefore match both English and German labels.

   Expect ~8 s per restaurant.  A random 3–5 s delay is inserted
   between requests to reduce the risk of rate-limiting.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

_MAPS_HOME = "https://www.google.com/maps"
_SEARCH_URL = "https://www.google.com/maps/search/{query}/@{lat},{lng},17z"
_NAV_TIMEOUT = 15_000  # ms
_DETAIL_WAIT = 3_000  # ms — wait for detail panel after navigation
_CONSENT_WAIT = 2_000  # ms — wait after clicking cookie consent
_MIN_DELAY = 3.0  # seconds between restaurants
_MAX_DELAY = 5.0

# Consent button labels (Google GDPR dialog)
_CONSENT_LABELS = (
    "Accept all",
    "Alle akzeptieren",
    "Reject all",
    "Alle ablehnen",
)

# User-agent mimicking a real Chrome 130 on Windows
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)


def _needs_rating(restaurant: dict[str, Any]) -> bool:
    """Return ``True`` if the restaurant still needs a rating."""
    if restaurant.get("rating") is not None:
        return False
    name = restaurant.get("name")
    lat = restaurant.get("latitude")
    lng = restaurant.get("longitude")
    return bool(name and lat is not None and lng is not None)


async def _accept_cookies(page: Any) -> None:
    """Dismiss the Google GDPR consent dialog if present."""
    for label in _CONSENT_LABELS:
        try:
            btn = page.locator(f'button:has-text("{label}")')
            if await btn.count() > 0:
                await btn.first.click()
                await page.wait_for_timeout(_CONSENT_WAIT)
                logger.debug("Cookie consent handled via '%s'", label)
                return
        except Exception:  # noqa: BLE001
            continue


async def _extract_rating(page: Any, name: str) -> float | None:
    """Extract the numeric rating from the Google Maps detail panel.

    Tries two strategies:
    1. ``aria-label`` on the star-rating image element.
    2. Fallback scan of prominent ``<span>`` elements for a ``\\d.\\d``
       pattern.
    """
    # Strategy 1: aria-label on star image
    try:
        rating_el = page.locator(
            '[role="img"][aria-label*="star"], '
            '[role="img"][aria-label*="Stern"]'
        )
        if await rating_el.count() > 0:
            aria = await rating_el.first.get_attribute("aria-label") or ""
            m = re.search(r"(\d[.,]\d)\s*(?:star|Stern)", aria)
            if m:
                return float(m.group(1).replace(",", "."))
    except Exception:  # noqa: BLE001
        pass

    # Strategy 2: visible span with rating number
    try:
        spans = page.locator(
            "span.ceNzKf, span.fontDisplayLarge, "
            'span[aria-hidden="true"]'
        )
        count = await spans.count()
        for i in range(min(count, 20)):
            text = (await spans.nth(i).inner_text(timeout=1000)).strip()
            if re.match(r"^\d[.,]\d$", text):
                return float(text.replace(",", "."))
    except Exception:  # noqa: BLE001
        pass

    logger.debug("No rating found for '%s'", name)
    return None


async def _extract_review_count(page: Any, name: str) -> int | None:
    """Extract the review count from the Google Maps detail panel.

    Tries three strategies:
    1. ``aria-label`` on the star-rating image (often includes count).
    2. Dedicated review button / link with count in aria-label.
    3. Parenthesized number ``(1,234)`` near the rating element.
    """
    # Strategy 1: from star-image aria-label
    try:
        rating_el = page.locator(
            '[role="img"][aria-label*="star"], '
            '[role="img"][aria-label*="Stern"]'
        )
        if await rating_el.count() > 0:
            aria = await rating_el.first.get_attribute("aria-label") or ""
            m = re.search(
                r"([\d.,]+)\s*(?:review|Rezension|Bewertung)",
                aria,
                re.IGNORECASE,
            )
            if m:
                raw = m.group(1).replace(".", "").replace(",", "")
                if raw.isdigit():
                    return int(raw)
    except Exception:  # noqa: BLE001
        pass

    # Strategy 2: review button aria-label
    try:
        review_btn = page.locator(
            'button[aria-label*="review"], '
            'button[aria-label*="Rezension"], '
            'button[aria-label*="Bewertung"], '
            'a[aria-label*="review"], '
            'a[aria-label*="Rezension"]'
        )
        if await review_btn.count() > 0:
            aria = (
                await review_btn.first.get_attribute("aria-label") or ""
            )
            m = re.search(
                r"([\d.,]+)\s*(?:review|Rezension|Bewertung)",
                aria,
                re.IGNORECASE,
            )
            if m:
                raw = m.group(1).replace(".", "").replace(",", "")
                if raw.isdigit():
                    return int(raw)
    except Exception:  # noqa: BLE001
        pass

    # Strategy 3: parenthesized numbers like "(1,234)"
    try:
        parens = page.locator(
            'span:has-text("("), button:has-text("(")'
        )
        count = await parens.count()
        for i in range(min(count, 15)):
            text = (await parens.nth(i).inner_text(timeout=500)).strip()
            m = re.match(r"^\(?([\d.,]+)\)?$", text)
            if m:
                raw = m.group(1).replace(".", "").replace(",", "")
                if raw.isdigit() and int(raw) > 0:
                    return int(raw)
    except Exception:  # noqa: BLE001
        pass

    logger.debug("No review count found for '%s'", name)
    return None


async def _scrape_one(
    page: Any,
    restaurant: dict[str, Any],
) -> tuple[float | None, int | None]:
    """Navigate to a restaurant's Google Maps page and extract data.

    Returns ``(rating, review_count)``.
    """
    name = restaurant["name"]
    lat = restaurant["latitude"]
    lng = restaurant["longitude"]

    search_q = quote(f"{name} Graz")
    url = _SEARCH_URL.format(query=search_q, lat=lat, lng=lng)

    try:
        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=_NAV_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Navigation failed for '%s': %s", name, exc)
        return None, None

    await page.wait_for_timeout(_DETAIL_WAIT)

    # If we landed on search results, click the first result
    try:
        feed_links = page.locator(
            'div[role="feed"] a[href*="/maps/place/"]'
        )
        if await feed_links.count() > 0:
            await feed_links.first.click()
            await page.wait_for_timeout(_DETAIL_WAIT)
    except Exception:  # noqa: BLE001
        pass

    rating = await _extract_rating(page, name)
    review_count = await _extract_review_count(page, name)
    return rating, review_count


async def enrich_ratings(
    restaurants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich restaurants with Google Maps ratings and review counts.

    Only processes restaurants that lack a ``rating`` value and have
    valid ``name``, ``latitude``, and ``longitude`` fields.  A headless
    Chromium browser is launched once and reused for all restaurants.

    Modifies the list **in-place** and returns it.

    Args:
        restaurants: Restaurant dicts to enrich.

    Returns:
        The same list with ``rating``, ``review_count``, and
        ``data_sources`` updated where data was found.
    """
    candidates = [r for r in restaurants if _needs_rating(r)]
    if not candidates:
        logger.info(
            "All restaurants already have ratings — skipping Google Maps"
        )
        return restaurants

    logger.info(
        "Scraping Google Maps ratings for %d/%d restaurants …",
        len(candidates),
        len(restaurants),
    )

    rating_count = 0
    review_count_total = 0

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error(
            "Playwright is not installed. "
            "Run 'pip install playwright && playwright install chromium'."
        )
        return restaurants

    pw_ctx = async_playwright()
    try:
        pw = await pw_ctx.__aenter__()
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="Europe/Vienna",
            user_agent=_USER_AGENT,
        )
        page = await context.new_page()

        # Initial navigation to handle cookie consent once
        logger.info("Opening Google Maps for cookie consent …")
        await page.goto(
            _MAPS_HOME,
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        await page.wait_for_timeout(2000)
        await _accept_cookies(page)
        await page.wait_for_timeout(1000)

        for idx, restaurant in enumerate(candidates, start=1):
            name = restaurant.get("name", "unknown")
            logger.info(
                "Processing %d/%d: %s", idx, len(candidates), name
            )

            try:
                rating, reviews = await _scrape_one(page, restaurant)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Unexpected error scraping '%s'",
                    name,
                    exc_info=True,
                )
                rating, reviews = None, None

            if rating is not None:
                restaurant["rating"] = rating
                rating_count += 1

            if reviews is not None:
                restaurant["review_count"] = reviews
                review_count_total += 1

            if rating is not None or reviews is not None:
                sources = restaurant.get("data_sources", [])
                if "google_maps" not in sources:
                    sources.append("google_maps")
                    restaurant["data_sources"] = sources

            # Log result
            parts: list[str] = []
            if rating is not None:
                parts.append(f"rating={rating:.1f}")
            if reviews is not None:
                parts.append(f"reviews={reviews}")
            if parts:
                logger.info("  → %s", ", ".join(parts))
            else:
                logger.info("  → No data extracted")

            # Random delay between restaurants (skip after last)
            if idx < len(candidates):
                delay = random.uniform(_MIN_DELAY, _MAX_DELAY)
                await asyncio.sleep(delay)

        await browser.close()
    except Exception:  # noqa: BLE001
        logger.error(
            "Browser error during rating enrichment", exc_info=True
        )
    finally:
        await pw_ctx.__aexit__(None, None, None)

    logger.info(
        "Rating enrichment complete: %d/%d with rating, "
        "%d/%d with review count",
        rating_count,
        len(candidates),
        review_count_total,
        len(candidates),
    )
    return restaurants
