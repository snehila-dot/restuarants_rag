"""Google Maps enrichment scraper using Playwright.

Enriches existing restaurant data with ratings and categories by
intercepting Google Maps XHR responses (more stable than DOM scraping).

Triggered via ``python -m scraper --google-maps``.

This is a one-off enrichment tool, NOT called at runtime.
Respects Google's pages by adding delays between requests.

**Known limitation**: Google strips review counts and price levels from
responses served to headless browsers.  Ratings and categories are
reliably extracted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from playwright.async_api import Browser, Page, Response, async_playwright

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_DELAY_BETWEEN_REQUESTS = 3.0  # seconds — be polite
_NAV_TIMEOUT = 20_000  # ms
_ELEMENT_TIMEOUT = 3_000  # ms


# ---------------------------------------------------------------------------
# Cookie consent
# ---------------------------------------------------------------------------


async def _accept_cookies(page: Page) -> None:
    """Handle Google's GDPR cookie consent dialog."""
    labels = ("Accept all", "Alle akzeptieren", "Accept All")
    for label in labels:
        try:
            btn = page.locator(f'button:has-text("{label}")')
            if await btn.count() > 0:
                await btn.first.click()
                await page.wait_for_timeout(2000)
                logger.debug("Cookie consent accepted ('%s')", label)
                return
        except Exception:
            continue

    for label in labels:
        try:
            btn = page.locator(f'form >> button:has-text("{label}")')
            if await btn.count() > 0:
                await btn.first.click()
                await page.wait_for_timeout(2000)
                return
        except Exception:
            continue


# ---------------------------------------------------------------------------
# XHR-based extraction (primary — stable against DOM changes)
# ---------------------------------------------------------------------------


def _parse_xhr_place(raw_text: str) -> dict[str, Any] | None:
    """Parse a Google Maps ``/search?tbm=map`` XHR into structured data.

    The response is JSON prefixed with ``)]}'`` (XSS protection).
    Place data lives at ``data[0][1][0][14]``.

    Known field indices (discovered empirically):
        [14][2]    — address parts [street, city]
        [14][4][7] — rating (float, e.g. 4.1)
        [14][7]    — website info [url, display_name, ...]
        [14][9]    — coordinates [None, None, lat, lng]
        [14][11]   — place name (str)
        [14][13]   — categories (list[str])
        [14][18]   — full formatted address (str)
        [14][39]   — formatted address (str, fallback)
    """
    text = raw_text
    if text.startswith(")]}'"):
        text = text[4:].strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    try:
        place = data[0][1][0][14]
    except (IndexError, TypeError, KeyError):
        return None

    if not isinstance(place, list):
        return None

    result: dict[str, Any] = {}

    # Name
    if len(place) > 11 and isinstance(place[11], str):
        result["name"] = place[11]

    # Rating
    try:
        if len(place) > 4 and place[4] and len(place[4]) > 7:
            rating = place[4][7]
            if isinstance(rating, (int, float)) and 0 < rating <= 5:
                result["rating"] = float(rating)
    except (IndexError, TypeError):
        pass

    # Categories
    if len(place) > 13 and isinstance(place[13], list):
        cats = [c for c in place[13] if isinstance(c, str)]
        if cats:
            result["categories"] = cats

    # Full address
    if len(place) > 18 and isinstance(place[18], str):
        result["address"] = place[18]
    elif len(place) > 39 and isinstance(place[39], str):
        result["address"] = place[39]

    # Website
    try:
        if len(place) > 7 and place[7] and isinstance(place[7], list):
            url = place[7][0]
            if isinstance(url, str) and url.startswith("http"):
                result["website"] = url
    except (IndexError, TypeError):
        pass

    return result if result else None


# ---------------------------------------------------------------------------
# DOM-based extraction (fallback — used when XHR doesn't fire)
# ---------------------------------------------------------------------------


async def _extract_rating_dom(page: Page) -> float | None:
    """Extract star rating from DOM aria-labels or visible text."""
    try:
        for sel in (
            '[role="img"][aria-label*="star"]',
            '[role="img"][aria-label*="Stern"]',
        ):
            el = page.locator(sel)
            if await el.count() > 0:
                aria = await el.first.get_attribute("aria-label") or ""
                m = re.search(r"(\d[.,]\d)", aria)
                if m:
                    return float(m.group(1).replace(",", "."))
    except Exception:
        pass

    try:
        for sel in ("span.ceNzKf", "span.fontDisplayLarge"):
            spans = page.locator(sel)
            count = await spans.count()
            for i in range(min(count, 10)):
                text = (await spans.nth(i).inner_text(timeout=1000)).strip()
                if re.match(r"^\d[.,]\d$", text):
                    return float(text.replace(",", "."))
    except Exception:
        pass

    return None


async def _extract_category_dom(page: Page) -> str | None:
    """Extract business category from DOM."""
    try:
        btn = page.locator('button[jsaction*="category"]')
        if await btn.count() > 0:
            return (await btn.first.inner_text(timeout=_ELEMENT_TIMEOUT)).strip()
    except Exception:
        pass
    return None


async def _extract_google_name_dom(page: Page) -> str | None:
    """Extract place name from DOM h1."""
    try:
        h1 = page.locator("h1")
        if await h1.count() > 0:
            text = (await h1.first.inner_text(timeout=_ELEMENT_TIMEOUT)).strip()
            if text and text.lower() not in ("ergebnisse", "results"):
                return text
    except Exception:
        pass
    return None


async def _extract_address_dom(page: Page) -> str | None:
    """Extract address from DOM."""
    try:
        for sel in (
            'button[data-item-id="address"]',
            '[data-item-id="address"]',
            'button[aria-label*="Address"]',
            'button[aria-label*="Adresse"]',
        ):
            el = page.locator(sel)
            if await el.count() > 0:
                aria = await el.first.get_attribute("aria-label") or ""
                addr = (
                    aria.replace("Address: ", "")
                    .replace("Adresse: ", "")
                    .strip()
                )
                if addr:
                    return addr
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Single-restaurant enrichment
# ---------------------------------------------------------------------------


async def _enrich_one(
    page: Page,
    restaurant: dict[str, Any],
    xhr_data: dict[str, str],
) -> dict[str, str | float | int | None]:
    """Navigate to a restaurant on Google Maps and extract enrichment data.

    Uses XHR interception as the primary method (stable against DOM changes),
    with DOM parsing as a fallback for fields not found in the XHR.
    """
    name = restaurant["name"]
    lat = restaurant.get("latitude")
    lng = restaurant.get("longitude")

    # Clear previous XHR capture
    xhr_data.clear()

    search_q = f"{name} Graz".replace(" ", "+")
    url = f"https://www.google.com/maps/search/{search_q}"
    if lat and lng:
        url += f"/@{lat},{lng},17z"

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT)
    except Exception as exc:
        logger.warning("Navigation failed for '%s': %s", name, exc)
        return {}

    await page.wait_for_timeout(3000)

    # If we landed on search results, click the first one
    first_result = page.locator('div[role="feed"] a[href*="/maps/place/"]')
    try:
        if await first_result.count() > 0:
            await first_result.first.click()
            await page.wait_for_timeout(3000)
    except Exception:
        pass

    # Wait for detail panel
    try:
        await page.wait_for_selector("h1", timeout=8000)
    except Exception:
        logger.debug("Detail panel didn't load for '%s'", name)

    await page.wait_for_timeout(1500)

    enrichment: dict[str, str | float | int | None] = {}

    # --- Primary: XHR-based extraction ---
    if "body" in xhr_data:
        xhr_place = _parse_xhr_place(xhr_data["body"])
        if xhr_place:
            if xhr_place.get("rating"):
                enrichment["rating"] = xhr_place["rating"]
            if xhr_place.get("categories"):
                enrichment["google_category"] = xhr_place["categories"][0]
            if xhr_place.get("name"):
                enrichment["google_name"] = xhr_place["name"]
            if xhr_place.get("address"):
                enrichment["google_address"] = xhr_place["address"]

    # --- Fallback: DOM-based extraction for missing fields ---
    if "rating" not in enrichment:
        rating = await _extract_rating_dom(page)
        if rating is not None:
            enrichment["rating"] = rating

    if "google_category" not in enrichment:
        cat = await _extract_category_dom(page)
        if cat:
            enrichment["google_category"] = cat

    if "google_name" not in enrichment:
        gname = await _extract_google_name_dom(page)
        if gname:
            enrichment["google_name"] = gname

    if "google_address" not in enrichment:
        addr = await _extract_address_dom(page)
        if addr:
            enrichment["google_address"] = addr

    return enrichment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def enrich_restaurants(
    restaurants: list[dict[str, Any]],
    *,
    headless: bool = True,
    max_failures: int = 10,
) -> list[dict[str, Any]]:
    """Enrich a list of restaurant dicts with Google Maps data.

    For each restaurant, visits its Google Maps page via Playwright,
    intercepts the XHR response, and extracts rating and category.

    Only overwrites fields that are currently empty/None in the restaurant.

    Args:
        restaurants: List of parsed restaurant dicts.
        headless: Whether to run the browser in headless mode.
        max_failures: Abort if this many consecutive navigations fail
            (indicates possible blocking/CAPTCHA).

    Returns:
        The same list with enriched data merged in.
    """
    total = len(restaurants)
    logger.info("Google Maps enrichment: %d restaurants to process", total)

    enriched_count = 0
    consecutive_failures = 0

    # Mutable dict shared with XHR interceptor
    xhr_data: dict[str, str] = {}

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(
            headless=headless,
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

        # Mask navigator.webdriver to reduce bot detection
        await context.add_init_script(
            """
            Object.defineProperty(navigator, "webdriver", { get: () => undefined });
            window.chrome = { runtime: {} };
            """
        )

        page = await context.new_page()

        # Set up XHR interceptor for Google Maps search responses
        async def _on_response(response: Response) -> None:
            if "/search?tbm=map" in response.url:
                try:
                    xhr_data["body"] = await response.text()
                except Exception:
                    pass

        page.on("response", _on_response)

        # Initial navigation to handle cookie consent once
        try:
            await page.goto(
                "https://www.google.com/maps",
                wait_until="domcontentloaded",
                timeout=_NAV_TIMEOUT,
            )
            await page.wait_for_timeout(2000)
            await _accept_cookies(page)
            await page.wait_for_timeout(1000)
        except Exception as exc:
            logger.warning("Initial Maps load failed: %s", exc)

        for idx, restaurant in enumerate(restaurants):
            name = restaurant.get("name", "unknown")
            logger.info("[%d/%d] %s", idx + 1, total, name)

            enrichment = await _enrich_one(page, restaurant, xhr_data)

            if not enrichment:
                consecutive_failures += 1
                logger.debug("  No data extracted for '%s'", name)
                if consecutive_failures >= max_failures:
                    logger.error(
                        "Aborting: %d consecutive failures — possible CAPTCHA/block",
                        max_failures,
                    )
                    break
                await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)
                continue

            consecutive_failures = 0

            # Merge — only fill empty/None fields
            if enrichment.get("rating") and not restaurant.get("rating"):
                restaurant["rating"] = enrichment["rating"]

            if enrichment.get("review_count") and restaurant.get("review_count", 0) == 0:
                restaurant["review_count"] = enrichment["review_count"]

            if enrichment.get("price_range") and restaurant.get("price_range") in (
                None,
                "€€",
            ):
                restaurant["price_range"] = enrichment["price_range"]

            # Track google-specific metadata
            if enrichment.get("google_category"):
                restaurant.setdefault("google_category", enrichment["google_category"])
            if enrichment.get("google_name"):
                restaurant.setdefault("google_name", enrichment["google_name"])
            if enrichment.get("google_address"):
                restaurant.setdefault("google_address", enrichment["google_address"])

            # Mark data source
            if "google_maps" not in restaurant.get("data_sources", []):
                restaurant.setdefault("data_sources", []).append("google_maps")

            enriched_count += 1
            fields = ", ".join(f"{k}={v}" for k, v in enrichment.items())
            logger.info("  ✓ %s", fields)

            # Polite delay
            await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)

        await browser.close()

    logger.info(
        "Google Maps enrichment done: %d/%d restaurants enriched",
        enriched_count,
        total,
    )
    return restaurants
