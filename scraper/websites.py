"""Individual restaurant website scraper using BeautifulSoup.

Enriches existing restaurant data with summaries and **menu items**
extracted from each restaurant's own website.  Triggered via
``python -m scraper --enrich``.

Pass ``use_playwright=True`` (or ``--enrich-js`` on the CLI) to enable
a Playwright headless-browser fallback for JavaScript-rendered menus.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, Tag

if TYPE_CHECKING:
    from playwright.async_api import Browser

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; GrazRestaurantChatbot/1.0; "
    "+https://github.com/example/graz-restaurants)"
)
_REQUEST_TIMEOUT = 15.0
_DELAY_BETWEEN_REQUESTS = 2.0  # seconds — be polite

# Keywords that hint a link leads to a menu / Speisekarte page
_MENU_KEYWORDS = frozenset(
    {"menu", "speisekarte", "karte", "gerichte", "dishes", "speisen"}
)

# Regex for prices in European format: €12, € 12.50, 12,90€, EUR 9.50, etc.
_PRICE_RE = re.compile(
    r"€\s*\d+(?:[.,]\d{1,2})?"  # €12  €12.50  €12,50
    r"|"
    r"\d+(?:[.,]\d{1,2})?\s*€"  # 12€  12.50€  12,50€
    r"|"
    r"EUR\s*\d+(?:[.,]\d{1,2})?",  # EUR 12
    re.IGNORECASE,
)

# Categories we try to infer from heading text
_CATEGORY_HINTS: dict[str, list[str]] = {
    "Starter": [
        "starter",
        "vorspeise",
        "antipast",
        "appetizer",
        "entrée",
        "small plate",
    ],
    "Main": [
        "main",
        "hauptgericht",
        "hauptspeise",
        "entrée",
        "second",
        "piatti",
        "gericht",
    ],
    "Dessert": [
        "dessert",
        "nachspeise",
        "süß",
        "dolci",
        "sweet",
        "nachtisch",
    ],
    "Drink": [
        "drink",
        "getränk",
        "beverage",
        "cocktail",
        "wein",
        "wine",
        "beer",
        "bier",
        "saft",
        "juice",
        "kaffee",
        "coffee",
    ],
    "Pizza": ["pizza"],
    "Pasta": ["pasta", "nudel"],
    "Sushi": ["sushi", "maki", "nigiri", "sashimi"],
    "Soup": ["soup", "suppe"],
    "Salad": ["salad", "salat"],
    "Burger": ["burger"],
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _fetch_html(
    client: httpx.AsyncClient,
    url: str,
) -> BeautifulSoup | None:
    """GET *url* and return parsed soup, or ``None`` on failure."""
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type:
        logger.debug("Skipping non-HTML response from %s", url)
        return None

    return BeautifulSoup(resp.text, "html.parser")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_summary(soup: BeautifulSoup) -> str | None:
    """Extract a short description from the page.

    Priority: ``<meta name="description">``, then first ``<p>`` with
    enough text content (≥ 40 chars).
    """
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        desc = str(meta["content"]).strip()
        if len(desc) >= 20:
            return desc[:300]

    for p in soup.find_all("p", limit=10):
        text = p.get_text(strip=True)
        if len(text) >= 40:
            return text[:300]

    return None


def _find_menu_url(soup: BeautifulSoup, base_url: str) -> str | None:
    """Return the absolute URL of the first HTML menu-like link found."""
    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"])
        text = a_tag.get_text(strip=True).lower()
        combined = f"{href.lower()} {text}"
        if any(kw in combined for kw in _MENU_KEYWORDS):
            # Skip anchors that point to PDFs / images (handled by vision)
            if any(
                href.lower().split("?")[0].endswith(ext)
                for ext in (".pdf", ".jpg", ".jpeg", ".png", ".webp")
            ):
                continue
            return urljoin(base_url, href)
    return None


def _find_menu_file_url(soup: BeautifulSoup, base_url: str) -> str | None:
    """Return the absolute URL of the first PDF/image menu link found."""
    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"])
        text = a_tag.get_text(strip=True).lower()
        combined = f"{href.lower()} {text}"
        if any(kw in combined for kw in _MENU_KEYWORDS):
            clean_href = href.lower().split("?")[0]
            if any(
                clean_href.endswith(ext)
                for ext in (".pdf", ".jpg", ".jpeg", ".png", ".webp")
            ):
                return urljoin(base_url, href)
    return None


def _infer_category(heading_text: str) -> str:
    """Map a heading string to a menu category."""
    lower = heading_text.lower()
    for category, keywords in _CATEGORY_HINTS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "Other"


def _extract_price(text: str) -> str | None:
    """Find the first €-price in *text*."""
    m = _PRICE_RE.search(text)
    if m:
        return m.group(0).strip()
    return None


def _extract_menu_items_from_soup(
    soup: BeautifulSoup,
) -> list[dict[str, str]]:
    """Best-effort extraction of structured menu items from HTML.

    Strategy (tried in order):
    1. **Schema.org / JSON-LD** — ``application/ld+json`` with ``Menu`` or
       ``MenuItem`` types.
    2. **Heuristic HTML parsing** — walk headings (h2/h3 = category) and
       sibling elements looking for dish-name + price patterns.

    Returns a list of ``{"name": …, "price": …, "category": …}`` dicts.
    Items without a recognisable name are skipped.
    """
    items: list[dict[str, str]] = []

    # --- Strategy 1: JSON-LD structured data --------------------------------
    items = _extract_from_jsonld(soup)
    if items:
        return items

    # --- Strategy 2: Heuristic HTML walk ------------------------------------
    items = _extract_from_html_heuristic(soup)
    return items


def _extract_from_jsonld(soup: BeautifulSoup) -> list[dict[str, str]]:
    """Parse Schema.org JSON-LD blocks for menu items."""
    import json as _json

    items: list[dict[str, str]] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(script.string or "")
        except (ValueError, TypeError):
            continue

        # Can be a single object or a list
        objects = data if isinstance(data, list) else [data]
        for obj in objects:
            _walk_jsonld(obj, items, category="Other")

    return items


def _walk_jsonld(
    obj: Any,
    items: list[dict[str, str]],
    category: str,
) -> None:
    """Recursively walk a JSON-LD structure looking for menu items."""
    if not isinstance(obj, dict):
        return

    obj_type = obj.get("@type", "")

    # MenuSection → use its name as category
    if obj_type == "MenuSection":
        section_name = obj.get("name", category)
        for child in obj.get("hasMenuItem", []):
            _walk_jsonld(child, items, category=section_name)
        for child in obj.get("hasMenuSection", []):
            _walk_jsonld(child, items, category=section_name)
        return

    # MenuItem
    if obj_type == "MenuItem":
        name = obj.get("name", "").strip()
        if not name:
            return
        price = ""
        offers = obj.get("offers", {})
        if isinstance(offers, dict):
            price = str(offers.get("price", ""))
            currency = offers.get("priceCurrency", "€")
            if price:
                price = f"{currency}{price}" if currency else price
        items.append(
            {
                "name": name,
                "price": price,
                "category": category,
            }
        )
        return

    # Menu — drill into sections / items
    if obj_type == "Menu":
        for child in obj.get("hasMenuSection", []):
            _walk_jsonld(child, items, category=category)
        for child in obj.get("hasMenuItem", []):
            _walk_jsonld(child, items, category=category)
        return

    # Generic: check common container keys
    for key in ("hasMenu", "menu", "hasMenuSection", "hasMenuItem"):
        val = obj.get(key)
        if isinstance(val, list):
            for child in val:
                _walk_jsonld(child, items, category=category)
        elif isinstance(val, dict):
            _walk_jsonld(val, items, category=category)


def _extract_from_html_heuristic(
    soup: BeautifulSoup,
) -> list[dict[str, str]]:
    """Walk headings + siblings to find dish-name / price pairs.

    Looks for patterns like:
        <h2>Hauptgerichte</h2>
        <div>Wiener Schnitzel ... €14.90</div>
        <div>Tafelspitz ... €18.50</div>
    or table rows, list items, etc.
    """
    items: list[dict[str, str]] = []
    current_category = "Other"

    # Collect all headings and the block-level elements between them
    for element in soup.find_all(["h1", "h2", "h3", "h4", "li", "tr", "div", "p"]):
        if not isinstance(element, Tag):
            continue

        tag_name = element.name
        text = element.get_text(separator=" ", strip=True)

        if not text or len(text) < 3:
            continue

        # Update category from headings
        if tag_name in ("h1", "h2", "h3", "h4"):
            current_category = _infer_category(text)
            continue

        # Skip very long blocks (likely paragraphs of prose, not menu items)
        if len(text) > 200:
            continue

        # A menu item typically has a price
        price = _extract_price(text)
        if not price:
            continue

        # Remove the price from the text to get the dish name
        name = _PRICE_RE.sub("", text).strip()
        # Clean up stray separators
        name = re.sub(r"[\.\-–—]+\s*$", "", name).strip()
        name = re.sub(r"^\s*[\.\-–—]+", "", name).strip()

        if not name or len(name) < 3 or len(name) > 120:
            continue

        items.append(
            {
                "name": name,
                "price": price,
                "category": current_category,
            }
        )

    # Deduplicate by name (keep first)
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in items:
        key = item["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


# ---------------------------------------------------------------------------
# Playwright fallback
# ---------------------------------------------------------------------------

_PW_NAV_TIMEOUT = 20_000  # ms — max time to wait for page load
_PW_IDLE_TIMEOUT = 10_000  # ms — max time to wait for network idle


async def _fetch_html_playwright(
    url: str,
    browser: Browser,
) -> BeautifulSoup | None:
    """Render *url* in a headless browser and return parsed soup.

    Opens a new page, navigates with ``networkidle`` wait strategy,
    extracts the fully-rendered HTML, then closes the page.
    Returns ``None`` on any failure without crashing the browser.
    """
    page = await browser.new_page()
    try:
        await page.goto(
            url,
            wait_until="networkidle",
            timeout=_PW_NAV_TIMEOUT,
        )
        # Give JS a brief moment to finish any late renders
        await page.wait_for_timeout(1000)
        html = await page.content()
        return BeautifulSoup(html, "html.parser")
    except Exception as exc:
        logger.debug("Playwright failed for %s: %s", url, exc)
        return None
    finally:
        await page.close()


async def _scrape_menu_playwright(
    url: str,
    browser: Browser,
) -> list[dict[str, str]]:
    """Fetch *url* with Playwright and extract menu items.

    Reuses the same ``_extract_menu_items_from_soup`` parser so
    all extraction logic stays in one place.
    """
    soup = await _fetch_html_playwright(url, browser)
    if soup is None:
        return []
    return _extract_menu_items_from_soup(soup)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def scrape_restaurant_website(
    url: str,
    client: httpx.AsyncClient,
    *,
    use_vision: bool = False,
    openai_api_key: str | None = None,
) -> dict[str, Any] | None:
    """Fetch a restaurant website and extract summary + menu data.

    If the homepage links to a dedicated menu page, that page is also
    fetched and parsed for menu items.

    When *use_vision* is True, PDF and image menu links are downloaded
    and processed via GPT-4o-mini vision as a final fallback.

    Returns:
        Dict with ``summary``, ``menu_items``, ``menu_url``, and
        ``menu_file_url`` keys, or ``None`` on failure.
    """
    soup = await _fetch_html(client, url)
    if soup is None:
        return None

    summary = _extract_summary(soup)
    menu_url = _find_menu_url(soup, url)
    menu_file_url = _find_menu_file_url(soup, url)

    # Try extracting menu items from the homepage first
    menu_items = _extract_menu_items_from_soup(soup)

    # If we found a separate menu page and didn't get items from homepage,
    # fetch and parse the menu page too
    if menu_url and menu_url != url and len(menu_items) < 3:
        await asyncio.sleep(1)  # polite delay before second request
        menu_soup = await _fetch_html(client, menu_url)
        if menu_soup is not None:
            menu_page_items = _extract_menu_items_from_soup(menu_soup)
            if len(menu_page_items) > len(menu_items):
                menu_items = menu_page_items

    # --- Vision fallback for PDF/image menus ---
    if use_vision and len(menu_items) < 3 and menu_file_url:
        from scraper.menu_vision import extract_menu_from_file_url

        logger.info("  → Vision extraction for %s", menu_file_url)
        await asyncio.sleep(1)
        try:
            vision_items = await extract_menu_from_file_url(
                menu_file_url, client, api_key=openai_api_key
            )
            if len(vision_items) > len(menu_items):
                menu_items = vision_items
        except Exception:
            logger.debug("Vision extraction failed for %s", menu_file_url, exc_info=True)

    return {
        "summary": summary,
        "menu_items": menu_items,
        "menu_url": menu_url or menu_file_url,
        "menu_file_url": menu_file_url,
    }


async def enrich_restaurants(
    restaurants: list[dict[str, Any]],
    *,
    use_playwright: bool = False,
    use_vision: bool = False,
    openai_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch each restaurant's website and merge extracted data in-place.

    Extracts summaries, menu items, and menu URLs.  Restaurants without
    a ``website`` field are skipped.  A 2-second delay is inserted
    between restaurants to respect rate limits.

    When *use_playwright* is ``True``, a headless Chromium browser is
    launched once and used as a fallback for any restaurant where the
    static BS4 pass found fewer than 3 menu items but a menu URL exists.

    When *use_vision* is ``True``, PDF and image menu links are
    downloaded and processed via GPT-4o-mini vision API. Requires
    an OpenAI API key (via *openai_api_key* or ``OPENAI_API_KEY`` env).

    Args:
        restaurants: List of restaurant dicts (modified in place).
        use_playwright: Enable Playwright JS-rendering fallback.
        use_vision: Enable GPT-4o vision extraction for PDF/image menus.
        openai_api_key: OpenAI API key for vision extraction.

    Returns:
        The same list with enriched data where available.
    """
    total = len(restaurants)
    enriched_count = 0
    menu_count = 0
    pw_count = 0
    vision_count = 0

    # Optionally launch Playwright browser
    browser: Browser | None = None
    pw_context = None
    if use_playwright:
        try:
            from playwright.async_api import async_playwright

            pw_context = async_playwright()
            pw = await pw_context.__aenter__()
            browser = await pw.chromium.launch(headless=True)
            logger.info("Playwright browser launched for JS-rendered menus")
        except Exception:
            logger.warning(
                "Failed to launch Playwright — falling back to BS4 only. "
                "Run 'playwright install chromium' if not installed.",
                exc_info=True,
            )
            browser = None

    try:
        async with httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            for idx, restaurant in enumerate(restaurants, start=1):
                url = restaurant.get("website")
                if not url:
                    continue

                # Skip malformed URLs (no scheme, relative paths, etc.)
                if not url.startswith(("http://", "https://")):
                    logger.debug("Skipping invalid URL: %s", url)
                    continue

                name = restaurant.get("name", "unknown")
                logger.info("Enriching %d/%d: %s", idx, total, name)

                try:
                    result = await scrape_restaurant_website(
                        url,
                        client,
                        use_vision=use_vision,
                        openai_api_key=openai_api_key,
                    )
                except Exception:
                    logger.warning("Unexpected error scraping %s", url, exc_info=True)
                    result = None
                if result is None:
                    continue

                # Merge summary
                if result.get("summary") and not restaurant.get("summary"):
                    restaurant["summary"] = result["summary"]

                # Merge menu URL
                menu_url = result.get("menu_url")
                if menu_url:
                    restaurant["menu_url"] = menu_url

                # Merge menu items from BS4
                menu_items = result.get("menu_items", [])

                # --- Playwright fallback ---
                # If BS4 found <3 items and we have a menu URL, try JS render
                if browser is not None and len(menu_items) < 3 and menu_url:
                    target = menu_url if menu_url != url else url
                    logger.info("  → Playwright fallback for %s", name)
                    try:
                        pw_items = await _scrape_menu_playwright(target, browser)
                    except Exception:
                        logger.debug("Playwright error for %s", target, exc_info=True)
                        pw_items = []
                    if len(pw_items) > len(menu_items):
                        menu_items = pw_items
                        pw_count += 1

                if menu_items:
                    restaurant["menu_items"] = menu_items
                    menu_count += 1
                    # Track if vision was used for this restaurant
                    if result.get("menu_file_url") and use_vision:
                        vision_count += 1

                # Track that we enriched from the website
                sources = restaurant.get("data_sources", [])
                if "website" not in sources:
                    sources.append("website")
                    restaurant["data_sources"] = sources

                enriched_count += 1

                # Polite delay between restaurants
                await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)
    finally:
        # Always clean up the browser
        if browser is not None:
            await browser.close()
        if pw_context is not None:
            await pw_context.__aexit__(None, None, None)

    extra_parts: list[str] = []
    if use_playwright:
        extra_parts.append(f"{pw_count} via Playwright")
    if use_vision:
        extra_parts.append(f"{vision_count} via Vision")
    extra_msg = f" ({', '.join(extra_parts)})" if extra_parts else ""
    logger.info(
        "Website enrichment complete: %d/%d restaurants enriched, "
        "%d with menu items%s",
        enriched_count,
        total,
        menu_count,
        extra_msg,
    )
    return restaurants
