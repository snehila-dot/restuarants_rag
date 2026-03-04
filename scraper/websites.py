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
_MENU_KEYWORDS = frozenset({
    # English
    "menu", "dishes", "food", "our food", "cuisine",
    "eat", "dine", "a la carte", "à la carte",
    # German
    "speisekarte", "karte", "gerichte", "speisen",
    "essen", "mittagstisch", "angebot", "wochenkarte",
    "tagesmenü", "tagesmenu", "küche",
})

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
    content_len = len(resp.content)
    if "text/html" not in content_type:
        logger.debug(
            "Skipping non-HTML response from %s (content-type: %s)",
            url,
            content_type,
        )
        return None

    logger.debug(
        "Fetched %s — %d bytes, content-type: %s",
        url,
        content_len,
        content_type,
    )
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
            logger.debug("Summary via <meta description> (%d chars)", len(desc))
            return desc[:300]
    for p in soup.find_all("p", limit=10):
        text = p.get_text(strip=True)
        if len(text) >= 40:
            logger.debug("Summary via <p> tag (%d chars)", len(text))
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
            resolved = urljoin(base_url, href)
            logger.debug("Found menu page link: %s", resolved)
            return resolved
    logger.debug("No menu page link found")
    return None


def _find_menu_file_url(soup: BeautifulSoup, base_url: str) -> str | None:
    """Return the absolute URL of the first PDF/image menu link found.
    Detects file links by extension OR by known CMS download patterns
    (e.g. WordPress ``?wpdmdl=``, ``?download=``, ``/download/``).
    """
    # Query-string patterns that indicate a file download
    _download_patterns = ("wpdmdl=", "download=", "/download/", "action=download")
    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"])
        text = a_tag.get_text(strip=True).lower()
        combined = f"{href.lower()} {text}"
        if any(kw in combined for kw in _MENU_KEYWORDS):
            href_lower = href.lower()
            clean_href = href_lower.split("?")[0]
            # Match by file extension
            if any(
                clean_href.endswith(ext)
                for ext in (".pdf", ".jpg", ".jpeg", ".png", ".webp")
            ):
                resolved = urljoin(base_url, href)
                logger.debug("Found menu file link (extension): %s", resolved)
                return resolved
            # Match by CMS download pattern in URL
            if any(pat in href_lower for pat in _download_patterns):
                resolved = urljoin(base_url, href)
                logger.debug("Found menu file link (CMS pattern): %s", resolved)
                return resolved
    logger.debug("No menu file link found")
    return None


_COMMON_MENU_PATHS = [
    "/menu", "/speisekarte", "/karte", "/food",
    "/essen", "/speisen", "/our-menu", "/the-menu",
]


async def _probe_menu_paths(
    client: httpx.AsyncClient,
    base_url: str,
) -> str | None:
    """Try common menu URL paths when no link was found in page HTML.

    Sends HEAD requests to avoid downloading full pages. Returns the
    first path that responds with HTTP 200, or ``None``.
    """
    for path in _COMMON_MENU_PATHS:
        probe_url = urljoin(base_url, path)
        try:
            resp = await client.head(probe_url)
            if resp.status_code == 200:
                logger.debug("Probed menu path found: %s", probe_url)
                return probe_url
        except httpx.RequestError:
            continue
    logger.debug("No common menu paths responded for %s", base_url)
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
    1. **Schema.org / JSON-LD** — ``application/ld+json`` with ``Menu`` or
       ``MenuItem`` types.
    2. **Heuristic HTML parsing** — walk headings (h2/h3 = category) and
       sibling elements looking for dish-name + price patterns.
    Items without a recognisable name are skipped.
    """
    items: list[dict[str, str]] = []
    items = _extract_from_jsonld(soup)
    if items:
        logger.debug("Menu extraction via JSON-LD: %d items", len(items))
        return items
    # --- Strategy 2: Heuristic HTML walk ------------------------------------
    items = _extract_from_html_heuristic(soup)
    if items:
        logger.debug("Menu extraction via HTML heuristic: %d items", len(items))
    else:
        logger.debug("Menu extraction: no items found (JSON-LD and heuristic both empty)")
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
        <h2>Hauptgerichte</h2>
        <div>Wiener Schnitzel ... €14.90</div>
        <div>Tafelspitz ... €18.50</div>
    or table rows, list items, etc.
    """
    items: list[dict[str, str]] = []
    current_category = "Other"
    elements_scanned = 0
    # Collect all headings and the block-level elements between them
    for element in soup.find_all(["h1", "h2", "h3", "h4", "li", "tr", "div", "p"]):
        if not isinstance(element, Tag):
            continue
        tag_name = element.name
        text = element.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            continue

        elements_scanned += 1
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
    dupes = len(items) - len(unique)
    logger.debug(
        "HTML heuristic: scanned %d elements, found %d items (%d duplicates removed)",
        elements_scanned,
        len(unique),
        dupes,
    )
    return unique


# ---------------------------------------------------------------------------
# LLM text extraction
# ---------------------------------------------------------------------------


async def _extract_menu_with_llm(
    visible_text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> list[dict[str, str]]:
    """Send visible page text to LLM for structured menu extraction.

    Delegates to :func:`scraper.menu_vision.parse_menu_text_with_llm`
    which handles the OpenAI API call and JSON parsing.

    Returns an empty list if the text is too short (<50 chars) or if
    the LLM call fails.
    """
    if not visible_text or len(visible_text.strip()) < 50:
        logger.debug("Text too short for LLM extraction (%d chars)", len(visible_text))
        return []
    from scraper.menu_vision import parse_menu_text_with_llm

    # Truncate to stay within token budget (~6000 chars ≈ ~2000 tokens)
    truncated = visible_text[:6000]
    logger.debug("Sending %d chars to LLM for menu extraction", len(truncated))
    return parse_menu_text_with_llm(truncated, api_key, model)


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


async def _scrape_menu_playwright_llm(
    url: str,
    browser: Browser,
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> list[dict[str, str]]:
    """Render *url* with Playwright, extract visible text, parse with LLM.

    Opens a new browser page, waits for JS to finish rendering,
    extracts the visible text content, then sends it to GPT-4o-mini
    for structured menu item extraction.

    Falls back to empty list on any error. Always closes the page.
    """
    page = await browser.new_page()
    try:
        await page.goto(
            url,
            wait_until="networkidle",
            timeout=_PW_NAV_TIMEOUT,
        )
        await page.wait_for_timeout(1500)
        visible_text = await page.inner_text("body")
        if not visible_text or len(visible_text.strip()) < 50:
            logger.debug(
                "Playwright page text too short for %s (%d chars)",
                url,
                len(visible_text or ""),
            )
            return []
        return await _extract_menu_with_llm(visible_text, api_key, model)
    except Exception as exc:
        logger.debug("Playwright+LLM failed for %s: %s", url, exc)
        return []
    finally:
        await page.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def scrape_restaurant_website(
    url: str,
    client: httpx.AsyncClient,
    *,
    browser: Browser | None = None,
    api_key: str | None = None,
    use_vision: bool = False,
) -> dict[str, Any] | None:
    """Fetch a restaurant website and extract summary + menu data.

    Extraction cascade (stops at first success with >=3 items):
    1. JSON-LD structured data (free, instant)
    2. BS4 HTML heuristic on homepage + menu page (free)
    3. Playwright + LLM text parsing (requires *browser* + *api_key*)
    4. Playwright + BS4 heuristic (when no *api_key*, fallback)
    5. Vision: PDF/image files via GPT-4o-mini (requires *use_vision*)
    6. Vision: embedded page images (requires *use_vision*)

    Returns:
        Dict with ``summary``, ``menu_items``, ``menu_url``, and
        ``menu_file_url`` keys, or ``None`` on failure.
    """
    soup = await _fetch_html(client, url)
    if soup is None:
        logger.debug("Homepage fetch failed, skipping: %s", url)
        return None

    summary = _extract_summary(soup)
    menu_url = _find_menu_url(soup, url)
    menu_file_url = _find_menu_file_url(soup, url)

    # Probe common paths if no menu link found in HTML
    if not menu_url and not menu_file_url:
        menu_url = await _probe_menu_paths(client, url)

    # --- Fast path 1: JSON-LD (free, instant) ---
    menu_items = _extract_from_jsonld(soup)
    if len(menu_items) >= 3:
        logger.debug("JSON-LD fast path: %d items", len(menu_items))
        return {
            "summary": summary,
            "menu_items": menu_items,
            "menu_url": menu_url or url,
            "menu_file_url": menu_file_url,
        }

    # --- Fast path 2: BS4 heuristic on homepage (free) ---
    heuristic_items = _extract_from_html_heuristic(soup)
    if len(heuristic_items) > len(menu_items):
        menu_items = heuristic_items

    # --- Fast path 3: BS4 on dedicated menu page (free) ---
    if menu_url and menu_url != url and len(menu_items) < 3:
        logger.debug(
            "Homepage yielded %d items (<3), fetching menu page: %s",
            len(menu_items),
            menu_url,
        )
        await asyncio.sleep(1)
        menu_soup = await _fetch_html(client, menu_url)
        if menu_soup is not None:
            jsonld_items = _extract_from_jsonld(menu_soup)
            if len(jsonld_items) > len(menu_items):
                menu_items = jsonld_items
            if len(menu_items) < 3:
                page_items = _extract_from_html_heuristic(menu_soup)
                if len(page_items) > len(menu_items):
                    menu_items = page_items

    # --- LLM text path: Playwright render + GPT-4o-mini ---
    if browser is not None and api_key and len(menu_items) < 3:
        target = menu_url if menu_url and menu_url != url else url
        logger.info("  -> Playwright+LLM for %s", target)
        llm_items = await _scrape_menu_playwright_llm(
            target, browser, api_key=api_key
        )
        if len(llm_items) > len(menu_items):
            logger.debug(
                "Playwright+LLM improved: %d -> %d items",
                len(menu_items),
                len(llm_items),
            )
            menu_items = llm_items
    # --- Playwright BS4 fallback (no API key, --no-llm mode) ---
    elif browser is not None and not api_key and len(menu_items) < 3:
        target = menu_url if menu_url and menu_url != url else url
        logger.info("  -> Playwright+BS4 for %s", target)
        pw_items = await _scrape_menu_playwright(target, browser)
        if len(pw_items) > len(menu_items):
            menu_items = pw_items

    # --- Vision fallback 1: PDF/image file links ---
    if use_vision and api_key and len(menu_items) < 3 and menu_file_url:
        from scraper.menu_vision import extract_menu_from_file_url

        logger.info("  -> Vision extraction (file) for %s", menu_file_url)
        await asyncio.sleep(1)
        try:
            vision_items = await extract_menu_from_file_url(
                menu_file_url, client, api_key=api_key
            )
            if len(vision_items) > len(menu_items):
                menu_items = vision_items
        except Exception:
            logger.debug(
                "Vision extraction failed for %s", menu_file_url, exc_info=True
            )

    # --- Vision fallback 2: embedded <img> tags on menu page ---
    if use_vision and api_key and len(menu_items) < 3 and menu_url:
        from scraper.menu_vision import extract_menu_from_page_images

        target_soup = None
        target_url = menu_url
        if menu_url and menu_url != url:
            await asyncio.sleep(1)
            target_soup = await _fetch_html(client, menu_url)
        else:
            target_soup = soup
            target_url = url
        if target_soup is not None:
            logger.info("  -> Vision extraction (page images) for %s", target_url)
            try:
                img_items = await extract_menu_from_page_images(
                    target_soup, target_url, client, api_key=api_key
                )
                if len(img_items) > len(menu_items):
                    menu_items = img_items
            except Exception:
                logger.debug(
                    "Page image extraction failed for %s",
                    target_url,
                    exc_info=True,
                )

    logger.debug(
        "Final result: summary=%s, menu_items=%d, menu_url=%s",
        "yes" if summary else "no",
        len(menu_items),
        menu_url or menu_file_url or "none",
    )
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
                    logger.debug(
                        "Skipping %d/%d: %s (no website)",
                        idx,
                        total,
                        restaurant.get("name", "unknown"),
                    )
                    continue
                # Skip malformed URLs (no scheme, relative paths, etc.)
                if not url.startswith(("http://", "https://")):
                    logger.warning(
                        "Skipping %d/%d: %s (invalid URL: %s)",
                        idx,
                        total,
                        restaurant.get("name", "unknown"),
                        url,
                    )
                    continue
                name = restaurant.get("name", "unknown")
                logger.info("Enriching %d/%d: %s", idx, total, name)
                try:
                    result = await scrape_restaurant_website(
                        url,
                        client,
                        browser=browser,
                        api_key=openai_api_key,
                        use_vision=use_vision,
                    )
                except Exception:
                    logger.warning("Unexpected error scraping %s", url, exc_info=True)
                    result = None
                if result is None:
                    logger.info("  [FAIL] %s: fetch failed", name)
                    continue
                # Merge summary
                if result.get("summary") and not restaurant.get("summary"):
                    restaurant["summary"] = result["summary"]
                menu_url = result.get("menu_url")
                if menu_url:
                    restaurant["menu_url"] = menu_url
                menu_items = result.get("menu_items", [])
                if menu_items:
                    restaurant["menu_items"] = menu_items
                    menu_count += 1
                    # Track if vision was used for this restaurant
                    if result.get("menu_file_url") and use_vision:
                        vision_count += 1
                sources = restaurant.get("data_sources", [])
                if "website" not in sources:
                    sources.append("website")
                    restaurant["data_sources"] = sources
                enriched_count += 1

                # Per-restaurant outcome line
                logger.info(
                    "  [OK] %s: summary=%s, menu_items=%d, menu_url=%s",
                    name,
                    "yes" if result.get("summary") else "no",
                    len(menu_items),
                    menu_url or "none",
                )
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
