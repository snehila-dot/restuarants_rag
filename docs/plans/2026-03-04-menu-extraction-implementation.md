# Menu Extraction Pipeline Improvement — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace fragile BS4 heuristic menu extraction with Playwright rendering + GPT-4o-mini text parsing, improving menu item yield from 0% to 40–60% of 711 restaurants.

**Architecture:** Three-tier extraction cascade: (1) free fast paths (JSON-LD, pdfplumber), (2) Playwright + LLM text parsing, (3) GPT-4o-mini vision fallback for PDFs/images. Playwright is always-on. LLM and vision require OPENAI_API_KEY.

**Tech Stack:** Python 3.12+, Playwright (async), OpenAI GPT-4o-mini, httpx, BeautifulSoup4, pdfplumber, pytest + pytest-asyncio

**Design doc:** `docs/plans/2026-03-04-menu-extraction-improvement.md`

---

### Task 1: Fix Windows Unicode in Log Messages

**Files:**
- Modify: `scraper/websites.py:672,718`

**Step 1: Fix the `✗` character (line 672)**

In `scraper/websites.py`, find the log line with `✗`:
```python
                    logger.info("  ✗ %s: fetch failed", name)
```
Replace with:
```python
                    logger.info("  [FAIL] %s: fetch failed", name)
```

**Step 2: Fix the `✓` character (line 718)**

Find the log line with `✓`:
```python
                logger.info(
                    "  ✓ %s: summary=%s, menu_items=%d, menu_url=%s",
```
Replace with:
```python
                logger.info(
                    "  [OK] %s: summary=%s, menu_items=%d, menu_url=%s",
```

**Step 3: Verify no other Unicode issues**

Run: `python -c "import scraper.websites"` on Windows
Expected: No UnicodeEncodeError

**Step 4: Commit**

```bash
git add scraper/websites.py
git commit -m "fix: replace Unicode log symbols with ASCII for Windows compat"
```

---

### Task 2: Create Scraper Test Infrastructure

**Files:**
- Create: `tests/test_websites.py`
- Create: `tests/test_menu_vision.py`

**Step 1: Create `tests/test_websites.py` with fixtures and first test**

```python
"""Tests for scraper.websites menu extraction."""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from scraper.websites import (
    _extract_from_html_heuristic,
    _extract_from_jsonld,
    _extract_menu_items_from_soup,
    _extract_price,
    _find_menu_url,
    _find_menu_file_url,
    _infer_category,
)


# ---------------------------------------------------------------------------
# Fixtures: Sample HTML snippets
# ---------------------------------------------------------------------------

MENU_HTML_WITH_PRICES = """
<html><body>
<h2>Hauptgerichte</h2>
<div>Wiener Schnitzel mit Kartoffelsalat €14,90</div>
<div>Tafelspitz mit Semmelkren €18,50</div>
<div>Rindsgulasch mit Nockerln €13,90</div>
<h2>Desserts</h2>
<div>Sachertorte €6,50</div>
<div>Apfelstrudel mit Vanillesoße €7,90</div>
</body></html>
"""

MENU_HTML_JSONLD = """
<html><head>
<script type="application/ld+json">
{
  "@type": "Menu",
  "hasMenuSection": [
    {
      "@type": "MenuSection",
      "name": "Mains",
      "hasMenuItem": [
        {"@type": "MenuItem", "name": "Schnitzel", "offers": {"price": "14.90", "priceCurrency": "€"}},
        {"@type": "MenuItem", "name": "Gulasch", "offers": {"price": "12.50", "priceCurrency": "€"}}
      ]
    }
  ]
}
</script>
</head><body></body></html>
"""

HOMEPAGE_WITH_MENU_LINK = """
<html><body>
<nav>
  <a href="/about">About</a>
  <a href="/speisekarte">Speisekarte</a>
  <a href="/contact">Contact</a>
</nav>
</body></html>
"""

HOMEPAGE_WITH_PDF_LINK = """
<html><body>
<nav>
  <a href="/menu.pdf">Download Menu</a>
</nav>
</body></html>
"""

HOMEPAGE_NO_MENU_LINK = """
<html><body>
<nav>
  <a href="/about">About</a>
  <a href="/contact">Contact</a>
</nav>
</body></html>
"""


# ---------------------------------------------------------------------------
# Tests: Price extraction
# ---------------------------------------------------------------------------


class TestExtractPrice:
    def test_euro_prefix(self) -> None:
        assert _extract_price("Schnitzel €14,90") == "€14,90"

    def test_euro_suffix(self) -> None:
        assert _extract_price("Schnitzel 14,90€") == "14,90€"

    def test_euro_with_dot(self) -> None:
        assert _extract_price("Schnitzel €14.90") == "€14.90"

    def test_eur_prefix(self) -> None:
        assert _extract_price("Schnitzel EUR 14.90") == "EUR 14.90"

    def test_no_price(self) -> None:
        assert _extract_price("Just a description") is None


# ---------------------------------------------------------------------------
# Tests: Category inference
# ---------------------------------------------------------------------------


class TestInferCategory:
    def test_main_course(self) -> None:
        assert _infer_category("Hauptgerichte") == "Main"

    def test_dessert(self) -> None:
        assert _infer_category("Desserts & Süßes") == "Dessert"

    def test_drinks(self) -> None:
        assert _infer_category("Getränke") == "Drink"

    def test_unknown(self) -> None:
        assert _infer_category("Specials") == "Other"


# ---------------------------------------------------------------------------
# Tests: Menu URL discovery
# ---------------------------------------------------------------------------


class TestFindMenuUrl:
    def test_finds_speisekarte_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_WITH_MENU_LINK, "html.parser")
        result = _find_menu_url(soup, "https://example.com")
        assert result == "https://example.com/speisekarte"

    def test_skips_pdf_links(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_WITH_PDF_LINK, "html.parser")
        result = _find_menu_url(soup, "https://example.com")
        assert result is None  # PDF links handled by _find_menu_file_url

    def test_returns_none_when_no_menu_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_NO_MENU_LINK, "html.parser")
        result = _find_menu_url(soup, "https://example.com")
        assert result is None


class TestFindMenuFileUrl:
    def test_finds_pdf_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_WITH_PDF_LINK, "html.parser")
        result = _find_menu_file_url(soup, "https://example.com")
        assert result == "https://example.com/menu.pdf"

    def test_returns_none_when_no_file_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_NO_MENU_LINK, "html.parser")
        result = _find_menu_file_url(soup, "https://example.com")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: JSON-LD extraction
# ---------------------------------------------------------------------------


class TestExtractFromJsonld:
    def test_extracts_menu_items(self) -> None:
        soup = BeautifulSoup(MENU_HTML_JSONLD, "html.parser")
        items = _extract_from_jsonld(soup)
        assert len(items) == 2
        assert items[0]["name"] == "Schnitzel"
        assert items[0]["price"] == "€14.90"
        assert items[0]["category"] == "Mains"

    def test_empty_when_no_jsonld(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_NO_MENU_LINK, "html.parser")
        items = _extract_from_jsonld(soup)
        assert items == []


# ---------------------------------------------------------------------------
# Tests: HTML heuristic extraction
# ---------------------------------------------------------------------------


class TestExtractFromHtmlHeuristic:
    def test_extracts_items_with_prices(self) -> None:
        soup = BeautifulSoup(MENU_HTML_WITH_PRICES, "html.parser")
        items = _extract_from_html_heuristic(soup)
        assert len(items) == 5
        names = [i["name"] for i in items]
        assert "Wiener Schnitzel mit Kartoffelsalat" in names

    def test_assigns_categories_from_headings(self) -> None:
        soup = BeautifulSoup(MENU_HTML_WITH_PRICES, "html.parser")
        items = _extract_from_html_heuristic(soup)
        mains = [i for i in items if i["category"] == "Main"]
        desserts = [i for i in items if i["category"] == "Dessert"]
        assert len(mains) == 3
        assert len(desserts) == 2

    def test_empty_when_no_prices(self) -> None:
        html = "<html><body><div>Schnitzel</div><div>Gulasch</div></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        items = _extract_from_html_heuristic(soup)
        assert items == []
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_websites.py -v`
Expected: All tests PASS (testing existing code, no changes needed)

**Step 3: Create `tests/test_menu_vision.py` stub**

```python
"""Tests for scraper.menu_vision extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scraper.menu_vision import _parse_json_response, is_menu_file_url


class TestIsMenuFileUrl:
    def test_pdf_url(self) -> None:
        assert is_menu_file_url("https://example.com/menu.pdf") is True

    def test_jpg_url(self) -> None:
        assert is_menu_file_url("https://example.com/menu.jpg") is True

    def test_html_url(self) -> None:
        assert is_menu_file_url("https://example.com/menu") is False

    def test_pdf_with_query_params(self) -> None:
        assert is_menu_file_url("https://example.com/menu.pdf?v=2") is True


class TestParseJsonResponse:
    def test_valid_json_array(self) -> None:
        result = _parse_json_response('[{"name": "Schnitzel", "price": "€14", "category": "Main"}]')
        assert len(result) == 1
        assert result[0]["name"] == "Schnitzel"

    def test_empty_array(self) -> None:
        result = _parse_json_response("[]")
        assert result == []

    def test_invalid_json(self) -> None:
        result = _parse_json_response("not json")
        assert result == []
```

**Step 4: Run tests**

Run: `pytest tests/test_menu_vision.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/test_websites.py tests/test_menu_vision.py
git commit -m "test: add unit tests for scraper menu extraction"
```

---

### Task 3: Expand Menu URL Discovery

**Files:**
- Modify: `scraper/websites.py:35-37` (expand `_MENU_KEYWORDS`)
- Modify: `scraper/websites.py:158-175` (after `_find_menu_url`, add `_probe_menu_paths`)
- Modify: `tests/test_websites.py` (add tests)

**Step 1: Write failing tests for new keywords**

Add to `tests/test_websites.py`:

```python
HOMEPAGE_WITH_FOOD_LINK = """
<html><body>
<nav>
  <a href="/about">About</a>
  <a href="/essen">Essen & Trinken</a>
</nav>
</body></html>
"""

HOMEPAGE_WITH_CUISINE_LINK = """
<html><body>
<nav>
  <a href="/our-food">Our Food</a>
</nav>
</body></html>
"""


class TestFindMenuUrlExpanded:
    def test_finds_essen_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_WITH_FOOD_LINK, "html.parser")
        result = _find_menu_url(soup, "https://example.com")
        assert result == "https://example.com/essen"

    def test_finds_our_food_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_WITH_CUISINE_LINK, "html.parser")
        result = _find_menu_url(soup, "https://example.com")
        assert result == "https://example.com/our-food"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_websites.py::TestFindMenuUrlExpanded -v`
Expected: FAIL — "essen" and "our food" not in current `_MENU_KEYWORDS`

**Step 3: Expand `_MENU_KEYWORDS` in `scraper/websites.py`**

Replace lines 35-37:
```python
_MENU_KEYWORDS = frozenset(
    {"menu", "speisekarte", "karte", "gerichte", "dishes", "speisen"}
)
```
With:
```python
_MENU_KEYWORDS = frozenset({
    # English
    "menu", "dishes", "food", "our food", "cuisine",
    "eat", "dine", "a la carte", "à la carte",
    # German
    "speisekarte", "karte", "gerichte", "speisen",
    "essen", "mittagstisch", "angebot", "wochenkarte",
    "tagesmenü", "tagesmenu", "küche",
})
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_websites.py -v`
Expected: All PASS

**Step 5: Add `_probe_menu_paths()` function**

Add after `_find_menu_url()` (after line 175) in `scraper/websites.py`:

```python
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
```

**Step 6: Write test for `_probe_menu_paths`**

Add to `tests/test_websites.py`:

```python
import httpx
import pytest

from scraper.websites import _probe_menu_paths


class TestProbeMenuPaths:
    @pytest.mark.asyncio
    async def test_finds_menu_path(self) -> None:
        async def mock_head(url: str) -> httpx.Response:
            if url.endswith("/speisekarte"):
                return httpx.Response(200)
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_head))
        async with client:
            result = await _probe_menu_paths(client, "https://example.com")
        assert result == "https://example.com/speisekarte"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_paths_found(self) -> None:
        async def mock_head(url: str) -> httpx.Response:
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_head))
        async with client:
            result = await _probe_menu_paths(client, "https://example.com")
        assert result is None
```

**Step 7: Run all tests**

Run: `pytest tests/test_websites.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add scraper/websites.py tests/test_websites.py
git commit -m "feat: expand menu URL discovery keywords and add path probing"
```

---

### Task 4: Make LLM Text Parser Public in `menu_vision.py`

**Files:**
- Modify: `scraper/menu_vision.py:98-129` (rename function)
- Modify: `tests/test_menu_vision.py` (add test)

**Step 1: Write test for the public function**

Add to `tests/test_menu_vision.py`:

```python
class TestParseMenuTextWithLlm:
    @patch("scraper.menu_vision.OpenAI")
    def test_returns_parsed_items(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='[{"name": "Schnitzel", "price": "€14,90", "category": "Main"}]'
                    )
                )
            ]
        )
        from scraper.menu_vision import parse_menu_text_with_llm

        result = parse_menu_text_with_llm("Schnitzel €14,90", api_key="test-key")
        assert len(result) == 1
        assert result[0]["name"] == "Schnitzel"

    @patch("scraper.menu_vision.OpenAI")
    def test_returns_empty_on_error(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        from scraper.menu_vision import parse_menu_text_with_llm

        result = parse_menu_text_with_llm("some text", api_key="test-key")
        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_menu_vision.py::TestParseMenuTextWithLlm -v`
Expected: FAIL — `parse_menu_text_with_llm` not importable (still private `_parse_menu_text_with_llm`)

**Step 3: Rename function in `scraper/menu_vision.py`**

Rename `_parse_menu_text_with_llm` → `parse_menu_text_with_llm` (remove leading underscore).

In `scraper/menu_vision.py`, change the function definition at line 98:
```python
def parse_menu_text_with_llm(
```

Also update ALL internal callers of this function within `menu_vision.py` — search for `_parse_menu_text_with_llm` and replace with `parse_menu_text_with_llm`.

**Step 4: Run tests**

Run: `pytest tests/test_menu_vision.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add scraper/menu_vision.py tests/test_menu_vision.py
git commit -m "refactor: make parse_menu_text_with_llm public for cross-module use"
```

---

### Task 5: Add LLM Text Extraction to `websites.py`

**Files:**
- Modify: `scraper/websites.py` (add `_extract_menu_with_llm` function)
- Modify: `tests/test_websites.py` (add test)

**Step 1: Write failing test**

Add to `tests/test_websites.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch


class TestExtractMenuWithLlm:
    @pytest.mark.asyncio
    @patch("scraper.menu_vision.OpenAI")
    async def test_returns_items_from_llm(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='[{"name": "Gulasch", "price": "€12,50", "category": "Main"}]'
                    )
                )
            ]
        )
        from scraper.websites import _extract_menu_with_llm

        result = await _extract_menu_with_llm(
            "Gulasch €12,50\nSchnitzel €14,90",
            api_key="test-key",
        )
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_returns_empty_for_short_text(self) -> None:
        from scraper.websites import _extract_menu_with_llm

        result = await _extract_menu_with_llm("hi", api_key="test-key")
        assert result == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_websites.py::TestExtractMenuWithLlm -v`
Expected: FAIL — `_extract_menu_with_llm` does not exist

**Step 3: Implement `_extract_menu_with_llm` in `scraper/websites.py`**

Add after the `_extract_from_html_heuristic` function (around line 391):

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_websites.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add scraper/websites.py tests/test_websites.py
git commit -m "feat: add LLM text extraction function to websites scraper"
```

---

### Task 6: Rewrite Playwright Extraction to Use LLM

**Files:**
- Modify: `scraper/websites.py:429-441` (rewrite `_scrape_menu_playwright`)
- Modify: `tests/test_websites.py` (add test — mocked, no real browser)

**Step 1: Write test with mocked Playwright**

Add to `tests/test_websites.py`:

```python
class TestScrapeMenuPlaywrightLlm:
    @pytest.mark.asyncio
    @patch("scraper.menu_vision.OpenAI")
    async def test_extracts_via_llm_from_rendered_text(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='[{"name": "Pizza Margherita", "price": "€9,50", "category": "Pizza"}]'
                    )
                )
            ]
        )
        # Mock Playwright browser and page
        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "Pizza Margherita €9,50\nPizza Salami €10,50"
        mock_browser = AsyncMock()
        mock_browser.new_page.return_value = mock_page

        from scraper.websites import _scrape_menu_playwright_llm

        result = await _scrape_menu_playwright_llm(
            "https://example.com/menu", mock_browser, api_key="test-key"
        )
        assert len(result) >= 1
        mock_page.goto.assert_called_once()
        mock_page.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_on_short_text(self) -> None:
        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "hi"
        mock_browser = AsyncMock()
        mock_browser.new_page.return_value = mock_page

        from scraper.websites import _scrape_menu_playwright_llm

        result = await _scrape_menu_playwright_llm(
            "https://example.com/menu", mock_browser, api_key="test-key"
        )
        assert result == []
        mock_page.close.assert_called_once()  # page always closed
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_websites.py::TestScrapeMenuPlaywrightLlm -v`
Expected: FAIL — `_scrape_menu_playwright_llm` does not exist

**Step 3: Add `_scrape_menu_playwright_llm` to `scraper/websites.py`**

Add after the existing `_scrape_menu_playwright` function (keep old one for `--no-llm` fallback):

```python
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
            logger.debug("Playwright page text too short for %s (%d chars)", url, len(visible_text or ""))
            return []
        return await _extract_menu_with_llm(visible_text, api_key, model)
    except Exception as exc:
        logger.debug("Playwright+LLM failed for %s: %s", url, exc)
        return []
    finally:
        await page.close()
```

**Step 4: Run tests**

Run: `pytest tests/test_websites.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add scraper/websites.py tests/test_websites.py
git commit -m "feat: add Playwright + LLM menu extraction function"
```

---

### Task 7: Update `scrape_restaurant_website()` Extraction Cascade

**Files:**
- Modify: `scraper/websites.py:449-575` (rewrite function signature + body)
- Modify: `tests/test_websites.py` (integration test with mocks)

**Step 1: Write integration test**

Add to `tests/test_websites.py`:

```python
class TestScrapeRestaurantWebsite:
    @pytest.mark.asyncio
    async def test_extracts_jsonld_without_api_key(self) -> None:
        """JSON-LD fast path works without OpenAI API key."""

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=MENU_HTML_JSONLD,
            )

        from scraper.websites import scrape_restaurant_website

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await scrape_restaurant_website(
                "https://example.com", client
            )
        assert result is not None
        assert len(result["menu_items"]) == 2

    @pytest.mark.asyncio
    async def test_probes_menu_paths_when_no_link(self) -> None:
        """When no menu link found, probes common paths."""
        call_urls: list[str] = []

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            call_urls.append(str(request.url))
            if request.method == "HEAD" and str(request.url).endswith("/speisekarte"):
                return httpx.Response(200)
            if request.method == "HEAD":
                return httpx.Response(404)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=HOMEPAGE_NO_MENU_LINK,
            )

        from scraper.websites import scrape_restaurant_website

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await scrape_restaurant_website(
                "https://example.com", client
            )
        assert result is not None
        assert result.get("menu_url") == "https://example.com/speisekarte"
```

**Step 2: Run tests to verify current behavior**

Run: `pytest tests/test_websites.py::TestScrapeRestaurantWebsite -v`
Expected: First test may PASS (JSON-LD already works), second test FAIL (no path probing yet)

**Step 3: Update `scrape_restaurant_website()` signature and body**

Modify the function in `scraper/websites.py` (line 449 onwards).

New signature:
```python
async def scrape_restaurant_website(
    url: str,
    client: httpx.AsyncClient,
    *,
    browser: Browser | None = None,
    api_key: str | None = None,
    use_vision: bool = False,
) -> dict[str, Any] | None:
```

New body (replace lines 468-575):
```python
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

    # --- Fast path 3: BS4 on menu page (free) ---
    if menu_url and menu_url != url and len(menu_items) < 3:
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
    openai_api_key = api_key
    if use_vision and openai_api_key and len(menu_items) < 3 and menu_file_url:
        from scraper.menu_vision import extract_menu_from_file_url

        logger.info("  -> Vision extraction (file) for %s", menu_file_url)
        await asyncio.sleep(1)
        try:
            vision_items = await extract_menu_from_file_url(
                menu_file_url, client, api_key=openai_api_key
            )
            if len(vision_items) > len(menu_items):
                menu_items = vision_items
        except Exception:
            logger.debug(
                "Vision extraction failed for %s", menu_file_url, exc_info=True
            )

    # --- Vision fallback 2: embedded <img> tags on menu page ---
    if use_vision and openai_api_key and len(menu_items) < 3 and menu_url:
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
                    target_soup, target_url, client, api_key=openai_api_key
                )
                if len(img_items) > len(menu_items):
                    menu_items = img_items
            except Exception:
                logger.debug(
                    "Page image extraction failed for %s",
                    target_url,
                    exc_info=True,
                )

    return {
        "summary": summary,
        "menu_items": menu_items,
        "menu_url": menu_url or menu_file_url,
        "menu_file_url": menu_file_url,
    }
```

**Step 4: Run tests**

Run: `pytest tests/test_websites.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add scraper/websites.py tests/test_websites.py
git commit -m "feat: rewrite scrape_restaurant_website with extraction cascade"
```

---

### Task 8: Update `enrich_restaurants()` — Playwright Always-On

**Files:**
- Modify: `scraper/websites.py:578-747` (update `enrich_restaurants`)

**Step 1: Update function signature**

Change signature from:
```python
async def enrich_restaurants(
    restaurants: list[dict[str, Any]],
    *,
    use_playwright: bool = False,
    use_vision: bool = False,
    openai_api_key: str | None = None,
) -> list[dict[str, Any]]:
```
To:
```python
async def enrich_restaurants(
    restaurants: list[dict[str, Any]],
    *,
    use_playwright: bool = True,
    use_vision: bool = True,
    use_llm: bool = True,
    openai_api_key: str | None = None,
) -> list[dict[str, Any]]:
```

**Step 2: Update Playwright launch logic**

Playwright is always launched (default `True`). Only skip if explicitly `False`.

**Step 3: Pass `browser` and `api_key` to `scrape_restaurant_website`**

Update the call at line 662:
```python
                    result = await scrape_restaurant_website(
                        url,
                        client,
                        browser=browser if use_llm else None,
                        api_key=openai_api_key if use_llm else None,
                        use_vision=use_vision,
                    )
```

When `use_llm=False`, `browser` is still passed but `api_key` is None — so the function will fall back to BS4 heuristic via the Playwright+BS4 path.

Update the `browser` usage to also support BS4-only Playwright:
```python
                    result = await scrape_restaurant_website(
                        url,
                        client,
                        browser=browser,
                        api_key=openai_api_key if use_llm else None,
                        use_vision=use_vision,
                    )
```

**Step 4: Remove the separate Playwright fallback block**

The old block at lines 681-704 that did `_scrape_menu_playwright` separately is no longer needed — it's now inside `scrape_restaurant_website`.

Remove this entire block:
```python
                # If BS4 found <3 items and we have a menu URL, try JS render
                bs4_count = len(menu_items)
                if browser is not None and bs4_count < 3 and menu_url:
                    ...
```

**Step 5: Update summary log to include LLM stats**

Add `llm_count` tracking alongside `pw_count` and `vision_count`.

**Step 6: Run all tests**

Run: `pytest -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add scraper/websites.py
git commit -m "feat: make Playwright always-on, add use_llm parameter"
```

---

### Task 9: Update CLI Flags in `__main__.py`

**Files:**
- Modify: `scraper/__main__.py` (add `--no-llm`, `--no-vision`, `--limit` flags)

**Step 1: Add new argument flags**

After the existing `--enrich-vision` arg, add:
```python
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM text parsing (BS4 heuristic only, free).",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip vision extraction for PDF/image menus.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit enrichment to first N restaurants (0 = all).",
    )
```

**Step 2: Update the `enrich_restaurants` call**

Find where `enrich_restaurants` is called and update:
```python
        restaurants_to_enrich = restaurants
        if args.limit > 0:
            restaurants_to_enrich = restaurants[:args.limit]
            logger.info("Limiting enrichment to first %d restaurants", args.limit)

        await enrich_restaurants(
            restaurants_to_enrich,
            use_playwright=True,  # always on
            use_vision=not args.no_vision,
            use_llm=not args.no_llm,
            openai_api_key=api_key,
        )
```

**Step 3: Handle deprecated flags gracefully**

When `--enrich-js` is passed, log a deprecation notice:
```python
    if args.enrich_js:
        logger.warning("--enrich-js is deprecated (Playwright is now always-on)")
```

**Step 4: Validate API key requirement**

```python
    if args.enrich and not args.no_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set — running with --no-llm (BS4 heuristic only). "
                "Set OPENAI_API_KEY for LLM-powered extraction."
            )
            args.no_llm = True
```

**Step 5: Run tests**

Run: `pytest -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add scraper/__main__.py
git commit -m "feat: add --no-llm, --no-vision, --limit CLI flags for enrichment"
```

---

### Task 10: Integration Test on Sample Data

**Files:** None modified — this is a manual validation step.

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS

**Step 2: Run ruff lint**

Run: `ruff check scraper/ tests/`
Expected: Clean (no errors)

**Step 3: Test pipeline on 5 restaurants (dry run)**

Run: `python -m scraper --enrich --limit 5`

Expected output pattern:
```
[INFO] Enriching 1/5: <name>
  -> Playwright+LLM for <url>
  [OK] <name>: summary=yes, menu_items=N, menu_url=<url>
...
Website enrichment complete: X/5 restaurants enriched, Y with menu items
```

**Step 4: Evaluate results**

Check `data/restaurants.json` for the first 5 restaurants:
- Do they have `menu_items` arrays with items?
- Are items structured correctly (`name`, `price`, `category`)?
- Are prices in expected format (€ prefix/suffix)?

**Step 5: Test --no-llm fallback**

Run: `python -m scraper --enrich --limit 5 --no-llm`
Expected: Pipeline works but uses BS4 heuristic only (fewer results)

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat: menu extraction pipeline v2 — Playwright + LLM text parsing"
```

---

## Summary of All Tasks

| # | Task | Files | Commit Message |
|---|------|-------|----------------|
| 1 | Fix Windows Unicode | `websites.py` | `fix: replace Unicode log symbols with ASCII` |
| 2 | Test infrastructure | `test_websites.py`, `test_menu_vision.py` | `test: add unit tests for scraper menu extraction` |
| 3 | Expand URL discovery | `websites.py`, `test_websites.py` | `feat: expand menu URL discovery keywords and path probing` |
| 4 | Public LLM parser | `menu_vision.py`, `test_menu_vision.py` | `refactor: make parse_menu_text_with_llm public` |
| 5 | LLM text extraction | `websites.py`, `test_websites.py` | `feat: add LLM text extraction function` |
| 6 | Playwright+LLM | `websites.py`, `test_websites.py` | `feat: add Playwright + LLM menu extraction` |
| 7 | Extraction cascade | `websites.py`, `test_websites.py` | `feat: rewrite scrape_restaurant_website cascade` |
| 8 | Playwright always-on | `websites.py` | `feat: make Playwright always-on, add use_llm param` |
| 9 | CLI flags | `__main__.py` | `feat: add --no-llm, --no-vision, --limit CLI flags` |
| 10 | Integration test | — | `feat: menu extraction pipeline v2` |
