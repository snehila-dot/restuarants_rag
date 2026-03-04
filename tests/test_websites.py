"""Tests for scraper.websites menu extraction."""

from __future__ import annotations

import httpx
import pytest
from bs4 import BeautifulSoup

from scraper.websites import (
    _extract_from_html_heuristic,
    _extract_from_jsonld,
    _extract_price,
    _find_menu_file_url,
    _find_menu_url,
    _infer_category,
    _probe_menu_paths,
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
        assert result is None

    def test_returns_none_when_no_menu_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_NO_MENU_LINK, "html.parser")
        result = _find_menu_url(soup, "https://example.com")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Menu file URL discovery
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: Menu URL discovery (expanded keywords)
# ---------------------------------------------------------------------------


class TestFindMenuUrlExpanded:
    def test_finds_essen_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_WITH_FOOD_LINK, "html.parser")
        result = _find_menu_url(soup, "https://example.com")
        assert result == "https://example.com/essen"

    def test_finds_our_food_link(self) -> None:
        soup = BeautifulSoup(HOMEPAGE_WITH_CUISINE_LINK, "html.parser")
        result = _find_menu_url(soup, "https://example.com")
        assert result == "https://example.com/our-food"


# ---------------------------------------------------------------------------
# Tests: Menu path probing
# ---------------------------------------------------------------------------


class TestProbeMenuPaths:
    @pytest.mark.asyncio
    async def test_finds_menu_path(self) -> None:
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/speisekarte"):
                return httpx.Response(200)
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await _probe_menu_paths(client, "https://example.com")
        assert result == "https://example.com/speisekarte"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_paths_found(self) -> None:
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await _probe_menu_paths(client, "https://example.com")
        assert result is None
