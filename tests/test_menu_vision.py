"""Tests for scraper.menu_vision extraction."""

from __future__ import annotations

from scraper.menu_vision import _parse_json_response, is_menu_file_url


# ---------------------------------------------------------------------------
# Tests: File URL detection
# ---------------------------------------------------------------------------


class TestIsMenuFileUrl:
    def test_pdf_url(self) -> None:
        assert is_menu_file_url("https://example.com/menu.pdf") is True

    def test_jpg_url(self) -> None:
        assert is_menu_file_url("https://example.com/menu.jpg") is True

    def test_html_url(self) -> None:
        assert is_menu_file_url("https://example.com/menu") is False

    def test_pdf_with_query_params(self) -> None:
        assert is_menu_file_url("https://example.com/menu.pdf?v=2") is True

    def test_png_url(self) -> None:
        assert is_menu_file_url("https://example.com/menu.png") is True

    def test_webp_url(self) -> None:
        assert is_menu_file_url("https://example.com/menu.webp") is True


# ---------------------------------------------------------------------------
# Tests: JSON response parsing
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_valid_json_array(self) -> None:
        result = _parse_json_response(
            '[{"name": "Schnitzel", "price": "€14", "category": "Main"}]'
        )
        assert len(result) == 1
        assert result[0]["name"] == "Schnitzel"

    def test_empty_array(self) -> None:
        result = _parse_json_response("[]")
        assert result == []

    def test_invalid_json(self) -> None:
        result = _parse_json_response("not json at all")
        assert result == []

    def test_json_with_markdown_fences(self) -> None:
        result = _parse_json_response(
            '```json\n[{"name": "Gulasch", "price": "€12", "category": "Main"}]\n```'
        )
        assert len(result) == 1
        assert result[0]["name"] == "Gulasch"

    def test_skips_items_without_name(self) -> None:
        result = _parse_json_response(
            '[{"name": "", "price": "€5", "category": "Other"}]'
        )
        assert result == []
