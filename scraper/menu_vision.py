"""Extract menu items from PDF and image files using vision LLM.

Provides two extraction strategies:
1. **pdfplumber** — fast, free text extraction for text-based PDFs.
2. **GPT-4o-mini vision** — fallback for scanned PDFs and image menus.

Triggered via ``python -m scraper --enrich --enrich-vision``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30.0

# Supported file extensions by type
_PDF_EXTENSIONS = frozenset({".pdf"})
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

# OpenAI vision extraction prompt
_EXTRACTION_PROMPT = """Extract ALL menu items from this restaurant menu.

Return a JSON array of objects with these fields:
- "name": dish/drink name (string, required)
- "price": price as shown on menu e.g. "€12,90" (string, optional)
- "category": one of: Starter, Main, Dessert, Drink, Pizza, Pasta, Sushi, Soup, Salad, Burger, Other (string)

Rules:
- Extract EVERY item visible on the menu
- Keep original dish names (preserve German/Italian/etc.)
- Include price exactly as displayed (with € symbol)
- If category is unclear, use "Other"
- If price is not visible for an item, set price to ""
- Do NOT make up items that aren't on the menu

Return ONLY the JSON array, no other text. Example:
[{"name": "Wiener Schnitzel", "price": "€14,90", "category": "Main"}]"""


def is_menu_file_url(url: str) -> bool:
    """Check if a URL points to a PDF or image file."""
    lower = url.lower().split("?")[0]  # Strip query params
    return any(lower.endswith(ext) for ext in _PDF_EXTENSIONS | _IMAGE_EXTENSIONS)


def _is_pdf_url(url: str) -> bool:
    lower = url.lower().split("?")[0]
    return any(lower.endswith(ext) for ext in _PDF_EXTENSIONS)


def _is_image_url(url: str) -> bool:
    lower = url.lower().split("?")[0]
    return any(lower.endswith(ext) for ext in _IMAGE_EXTENSIONS)


# ---------------------------------------------------------------------------
# PDF text extraction (pdfplumber — free, fast)
# ---------------------------------------------------------------------------


def _extract_text_from_pdf(pdf_bytes: bytes) -> str | None:
    """Extract text from a PDF using pdfplumber.

    Returns concatenated text from all pages, or None if extraction
    fails or yields no meaningful text.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber not installed — skipping text extraction")
        return None

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            full_text = "\n".join(pages_text).strip()
            # Only useful if we got a reasonable amount of text
            if len(full_text) < 20:
                return None
            return full_text
    except Exception as exc:
        logger.debug("pdfplumber extraction failed: %s", exc)
        return None


def _parse_menu_text_with_llm(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> list[dict[str, str]]:
    """Send extracted PDF text to LLM for structured menu parsing.

    Uses the chat completions API (non-vision) since we already
    have text — cheaper and faster than vision.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Here is the menu text:\n\n{text[:8000]}",
                },
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        content = response.choices[0].message.content or "[]"
        return _parse_json_response(content)
    except Exception as exc:
        logger.warning("LLM text parsing failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Vision extraction (GPT-4o-mini — for images and scanned PDFs)
# ---------------------------------------------------------------------------


def _pdf_to_images(pdf_bytes: bytes, max_pages: int = 5) -> list[bytes]:
    """Convert PDF pages to PNG images for vision API.

    Uses pdfplumber to render pages. Returns a list of PNG byte
    buffers, limited to *max_pages* to control cost.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber required for PDF-to-image conversion")
        return []

    images: list[bytes] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:max_pages]:
                img = page.to_image(resolution=200)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                images.append(buf.getvalue())
    except Exception as exc:
        logger.debug("PDF-to-image conversion failed: %s", exc)
    return images


def _extract_menu_from_image(
    image_bytes: bytes,
    api_key: str,
    model: str = "gpt-4o-mini",
    mime_type: str = "image/png",
) -> list[dict[str, str]]:
    """Send a single image to GPT-4o-mini vision for menu extraction."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        content = response.choices[0].message.content or "[]"
        return _parse_json_response(content)
    except Exception as exc:
        logger.warning("Vision extraction failed: %s", exc)
        return []


def _extract_menu_from_images(
    image_list: list[bytes],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> list[dict[str, str]]:
    """Extract menu items from multiple images, deduplicate results."""
    all_items: list[dict[str, str]] = []
    seen_names: set[str] = set()

    for img_bytes in image_list:
        items = _extract_menu_from_image(img_bytes, api_key, model)
        for item in items:
            key = item.get("name", "").lower().strip()
            if key and key not in seen_names:
                seen_names.add(key)
                all_items.append(item)

    return all_items


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_json_response(content: str) -> list[dict[str, str]]:
    """Parse LLM JSON response into a list of menu item dicts.

    Handles common issues: markdown code fences, trailing text, etc.
    """
    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON array in the response
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            try:
                data = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                logger.debug("Failed to parse LLM response as JSON")
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    # Validate and clean items
    items: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name or len(name) < 2:
            continue
        items.append(
            {
                "name": name[:200],
                "price": str(entry.get("price", ""))[:50],
                "category": str(entry.get("category", "Other"))[:50],
            }
        )

    return items


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------


async def _download_file(
    client: httpx.AsyncClient,
    url: str,
) -> tuple[bytes, str] | None:
    """Download a file and return (raw bytes, content_type), or None on failure."""
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        return resp.content, content_type
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return None


def _mime_from_url(url: str) -> str:
    """Infer MIME type from URL extension."""
    lower = url.lower().split("?")[0]
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_menu_from_file_url(
    url: str,
    http_client: httpx.AsyncClient,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> list[dict[str, str]]:
    """Download a PDF/image menu and extract structured menu items.

    Strategy:
    1. Download the file.
    2. If PDF → try pdfplumber text extraction first (free).
    3. If text extraction succeeds → send text to LLM (cheaper).
    4. If text extraction fails or yields <3 items → use vision API.
    5. If image → send directly to vision API.

    Args:
        url: URL to a PDF or image file.
        http_client: Async HTTP client for downloading.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: Vision-capable model to use.

    Returns:
        List of ``{"name": …, "price": …, "category": …}`` dicts.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        logger.warning("No OpenAI API key — cannot use vision extraction")
        return []

    result = await _download_file(http_client, url)
    if result is None:
        return []

    file_bytes, content_type = result

    # Cap file size at 20 MB
    if len(file_bytes) > 20 * 1024 * 1024:
        logger.warning("File too large (>20MB), skipping: %s", url)
        return []

    # Detect file type: prefer Content-Type header, fall back to URL extension
    is_pdf = (
        "application/pdf" in content_type
        or file_bytes[:5] == b"%PDF-"
        or _is_pdf_url(url)
    )
    is_image = (
        content_type.startswith("image/")
        or _is_image_url(url)
    )

    if is_pdf:
        logger.debug("Detected PDF content for %s", url)
        return _extract_menu_from_pdf(file_bytes, key, model)
    elif is_image:
        mime = content_type if content_type.startswith("image/") else _mime_from_url(url)
        logger.debug("Detected image content (%s) for %s", mime, url)
        return _extract_menu_from_image(file_bytes, key, model, mime_type=mime)
    else:
        logger.debug("Unsupported content type '%s' for %s", content_type, url)
        return []


def _extract_menu_from_pdf(
    pdf_bytes: bytes,
    api_key: str,
    model: str,
) -> list[dict[str, str]]:
    """Extract menu from PDF: try text first, fall back to vision."""
    # Strategy 1: text extraction (free)
    text = _extract_text_from_pdf(pdf_bytes)
    if text:
        items = _parse_menu_text_with_llm(text, api_key, model)
        if len(items) >= 3:
            logger.debug("PDF text extraction yielded %d items", len(items))
            return items

    # Strategy 2: convert to images and use vision (costs more)
    images = _pdf_to_images(pdf_bytes, max_pages=5)
    if not images:
        # If pdfplumber can't render either, try text result if we got any
        if text:
            return _parse_menu_text_with_llm(text, api_key, model)
        logger.debug("Cannot extract images from PDF")
        return []

    logger.debug("Using vision API for %d PDF page(s)", len(images))
    return _extract_menu_from_images(images, api_key, model)
