"""Debug: Extract raw embedded data from Google Maps page source.

Google Maps embeds structured data in script tags as JS arrays/objects.
This finds review counts and price data even when not visually displayed.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys

from playwright.async_api import Page, async_playwright

sys.stdout.reconfigure(encoding="utf-8")

TARGETS = [
    ("Gösser Bräu", 47.0669485, 15.43779),
    ("Der Steirer", 47.0695245, 15.4342011),
    ("McDonald's Jakominiplatz Graz", 47.066591, 15.441518),
    ("KFC Graz", 47.0668, 15.4420),
    ("Aiola upstairs", 47.0740742, 15.4375989),
]


async def _accept_cookies(page: Page) -> None:
    for label in ("Accept all", "Alle akzeptieren", "Reject all", "Alle ablehnen"):
        try:
            btn = page.locator(f'button:has-text("{label}")')
            if await btn.count() > 0:
                await btn.first.click()
                await page.wait_for_timeout(2000)
                return
        except Exception:
            continue


async def _dump_raw(page: Page, name: str) -> None:
    """Extract embedded data from page source."""

    search_q = f"{name} Graz".replace(" ", "+")
    url = f"https://www.google.com/maps/search/{search_q}"
    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
    await page.wait_for_timeout(4000)

    # Click first result if on search page
    first = page.locator('div[role="feed"] a[href*="/maps/place/"]')
    if await first.count() > 0:
        await first.first.click()
        await page.wait_for_timeout(4000)

    html = await page.content()

    # --- 1. Find review count patterns in raw HTML ---
    print("\n  === REVIEW COUNT PATTERNS ===")
    # Google often embeds review data as numbers near "rezension/review" text
    for pattern, desc in [
        (r'(\d[\d.,]*)\s*(?:Rezension|review|Bewertung)', "review word"),
        (r'(?:Rezension|review|Bewertung).*?(\d[\d.,]+)', "review word (after)"),
        (r'"userRatingCount"[:\s]*(\d+)', "userRatingCount"),
        (r'"reviewCount"[:\s]*(\d+)', "reviewCount"),
        (r'"ratingCount"[:\s]*(\d+)', "ratingCount"),
        (r'"totalReviews?"[:\s]*(\d+)', "totalReview(s)"),
        (r'"aggregateRating".*?"ratingCount"[:\s]*"?(\d+)', "aggregateRating.ratingCount"),
        (r'"aggregateRating".*?"reviewCount"[:\s]*"?(\d+)', "aggregateRating.reviewCount"),
        (r'\\"reviewCount\\"[:\s]*(\d+)', "escaped reviewCount"),
        (r'\\"ratingCount\\"[:\s]*(\d+)', "escaped ratingCount"),
        (r'"ratingValue"[:\s]*"?([\d.,]+)', "ratingValue"),
    ]:
        matches = re.findall(pattern, html, re.IGNORECASE)
        if matches:
            print(f"    {desc}: {matches[:5]}")

    # --- 2. Find price patterns in raw HTML ---
    print("\n  === PRICE PATTERNS ===")
    for pattern, desc in [
        (r'"priceLevel"[:\s]*"?(\w+)"?', "priceLevel"),
        (r'"price_level"[:\s]*(\d)', "price_level (numeric)"),
        (r'"priciness"[:\s]*(\d)', "priciness"),
        (r'\\"priceLevel\\"[:\s]*\\"?(\w+)', "escaped priceLevel"),
        (r'\\"price_level\\"[:\s]*(\d)', "escaped price_level"),
        (r'PRICE_LEVEL_(\w+)', "PRICE_LEVEL_*"),
        (r'"inexpensive|moderate|expensive|very.expensive"', "price word"),
        (r'["\s](€{1,4}|[\$]{1,4})["\s,\]]', "€/$$ symbols in data"),
    ]:
        matches = re.findall(pattern, html, re.IGNORECASE)
        if matches:
            print(f"    {desc}: {matches[:5]}")

    # --- 3. Find all large number arrays that might be review-related ---
    print("\n  === RATING VALUE IN DATA ===")
    # Google Maps embeds data as arrays: [null,4.4,null,null,1234,...]
    # Look for the rating value we know (to find the right data array)
    # Then check nearby numbers for review count
    for pattern, desc in [
        (r'\[(?:null,)*?(4[.,]\d)(?:,(?:null|\d+))*?\]', "array with rating"),
        (r',4[.,]\d,(\d{2,6}),', "number after 4.x rating"),
        (r'(\d{2,6}),4[.,]\d,', "number before 4.x rating"),
    ]:
        matches = re.findall(pattern, html)
        if matches:
            print(f"    {desc}: {matches[:8]}")

    # --- 4. Search for known structured data formats ---
    print("\n  === JSON-LD / STRUCTURED DATA ===")
    ld_blocks = re.findall(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    for i, block in enumerate(ld_blocks):
        try:
            data = json.loads(block)
            print(f"    JSON-LD block {i}: type={data.get('@type', '?')}")
            if "aggregateRating" in str(data):
                print(f"      aggregateRating: {json.dumps(data.get('aggregateRating', {}))}")
            if "priceRange" in str(data):
                print(f"      priceRange: {data.get('priceRange', '?')}")
        except Exception:
            pass

    # --- 5. Dump interesting data blobs near "review" keyword ---
    print("\n  === DATA CONTEXT AROUND 'review' ===")
    for m in re.finditer(r'review', html, re.IGNORECASE):
        start = max(0, m.start() - 100)
        end = min(len(html), m.end() + 100)
        snippet = html[start:end].replace("\n", " ")
        # Only show if it contains numbers
        if re.search(r'\d{2,}', snippet):
            print(f"    ...{snippet}...")
            print()


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="Europe/Vienna",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        await page.goto(
            "https://www.google.com/maps",
            wait_until="domcontentloaded",
            timeout=20000,
        )
        await page.wait_for_timeout(2000)
        await _accept_cookies(page)

        for idx, (name, lat, lng) in enumerate(TARGETS):
            print(f"\n{'='*70}")
            print(f"[{idx+1}] {name}")
            print(f"{'='*70}")
            await _dump_raw(page, name)
            await asyncio.sleep(2)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
