"""Entry point for the Graz restaurant scraper pipeline.

Usage::

    python -m scraper                                    # OSM data only
    python -m scraper --discover-websites                # + find missing websites via DuckDuckGo
    python -m scraper --enrich                           # + scrape websites (Playwright + LLM + Vision)
    python -m scraper --enrich --no-llm                  # + scrape websites (Playwright + BS4 only, free)
    python -m scraper --enrich --no-vision               # + scrape websites (skip PDF/image vision)
    python -m scraper --enrich --limit 20                # + test on first 20 restaurants
    python -m scraper --google-maps                      # + enrich ratings/price from Google Maps (Playwright)
    python -m scraper --google-places                    # + enrich via Google Places API (recommended)
    python -m scraper --discover-websites --enrich --google-places  # Full pipeline
    python -m scraper --output-dir data/
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scraper.google_maps import enrich_restaurants as enrich_from_google_maps
from scraper.google_places import enrich_restaurants as enrich_from_google_places
from scraper.output import write_clean, write_raw
from scraper.overpass import scrape as scrape_overpass
from scraper.parsers import raw_to_restaurant
from scraper.website_discovery import discover_missing_websites
from scraper.websites import enrich_restaurants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy httpx request/response logs (they log every HTTP call at INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _deduplicate(restaurants: list[dict]) -> list[dict]:
    """Remove duplicates by name (case-insensitive, keep first occurrence)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for r in restaurants:
        key = r["name"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    dupes = len(restaurants) - len(unique)
    if dupes:
        logger.info("Removed %d duplicate entries", dupes)
    return unique


async def run(
    enrich: bool = False,
    enrich_js: bool = False,
    enrich_vision: bool = False,
    no_llm: bool = False,
    no_vision: bool = False,
    discover_websites: bool = False,
    google_maps: bool = False,
    google_places: bool = False,
    output_dir: str = "data",
    limit: int = 0,
) -> None:
    """Execute the full scraping pipeline.

    1. Query Overpass API for raw OSM data.
    2. Save raw data.
    3. Parse into restaurant schema dicts.
    4. Deduplicate by name.
    4b. (Optional) Discover missing websites via DuckDuckGo.
    5. (Optional) Enrich from individual restaurant websites.
    5b. (Optional) Enrich ratings/price/reviews from Google Maps.
    6. Save clean data.
    7. Log summary statistics.
    """
    out = Path(output_dir)

    # --- Step 1: Scrape OSM ---------------------------------------------------
    logger.info("Step 1/7: Querying OpenStreetMap Overpass API …")
    raw_elements = await scrape_overpass()
    if not raw_elements:
        logger.warning("No restaurant data returned from Overpass API")
        return

    # --- Step 2: Save raw data ------------------------------------------------
    logger.info("Step 2/7: Writing raw data …")
    write_raw(raw_elements, output_dir=out)

    # --- Step 3: Parse --------------------------------------------------------
    logger.info("Step 3/7: Parsing %d raw elements …", len(raw_elements))
    restaurants = [raw_to_restaurant(el) for el in raw_elements]

    # --- Step 4: Deduplicate --------------------------------------------------
    logger.info("Step 4/7: Deduplicating …")
    restaurants = _deduplicate(restaurants)

    # --- Optional limit (for testing) -----------------------------------------
    if limit > 0:
        logger.info("Limiting to %d restaurants (--limit flag)", limit)
        logger.warning(
            "[WARNING] Using --limit writes truncated output. "
            "Use --output-dir to avoid overwriting production data."
        )
        restaurants = restaurants[:limit]

    # --- Step 4b: Discover missing websites (optional) -------------------------
    if discover_websites:
        missing = sum(1 for r in restaurants if not r.get("website"))
        logger.info(
            "Step 4b: Discovering missing websites via DuckDuckGo (%d to find) …",
            missing,
        )
        restaurants = await discover_missing_websites(restaurants)
    else:
        missing = sum(1 for r in restaurants if not r.get("website"))
        if missing:
            logger.info(
                "Step 4b: Skipping website discovery (%d without website, "
                "use --discover-websites to find them)",
                missing,
            )

    # --- Step 5: Enrich (optional) --------------------------------------------
    if enrich:
        # Deprecation warnings for old flags
        if enrich_js:
            logger.warning(
                "--enrich-js is deprecated (Playwright is now always-on). "
                "Flag will be ignored."
            )
        if enrich_vision:
            logger.warning(
                "--enrich-vision is deprecated (vision is now on by default). "
                "Use --no-vision to disable. Flag will be ignored."
            )

        # Determine effective settings
        use_llm = not no_llm
        use_vision_flag = not no_vision

        # Resolve OpenAI API key for LLM/vision extraction
        openai_key: str | None = None
        if use_llm or use_vision_flag:
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if not openai_key:
                logger.warning(
                    "OPENAI_API_KEY not set — running with BS4 heuristic only. "
                    "Set OPENAI_API_KEY for LLM-powered extraction."
                )
                use_llm = False
                use_vision_flag = False

        modes = ["Playwright"]
        if use_llm:
            modes.append("LLM")
        else:
            modes.append("BS4-only")
        if use_vision_flag:
            modes.append("Vision")
        mode = " + ".join(modes)
        logger.info("Step 5/7: Enriching from restaurant websites (%s) …", mode)

        restaurants = await enrich_restaurants(
            restaurants,
            use_playwright=True,
            use_vision=use_vision_flag,
            use_llm=use_llm,
            openai_api_key=openai_key,
        )
    else:
        logger.info("Step 5/7: Skipping website enrichment (use --enrich to enable)")

    # --- Step 5b: Google Maps enrichment (optional) ----------------------------
    if google_maps:
        without_rating = sum(1 for r in restaurants if not r.get("rating"))
        logger.info(
            "Step 5b/7: Enriching from Google Maps via Playwright "
            "(%d without rating) …",
            without_rating,
        )
        restaurants = await enrich_from_google_maps(restaurants)
    else:
        without_rating = sum(1 for r in restaurants if not r.get("rating"))
        if without_rating:
            logger.info(
                "Step 5b/7: Skipping Google Maps enrichment "
                "(%d without rating, use --google-maps to enable)",
                without_rating,
            )

    # --- Step 5c: Google Places API enrichment (optional) -------------------------
    if google_places:
        without_rating = sum(1 for r in restaurants if not r.get("rating"))
        without_website = sum(1 for r in restaurants if not r.get("website"))
        logger.info(
            "Step 5c: Enriching from Google Places API "
            "(%d without rating, %d without website) …",
            without_rating,
            without_website,
        )
        restaurants = await enrich_from_google_places(restaurants)
    else:
        logger.info(
            "Step 5c: Skipping Google Places API enrichment "
            "(use --google-places to enable)"
        )

    # --- Step 6: Save clean data ------------------------------------------------
    logger.info("Step 6/7: Writing clean data …")
    write_clean(restaurants, output_dir=out)

    # --- Summary --------------------------------------------------------------
    _log_summary(restaurants)


def _pct(part: int, total: int) -> float:
    return (part / total * 100) if total else 0.0


def _log_summary(restaurants: list[dict]) -> None:
    """Log summary statistics about scraped data."""
    total = len(restaurants)
    with_address = sum(1 for r in restaurants if r.get("address"))
    with_phone = sum(1 for r in restaurants if r.get("phone"))
    with_website = sum(1 for r in restaurants if r.get("website"))
    with_hours = sum(1 for r in restaurants if r.get("opening_hours"))
    with_cuisine = sum(1 for r in restaurants if r.get("cuisine"))
    with_summary = sum(1 for r in restaurants if r.get("summary"))
    with_menu = sum(1 for r in restaurants if r.get("menu_items"))
    with_rating = sum(1 for r in restaurants if r.get("rating"))
    with_reviews = sum(1 for r in restaurants if r.get("review_count", 0) > 0)
    with_real_price = sum(
        1 for r in restaurants if r.get("price_range") and r["price_range"] != "€€"
    )

    logger.info("=" * 50)
    logger.info("SCRAPE SUMMARY")
    logger.info("-" * 50)
    logger.info("Total restaurants: %d", total)
    logger.info(
        "  With address:       %d (%.0f%%)",
        with_address,
        _pct(with_address, total),
    )
    logger.info(
        "  With phone:         %d (%.0f%%)",
        with_phone,
        _pct(with_phone, total),
    )
    logger.info(
        "  With website:       %d (%.0f%%)",
        with_website,
        _pct(with_website, total),
    )
    logger.info(
        "  With opening hours: %d (%.0f%%)",
        with_hours,
        _pct(with_hours, total),
    )
    logger.info(
        "  With cuisine info:  %d (%.0f%%)",
        with_cuisine,
        _pct(with_cuisine, total),
    )
    logger.info(
        "  With rating:        %d (%.0f%%)",
        with_rating,
        _pct(with_rating, total),
    )
    logger.info(
        "  With reviews:       %d (%.0f%%)",
        with_reviews,
        _pct(with_reviews, total),
    )
    logger.info(
        "  With real price:    %d (%.0f%%)",
        with_real_price,
        _pct(with_real_price, total),
    )
    logger.info(
        "  With summary:       %d (%.0f%%)",
        with_summary,
        _pct(with_summary, total),
    )
    logger.info(
        "  With menu items:    %d (%.0f%%)",
        with_menu,
        _pct(with_menu, total),
    )
    logger.info("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Graz restaurant data from OpenStreetMap."
    )
    parser.add_argument(
        "--discover-websites",
        action="store_true",
        help=(
            "Search DuckDuckGo for missing restaurant websites before"
            " enrichment (requires 'pip install duckduckgo-search')."
        ),
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help=("Also scrape individual restaurant websites for summaries (slower)."),
    )
    parser.add_argument(
        "--enrich-js",
        action="store_true",
        help=(
            "Use Playwright headless browser as fallback for JS-rendered"
            " menus (requires 'playwright install chromium')."
        ),
    )
    parser.add_argument(
        "--enrich-vision",
        action="store_true",
        help=(
            "[DEPRECATED] Vision is now on by default. Use --no-vision to disable."
        ),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Skip LLM text parsing (use BS4 heuristic only, free but lower yield)."
        ),
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip vision extraction for PDF/image menus.",
    )
    parser.add_argument(
        "--google-maps",
        action="store_true",
        help=(
            "Enrich restaurants with ratings, review counts, and price levels"
            " from Google Maps via Playwright (slow — ~3s per restaurant)."
        ),
    )
    parser.add_argument(
        "--google-places",
        action="store_true",
        help=(
            "Enrich restaurants with ratings, reviews, price, website, phone,"
            " hours, and features from Google Places API (New)."
            " Requires GOOGLE_PLACES_API_KEY env variable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for JSON files (default: data/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of restaurants to process (0 = all, for testing).",
    )
    args = parser.parse_args()

    if args.no_llm and not args.enrich:
        parser.error("--no-llm requires --enrich")
    if args.no_vision and not args.enrich:
        parser.error("--no-vision requires --enrich")

    asyncio.run(
        run(
            enrich=args.enrich,
            enrich_js=args.enrich_js,
            enrich_vision=args.enrich_vision,
            no_llm=args.no_llm,
            no_vision=args.no_vision,
            discover_websites=args.discover_websites,
            google_maps=args.google_maps,
            google_places=args.google_places,
            output_dir=args.output_dir,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
