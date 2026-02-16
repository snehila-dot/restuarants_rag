"""Entry point for the Graz restaurant scraper pipeline.

Usage::

    python -m scraper                                    # OSM data only
    python -m scraper --enrich                           # Also scrape restaurant websites
    python -m scraper --enrich --enrich-js               # + Playwright for JS-rendered menus
    python -m scraper --enrich --enrich-vision           # + GPT-4o vision for PDF/image menus
    python -m scraper --enrich --enrich-js --enrich-vision  # All extraction methods
    python -m scraper --output-dir data/
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scraper.output import write_clean, write_raw
from scraper.overpass import scrape as scrape_overpass
from scraper.parsers import raw_to_restaurant
from scraper.websites import enrich_restaurants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    output_dir: str = "data",
    limit: int = 0,
) -> None:
    """Execute the full scraping pipeline.

    1. Query Overpass API for raw OSM data.
    2. Save raw data.
    3. Parse into restaurant schema dicts.
    4. Deduplicate by name.
    5. (Optional) Enrich from individual restaurant websites.
    6. Save clean data.
    7. Log summary statistics.
    """
    out = Path(output_dir)

    # --- Step 1: Scrape OSM ---------------------------------------------------
    logger.info("Step 1/6: Querying OpenStreetMap Overpass API …")
    raw_elements = await scrape_overpass()
    if not raw_elements:
        logger.warning("No restaurant data returned from Overpass API")
        return

    # --- Step 2: Save raw data ------------------------------------------------
    logger.info("Step 2/6: Writing raw data …")
    write_raw(raw_elements, output_dir=out)

    # --- Step 3: Parse --------------------------------------------------------
    logger.info("Step 3/6: Parsing %d raw elements …", len(raw_elements))
    restaurants = [raw_to_restaurant(el) for el in raw_elements]

    # --- Step 4: Deduplicate --------------------------------------------------
    logger.info("Step 4/6: Deduplicating …")
    restaurants = _deduplicate(restaurants)

    # --- Optional limit (for testing) -----------------------------------------
    if limit > 0:
        logger.info("Limiting to %d restaurants (--limit flag)", limit)
        logger.warning(
            "⚠ Using --limit writes truncated output. "
            "Use --output-dir to avoid overwriting production data."
        )
        restaurants = restaurants[:limit]

    # --- Step 5: Enrich (optional) --------------------------------------------
    if enrich:
        modes = ["BS4"]
        if enrich_js:
            modes.append("Playwright")
        if enrich_vision:
            modes.append("Vision")
        mode = " + ".join(modes)
        logger.info("Step 5/6: Enriching from restaurant websites (%s) …", mode)

        # Resolve OpenAI API key for vision extraction
        openai_key: str | None = None
        if enrich_vision:
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if not openai_key:
                logger.warning(
                    "OPENAI_API_KEY not set — vision extraction will be skipped. "
                    "Set it in .env or export it before running."
                )

        restaurants = await enrich_restaurants(
            restaurants,
            use_playwright=enrich_js,
            use_vision=enrich_vision,
            openai_api_key=openai_key,
        )
    else:
        logger.info("Step 5/6: Skipping website enrichment (use --enrich to enable)")

    # --- Step 6: Save clean data ------------------------------------------------
    logger.info("Step 6/6: Writing clean data …")
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
        "--enrich",
        action="store_true",
        help=("Also scrape individual restaurant websites for summaries" " (slower)."),
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
            "Use GPT-4o-mini vision API to extract menus from PDF and image"
            " files (requires OPENAI_API_KEY env variable)."
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

    if args.enrich_js and not args.enrich:
        parser.error("--enrich-js requires --enrich")
    if args.enrich_vision and not args.enrich:
        parser.error("--enrich-vision requires --enrich")

    asyncio.run(
        run(
            enrich=args.enrich,
            enrich_js=args.enrich_js,
            enrich_vision=args.enrich_vision,
            output_dir=args.output_dir,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
