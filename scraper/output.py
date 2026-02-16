"""Write scraped restaurant data to JSON files."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_raw(
    restaurants: list[dict[str, Any]],
    output_dir: Path = Path("data"),
) -> Path:
    """Write raw Overpass API response data to ``restaurants_raw.json``.

    Args:
        restaurants: Raw element dicts straight from the Overpass scraper.
        output_dir: Directory to write into (created if missing).

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "restaurants_raw.json"

    payload = {
        "restaurants": restaurants,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": "openstreetmap",
        "count": len(restaurants),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d raw entries → %s", len(restaurants), path)
    return path


def write_clean(
    restaurants: list[dict[str, Any]],
    output_dir: Path = Path("data"),
) -> Path:
    """Write cleaned/parsed restaurant data to ``restaurants.json``.

    This is the file consumed by ``app/seed.py --from-scraped``.

    Args:
        restaurants: Parsed restaurant dicts matching the Restaurant schema.
        output_dir: Directory to write into (created if missing).

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "restaurants.json"

    payload = {
        "restaurants": restaurants,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": "openstreetmap",
        "count": len(restaurants),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d clean entries → %s", len(restaurants), path)
    return path
