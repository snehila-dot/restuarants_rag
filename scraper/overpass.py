"""OpenStreetMap Overpass API scraper for Graz restaurants."""

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass QL: all food/drink amenities within Graz administrative boundary
OVERPASS_QUERY = """
[out:json][timeout:{timeout}];
area["name"="Graz"]["admin_level"="6"]->.graz;
(
  nwr["amenity"~"restaurant|cafe|fast_food|bar|pub|biergarten"](area.graz);
);
out center tags;
""".strip()

# Amenity types we care about
AMENITY_TYPES = frozenset(
    {"restaurant", "cafe", "fast_food", "bar", "pub", "biergarten"}
)


def _extract_coords(element: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract latitude/longitude from an Overpass element.

    Nodes have lat/lon directly. Ways and relations use the 'center'
    field produced by ``out center``.
    """
    if element.get("type") == "node":
        return element.get("lat"), element.get("lon")
    center = element.get("center", {})
    return center.get("lat"), center.get("lon")


_MAX_RETRIES = 3
_RETRY_BACKOFF = [5, 15, 30]  # seconds between retries


async def scrape(timeout: int = 90) -> list[dict[str, Any]]:
    """Query the Overpass API for all food/drink amenities in Graz.

    Retries up to 3 times with exponential backoff on server errors.

    Args:
        timeout: Overpass server-side query timeout in seconds.

    Returns:
        List of raw element dicts with tags and coordinates preserved.
    """
    query = OVERPASS_QUERY.format(timeout=timeout)
    logger.info("Querying Overpass API for Graz restaurants …")

    # Respect Overpass fair-use policy — brief pause before request
    await asyncio.sleep(1)

    resp: httpx.Response | None = None

    for attempt in range(_MAX_RETRIES):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        ) as client:
            try:
                resp = await client.post(
                    OVERPASS_URL,
                    data={"data": query},
                    headers={"User-Agent": "GrazRestaurantChatbot/1.0"},
                )
                resp.raise_for_status()
                break  # Success
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in {429, 504} and attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF[attempt]
                    logger.warning(
                        "Overpass API returned %d — retrying in %ds (attempt %d/%d)",
                        status,
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("Overpass API HTTP error: %s", status)
                raise
            except httpx.RequestError as exc:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF[attempt]
                    logger.warning(
                        "Overpass API request failed: %s — retrying in %ds",
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("Overpass API request failed: %s", exc)
                raise

    if resp is None:
        msg = "Overpass API: all retries exhausted"
        raise RuntimeError(msg)

    data = resp.json()
    elements: list[dict[str, Any]] = data.get("elements", [])

    total = len(elements)
    named = [e for e in elements if e.get("tags", {}).get("name")]
    skipped = total - len(named)

    logger.info(
        "Overpass returned %d elements — %d with name, %d skipped (no name)",
        total,
        len(named),
        skipped,
    )

    # Attach coordinates at top level for convenience
    results: list[dict[str, Any]] = []
    for element in named:
        lat, lon = _extract_coords(element)
        results.append(
            {
                "osm_id": element.get("id"),
                "osm_type": element.get("type"),
                "tags": element.get("tags", {}),
                "latitude": lat,
                "longitude": lon,
            }
        )

    return results
