"""Reverse geocode restaurants with bare addresses via Nominatim.

Restaurants scraped from OpenStreetMap sometimes lack ``addr:*`` tags,
resulting in a generic ``"Graz"`` address.  This module fills those gaps
by reverse-geocoding the restaurant's lat/lng coordinates through the
free `Nominatim API <https://nominatim.org/>`_.

Triggered via ``python -m scraper --geocode`` or called from the
pipeline after website enrichment.

.. note::

   Nominatim's usage policy requires **max 1 request per second** and a
   custom ``User-Agent`` header.  All requests are therefore sequential
   with a 1.1 s delay between calls.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_USER_AGENT = "GrazRestaurantChatbot/1.0 (graz-restaurant-scraper)"
_ACCEPT_LANGUAGE = "de,en;q=0.9"
_REQUEST_TIMEOUT = 10.0

# OSM fair-use: max 1 req/s.  Use 1.1 s to stay safely under the limit.
_DELAY_BETWEEN_REQUESTS = 1.1

# Addresses that are effectively "no street address".
_BARE_ADDRESS_RE = re.compile(
    r"^Graz(?:,?\s*(?:Austria|Österreich))?$",
    re.IGNORECASE,
)


def _needs_geocoding(restaurant: dict[str, Any]) -> bool:
    """Return ``True`` if the restaurant's address is bare / uninformative."""
    address = restaurant.get("address", "")
    if not address:
        return True
    return bool(_BARE_ADDRESS_RE.fullmatch(address.strip()))


def _format_address(data: dict[str, Any]) -> str | None:
    """Extract a human-readable address from a Nominatim ``jsonv2`` response.

    Returns a string like ``"Hauptplatz 1, 8010 Graz"`` or ``None`` when
    the response lacks essential fields.
    """
    addr = data.get("address", {})
    if not isinstance(addr, dict):
        return None

    road = addr.get("road")
    house_number = addr.get("house_number")
    postcode = addr.get("postcode")
    # Fallback chain: city → town → village → municipality
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
    )

    if not road:
        logger.debug(
            "Nominatim returned no road: display_name=%r",
            data.get("display_name"),
        )
        return None

    street = f"{road} {house_number}" if house_number else road
    city_part = f"{postcode} {city}" if postcode and city else (city or "Graz")
    return f"{street}, {city_part}"


async def _reverse_geocode_one(
    lat: float,
    lng: float,
    client: httpx.AsyncClient,
) -> str | None:
    """Reverse-geocode a single coordinate pair.

    The caller is responsible for enforcing the inter-request delay.
    """
    params: dict[str, str | int | float] = {
        "lat": lat,
        "lon": lng,
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
        "layer": "address,poi",
    }

    try:
        resp = await client.get(
            _NOMINATIM_URL,
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("Nominatim timeout for lat=%s lon=%s", lat, lng)
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Nominatim HTTP %d for lat=%s lon=%s",
            exc.response.status_code,
            lat,
            lng,
        )
        return None
    except httpx.RequestError as exc:
        logger.warning("Nominatim request error: %s", exc)
        return None

    data = resp.json()
    if "error" in data:
        logger.debug(
            "Nominatim error for lat=%s lon=%s: %s",
            lat,
            lng,
            data["error"],
        )
        return None

    return _format_address(data)


async def enrich_addresses(
    restaurants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill bare addresses by reverse-geocoding restaurant coordinates.

    Only processes restaurants whose address matches the "bare Graz"
    pattern (no street info).  Requests are sent sequentially with a
    1.1 s delay to respect Nominatim's fair-use policy.

    Modifies the list **in-place** and returns it.

    Args:
        restaurants: Restaurant dicts (must contain ``latitude``,
            ``longitude``, and ``address`` keys).

    Returns:
        The same list with updated addresses where geocoding succeeded.
    """
    candidates = [r for r in restaurants if _needs_geocoding(r)]
    if not candidates:
        logger.info(
            "All restaurants already have street addresses — skipping geocoding"
        )
        return restaurants

    logger.info(
        "Reverse-geocoding %d/%d restaurants with bare addresses …",
        len(candidates),
        len(restaurants),
    )

    updated = 0
    async with httpx.AsyncClient(
        headers={
            "User-Agent": _USER_AGENT,
            "Accept-Language": _ACCEPT_LANGUAGE,
        },
    ) as client:
        for idx, restaurant in enumerate(candidates, start=1):
            lat = restaurant.get("latitude")
            lng = restaurant.get("longitude")
            name = restaurant.get("name", "unknown")

            if lat is None or lng is None:
                logger.debug("Skipping %s — no coordinates", name)
                continue

            logger.info("Geocoding %d/%d: %s", idx, len(candidates), name)

            address = await _reverse_geocode_one(lat, lng, client)
            if address:
                old = restaurant.get("address", "")
                restaurant["address"] = address
                updated += 1
                logger.info("  → %s: %r → %r", name, old, address)

                # Track data source
                sources = restaurant.get("data_sources", [])
                if "nominatim" not in sources:
                    sources.append("nominatim")
                    restaurant["data_sources"] = sources

            # Enforce rate limit (skip delay after last request)
            if idx < len(candidates):
                await asyncio.sleep(_DELAY_BETWEEN_REQUESTS)

    logger.info(
        "Geocoding complete: updated %d/%d addresses",
        updated,
        len(candidates),
    )
    return restaurants
