"""Discover missing restaurant websites via DuckDuckGo search.

For restaurants scraped from OpenStreetMap that lack a ``website`` tag,
this module searches DuckDuckGo for the restaurant name + "Graz" and
returns the most likely official website URL.

**Precision over recall**: It is better to return ``None`` than a wrong
website, because wrong URLs waste downstream vision API calls and
pollute the database with unrelated content.

Triggered via ``python -m scraper --discover-websites``.

Requires: ``pip install duckduckgo-search``
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urlparse

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# Delay between searches to avoid rate-limiting
_DELAY_BETWEEN_SEARCHES = 3.0

# Minimum score a candidate must reach to be accepted.
# Prevents garbage results (zhihu.com, ford-torino.de, etc.).
_MIN_SCORE = 3

# ---------------------------------------------------------------------------
# Domain blocklists
# ---------------------------------------------------------------------------

# Aggregator / review / social platforms — never an official restaurant site
_SKIP_DOMAINS = frozenset(
    {
        # Search / maps
        "google.com",
        "google.at",
        "maps.google.com",
        # Social media
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "tiktok.com",
        "twitter.com",
        "x.com",
        "linkedin.com",
        "pinterest.com",
        "reddit.com",
        # Review / booking platforms
        "tripadvisor.com",
        "tripadvisor.at",
        "tripadvisor.de",
        "yelp.com",
        "yelp.at",
        "thefork.com",
        "quandoo.at",
        "quandoo.com",
        # Delivery platforms
        "lieferando.at",
        "lieferando.de",
        "mjam.net",
        "wolt.com",
        "uber.com",
        "ubereats.com",
        # Directories / reference / guides
        "foursquare.com",
        "wikipedia.org",
        "de.wikipedia.org",
        "en.wikipedia.org",
        "herold.at",
        "falter.at",
        "falstaff.com",
        "falstaff.at",
        "graz.at",
        "stadtbekannt.at",
        "gastrofinder.at",
        "restaurant.info",
        "speisekarte.de",
        "restaurantguru.com",
        "smartgastro.at",
    }
)

# Generic / non-restaurant domains that slipped through in testing.
# Separate from _SKIP_DOMAINS for clarity — these are not aggregators,
# just sites that are obviously not restaurant websites.
_GENERIC_BLOCKLIST = frozenset(
    {
        # Chinese / non-DACH sites
        "zhihu.com",
        "baidu.com",
        "weibo.com",
        # Tech / forums / Q&A
        "answers.microsoft.com",
        "learn.microsoft.com",
        "microsoft.com",
        "stackoverflow.com",
        "github.com",
        "gitlab.com",
        "nuwaves.com",
        # Finance
        "finanztip.de",
        "check24.de",
        # Gaming / sports
        "sofifa.com",
        "transfermarkt.at",
        "transfermarkt.de",
        # Generic coffee / food forums (not a specific restaurant)
        "kaffee-netz.de",
        "chefkoch.de",
        # Generic regional portals (too broad)
        "ganz-wien.at",
        "meinbezirk.at",
        "wogibtswas.at",
        # Car / vehicle sites
        "ford-torino.de",
        "autoscout24.at",
        "autoscout24.de",
        # Non-DACH / non-restaurant sites that slipped through
        "pantip.com",
        "bergfex.at",
        "bergfex.com",
        # City/government portals (match geographic names, not restaurants)
        "stadt-salzburg.at",
        "wien.gv.at",
        "graz.gv.at",
    }
)

# ---------------------------------------------------------------------------
# TLD preferences
# ---------------------------------------------------------------------------

_PREFERRED_TLDS = frozenset({".at", ".co.at", ".or.at"})
_ACCEPTABLE_TLDS = frozenset({".com", ".eu", ".de", ".net", ".org", ".io"})
_SUSPECT_TLDS = frozenset(
    {".cn", ".ru", ".fi", ".kr", ".jp", ".br", ".pl", ".cz", ".sk"}
)

# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

# Words in the domain that are strong negative signals (not a restaurant)
_NEGATIVE_DOMAIN_WORDS = re.compile(
    r"(microsoft|amazon|apple|google|github|gitlab|stackoverflow|"
    r"wikipedia|reddit|forum|wiki|news|finance|finanz|"
    r"sport|fifa|football|soccer|auto|ford|vehicle|"
    r"electronics|semiconductor|nuwaves|answers|learn)",
    re.IGNORECASE,
)

# Restaurant / food / hospitality signals in title or snippet
_FOOD_SIGNALS = re.compile(
    r"(restaurant|gasthaus|gasthof|wirtshaus|pizzeria|trattoria|ristorante|"
    r"bistro|brasserie|lokal|küche|speisekarte|menü|menu|essen|"
    r"reservier|tischreservierung|brunch|mittag|abendessen|"
    r"café|cafe|konditorei|bäckerei|kebab|sushi|pizza|burger|"
    r"buschenschank|heuriger|beisl|stüberl|kantine|mensa|"
    r"graz|steiermark|styria)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_blocked(url: str) -> bool:
    """Check if a URL belongs to a blocked domain (aggregator or generic)."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        all_blocked = _SKIP_DOMAINS | _GENERIC_BLOCKLIST
        if domain in all_blocked:
            return True
        # Also check parent domains (e.g., "de.wikipedia.org" → "wikipedia.org")
        return any(domain.endswith(f".{b}") for b in all_blocked)
    except Exception:
        return True


def _get_tld(domain: str) -> str:
    """Extract TLD from a domain (e.g., '.at', '.co.at')."""
    parts = domain.rsplit(".", maxsplit=2)
    if len(parts) >= 3 and parts[-2] in ("co", "or", "ac", "gv"):
        return f".{parts[-2]}.{parts[-1]}"
    if len(parts) >= 2:
        return f".{parts[-1]}"
    return ""


def _name_matches_domain(name_tokens: list[str], domain: str) -> int:
    """Score how well restaurant name tokens match the domain.

    Uses word-boundary-aware matching to prevent false positives like
    "torino" matching "ford-torino.de" (the token is a minor part of
    the domain, not its primary identity).

    Returns:
        Score (0 = no match, higher = better).
    """
    # Strip TLD: "ristorantecorti.at" → "ristorantecorti"
    domain_base = domain.rsplit(".", maxsplit=1)[0] if "." in domain else domain
    # Split domain into words: "konditorei-philipp" → {"konditorei", "philipp"}
    domain_words = set(re.split(r"[\-._]+", domain_base))

    score = 0
    for token in name_tokens:
        if token in domain_words:
            # Exact word match: "corti" in {"ristorante", "corti"}
            score += 4
        elif token == domain_base:
            # Token IS the domain: "schwalben" == "schwalben"
            score += 5
        elif domain_base.endswith(token) or domain_base.startswith(token):
            # Token is a prefix/suffix of the unsplit domain base.
            # Catches "corti" in "ristorantecorti" (common pattern where
            # the restaurant name is appended to a generic word).
            score += 3
        elif any(
            token in word and len(token) / len(word) > 0.6
            for word in domain_words
            if len(word) > 2
        ):
            # Token is a substantial part of a domain word (>60% of length).
            # Catches "philipp" in "konditorei-philipp" but blocks
            # "torino" in "ford-torino" (40% — too small).
            score += 2

    return score


def _pick_best_url(results: list[dict[str, str]], restaurant_name: str) -> str | None:
    """Pick the most likely official website from search results.

    Strategy:
    1. Skip blocked domains (aggregators, generic sites)
    2. Skip domains with negative signals (tech, finance, gaming, etc.)
    3. Skip results from suspect TLDs (.cn, .ru, .fi, etc.)
    4. Score by: name match in domain/title, TLD, food signals, "graz"
    5. Require at least one relevance signal (name, graz, or food keyword)
    6. Require minimum score to accept (precision > recall)
    """
    name_lower = restaurant_name.lower().strip()
    # Generic words that appear in nearly every restaurant search result.
    # Matching on these causes false positives (e.g., "Gasthaus Doppler"
    # matches pfingstl.at because its title also contains "Gasthaus").
    stop_words = {
        # Articles / conjunctions
        "the", "das", "der", "die", "und", "and", "von", "zum", "zur",
        # Generic restaurant / venue words (too common to be distinctive)
        "bar", "cafe", "café", "restaurant", "gasthaus", "gasthof",
        "wirtshaus", "pizzeria", "trattoria", "ristorante", "bistro",
        "brasserie", "lokal", "stüberl", "beisl", "kantine", "mensa",
        "buschenschank", "heuriger", "konditorei", "bäckerei",
        # Location
        "graz", "steiermark",
    }
    name_tokens = [
        t
        for t in re.split(r"[\s\-/&'+.,()]+", name_lower)
        if t and len(t) > 2 and t not in stop_words
    ]

    candidates: list[tuple[int, str]] = []

    for result in results:
        url = result.get("href", "")
        title = result.get("title", "").lower()
        snippet = result.get("body", "").lower()

        if not url or _is_blocked(url):
            continue

        domain = urlparse(url).netloc.lower().replace("www.", "")
        tld = _get_tld(domain)

        # Hard skip: suspect TLDs
        if tld in _SUSPECT_TLDS:
            continue

        # Hard skip: negative domain words
        if _NEGATIVE_DOMAIN_WORDS.search(domain):
            continue

        # --- Scoring ---
        score = 0

        # 1. Name match in domain (word-boundary-aware)
        domain_name_score = _name_matches_domain(name_tokens, domain)
        score += domain_name_score

        # 2. Name tokens in title
        title_name_hits = sum(1 for t in name_tokens if t in title)
        score += title_name_hits

        # 3. "Graz" in title or snippet
        if "graz" in title:
            score += 2
        elif "graz" in snippet:
            score += 1

        # 4. TLD preference
        if tld in _PREFERRED_TLDS:
            score += 2
        elif tld in _ACCEPTABLE_TLDS:
            score += 1

        # 5. Food / restaurant signals in title or snippet
        if _FOOD_SIGNALS.search(title):
            score += 1
        if _FOOD_SIGNALS.search(snippet):
            score += 1

        # --- Relevance gate ---
        # The restaurant's distinctive name MUST appear somewhere in the
        # result (domain or title). Without this, location + food signals
        # alone cause false positives (e.g., pfingstl.at for "Gasthaus
        # Doppler" just because both are Graz restaurants).
        has_name_signal = domain_name_score > 0 or title_name_hits > 0
        has_location_signal = "graz" in title or "graz" in snippet

        if not has_name_signal:
            # No name match at all — almost certainly the wrong website.
            # Only exception: if we have no distinctive name tokens (all
            # were stop words), fall back to requiring location signal.
            if name_tokens:
                continue
            if not has_location_signal:
                continue

        candidates.append((score, url))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_url = candidates[0]

    # Minimum score threshold — reject low-confidence matches
    if best_score < _MIN_SCORE:
        logger.debug(
            "Best candidate scored %d (below threshold %d) — rejecting: %s",
            best_score,
            _MIN_SCORE,
            best_url,
        )
        return None

    return best_url


def discover_website(restaurant_name: str, address: str = "") -> str | None:
    """Search DuckDuckGo for a restaurant's official website.

    Args:
        restaurant_name: Name of the restaurant.
        address: Optional address for disambiguation.

    Returns:
        URL string of the most likely official website, or None.
    """
    # Build search query
    query = f'"{restaurant_name}" Graz Restaurant'
    if address and "graz" not in address.lower():
        query += f" {address}"

    try:
        results = DDGS().text(query, region="at-de", max_results=8)
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for '%s': %s", restaurant_name, exc)
        return None

    if not results:
        return None

    return _pick_best_url(results, restaurant_name)


async def discover_missing_websites(
    restaurants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find websites for restaurants that don't have one.

    Searches DuckDuckGo for each restaurant without a ``website`` field
    and fills it in if a likely official website is found.

    A delay is inserted between searches to respect rate limits.

    Args:
        restaurants: List of restaurant dicts (modified in place).

    Returns:
        The same list with discovered websites filled in.
    """
    missing = [r for r in restaurants if not r.get("website")]
    if not missing:
        logger.info("All restaurants already have websites — skipping discovery")
        return restaurants

    logger.info(
        "Discovering websites for %d/%d restaurants without one…",
        len(missing),
        len(restaurants),
    )

    found_count = 0
    for idx, restaurant in enumerate(missing, start=1):
        name = restaurant.get("name", "unknown")
        address = restaurant.get("address", "")

        logger.info("Searching %d/%d: %s", idx, len(missing), name)

        # DuckDuckGo search is synchronous — run in executor to not block
        url = await asyncio.get_event_loop().run_in_executor(
            None, discover_website, name, address
        )

        if url:
            restaurant["website"] = url
            sources = restaurant.get("data_sources", [])
            if "web_search" not in sources:
                sources.append("web_search")
                restaurant["data_sources"] = sources
            found_count += 1
            logger.info("  → Found: %s", url)
        else:
            logger.debug("  → No website found for %s", name)

        # Polite delay between searches
        await asyncio.sleep(_DELAY_BETWEEN_SEARCHES)

    logger.info(
        "Website discovery complete: found %d/%d missing websites",
        found_count,
        len(missing),
    )
    return restaurants
