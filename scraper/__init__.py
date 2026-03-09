"""Graz restaurant data scraping module.

Usage::

    python -m scraper                              # Scrape OSM data only
    python -m scraper --google-places              # + enrich via Google Places API (recommended)
    python -m scraper --enrich                     # + scrape restaurant websites
    python -m scraper --google-places --enrich     # Full pipeline
"""
