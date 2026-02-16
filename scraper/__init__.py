"""Graz restaurant data scraping module.

Usage::

    python -m scraper                              # Scrape OSM data only
    python -m scraper --discover-websites          # + find missing websites via DuckDuckGo
    python -m scraper --enrich                     # + scrape restaurant websites
    python -m scraper --enrich --enrich-vision     # + GPT-4o vision for PDF/image menus
"""
