# Menu Extraction Pipeline Improvement

**Date**: 2026-03-04
**Status**: Proposed
**Approach**: B — Playwright + LLM Text Parser

---

## Problem

1016 restaurants in dataset, 711 have websites, **0 have menu items**.
`--enrich` was never run, and when tested on 15 restaurants only **1/15 succeeded**.

### Root Causes (from live test on 15 restaurants)

| Failure Mode           | Count | Why Current Pipeline Fails                          |
|------------------------|-------|-----------------------------------------------------|
| No menu page found     | 7/15  | `_MENU_KEYWORDS` too narrow, no path probing        |
| JS-rendered menu       | 3/15  | BS4 can't parse JS — Playwright is opt-in only      |
| PDF/image menu         | 2/15  | Vision extraction exists but requires `--enrich-vision` |
| No menu on website     | 2/15  | Genuinely no menu data (McDonald's, etc.)           |
| **Success** (static)   | 1/15  | Static HTML with `€` prices — only working mode     |

### Current Pipeline Flow

```
Homepage (httpx/BS4)
  → _find_menu_url()         # narrow keyword match on <a> tags
  → _extract_menu_items_from_soup()   # JSON-LD or HTML heuristic (needs € in text)
  → IF <3 items AND --enrich-js: Playwright fallback (same BS4 heuristic on rendered HTML)
  → IF <3 items AND --enrich-vision: vision fallback on PDF/image links
```

**Core issue**: The BS4 HTML heuristic (`_extract_from_html_heuristic`) requires a `€` price pattern in visible text. Most Graz restaurants use JS-rendered menus, PDFs, or images — none of which contain parseable `€` text in raw HTML.

---

## Design: Approach B — Playwright + LLM Text Parser

### Principle

Replace the fragile BS4 heuristic with a two-stage approach:
1. **Playwright** renders every page (handles JS)
2. **GPT-4o-mini** parses the visible text into structured menu items (handles messy HTML)

Keep the free fast-paths (JSON-LD, pdfplumber) as pre-checks to avoid unnecessary API calls.

### New Pipeline Flow

```
For each restaurant with website:
  1. Fetch homepage with httpx (for summary + link discovery only)
  2. Find menu URL (improved keyword list + common path probing)
  3. Find menu file URL (PDF/image links)

  --- Fast path (free, no API) ---
  4. IF JSON-LD menu data on homepage → extract → DONE
  5. IF PDF file URL → pdfplumber text extraction → DONE (if ≥3 items)

  --- LLM text path (~$0.001/restaurant) ---
  6. Playwright renders (menu_url OR homepage)
  7. Extract page.inner_text() (visible text only, no HTML tags)
  8. Send text to GPT-4o-mini → [{name, price, category}]

  --- Vision fallback (~$0.002/restaurant) ---
  9. IF <3 items AND PDF file URL → GPT-4o-mini vision on PDF
  10. IF <3 items AND large images on page → GPT-4o-mini vision on images
```

### Changes by File

#### `scraper/websites.py`

**1. Improve menu URL discovery** — `_find_menu_url()`

Add keywords to `_MENU_KEYWORDS`:
```python
_MENU_KEYWORDS = frozenset({
    # Existing
    "menu", "speisekarte", "karte", "gerichte", "dishes", "speisen",
    # New
    "food", "essen", "mittagstisch", "angebot", "wochenkarte",
    "tagesmenü", "tagesmenu", "our food", "küche", "cuisine",
    "eat", "dine", "à la carte", "a la carte",
})
```

Add common path probing when no link found:
```python
_COMMON_MENU_PATHS = [
    "/menu", "/speisekarte", "/karte", "/food",
    "/essen", "/speisen", "/our-menu", "/the-menu",
]

async def _probe_menu_paths(client, base_url) -> str | None:
    """Try common menu URL paths when no link was found in HTML."""
    for path in _COMMON_MENU_PATHS:
        url = urljoin(base_url, path)
        try:
            resp = await client.head(url)
            if resp.status_code == 200:
                return url
        except httpx.RequestError:
            continue
    return None
```

**2. Add LLM text extraction function** — new `_extract_menu_with_llm()`

```python
async def _extract_menu_with_llm(
    visible_text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> list[dict[str, str]]:
    """Send visible page text to LLM for structured menu extraction."""
    # Reuse the same prompt and parsing as menu_vision._parse_menu_text_with_llm
    from scraper.menu_vision import _parse_menu_text_with_llm
    return _parse_menu_text_with_llm(visible_text, api_key, model)
```

**3. New Playwright extraction** — replace `_scrape_menu_playwright()`

Current behavior: Playwright renders → BS4 heuristic (same problem).
New behavior: Playwright renders → extract visible text → LLM parses.

```python
async def _scrape_menu_playwright_llm(
    url: str,
    browser: Browser,
    api_key: str,
) -> list[dict[str, str]]:
    """Render page with Playwright, extract visible text, parse with LLM."""
    page = await browser.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=_PW_NAV_TIMEOUT)
        await page.wait_for_timeout(1500)  # let late JS finish
        visible_text = await page.inner_text("body")
        if not visible_text or len(visible_text) < 50:
            return []
        # Truncate to ~6000 chars to stay within token limits
        return await _extract_menu_with_llm(visible_text[:6000], api_key)
    except Exception as exc:
        logger.debug("Playwright+LLM failed for %s: %s", url, exc)
        return []
    finally:
        await page.close()
```

**4. Update `scrape_restaurant_website()`** — new extraction cascade

```python
async def scrape_restaurant_website(url, client, *, browser=None, api_key=None, use_vision=False):
    # 1. Fetch homepage (httpx) for summary + link discovery
    soup = await _fetch_html(client, url)
    if soup is None:
        return None

    summary = _extract_summary(soup)
    menu_url = _find_menu_url(soup, url)
    menu_file_url = _find_menu_file_url(soup, url)

    # 2. Probe common paths if no menu link found
    if not menu_url:
        menu_url = await _probe_menu_paths(client, url)

    # 3. Fast path: JSON-LD (free)
    menu_items = _extract_from_jsonld(soup)
    if len(menu_items) >= 3:
        return {...}  # done, no API cost

    # 4. Fast path: pdfplumber for text PDFs (free)
    if menu_file_url and len(menu_items) < 3:
        pdf_items = await _try_pdfplumber(menu_file_url, client)
        if len(pdf_items) > len(menu_items):
            menu_items = pdf_items
        if len(menu_items) >= 3:
            return {...}  # done, no API cost

    # 5. LLM text path: Playwright + GPT-4o-mini
    if browser and api_key and len(menu_items) < 3:
        target = menu_url or url
        llm_items = await _scrape_menu_playwright_llm(target, browser, api_key)
        if len(llm_items) > len(menu_items):
            menu_items = llm_items

    # 6. Vision fallback: PDF/image
    if use_vision and api_key and len(menu_items) < 3:
        # existing vision logic (unchanged)
        ...

    return {summary, menu_items, menu_url, menu_file_url}
```

**5. Update `enrich_restaurants()`** — always use Playwright

- Playwright is always launched (not gated by `use_playwright`)
- Pass `browser` and `api_key` to `scrape_restaurant_website()`
- Vision enabled by default when `OPENAI_API_KEY` is set

**6. Fix Windows Unicode** — log messages

Replace `✓` and `✗` with `[OK]` and `[FAIL]` in log strings (lines 672, 718).

#### `scraper/__main__.py`

**7. Simplify CLI flags**

- `--enrich` now does Playwright + LLM by default (requires `OPENAI_API_KEY`)
- `--enrich-js` deprecated (Playwright is always-on)
- `--enrich-vision` deprecated (vision enabled by default when API key present)
- Add `--no-llm` flag to skip LLM parsing (BS4 heuristic only, for cost-free runs)
- Add `--no-vision` flag to skip vision extraction
- Add `--limit N` flag for testing on N restaurants

```
python -m scraper --enrich                    # full pipeline (Playwright + LLM + vision)
python -m scraper --enrich --no-llm           # free mode (Playwright + BS4 only)
python -m scraper --enrich --no-vision        # skip PDF/image vision
python -m scraper --enrich --limit 20         # test on 20 restaurants
```

#### `scraper/menu_vision.py`

**8. Make `_parse_menu_text_with_llm` importable**

Currently a private function. Add `__all__` or rename without underscore since `websites.py` will call it for LLM text parsing.

No other changes needed — vision logic is already solid.

### What We Keep (Unchanged)

- `_extract_from_jsonld()` — free, fast, accurate when available
- `_extract_summary()` — works fine
- `_find_menu_file_url()` — PDF/image detection works
- `menu_vision.py` vision extraction — already handles PDFs and images
- `_CATEGORY_HINTS` — still used for JSON-LD and as fallback
- pdfplumber text extraction — free pre-check before vision

### What We Remove / Deprecate

- `_extract_from_html_heuristic()` — keep as `--no-llm` fallback, but no longer primary
- `--enrich-js` CLI flag — deprecated, Playwright is always-on
- `--enrich-vision` CLI flag — deprecated, vision is default

---

## Cost Estimate

| Stage                    | Cost per restaurant | Total (711 restaurants) |
|--------------------------|--------------------:|------------------------:|
| JSON-LD fast path        | $0                  | $0                      |
| pdfplumber fast path     | $0                  | $0                      |
| LLM text (GPT-4o-mini)  | ~$0.001             | ~$0.70                  |
| Vision fallback          | ~$0.002             | ~$0.50 (est. 250 need)  |
| **Total**                |                     | **~$1.20**              |

*Assumes ~60% hit LLM text path, ~35% need vision fallback, ~5% have JSON-LD or no menu.*

### Compute Time

- Playwright: ~3s per restaurant × 711 = ~35 minutes (sequential with 2s delay)
- Could parallelize with browser contexts (4-5 concurrent) → ~10 minutes

---

## Expected Results

| Metric                                | Before | After (estimated) |
|---------------------------------------|-------:|-------------------:|
| Restaurants with ≥1 menu item         | 0      | 280–420 (40–60%)   |
| Average items per successful extract  | 0      | 10–25              |
| Cost                                  | $0     | ~$1.20             |

**Why not 100%?** Some restaurants genuinely don't have menus online (chains with app-only menus, social-media-only presence, broken websites, non-restaurant businesses mis-tagged in OSM).

---

## Success Criteria

1. **≥30% of 711 restaurants** get ≥1 menu item (up from 0%)
2. **No regressions** — restaurants that had data keep it
3. **Pipeline completes** in <60 minutes for all 711 restaurants
4. **Cost stays under $5** per full run
5. `pytest` still passes after changes

---

## Implementation Order

1. Fix Windows Unicode in log messages (`✓`/`✗` → `[OK]`/`[FAIL]`)
2. Expand `_MENU_KEYWORDS` + add `_probe_menu_paths()`
3. Add `_extract_menu_with_llm()` in `websites.py`
4. Rewrite `_scrape_menu_playwright()` → `_scrape_menu_playwright_llm()`
5. Update `scrape_restaurant_website()` extraction cascade
6. Update `enrich_restaurants()` — Playwright always-on, pass API key
7. Update `__main__.py` CLI flags
8. Make `_parse_menu_text_with_llm` importable from `menu_vision.py`
9. Test on 20 restaurants (`--limit 20`), evaluate results
10. Run full pipeline on 711 restaurants
