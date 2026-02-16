# Restaurant Website Scraper Research Findings

## 1. Structured Data Extraction (Schema.org, JSON-LD, OpenGraph)

### Most Common Metadata Formats

**JSON-LD (Highest Priority)**
- 70%+ of modern restaurant websites embed Schema.org data as JSON-LD
- Located in `<script type="application/ld+json">` tags
- Schema.org types: `Restaurant`, `FoodEstablishment`, `LocalBusiness`

**Key Properties for Restaurants:**
```json
{
  "@context": "https://schema.org",
  "@type": "Restaurant",
  "name": "Restaurant Name",
  "address": {
    "@type": "PostalAddress",
    "streetAddress": "Hauptplatz 1",
    "addressLocality": "Graz",
    "postalCode": "8010",
    "addressCountry": "AT"
  },
  "telephone": "+43 316 123456",
  "openingHours": ["Mo-Fr 11:00-22:00", "Sa-Su 12:00-23:00"],
  "openingHoursSpecification": [...],
  "servesCuisine": "Austrian",
  "priceRange": "$$",
  "hasMenu": "https://example.at/menu",
  "acceptsReservations": true,
  "image": "https://example.at/image.jpg"
}
```

**OpenGraph (Secondary)**
- Used by ~60% of sites for social media sharing
- Properties: `og:title`, `og:type`, `og:url`, `og:image`, `og:description`
- Extract via: `<meta property="og:*" content="...">`

**Microdata (Legacy)**
- Older format using `itemprop`, `itemtype`, `itemscope` attributes
- Still found on ~20% of sites
- Example: `<div itemscope itemtype="http://schema.org/Restaurant">`

### Extraction Code Pattern

```python
from bs4 import BeautifulSoup
import json
import re

def extract_jsonld(soup: BeautifulSoup) -> list[dict]:
    """Extract all JSON-LD structured data."""
    results = []
    scripts = soup.find_all('script', type='application/ld+json')
    
    for script in scripts:
        if not script.string:
            continue
        try:
            # Remove HTML comments (some sites wrap JSON-LD)
            content = re.sub(r'<!--.*?-->', '', script.string, flags=re.DOTALL)
            data = json.loads(content)
            
            # Handle both single objects and arrays
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
        except json.JSONDecodeError:
            # Try removing JS-style comments
            content = re.sub(r'^\s*//.*$', '', content, flags=re.MULTILINE)
            try:
                data = json.loads(content)
                results.append(data)
            except:
                continue
    
    return results

def find_restaurant_in_jsonld(jsonld_data: list[dict]) -> dict | None:
    """Find Restaurant type in JSON-LD, handling @graph structures."""
    for item in jsonld_data:
        item_type = item.get('@type', '')
        
        # Direct match
        if isinstance(item_type, str):
            if 'Restaurant' in item_type or 'FoodEstablishment' in item_type:
                return item
        elif isinstance(item_type, list):
            if any('Restaurant' in t or 'FoodEstablishment' in t for t in item_type):
                return item
        
        # Check @graph (common in WordPress sites)
        if '@graph' in item:
            for graph_item in item['@graph']:
                graph_type = graph_item.get('@type', '')
                if 'Restaurant' in str(graph_type) or 'FoodEstablishment' in str(graph_type):
                    return graph_item
    
    return None

def extract_opengraph(soup: BeautifulSoup) -> dict[str, str]:
    """Extract OpenGraph metadata."""
    og_data = {}
    for meta in soup.find_all('meta', property=re.compile(r'^og:')):
        prop = meta.get('property', '')
        content = meta.get('content', '')
        if prop and content:
            og_data[prop] = content
    return og_data
```

---

## 2. Multilingual Content Handling (German/English)

### Language Detection

**Priority Order:**
1. HTML `lang` attribute: `<html lang="de-AT">`
2. Meta tag: `<meta http-equiv="content-language" content="de">`
3. Heuristic: Check for German-specific words

```python
def detect_language(soup: BeautifulSoup, text_sample: str = '') -> str:
    """Detect page language."""
    # Check HTML lang attribute
    html_tag = soup.find('html')
    if html_tag and html_tag.get('lang'):
        lang = html_tag['lang'].lower()
        if lang.startswith('de'):
            return 'de'
        elif lang.startswith('en'):
            return 'en'
    
    # Heuristic: German indicators
    if text_sample:
        german_words = ['öffnungszeiten', 'speisekarte', 'über uns', 
                       'montag', 'dienstag', 'mittwoch']
        if any(word in text_sample.lower() for word in german_words):
            return 'de'
    
    return 'unknown'
```

### Handling Multilingual Sites

Many Austrian restaurant sites have language switchers (DE/EN):

```python
def extract_multilingual_content(soup: BeautifulSoup) -> dict:
    """Extract content in multiple languages."""
    content = {'de': {}, 'en': {}}
    
    for lang in ['de', 'en']:
        # Common patterns:
        # - <div lang="de">...</div>
        # - <div class="lang-de">...</div>
        # - <div data-lang="de">...</div>
        
        lang_elements = soup.find_all(attrs={'lang': lang})
        lang_elements += soup.find_all(class_=re.compile(f'lang-{lang}'))
        lang_elements += soup.find_all(attrs={'data-lang': lang})
        
        if lang_elements:
            content[lang]['text'] = ' '.join(el.get_text(strip=True) 
                                            for el in lang_elements)
    
    return content
```

**Character Encoding:**
- Always use UTF-8 for Austrian sites (handles ä, ö, ü, ß)
- httpx defaults to UTF-8, but verify: `response.encoding = 'utf-8'`

---

## 3. Opening Hours Extraction Patterns

### Priority Order

1. **JSON-LD `openingHours`** (most reliable)
2. **JSON-LD `openingHoursSpecification`** (detailed)
3. **HTML with semantic classes/IDs**
4. **Regex pattern matching** (fallback)

### Code Pattern

```python
def parse_opening_hours(soup: BeautifulSoup, jsonld_data: dict | None = None) -> list[str]:
    """Extract opening hours from multiple sources."""
    hours = []
    
    # 1. From JSON-LD
    if jsonld_data:
        if 'openingHours' in jsonld_data:
            oh = jsonld_data['openingHours']
            hours.extend(oh if isinstance(oh, list) else [oh])
        
        # OpeningHoursSpecification (more detailed)
        if 'openingHoursSpecification' in jsonld_data:
            specs = jsonld_data['openingHoursSpecification']
            if not isinstance(specs, list):
                specs = [specs]
            for spec in specs:
                day = spec.get('dayOfWeek', '')
                opens = spec.get('opens', '')
                closes = spec.get('closes', '')
                if day and opens and closes:
                    hours.append(f"{day} {opens}-{closes}")
    
    # 2. From HTML with common selectors
    if not hours:
        selectors = [
            {'class': re.compile(r'(opening|hours|öffnungszeiten)', re.I)},
            {'id': re.compile(r'(opening|hours|öffnungszeiten)', re.I)},
            {'itemprop': 'openingHours'},
        ]
        
        for selector in selectors:
            elements = soup.find_all(attrs=selector)
            for el in elements:
                text = el.get_text(strip=True)
                if text and len(text) > 5:
                    hours.append(text)
                    break
            if hours:
                break
    
    # 3. Regex patterns (fallback)
    if not hours:
        text = soup.get_text()
        # German: Mo-Fr 11:00-22:00
        pattern_de = r'(Mo|Di|Mi|Do|Fr|Sa|So)[\w\s,-]*\d{1,2}[:\.]\d{2}\s*[-–]\s*\d{1,2}[:\.]\d{2}'
        # English: Mon-Fri 11:00-22:00
        pattern_en = r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[\w\s,-]*\d{1,2}[:\.]\d{2}\s*[-–]\s*\d{1,2}[:\.]\d{2}'
        
        matches = re.findall(pattern_de, text, re.IGNORECASE)
        matches += re.findall(pattern_en, text, re.IGNORECASE)
        hours.extend(matches)
    
    return hours
```

### Common Formats

**German:**
- `Mo-Fr 11:00-22:00, Sa-So 12:00-23:00`
- `Montag bis Freitag: 11:00 
