# AGENTS.md — Graz Restaurant Chatbot

## Project Overview

A chatbot answering questions about restaurants in Graz, Austria. Uses a curated local dataset as its only source of truth — no live API calls, no scraping, no hallucinated data at runtime.

**Stack**: Python 3.12+ · FastAPI · Jinja2 · Pydantic v2 · SQLAlchemy 2.x (async) · PostgreSQL (SQLite for dev/test) · BeautifulSoup4 + Playwright (scraping) · LLM for response generation only

---

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt        # production
pip install -r requirements-dev.txt    # includes test/lint deps

# Run dev server
uvicorn app.main:app --reload --port 8000

# Database
alembic upgrade head                   # run migrations
python -m app.seed                     # seed sample restaurant data

# Data scraping
python -m scraper                      # run all scrapers, output to data/
python -m scraper.google_maps          # scrape Google Maps only
python -m scraper.websites             # scrape restaurant websites only
python -m app.seed --from-scraped      # seed DB from scraped JSON files

# Lint & format
ruff check .                           # lint
ruff check . --fix                     # lint + autofix
ruff format .                          # format

# Type checking
mypy app/

# Tests
pytest                                 # all tests
pytest tests/test_chat.py              # single file
pytest tests/test_chat.py::test_name   # single test
pytest -x                              # stop on first failure
pytest -k "vegan"                      # run tests matching keyword
pytest --cov=app                       # with coverage
```

---

## Project Structure

```
app/
├── main.py              # FastAPI app factory, CORS, lifespan, template mount
├── config.py            # Settings via pydantic-settings (env-based)
├── database.py          # SQLAlchemy engine, session factory
├── models/              # SQLAlchemy ORM models
│   └── restaurant.py
├── schemas/             # Pydantic request/response schemas
│   └── restaurant.py
├── api/                 # Route handlers (thin — delegate to services)
│   └── chat.py
├── services/            # Business logic
│   ├── query_parser.py  # Intent extraction from user message
│   ├── retrieval.py     # DB query building, filtering, ranking
│   └── llm.py           # LLM prompt construction & response generation
├── templates/           # Jinja2 HTML templates
│   ├── base.html        # Base layout (head, nav, footer, scripts)
│   ├── index.html       # Chat interface page
│   └── components/      # Reusable template fragments
│       └── message.html # Single chat message bubble
├── static/              # Static assets served by FastAPI
│   ├── css/
│   │   └── style.css    # App styles (single file, no framework needed)
│   └── js/
│       └── chat.js      # Chat UI logic (fetch API, DOM updates)
├── seed.py              # Database seeder (manual + scraped JSON)
└── utils/               # Shared helpers (language detection, etc.)
scraper/                 # One-time data extraction scripts (NOT used at runtime)
├── __main__.py          # Entry point — runs all scrapers
├── google_maps.py       # Google Maps scraper (Playwright for JS-rendered pages)
├── websites.py          # Individual restaurant website scraper (BeautifulSoup)
├── parsers.py           # HTML → structured data extraction helpers
└── output.py            # Write scraped data to data/*.json
data/                    # Scraped output (JSON files, git-tracked)
├── restaurants_raw.json # Raw scraper output
└── restaurants.json     # Cleaned/validated, ready for seeding
tests/
├── conftest.py          # Fixtures: test DB, client, seeded data
├── test_chat.py
├── test_query_parser.py
├── test_retrieval.py
└── test_llm.py
alembic/                 # Database migrations
```

---

## Code Style Guidelines

### General

- **Python 3.12+** — use modern syntax (`type` aliases, `X | Y` unions, f-strings)
- **Line length**: 88 characters (ruff default)
- **Formatter/Linter**: ruff (replaces black + isort + flake8)
- **Type checker**: mypy in strict mode
- All functions and methods MUST have type annotations (params + return)

### Imports

Order enforced by ruff (isort-compatible):
1. Standard library
2. Third-party packages
3. Local app modules

```python
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.restaurant import Restaurant
```

- Use absolute imports (`from app.models.restaurant import ...`)
- No wildcard imports (`from x import *`)
- No unused imports (ruff enforces this)

### Naming Conventions

| Element         | Convention       | Example                     |
|-----------------|------------------|-----------------------------|
| Files/modules   | snake_case       | `query_parser.py`           |
| Classes         | PascalCase       | `RestaurantResponse`        |
| Functions/vars  | snake_case       | `parse_user_query`          |
| Constants       | UPPER_SNAKE_CASE | `MAX_RESULTS = 5`           |
| Pydantic models | PascalCase       | `ChatRequest`               |
| SQLAlchemy models | PascalCase     | `Restaurant`                |
| API routes      | snake_case       | `/api/chat`                 |
| Test functions  | `test_` prefix   | `test_vegan_filter_works`   |

### Pydantic Models (Schemas)

- Use Pydantic v2 style (`model_config = ConfigDict(...)`)
- Request models: suffix `Request` (e.g., `ChatRequest`)
- Response models: suffix `Response` (e.g., `ChatResponse`)
- Keep schemas in `app/schemas/`, separate from ORM models
- Use `Field()` for validation constraints and descriptions

```python
from pydantic import BaseModel, Field, ConfigDict

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    language: str | None = Field(default=None, pattern="^(de|en)$")

class RestaurantResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    cuisine: list[str]
    price_range: str
    summary: str
```

### SQLAlchemy Models

- Use SQLAlchemy 2.x `Mapped[]` / `mapped_column()` syntax
- Models go in `app/models/`
- Table names: plural snake_case (`restaurants`)

```python
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase

class Base(DeclarativeBase):
    pass

class Restaurant(Base):
    __tablename__ = "restaurants"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
```

### Error Handling

- Use `HTTPException` for API-level errors with appropriate status codes
- Service layer raises domain-specific exceptions; API layer catches and maps
- Never expose internal details (tracebacks, DB errors) to the client
- Log errors with `logging` (stdlib) — structured where possible
- No bare `except:` — always catch specific exceptions

```python
# Service layer
class RestaurantNotFoundError(Exception):
    pass

# API layer
try:
    result = await retrieval_service.search(filters)
except RestaurantNotFoundError:
    raise HTTPException(status_code=404, detail="No restaurants found matching your criteria.")
```

### Testing

- Framework: **pytest** with **pytest-asyncio** for async tests
- Use fixtures in `conftest.py` for DB sessions, test client, seed data
- Test files mirror source structure (`app/services/retrieval.py` → `tests/test_retrieval.py`)
- Use `httpx.AsyncClient` for API integration tests
- Mock the LLM layer in tests — never call real LLM in CI
- Each test should be independent and idempotent

### Async

- All DB operations are async (`AsyncSession`, `async with`)
- All API route handlers are `async def`
- Use `asyncio` patterns — no `run_in_executor` unless wrapping sync-only libs

### Configuration

- Use `pydantic-settings` for env-based config
- All secrets (LLM API keys, DB URLs) via environment variables
- Never hardcode secrets — `.env` file for local dev, excluded from git

### Data Scraping (`scraper/`)

- **Offline only** — scrapers run manually before deployment, never at chat time
- **Playwright** for JS-rendered pages (Google Maps) — headless Chromium
- **BeautifulSoup4** for static HTML (restaurant websites, blogs)
- Output raw data to `data/restaurants_raw.json`, cleaned data to `data/restaurants.json`
- Each scraper module exposes a `scrape() -> list[dict]` function
- All scraped fields must validate against `app/schemas/restaurant.py` before seeding
- Respect `robots.txt` — add delays between requests (`time.sleep`)
- Log every scraped URL and timestamp for traceability
- No copyrighted text — summaries must be original (rewrite if needed)
- Track `data_sources` and `last_verified` per restaurant entry

```python
# scraper/google_maps.py — example structure
from playwright.sync_api import sync_playwright

def scrape(query: str = "restaurants in Graz") -> list[dict]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # ... navigate, extract, return structured dicts
```

### Domain Rules (Critical)

- **Database is the single source of truth** — LLM receives only filtered DB results
- **No live API calls** at chat time (no Google Maps, no Tripadvisor)
- **No hallucinated restaurants** — if data is missing, say so explicitly
- **Scope: Graz only** — politely decline out-of-scope queries
- **LLM temperature ≤ 0.4** — deterministic, factual responses
- **Max 3–5 restaurants** per response
- **Bilingual**: German and English — match user's language

### Frontend (Jinja2 + Vanilla JS)

- **Templating**: Jinja2 served by FastAPI — no SPA framework
- **Base layout**: `app/templates/base.html` defines `{% block content %}` and loads static assets
- **Static files**: mounted at `/static` via `StaticFiles` in `main.py`
- **JS**: vanilla JavaScript only — no build step, no npm, no bundler
- **CSS**: single `style.css` — no Tailwind, no Bootstrap (keep it minimal)
- **Chat UI**: `fetch()` calls to `/api/chat`, responses rendered as DOM elements
- **No inline scripts/styles** — keep JS in `.js` files, CSS in `.css` files
- Template fragments in `components/` for reuse (`{% include "components/message.html" %}`)
- All user input must be escaped by Jinja2 autoescaping (enabled by default)

```python
# main.py — template & static setup
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})
```

### Git Conventions

- Commit messages: imperative mood, concise (`add vegan filter to retrieval service`)
- Branch naming: `feature/description`, `fix/description`
- No commits of `.env`, credentials, or API keys
- Keep commits atomic — one logical change per commit
