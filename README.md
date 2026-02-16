# Graz Restaurant Chatbot

A chatbot answering questions about restaurants in Graz, Austria using a curated local dataset.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run migrations
alembic upgrade head

# Seed database
python -m app.seed

# Start server
uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000 to use the chat interface.

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Lint and format
ruff check .
ruff format .

# Type check
mypy app/
```

## Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy 2.x
- **Database**: PostgreSQL (SQLite for dev)
- **Frontend**: Jinja2, vanilla JavaScript, CSS
- **Data**: BeautifulSoup4 + Playwright for scraping
- **LLM**: OpenAI GPT-4 for response generation
