# Setup Instructions

## Prerequisites

- Python 3.12 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   cd C:\Users\snehi\Documents\restuarants
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env
     ```
   - Edit `.env` and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-actual-api-key-here
     ```

4. **Seed the database**:
   ```bash
   python -m app.seed
   ```
   This creates a SQLite database with 6 sample restaurants in Graz.

## Running the Application

### Development Server

```bash
uvicorn app.main:app --reload --port 8000
```

The application will be available at: **http://localhost:8000**

### Using the Chat Interface

1. Open your browser and navigate to http://localhost:8000
2. Type questions about restaurants in Graz, such as:
   - "vegan restaurants"
   - "cheap Italian food"
   - "restaurants with outdoor seating"
   - "fine dining in Graz"

The chatbot will:
- Parse your query to extract filters (cuisine, price, features)
- Search the database for matching restaurants
- Generate a natural language response using OpenAI GPT-4
- Display the results with restaurant details

## Development Commands

### Run Tests
```bash
pytest
```

### Lint and Format
```bash
ruff check .           # Check for issues
ruff check . --fix     # Auto-fix issues
ruff format .          # Format code
```

### Type Checking
```bash
mypy app/
```

### Database Migrations
```bash
alembic upgrade head   # Apply migrations
alembic revision --autogenerate -m "description"  # Create new migration
```

## Project Structure

- `app/` - Main application code
  - `main.py` - FastAPI app factory
  - `models/` - SQLAlchemy database models
  - `schemas/` - Pydantic request/response schemas
  - `services/` - Business logic (query parser, retrieval, LLM)
  - `api/` - API route handlers
  - `templates/` - Jinja2 HTML templates
  - `static/` - CSS and JavaScript files
- `tests/` - Test files
- `scraper/` - Data scraping scripts (placeholder implementations)
- `alembic/` - Database migrations

## Notes

- The application uses SQLite by default for development
- Sample data includes 6 real restaurants in Graz, Austria
- The LLM is configured with temperature 0.3 for deterministic responses
- All restaurant data is from the local database - no live API calls
- The application supports both English and German responses
