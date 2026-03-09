# Streaming Chat Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the blocking JSON chat response with SSE streaming so users see restaurant cards immediately and LLM text token-by-token.

**Architecture:** Two-phase SSE stream — Phase 1 emits restaurant data after query parsing + DB lookup (~2-4s), Phase 2 streams LLM tokens via OpenAI's `stream=True`. Frontend reads the stream with `ReadableStream` and updates the DOM incrementally.

**Tech Stack:** FastAPI `StreamingResponse`, OpenAI async streaming, vanilla JS `ReadableStream`

**Design doc:** `docs/plans/2026-03-09-streaming-chat-design.md`

---

### Task 1: Add streaming generator to LLM service

**Files:**
- Modify: `app/services/llm.py` (add `generate_response_stream` after line 175)

**Step 1: Add the streaming async generator**

Add this function after `generate_response()`:

```python
async def generate_response_stream(
    user_message: str,
    restaurants: list[Restaurant],
    language: str = "en",
    location_hint: str = "",
) -> AsyncIterator[str]:
    """Stream LLM response tokens.

    Yields individual text chunks as they arrive from the OpenAI API.
    Falls back to a single-chunk fallback message on error.
    """
    system_prompt = SYSTEM_PROMPT_DE if language == "de" else SYSTEM_PROMPT_EN
    restaurant_data = format_restaurant_data(restaurants)

    location_ctx = ""
    if location_hint:
        location_ctx = (
            f"\nNote: These restaurants are located{location_hint}. "
            "Mention their proximity to this area in your response."
        )

    user_prompt = f"""User question: {user_message}

Available restaurant data:
{restaurant_data}
{location_ctx}
Please answer the user's question using ONLY the information provided above. Be concise and helpful."""

    try:
        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=500,
            timeout=30.0,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        logger.error("LLM streaming error: %s", e)
        if language == "de":
            yield f"Ich habe {len(restaurants)} Restaurant(s) für Sie gefunden. Bitte sehen Sie sich die Details unten an."
        else:
            yield f"I found {len(restaurants)} restaurant(s) for you. Please see the details below."
```

Also add `AsyncIterator` to the imports at the top:

```python
from collections.abc import AsyncIterator
```

**Step 2: Verify import works**

Run: `python -c "from app.services.llm import generate_response_stream; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/services/llm.py
git commit -m "feat: add streaming LLM response generator"
```

---

### Task 2: Convert chat endpoint to SSE streaming

**Files:**
- Modify: `app/api/chat.py` (rewrite the `chat` function)

**Step 1: Rewrite the chat endpoint**

Replace the entire contents of `app/api/chat.py` with:

```python
"""Chat API endpoint with Server-Sent Events streaming."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.limiter import limiter
from app.schemas.restaurant import RestaurantResponse
from app.services import llm, query_parser, retrieval

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _sse_event(event_type: str, data: object) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _chat_stream(
    message: str,
    language: str | None,
    session: AsyncSession,
) -> AsyncIterator[str]:
    """Async generator that yields SSE events for the chat response.

    Phase 1: Parse query, fetch restaurants, emit restaurant data.
    Phase 2: Stream LLM response tokens.
    """
    try:
        # --- Phase 1: Parse + DB lookup ---
        filters = await query_parser.parse_query(message)
        detected_language = language or filters.language

        try:
            restaurants = await retrieval.search_restaurants(session, filters)
        except retrieval.NoRestaurantsFoundError:
            if filters.has_location:
                filters.location_radius_m = 1500
                try:
                    restaurants = await retrieval.search_restaurants(session, filters)
                except retrieval.NoRestaurantsFoundError:
                    restaurants = []
            else:
                restaurants = await retrieval.get_all_restaurants(session, limit=3)

        if not restaurants:
            yield _sse_event("error", "No restaurants found matching your criteria.")
            return

        # Emit restaurant cards immediately
        restaurant_responses = [
            RestaurantResponse.model_validate(r).model_dump(mode="json")
            for r in restaurants
        ]
        yield _sse_event("restaurants", restaurant_responses)

        # --- Phase 2: Stream LLM response ---
        yield _sse_event("status", "generating")

        location_hint = (
            f" near {filters.location_name}" if filters.location_name else ""
        )

        async for token in llm.generate_response_stream(
            user_message=message,
            restaurants=restaurants,
            language=detected_language,
            location_hint=location_hint,
        ):
            yield _sse_event("token", token)

        yield _sse_event("done", {"language": detected_language})

    except Exception as e:
        logger.error("Error in chat stream: %s", e, exc_info=True)
        yield _sse_event("error", "An error occurred while processing your request.")


@router.post("/chat")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: dict,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> StreamingResponse:
    """Stream chat response as Server-Sent Events.

    The client receives:
    - ``restaurants`` event with matched restaurant data
    - ``token`` events with LLM response chunks
    - ``done`` event when complete
    - ``error`` event on failure
    """
    message = body.get("message", "").strip()
    if not message or len(message) > 1000:
        return StreamingResponse(
            iter([_sse_event("error", "Message must be 1-1000 characters.")]),
            media_type="text/event-stream",
        )

    language = body.get("language")

    return StreamingResponse(
        _chat_stream(message, language, session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

Key changes:
- `body` is now `dict` instead of `ChatRequest` because `StreamingResponse` can't use Pydantic's `response_model`. Validation is done manually.
- `_sse_event()` helper formats SSE-compliant messages.
- `_chat_stream()` is the async generator that yields events in order.
- The `X-Accel-Buffering: no` header prevents nginx from buffering SSE.

**Step 2: Verify server starts**

Run: `python -c "from app.api.chat import router; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/api/chat.py
git commit -m "feat: convert chat endpoint to SSE streaming"
```

---

### Task 3: Update frontend to consume SSE stream

**Files:**
- Modify: `app/static/js/chat.js` (rewrite form submit handler)

**Step 1: Rewrite chat.js**

Replace the entire contents of `app/static/js/chat.js` with:

```javascript
// Chat interface with SSE streaming
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const chatMessages = document.getElementById('chat-messages');
const loadingIndicator = document.getElementById('loading');

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add a message bubble to the chat
function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return contentDiv;
}

// Format restaurant cards as HTML
function formatRestaurants(restaurants) {
    if (!restaurants || restaurants.length === 0) return '';

    let html = '<div class="restaurants-list">';
    restaurants.forEach(restaurant => {
        html += `<div class="restaurant-card">
            <h4>${escapeHtml(restaurant.name)}</h4>
            <p><strong>Address:</strong> ${escapeHtml(restaurant.address)}</p>
            <p class="cuisine"><strong>Cuisine:</strong> ${restaurant.cuisine.map(c => escapeHtml(c)).join(', ')}</p>
            <p><strong>Price:</strong> ${escapeHtml(restaurant.price_range)}${restaurant.price_range_text ? ' (' + escapeHtml(restaurant.price_range_text) + ')' : ''}</p>`;

        if (restaurant.rating) {
            html += `<p class="rating"><strong>Rating:</strong> ${restaurant.rating}/5.0 (${restaurant.review_count} reviews)</p>`;
        }
        if (restaurant.phone) {
            html += `<p><strong>Phone:</strong> ${escapeHtml(restaurant.phone)}</p>`;
        }
        if (restaurant.website) {
            html += `<p><strong>Website:</strong> <a href="${escapeHtml(restaurant.website)}" target="_blank" rel="noopener">Visit</a></p>`;
        }
        if (restaurant.features && restaurant.features.length > 0) {
            html += `<p><strong>Features:</strong> ${restaurant.features.map(f => escapeHtml(f)).join(', ')}</p>`;
        }
        if (restaurant.menu_url) {
            html += `<p><strong>Menu:</strong> <a href="${escapeHtml(restaurant.menu_url)}" target="_blank" rel="noopener">View full menu</a></p>`;
        }
        html += '</div>';
    });
    html += '</div>';
    return html;
}

// Parse SSE events from a text chunk (may contain multiple events)
function parseSSEEvents(text) {
    const events = [];
    const lines = text.split('\n');
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            try {
                events.push(JSON.parse(line.slice(6)));
            } catch (e) {
                // Skip malformed events
            }
        }
    }
    return events;
}

// Handle form submission with streaming
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const message = userInput.value.trim();
    if (!message) return;

    // Show user message
    addMessage(`<p>${escapeHtml(message)}</p>`, 'user');

    // Clear input, disable send
    userInput.value = '';
    sendButton.disabled = true;
    loadingIndicator.style.display = 'block';

    // Create assistant message bubble (will be filled by stream)
    let assistantContent = null;
    let textContainer = null;
    let accumulatedText = '';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE events in buffer
            const events = parseSSEEvents(buffer);
            // Keep only the incomplete trailing part
            const lastNewline = buffer.lastIndexOf('\n\n');
            buffer = lastNewline >= 0 ? buffer.slice(lastNewline + 2) : buffer;

            for (const event of events) {
                switch (event.type) {
                    case 'restaurants':
                        // Phase 1: Show restaurant cards immediately
                        loadingIndicator.style.display = 'none';
                        assistantContent = addMessage('', 'assistant');
                        assistantContent.innerHTML = formatRestaurants(event.data);
                        // Create text container for the LLM narrative
                        textContainer = document.createElement('div');
                        textContainer.className = 'response-text';
                        assistantContent.appendChild(textContainer);
                        break;

                    case 'status':
                        // Show typing indicator
                        if (textContainer) {
                            textContainer.innerHTML = '<em>Generating response...</em>';
                        }
                        break;

                    case 'token':
                        // Phase 2: Append streamed tokens
                        if (textContainer) {
                            if (accumulatedText === '') {
                                textContainer.innerHTML = '';  // Clear "Generating..."
                            }
                            accumulatedText += event.data;
                            textContainer.innerHTML = `<p>${escapeHtml(accumulatedText)}</p>`;
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }
                        break;

                    case 'done':
                        // Stream complete
                        break;

                    case 'error':
                        loadingIndicator.style.display = 'none';
                        if (!assistantContent) {
                            addMessage(`<p>${escapeHtml(event.data)}</p>`, 'assistant');
                        } else if (textContainer) {
                            textContainer.innerHTML = `<p>${escapeHtml(event.data)}</p>`;
                        }
                        break;
                }
            }
        }

        // If no events were received at all
        if (!assistantContent) {
            addMessage('<p>No response received. Please try again.</p>', 'assistant');
        }

    } catch (error) {
        console.error('Stream error:', error);
        loadingIndicator.style.display = 'none';
        if (!assistantContent) {
            addMessage(
                `<p>Sorry, I encountered an error: ${escapeHtml(error.message)}. Please try again.</p>`,
                'assistant'
            );
        }
    } finally {
        sendButton.disabled = false;
        loadingIndicator.style.display = 'none';
        userInput.focus();
    }
});

// Focus input on load
userInput.focus();
```

**Step 2: Verify no syntax errors**

Open `http://localhost:8000` in browser, check console for JS errors.

**Step 3: Commit**

```bash
git add app/static/js/chat.js
git commit -m "feat: update chat UI to consume SSE stream"
```

---

### Task 4: Update tests for streaming endpoint

**Files:**
- Modify: `tests/test_chat.py` (rewrite tests to handle SSE responses)

**Step 1: Rewrite test file**

Replace the contents of `tests/test_chat.py` with:

```python
"""Tests for streaming chat API endpoint."""

import json
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient

from app.models.restaurant import Restaurant
from app.services.query_parser import ParsedQuery


def _mock_parsed_query(**overrides: object) -> ParsedQuery:
    """Helper to create a mock ParsedQuery with defaults."""
    defaults: dict[str, object] = {
        "cuisine_types": [],
        "excluded_cuisines": [],
        "price_ranges": [],
        "excluded_price_ranges": [],
        "features": [],
        "dish_keywords": [],
        "location_name": None,
        "mood": None,
        "group_size": None,
        "time_preference": None,
        "sort_by": None,
        "language": "en",
    }
    defaults.update(overrides)
    return ParsedQuery(**defaults)  # type: ignore[arg-type]


def _parse_sse_events(content: str) -> list[dict]:
    """Parse SSE event stream into list of event dicts."""
    events = []
    for line in content.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


async def _stream_response(response) -> list[dict]:
    """Read full SSE response and parse events."""
    content = response.text
    return _parse_sse_events(content)


async def test_chat_streams_restaurants_then_tokens(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Test that SSE stream emits restaurants first, then tokens, then done."""

    async def mock_llm_stream(*args, **kwargs):
        yield "Hello "
        yield "world"

    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
            return_value=mock_llm_stream(),
        ),
    ):
        mock_parser.return_value = _mock_parsed_query(cuisine_types=["italian"])

        response = await client.post(
            "/api/chat", json={"message": "Italian restaurants"}
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = _parse_sse_events(response.text)
        event_types = [e["type"] for e in events]

        # Must have restaurants, then status, then tokens, then done
        assert "restaurants" in event_types
        assert "done" in event_types

        # Restaurants event should have a list
        restaurants_event = next(e for e in events if e["type"] == "restaurants")
        assert isinstance(restaurants_event["data"], list)
        assert len(restaurants_event["data"]) > 0

        # Token events should contain our mock text
        token_events = [e for e in events if e["type"] == "token"]
        full_text = "".join(e["data"] for e in token_events)
        assert full_text == "Hello world"


async def test_chat_validation_error(client: AsyncClient) -> None:
    """Test that empty message returns an SSE error event."""
    response = await client.post(
        "/api/chat",
        json={"message": ""},
    )

    assert response.status_code == 200  # SSE always returns 200
    events = _parse_sse_events(response.text)
    assert any(e["type"] == "error" for e in events)


async def test_chat_language_detection(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Test that detected language is included in the done event."""

    async def mock_llm_stream(*args, **kwargs):
        yield "Antwort"

    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
            return_value=mock_llm_stream(),
        ),
    ):
        mock_parser.return_value = _mock_parsed_query(language="de")

        response = await client.post(
            "/api/chat", json={"message": "Ich suche ein Restaurant"}
        )

        events = _parse_sse_events(response.text)
        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["data"]["language"] == "de"


async def test_chat_no_restaurants_found(
    client: AsyncClient,
) -> None:
    """Test error event when no restaurants match."""

    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.retrieval.search_restaurants",
            new_callable=AsyncMock,
            side_effect=__import__("app.services.retrieval", fromlist=["NoRestaurantsFoundError"]).NoRestaurantsFoundError("none"),
        ),
        patch(
            "app.services.retrieval.get_all_restaurants",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        mock_parser.return_value = _mock_parsed_query()

        response = await client.post(
            "/api/chat", json={"message": "martian food"}
        )

        events = _parse_sse_events(response.text)
        assert any(e["type"] == "error" for e in events)
```

**Step 2: Run tests**

Run: `pytest tests/test_chat.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_chat.py
git commit -m "test: update chat tests for SSE streaming"
```

---

### Task 5: Manual verification and final commit

**Step 1: Start the dev server**

Run: `uvicorn app.main:app --reload --port 8000`

**Step 2: Test in browser**

1. Open `http://localhost:8000`
2. Type "Italian restaurants" and send
3. Verify: restaurant cards appear first (~2-4s)
4. Verify: LLM text streams in token-by-token after cards
5. Verify: input re-enables after stream completes

**Step 3: Run full test suite**

Run: `pytest -x`
Expected: All tests pass

**Step 4: Lint**

Run: `ruff check app/api/chat.py app/services/llm.py`
Expected: No errors
