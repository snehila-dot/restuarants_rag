# Conversation Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add client-side conversation memory so users can refine searches, ask follow-ups, and compare restaurants within a browser session.

**Architecture:** The browser keeps the last 5 user+assistant message pairs in JS memory and sends them with each `/api/chat` request. The query parser uses history to resolve references ("but cheaper"). The LLM receives the full conversation for contextual responses. No database changes. No server-side state.

**Tech Stack:** FastAPI (Pydantic request model), OpenAI chat completions (multi-turn messages), vanilla JavaScript (in-memory array)

**Design doc:** `docs/plans/2026-03-17-conversation-memory-design.md`

---

## Task 1: Add ConversationMessage Schema and Update ChatRequest

**Files:**
- Modify: `app/schemas/restaurant.py`
- Test: `tests/test_chat.py`

**Step 1: Write the failing test**

Add to `tests/test_chat.py`:

```python
async def test_chat_accepts_conversation_history(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Chat endpoint accepts conversation_history in request body."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
            return_value=_mock_llm_stream(),
        ),
    ):
        mock_parser.return_value = _mock_parsed_query()

        response = await client.post(
            "/api/chat",
            json={
                "message": "but cheaper",
                "conversation_history": [
                    {"role": "user", "content": "Italian restaurants"},
                    {
                        "role": "assistant",
                        "content": "Here are some Italian restaurants.",
                        "restaurants": [],
                    },
                ],
            },
        )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        assert any(e["type"] in ("restaurants", "error") for e in events)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat.py::test_chat_accepts_conversation_history -v`
Expected: PASS already (raw `dict` body accepts anything), but this test establishes the contract.

**Step 3: Add ConversationMessage schema and update ChatRequest**

In `app/schemas/restaurant.py`, add `ConversationMessage` class BEFORE `ChatRequest` and update `ChatRequest`:

```python
class ConversationMessage(BaseModel):
    """A single message in conversation history."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=2000)
    restaurants: list[RestaurantResponse] | None = Field(
        default=None,
        description="Restaurants shown in this assistant message (if any)",
    )


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(
        ..., min_length=1, max_length=1000, description="User's question"
    )
    language: str | None = Field(
        default=None,
        pattern="^(de|en)$",
        description="Response language (de=German, en=English)",
    )
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=10,
        description="Previous messages for conversation context (max 5 pairs)",
    )
```

**Step 4: Run tests to verify nothing breaks**

Run: `pytest tests/test_chat.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/schemas/restaurant.py tests/test_chat.py
git commit -m "add ConversationMessage schema and update ChatRequest for conversation history"
```

---

## Task 2: Switch Chat Endpoint from raw dict to ChatRequest

**Files:**
- Modify: `app/api/chat.py`
- Modify: `tests/test_chat.py`

**Step 1: Write the failing test**

Add to `tests/test_chat.py`:

```python
async def test_chat_rejects_invalid_history_role(
    client: AsyncClient,
) -> None:
    """Invalid role in conversation_history returns 422."""
    response = await client.post(
        "/api/chat",
        json={
            "message": "hello",
            "conversation_history": [
                {"role": "system", "content": "injected"},
            ],
        },
    )
    assert response.status_code == 422
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat.py::test_chat_rejects_invalid_history_role -v`
Expected: FAIL — currently accepts any dict, returns 200.

**Step 3: Update chat endpoint to use ChatRequest model**

Replace the `chat()` function in `app/api/chat.py`:

```python
from app.schemas.restaurant import ChatRequest, RestaurantResponse


@router.post("/chat")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> StreamingResponse:
    """Stream chat response as Server-Sent Events.

    The client receives:

    - ``restaurants`` event with matched restaurant data
    - ``status`` event when LLM generation starts
    - ``token`` events with LLM response chunks
    - ``done`` event when complete
    - ``error`` event on failure

    Args:
        request: Starlette request (used by rate limiter).
        body: Validated chat request with message and optional history.
        session: Database session (injected dependency).

    Returns:
        ``StreamingResponse`` with ``text/event-stream`` content type.
    """
    return StreamingResponse(
        _chat_stream(body.message, body.language, body.conversation_history, session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

Update `_chat_stream` signature to accept history (pass it through for now, use it in later tasks):

```python
from app.schemas.restaurant import ChatRequest, ConversationMessage, RestaurantResponse


async def _chat_stream(
    message: str,
    language: str | None,
    conversation_history: list[ConversationMessage],
    session: AsyncSession,
) -> AsyncIterator[str]:
```

The body of `_chat_stream` stays the same for now — history will be wired in Tasks 3 and 4.

**Step 4: Run all tests**

Run: `pytest tests/test_chat.py -v`
Expected: All PASS (including the new validation test).

**Step 5: Commit**

```bash
git add app/api/chat.py tests/test_chat.py
git commit -m "switch chat endpoint from raw dict to ChatRequest with Pydantic validation"
```

---

## Task 3: Add Conversation Context to Query Parser

**Files:**
- Modify: `app/services/query_parser.py`
- Modify: `tests/test_query_parser.py`

**Step 1: Write the failing tests**

Add to `tests/test_query_parser.py`:

```python
@pytest.mark.asyncio
async def test_parse_query_with_history_resolves_refinement() -> None:
    """Follow-up 'but cheaper' with history produces lower price filter."""
    history = [
        {"role": "user", "content": "Italian restaurants"},
        {
            "role": "assistant",
            "content": "Here are some Italian restaurants in the €€ range.",
            "restaurants": [{"name": "Test Place", "cuisine": ["Italian"], "price_range": "€€"}],
        },
    ]

    mock_parsed = ParsedQuery(
        cuisine_types=["italian"],
        excluded_cuisines=[],
        price_ranges=["€"],
        excluded_price_ranges=[],
        features=[],
        dish_keywords=[],
        location_name=None,
        mood=None,
        group_size=None,
        time_preference=None,
        sort_by=None,
        language="en",
    )

    with patch(
        "app.services.query_parser._llm_extract", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.return_value = mock_parsed
        filters = await parse_query("but cheaper", conversation_history=history)

    # The mock returns what the LLM would produce given history context
    assert "italian" in filters.cuisine_types
    assert "€" in filters.price_ranges
    # Verify _llm_extract was called with the message (history is in the prompt)
    mock_extract.assert_called_once()


@pytest.mark.asyncio
async def test_parse_query_without_history_still_works() -> None:
    """parse_query with no history works exactly as before."""
    mock_parsed = ParsedQuery(
        cuisine_types=["italian"],
        excluded_cuisines=[],
        price_ranges=[],
        excluded_price_ranges=[],
        features=[],
        dish_keywords=[],
        location_name=None,
        mood=None,
        group_size=None,
        time_preference=None,
        sort_by=None,
        language="en",
    )

    with patch(
        "app.services.query_parser._llm_extract", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.return_value = mock_parsed
        filters = await parse_query("Italian food")

    assert filters.cuisine_types == ["italian"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_query_parser.py::test_parse_query_with_history_resolves_refinement -v`
Expected: FAIL — `parse_query()` doesn't accept `conversation_history` parameter.

**Step 3: Update parse_query and _llm_extract to accept history**

In `app/services/query_parser.py`:

1. Add a helper to format conversation history for the parser prompt:

```python
def _format_history_for_parser(
    conversation_history: list[dict[str, object]],
) -> str:
    """Build a conversation context summary for the parser LLM.

    Args:
        conversation_history: List of message dicts with role, content,
            and optional restaurants.

    Returns:
        Formatted context string, or empty string if no history.
    """
    if not conversation_history:
        return ""

    lines: list[str] = [
        "\nCONVERSATION CONTEXT (use to resolve references like "
        '"cheaper", "more like that", "which one", etc.):'
    ]
    for msg in conversation_history:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))[:200]  # Truncate long messages
        if role == "user":
            lines.append(f"- User: {content}")
        elif role == "assistant":
            restaurants = msg.get("restaurants") or []
            if restaurants and isinstance(restaurants, list):
                names = [
                    str(r.get("name", "?")) if isinstance(r, dict) else "?"
                    for r in restaurants[:5]
                ]
                lines.append(f"- Assistant: {content[:100]}...")
                lines.append(f"  (Showed: {', '.join(names)})")
            else:
                lines.append(f"- Assistant: {content}")

    lines.append(
        "The user's NEW message follows. Resolve any references "
        "using the context above."
    )
    return "\n".join(lines)
```

2. Update `_llm_extract` to accept and pass history context:

```python
async def _llm_extract(
    message: str,
    conversation_history: list[dict[str, object]] | None = None,
) -> ParsedQuery:
    """Extract structured query filters from a user message using the LLM."""
    try:
        system_content = PARSER_SYSTEM_PROMPT
        if conversation_history:
            system_content += _format_history_for_parser(conversation_history)

        response = await _parser_client.beta.chat.completions.parse(
            model=settings.parser_model,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        "Extract restaurant search filters from this"
                        f' message: "{message}"'
                    ),
                },
            ],
            response_format=ParsedQuery,
            temperature=0.0,
            timeout=5.0,
        )
        # ... rest stays the same
```

3. Update `parse_query` signature:

```python
async def parse_query(
    message: str,
    conversation_history: list[dict[str, object]] | None = None,
) -> QueryFilters:
    """Parse user query to extract structured filters.

    Uses LLM structured extraction (gpt-4o-mini) with keyword fallback
    on API failure.

    Args:
        message: User's natural language query
        conversation_history: Previous messages for resolving follow-ups

    Returns:
        QueryFilters with extracted and enriched filters
    """
    try:
        parsed = await _llm_extract(message, conversation_history)
    except Exception as e:
        logger.warning(
            "LLM extraction failed (%s), using keyword fallback", type(e).__name__
        )
        parsed = _keyword_fallback(message)
    # ... rest stays the same
```

**Step 4: Run all query parser tests**

Run: `pytest tests/test_query_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/services/query_parser.py tests/test_query_parser.py
git commit -m "add conversation history context to query parser for follow-up resolution"
```

---

## Task 4: Add Conversation History to LLM Service

**Files:**
- Modify: `app/services/llm.py`
- Create: `tests/test_llm.py`

**Step 1: Write the failing test**

Create `tests/test_llm.py`:

```python
"""Tests for LLM service."""

from unittest.mock import AsyncMock, MagicMock, patch

from app.services.llm import generate_response_stream


async def test_generate_response_stream_includes_history() -> None:
    """Conversation history is included in LLM messages."""
    history = [
        {"role": "user", "content": "Italian restaurants"},
        {"role": "assistant", "content": "Here are some Italian restaurants."},
    ]

    captured_messages: list[dict] = []

    async def mock_create(**kwargs: object) -> MagicMock:
        captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="ok"))]

        async def single_chunk():
            yield mock_chunk

        return single_chunk()

    with patch("app.services.llm.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)

        tokens = []
        async for token in generate_response_stream(
            user_message="but cheaper",
            restaurants=[],
            language="en",
            conversation_history=history,
        ):
            tokens.append(token)

    # Should have: system, user1, assistant1, user2
    assert len(captured_messages) >= 4
    assert captured_messages[0]["role"] == "system"
    assert captured_messages[1]["role"] == "user"
    assert captured_messages[1]["content"] == "Italian restaurants"
    assert captured_messages[2]["role"] == "assistant"
    assert captured_messages[-1]["role"] == "user"
    assert captured_messages[-1]["content"] == "but cheaper"


async def test_generate_response_stream_works_without_history() -> None:
    """Stream works with empty history (backward compat)."""
    async def mock_create(**kwargs: object) -> MagicMock:
        messages = kwargs.get("messages", [])
        assert len(messages) == 2  # system + user only

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="hi"))]

        async def single_chunk():
            yield mock_chunk

        return single_chunk()

    with patch("app.services.llm.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)

        tokens = []
        async for token in generate_response_stream(
            user_message="hello",
            restaurants=[],
            language="en",
        ):
            tokens.append(token)

    assert tokens == ["hi"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL — `generate_response_stream` doesn't accept `conversation_history`.

**Step 3: Update generate_response_stream to accept and use history**

In `app/services/llm.py`, update `generate_response_stream`:

```python
async def generate_response_stream(
    user_message: str,
    restaurants: list[Restaurant],
    language: str = "en",
    location_hint: str = "",
    conversation_history: list[dict[str, object]] | None = None,
) -> AsyncIterator[str]:
    """Stream LLM response tokens.

    Yields individual text chunks as they arrive from the OpenAI API.
    Falls back to a single-chunk fallback message on error.

    Args:
        user_message: The user's original question.
        restaurants: List of relevant restaurants.
        language: Response language ('en' or 'de').
        location_hint: Optional location context.
        conversation_history: Previous messages for multi-turn conversation.

    Yields:
        Text chunks from the LLM response.
    """
    system_prompt = SYSTEM_PROMPT_DE if language == "de" else SYSTEM_PROMPT_EN
    restaurant_data = format_restaurant_data(restaurants)

    system_content = system_prompt + "\n\n--- RESTAURANT DATA ---\n" + restaurant_data
    if location_hint:
        system_content += (
            f"\nNote: These restaurants are located{location_hint}. "
            "Mention their proximity to this area in your response."
        )
    system_content += (
        "\n--- END DATA ---\n\n"
        "Answer the user's question using ONLY the restaurant data above. "
        "Be concise and helpful."
    )

    # Build messages: system + history + current user message
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

    if conversation_history:
        for msg in conversation_history:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    try:
        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
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
            yield (
                f"Ich habe {len(restaurants)} Restaurant(s) für Sie gefunden. "
                "Bitte sehen Sie sich die Details unten an."
            )
        else:
            yield (
                f"I found {len(restaurants)} restaurant(s) for you. "
                "Please see the details below."
            )
```

Also update `generate_response` (non-streaming) with the same pattern for consistency, adding `conversation_history: list[dict[str, object]] | None = None` parameter.

**Step 4: Run tests**

Run: `pytest tests/test_llm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/services/llm.py tests/test_llm.py
git commit -m "add conversation history to LLM service for multi-turn responses"
```

---

## Task 5: Wire History Through the Chat Stream

**Files:**
- Modify: `app/api/chat.py`
- Modify: `tests/test_chat.py`

**Step 1: Write the failing test**

Add to `tests/test_chat.py`:

```python
async def test_chat_passes_history_to_llm(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Conversation history is forwarded to LLM service."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
        ) as mock_llm,
    ):
        mock_parser.return_value = _mock_parsed_query()
        mock_llm.return_value = _mock_llm_stream()

        history = [
            {"role": "user", "content": "Italian food"},
            {"role": "assistant", "content": "Here are restaurants."},
        ]

        await client.post(
            "/api/chat",
            json={"message": "but cheaper", "conversation_history": history},
        )

        # Verify LLM was called with conversation_history
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        assert "conversation_history" in call_kwargs.kwargs or len(call_kwargs.args) > 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat.py::test_chat_passes_history_to_llm -v`
Expected: FAIL — `_chat_stream` doesn't pass history to services yet.

**Step 3: Wire history through `_chat_stream`**

Update `_chat_stream` in `app/api/chat.py` to pass history to both services:

```python
async def _chat_stream(
    message: str,
    language: str | None,
    conversation_history: list[ConversationMessage],
    session: AsyncSession,
) -> AsyncIterator[str]:
    try:
        # Convert Pydantic models to dicts for service layer
        history_dicts = [
            msg.model_dump(mode="json") for msg in conversation_history
        ]

        # --- Phase 1: Parse + DB lookup ---
        filters = await query_parser.parse_query(
            message, conversation_history=history_dicts
        )
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
            conversation_history=history_dicts,
        ):
            yield _sse_event("token", token)

        yield _sse_event("done", {"language": detected_language})

    except Exception as e:
        logger.error("Error in chat stream: %s", e, exc_info=True)
        yield _sse_event(
            "error",
            "An error occurred while processing your request.",
        )
```

**Step 4: Run all chat tests**

Run: `pytest tests/test_chat.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add app/api/chat.py tests/test_chat.py
git commit -m "wire conversation history through chat stream to parser and LLM"
```

---

## Task 6: Frontend — Track Conversation History

**Files:**
- Modify: `app/static/js/chat.js`

**Step 1: No automated test (JS is untested — manual verification)**

**Step 2: Add conversation history tracking to chat.js**

At the top of `chat.js`, add:

```javascript
// Conversation history — last 5 pairs (10 messages), client-side only
let conversationHistory = [];
const MAX_HISTORY_PAIRS = 5;
```

In the submit handler, update to:

1. Push user message to history before sending
2. Include `conversation_history` in the fetch body
3. On `"done"` event, push assistant message + restaurants to history

Key changes in the submit handler:

```javascript
// Before fetch:
conversationHistory.push({ role: 'user', content: message });

// In fetch body:
body: JSON.stringify({
    message: message,
    conversation_history: conversationHistory.slice(-(MAX_HISTORY_PAIRS * 2)),
}),

// Track restaurants received during stream:
var streamedRestaurants = [];
// In 'restaurants' case:
streamedRestaurants = event.data;

// In 'done' case:
conversationHistory.push({
    role: 'assistant',
    content: accumulatedText,
    restaurants: streamedRestaurants,
});
// Trim to max pairs
if (conversationHistory.length > MAX_HISTORY_PAIRS * 2) {
    conversationHistory = conversationHistory.slice(-(MAX_HISTORY_PAIRS * 2));
}
```

**Step 3: Manual verification**

1. Start server: `uvicorn app.main:app --reload --port 8000`
2. Open http://localhost:8000
3. Send: "Italian restaurants"
4. Send: "but cheaper" — should refine results
5. Send: "which one has vegan options?" — should answer from prior results
6. Verify browser console shows no errors

**Step 4: Commit**

```bash
git add app/static/js/chat.js
git commit -m "track conversation history client-side and send with each request"
```

---

## Task 7: Add "New Conversation" Button

**Files:**
- Modify: `app/templates/index.html`
- Modify: `app/static/css/style.css`
- Modify: `app/static/js/chat.js`

**Step 1: Add button to template**

In `app/templates/index.html`, add a "New conversation" button inside the chat header:

```html
<div class="chat-header">
    <h2>Ask me about restaurants in Graz!</h2>
    <p>Try: "vegan restaurants", "cheap Italian food", "restaurants with outdoor seating"</p>
    <button id="new-chat-button" class="new-chat-button" type="button" style="display: none;">
        New conversation
    </button>
</div>
```

**Step 2: Add CSS for the button**

In `app/static/css/style.css`, add:

```css
.new-chat-button {
    margin-top: 0.75rem;
    padding: 0.4rem 1rem;
    background: rgba(255, 255, 255, 0.2);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 16px;
    font-size: 0.85rem;
    cursor: pointer;
    transition: background 0.2s;
}

.new-chat-button:hover {
    background: rgba(255, 255, 255, 0.3);
}
```

**Step 3: Add JS handler**

In `app/static/js/chat.js`, add:

```javascript
const newChatButton = document.getElementById('new-chat-button');

function startNewConversation() {
    conversationHistory = [];
    newChatButton.style.display = 'none';
    // Clear chat messages except the welcome message
    while (chatMessages.children.length > 1) {
        chatMessages.removeChild(chatMessages.lastChild);
    }
    userInput.focus();
}

newChatButton.addEventListener('click', startNewConversation);
```

Show the button after the first response:

```javascript
// In the 'done' case of the event handler:
if (conversationHistory.length >= 2) {
    newChatButton.style.display = 'inline-block';
}
```

**Step 4: Manual verification**

1. Refresh page — button should be hidden
2. Send a message — button appears after response
3. Click "New conversation" — chat clears, history resets
4. Send another message — works as fresh conversation

**Step 5: Commit**

```bash
git add app/templates/index.html app/static/css/style.css app/static/js/chat.js
git commit -m "add 'New conversation' button to reset chat context"
```

---

## Task 8: Final Integration Test and Cleanup

**Files:**
- Modify: `tests/test_chat.py`

**Step 1: Add integration test for full history flow**

```python
async def test_chat_full_conversation_flow(
    client: AsyncClient,
    sample_restaurants: list[Restaurant],
) -> None:
    """Full multi-turn conversation produces valid SSE events at each step."""
    with (
        patch(
            "app.services.query_parser._llm_extract", new_callable=AsyncMock
        ) as mock_parser,
        patch(
            "app.services.llm.generate_response_stream",
        ) as mock_llm,
    ):
        # Turn 1: Initial query
        mock_parser.return_value = _mock_parsed_query(cuisine_types=["italian"])
        mock_llm.return_value = _mock_llm_stream()

        r1 = await client.post("/api/chat", json={"message": "Italian food"})
        events1 = _parse_sse_events(r1.text)
        assert any(e["type"] == "restaurants" for e in events1)

        # Turn 2: Follow-up with history
        mock_parser.return_value = _mock_parsed_query(
            cuisine_types=["italian"], price_ranges=["€"]
        )
        mock_llm.return_value = _mock_llm_stream()

        r2 = await client.post(
            "/api/chat",
            json={
                "message": "but cheaper",
                "conversation_history": [
                    {"role": "user", "content": "Italian food"},
                    {"role": "assistant", "content": "Here are Italian restaurants."},
                ],
            },
        )
        events2 = _parse_sse_events(r2.text)
        assert any(e["type"] == "restaurants" for e in events2)
        assert any(e["type"] == "done" for e in events2)
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Run linter and type checker**

Run: `ruff check app/ tests/ && ruff format app/ tests/`
Expected: Clean

Run: `mypy app/`
Expected: Clean (or only pre-existing errors)

**Step 4: Commit**

```bash
git add tests/test_chat.py
git commit -m "add integration test for multi-turn conversation flow"
```

---

## Summary

| Task | What | Files Changed |
|------|------|---------------|
| 1 | ConversationMessage schema + ChatRequest update | schemas, test_chat |
| 2 | Switch endpoint from raw dict to ChatRequest | chat.py, test_chat |
| 3 | Query parser accepts history context | query_parser, test_query_parser |
| 4 | LLM service accepts history as messages | llm.py, test_llm |
| 5 | Wire history through _chat_stream | chat.py, test_chat |
| 6 | Frontend tracks and sends history | chat.js |
| 7 | "New conversation" button | index.html, style.css, chat.js |
| 8 | Integration test + lint/type check | test_chat |

**Total**: 8 tasks, ~8 commits. Backend is fully TDD. Frontend is manually verified.
