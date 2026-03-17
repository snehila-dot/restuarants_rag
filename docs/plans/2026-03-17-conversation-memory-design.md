# Conversation Memory — Design Document

**Date**: 2026-03-17
**Status**: Approved
**Scope**: Add multi-turn conversation memory to the Graz restaurant chatbot

---

## Overview

Add client-side conversation memory so users can refine searches, ask follow-up questions about results, and compare restaurants within a single browser session. History lives in JavaScript memory (not persisted), sent with each API request.

**Supported follow-up interactions**:
- **Refine filters**: "But cheaper" / "Without Italian" / "Add outdoor seating"
- **Ask about results**: "Which one has vegan options?" / "What are their hours?"
- **Compare**: "Which is closest?" / "Compare the first two"

**Context window**: Last 5 user+assistant message pairs (10 messages total). Resets on page refresh.

---

## Data Flow

```
Browser (chat.js)                          Server (FastAPI)
─────────────────                          ─────────────────
conversationHistory: [                     POST /api/chat
  {role: "user", content: "..."},            body: {
  {role: "assistant", content: "...",          message: "but cheaper",
   restaurants: [...]},                        conversation_history: [
  ...                                           {role:"user", content:"italian food"},
]                                               {role:"assistant", content:"...",
                                                   restaurants: [{name:..., ...}]},
Keep last 5 pairs (10 messages).               ],
On each send: slice last 10,                 language: "en"  (optional)
attach to request body.                    }

                                           → query_parser receives current message
                                             + conversation history for context
                                           → retrieval runs as before
                                           → llm.generate_response_stream gets
                                             full conversation as messages[]
                                           → SSE stream back
```

The query parser always parses the *current* message but uses history to resolve references ("cheaper" → relative to previous price filters). The LLM receives the full conversation for natural, contextual responses.

---

## Backend Changes

### Schemas (`app/schemas/restaurant.py`)

New model for conversation history entries:

```python
class ConversationMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=2000)
    restaurants: list[RestaurantResponse] | None = None
```

Updated `ChatRequest` (replaces the current raw `dict` parameter — fixes existing tech debt):

```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    language: str | None = Field(default=None, pattern="^(de|en)$")
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=10,
        description="Previous conversation messages for context"
    )
```

### API Layer (`app/api/chat.py`)

1. Accept `ChatRequest` model instead of raw `dict`
2. Pass `conversation_history` to `query_parser.parse_query()` and `llm.generate_response_stream()`

### Query Parser (`app/services/query_parser.py`)

The parser prompt gains conversation context. Previous user messages and applied filters are appended to the system prompt so the LLM can resolve references:

```
CONVERSATION CONTEXT (use to resolve references):
- User asked: "Italian restaurants near Hauptplatz"
  → Filters applied: cuisine=italian, location=hauptplatz
  → Results: 3 restaurants shown (Trattoria Roma, Pizza Express, La Cucina)
- User now asks: "but cheaper"
  → Resolve: keep cuisine=italian, location=hauptplatz, add price_range=€
```

For "ask about results" and "compare" queries, the parser may return empty filters, meaning "re-use results from context." The chat handler detects this and uses cached results from the history instead of re-querying the DB.

### LLM Service (`app/services/llm.py`)

The LLM conversation includes actual history:

- Instead of `[system, user]`, send `[system, user₁, assistant₁, user₂, assistant₂, ..., userₙ]`
- Restaurant data from the *current* query stays in the system prompt
- The LLM naturally handles "which one has vegan options?" because it sees its own prior response

---

## Frontend Changes

### `app/static/js/chat.js`

New state variable:

```javascript
let conversationHistory = [];  // Array of {role, content, restaurants?}
```

On send:
1. Push `{role: "user", content: message}` to `conversationHistory`
2. Slice last 10 entries (5 pairs)
3. POST `/api/chat` with `{message, conversation_history: slicedHistory}`
4. On receiving response:
   - Accumulate streamed tokens into `fullText`
   - On `"restaurants"` event: capture restaurant data
   - On `"done"`: push `{role: "assistant", content: fullText, restaurants: [...]}` to history

### `app/templates/index.html`

Add a "New conversation" button that clears `conversationHistory` and optionally clears the chat DOM.

---

## Error Handling & Edge Cases

| Scenario | Behavior |
|----------|----------|
| First message (no history) | Works exactly as today — no change |
| History exceeds 5 pairs | Oldest pair silently dropped (FIFO) |
| Contradictory follow-up ("Italian" then "no Italian") | Parser treats latest message as authoritative |
| "Compare" with no prior results | LLM responds: "I don't have any restaurants to compare yet" |
| Page refresh | History clears, fresh start (expected) |
| Long conversation | Token cost bounded at ~2000 extra tokens max (5 pairs) |

---

## Testing Strategy

- **`test_chat.py`**: Test `conversation_history` is accepted and passed through; SSE events remain correct with history present
- **`test_query_parser.py`**: Test follow-up messages ("but cheaper", "which has vegan?") produce correct filters with prior context
- **`test_llm.py`**: Test conversation history is included in LLM messages array
- **Frontend**: Manual testing for history accumulation, "New conversation" button, refresh clearing

---

## Out of Scope (YAGNI)

- No database persistence for conversations
- No user accounts or authentication
- No "browse past chats" UI
- No token counting or smart truncation
- No streaming of the parser step
