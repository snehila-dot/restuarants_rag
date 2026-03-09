# Streaming Chat Response Design

**Date**: 2026-03-09
**Status**: Approved
**Branch**: `feature/streaming-chat`

## Problem

The chat UI has a 5-12 second delay between sending a message and seeing any response. The endpoint makes two sequential OpenAI API calls (query parsing ~2-4s, response generation ~3-8s) and returns the complete result as JSON. The frontend waits for the full response before rendering anything.

## Solution

Replace the JSON response with Server-Sent Events (SSE) streaming. The response is sent in two phases:

1. **Restaurant cards** (fast, after query parsing + DB query, ~2-4s) — user sees results immediately
2. **LLM narrative** (streamed token-by-token) — text appears in real-time as it's generated

## SSE Event Protocol

```
← {"type": "restaurants", "data": [...]}     # Phase 1: DB results
← {"type": "status", "data": "generating"}   # Typing indicator
← {"type": "token", "data": "I"}             # Phase 2: streamed tokens
← {"type": "token", "data": " found"}
← ...
← {"type": "done", "data": {"language": "en"}}  # Stream complete
← {"type": "error", "data": "error message"}    # On failure
```

## Backend Changes

### `app/services/llm.py`
- Add `generate_response_stream()` — async generator yielding tokens via OpenAI `stream=True`
- Keep existing `generate_response()` unchanged (used by tests, fallback)

### `app/api/chat.py`
- Replace `ChatResponse` return with `StreamingResponse(media_type="text/event-stream")`
- Async generator: parse query → fetch restaurants → emit `restaurants` event → stream LLM tokens
- Rate limiter stays on the endpoint

### `app/schemas/restaurant.py`
- No schema changes needed — SSE events are ad-hoc JSON, not Pydantic models

## Frontend Changes

### `app/static/js/chat.js`
- Replace `fetch()` + `response.json()` with `fetch()` + `ReadableStream` reader
- Parse SSE events line-by-line from the stream
- On `restaurants` event: render restaurant cards immediately
- On `token` events: append text to assistant message bubble
- On `done` event: finalize message, re-enable input
- On `error` event: show error message

## Error Handling

- Stream failure mid-way: show whatever tokens arrived + "response interrupted"
- Query parsing failure: emit single `error` event, close stream
- OpenAI timeout (30s): emit `error` event
- Rate limit exceeded: return 429 before stream starts (existing behavior)

## What Doesn't Change

- `query_parser.py` — still non-streaming (structured extraction, not prose)
- Rate limiting — still `10/minute` on `/api/chat`
- Template HTML (`index.html`) — no changes needed
- CSS (`style.css`) — existing message bubble styles work
- Database layer — unchanged
