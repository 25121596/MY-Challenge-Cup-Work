"""SSE (Server-Sent Events) streaming utilities.

All inference endpoints return token-by-token SSE streams in the format:

    data: {"choices":[{"delta":{"content":"token"},"finish_reason":null}]}
    data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":N}}
    data: [DONE]
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator


async def sse_event(data: dict | str) -> str:
    """Format a single SSE event line."""
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_done_event() -> str:
    """Return the [DONE] marker string."""
    return "data: [DONE]\n\n"


def sse_error_event(message: str) -> str:
    """Return an SSE error event."""
    return f"data: {json.dumps({'error': {'message': message}})}\n\n"


async def token_sse_generator(
    tokens: list[str],
    delay: float = 0.0,
    finish_reason: str = "stop",
) -> AsyncIterator[str]:
    """Generate an SSE stream from a list of token strings.

    Args:
        tokens: The token strings to emit one-by-one.
        delay: Optional artificial delay between tokens (seconds).
        finish_reason: The finish_reason value ("stop", "length", etc.).
    """
    total_tokens = len(tokens)
    for i, tok in enumerate(tokens):
        if delay > 0:
            await asyncio.sleep(delay)
        event = {
            "choices": [
                {
                    "delta": {"content": tok},
                    "finish_reason": None,
                }
            ]
        }
        yield await sse_event(event)

    # Final event with usage
    final = {
        "choices": [
            {
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"total_tokens": total_tokens},
    }
    yield await sse_event(final)
    yield sse_done_event()


async def sse_stream(
    token_iterable,
    delay: float = 0.0,
) -> AsyncIterator[str]:
    """Yield SSE events from any iterable of token strings."""
    tokens = list(token_iterable)
    async for chunk in token_sse_generator(tokens, delay=delay):
        yield chunk
