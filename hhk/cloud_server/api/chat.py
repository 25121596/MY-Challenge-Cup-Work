"""`POST /v1/chat/completions` — Text offload (OpenAI-compatible).

The edge device sends standard OpenAI chat-completion requests when it
wants to offload text-based inference to a larger cloud model.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from inference.model_manager import model_manager
from storage.models import ChatCompletionRequest

logger = logging.getLogger("cloud_server.api.chat")

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """Handle a text offload request from an edge device.

    The request format follows the OpenAI Chat Completions API spec.
    The response is an SSE stream of token delta events, compatible with
    OpenAI's streaming format:

        data: {"choices":[{"delta":{"content":"token"},"finish_reason":null}]}
        data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":N}}
        data: [DONE]
    """
    logger.info(
        "Chat completions: model=%s, messages=%d, max_tokens=%d, stream=%s",
        body.model or "(default)",
        len(body.messages),
        body.max_tokens,
        body.stream,
    )

    # Convert messages to simple dicts for the backend
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    async def event_stream():
        try:
            async for chunk in model_manager.chat_completions(
                messages=messages,
                model=body.model,
                temperature=body.temperature,
                top_p=body.top_p,
                max_tokens=body.max_tokens,
                seed=body.seed,
            ):
                yield chunk
        except Exception as exc:
            logger.exception("Chat completions streaming failed: %s", exc)
            from inference.streaming import sse_error_event, sse_done_event

            yield sse_error_event(str(exc))
            yield sse_done_event()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
