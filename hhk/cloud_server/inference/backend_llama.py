"""llama-server backend adapter.

Proxies requests to a running llama-server instance that exposes an
OpenAI-compatible /v1/chat/completions endpoint.

The llama-server is launched separately (e.g. via ``llama-server -m model.gguf``).
This adapter simply forwards requests and streams responses back.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger("cloud_server.inference.llama")


class LlamaServerBackend:
    """Adapter for llama-server's OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8081") -> None:
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_keepalive_connections=10),
            )
        return self._client

    async def health(self) -> bool:
        """Check if llama-server is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        seed: int = 42,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Forward a chat completion request to llama-server and yield SSE chunks."""
        from inference.streaming import sse_event, sse_done_event

        client = await self._get_client()

        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": stream,
        }
        if model:
            payload["model"] = model

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    logger.error(
                        "llama-server returned %s: %s",
                        response.status_code,
                        text[:500],
                    )
                    yield await sse_event(
                        {
                            "error": {
                                "message": f"llama-server error {response.status_code}"
                            }
                        }
                    )
                    yield sse_done_event()
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
                    elif line.strip() == "":
                        continue
        except Exception as exc:
            logger.exception("llama-server request failed: %s", exc)
            yield await sse_event({"error": {"message": str(exc)}})
            yield sse_done_event()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
