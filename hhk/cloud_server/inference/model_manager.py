"""Unified model backend manager.

Selects the active backend based on config.model_backend and provides a
single interface for chat completions (used by the chat, split-inference,
and feature-upload endpoints).

Supported backends:
- "llama"  — proxies to llama-server
- "vllm"   — proxies to vLLM
- "mock"   — returns simulated tokens (for testing without a real model)
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from config import config

logger = logging.getLogger("cloud_server.inference.manager")


class ModelManager:
    """Manages the active inference backend."""

    def __init__(self) -> None:
        self.backend_name = config.model_backend
        self._backend: object = None
        self._initialized = False

    async def initialize(self) -> None:
        """Lazy-initialize the active backend."""
        if self._initialized:
            return
        self._initialized = True

        if self.backend_name == "llama":
            from inference.backend_llama import LlamaServerBackend

            self._backend = LlamaServerBackend(base_url=config.llama_server_url)
            logger.info("Using llama-server backend at %s", config.llama_server_url)
        elif self.backend_name == "vllm":
            from inference.backend_vllm import VLLMBackend

            self._backend = VLLMBackend(base_url=config.vllm_server_url)
            logger.info("Using vLLM backend at %s", config.vllm_server_url)
        else:
            logger.info("Using mock backend (no real model)")
            self._backend = None

    async def health(self) -> bool:
        """Check if the real model backend is healthy."""
        await self.initialize()
        if self._backend is None:
            return True  # mock is always "healthy"
        return await self._backend.health()  # type: ignore[union-attr]

    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        seed: int = 42,
    ) -> AsyncIterator[str]:
        """Stream chat completions as SSE text.

        If the backend is "mock", simulated tokens are generated.
        """
        await self.initialize()

        if self._backend is not None:
            async for chunk in self._backend.chat_completions(  # type: ignore[union-attr]
                messages=messages,
                model=model or config.default_model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                stream=True,
            ):
                yield chunk
        else:
            # Mock mode — generate simulated tokens
            from inference.streaming import token_sse_generator

            # Build a simple mock response from the last user message
            user_content = ""
            for m in messages:
                if m.get("role") == "user":
                    user_content = m.get("content", "")
            mock_text = f'[Mock] 基于"{user_content[:60]}..."的云端分析结果：系统运行正常。'
            tokens = list(mock_text)
            async for chunk in token_sse_generator(tokens, delay=0.02):
                yield chunk

    async def generate_from_hidden(
        self,
        model: str,
        seq_len: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
    ) -> AsyncIterator[str]:
        """Generate tokens from hidden states (split inference).

        In mock mode this simulates the continued inference.  With real
        backends, this would inject the hidden states into the model's
        intermediate layer.
        """
        await self.initialize()

        from inference.streaming import token_sse_generator

        if self._backend is not None:
            # Real backends: forward as a special prompt to chat completions
            # (a full implementation would do mid-layer injection)
            messages = [
                {
                    "role": "user",
                    "content": f"[Split] Continue from layer {seq_len}",
                }
            ]
            async for chunk in self._backend.chat_completions(  # type: ignore[union-attr]
                messages=messages,
                model=model or config.default_model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            ):
                yield chunk
        else:
            mock_text = "云端继续推理结果：确认为正常状态，无需干预。"
            tokens = list(mock_text)
            async for chunk in token_sse_generator(tokens, delay=0.03):
                yield chunk

    async def generate_from_features(
        self,
        prompt: str,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> AsyncIterator[str]:
        """Generate from visual features (feature upload endpoint)."""
        await self.initialize()

        from inference.streaming import token_sse_generator

        if self._backend is not None:
            messages = [{"role": "user", "content": prompt}]
            async for chunk in self._backend.chat_completions(  # type: ignore[union-attr]
                messages=messages,
                model=model or config.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk
        else:
            mock_text = f'[Mock视觉分析] 针对"{prompt[:40]}..."的分析：未检测到异常。'
            tokens = list(mock_text)
            async for chunk in token_sse_generator(tokens, delay=0.02):
                yield chunk

    async def close(self) -> None:
        if self._backend is not None and hasattr(self._backend, "close"):
            await self._backend.close()  # type: ignore[union-attr]


# Singleton
model_manager = ModelManager()
