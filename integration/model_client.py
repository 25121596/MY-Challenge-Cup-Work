from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from profiling import parse_llama_profile


class ModelClient:
    """Minimal llama-server OpenAI-compatible client with runtime profiling."""

    def __init__(self, endpoint: str, timeout_s: int = 180):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = int(timeout_s)

    def _get_json(self, path: str) -> dict[str, Any]:
        url = self.endpoint + path
        with urllib.request.urlopen(url, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> dict[str, Any]:
        return self._get_json("/health")

    def get_model_id(self) -> str:
        payload = self._get_json("/v1/models")
        data = payload.get("data", [])
        if not data:
            raise RuntimeError(f"{self.endpoint}/v1/models 未返回模型")
        model_id = data[0].get("id")
        if not model_id:
            raise RuntimeError(f"{self.endpoint}/v1/models 返回结果中缺少 id")
        return str(model_id)

    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        model_id = self.get_model_id()
        url = self.endpoint + "/v1/chat/completions"
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 {url}: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        payload = json.loads(raw)
        try:
            message = payload["choices"][0]["message"]
            answer = message.get("content")
            if answer is None:
                # Some reasoning-capable builds may expose reasoning separately.
                answer = message.get("reasoning_content", "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"模型返回格式不是预期的 chat/completions 格式: {payload}") from exc

        profile = parse_llama_profile(payload, client_latency_ms=latency_ms)
        profile["request_prompt_bytes"] = len(prompt.encode("utf-8"))
        profile["response_answer_bytes"] = len(str(answer or "").encode("utf-8"))
        return {
            "endpoint": self.endpoint,
            "model": model_id,
            "latency_ms": round(latency_ms, 2),
            "answer": answer,
            "usage": payload.get("usage", {}),
            "timings": payload.get("timings", {}),
            "profile": profile,
            # Retain a small raw response for later parser upgrades without
            # needing to re-run the expensive model call.
            "raw_response": payload,
        }
