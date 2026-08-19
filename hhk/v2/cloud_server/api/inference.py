"""Inference endpoints: split inference and feature upload.

``POST /api/v1/infer/split`` — Split inference (hidden states)
``POST /api/v1/infer/features`` — Feature upload (visual features)
"""

from __future__ import annotations

import base64
import hashlib
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from inference.model_manager import model_manager
from storage.models import SplitInferenceRequest, FeatureUploadRequest

logger = logging.getLogger("cloud_server.api.inference")

router = APIRouter(tags=["inference"])


# ── Split Inference ──────────────────────────────────────────────


@router.post("/api/v1/infer/split")
async def split_inference(request: Request, body: SplitInferenceRequest):
    """Accept hidden states from an edge device and continue inference in the cloud.

    The edge runs the first K layers locally, then sends the intermediate hidden
    states (base64-encoded) along with inference context.  The cloud picks up
    from layer K+1 and streams tokens back via SSE.

    The hidden-state tensor shape is [batch, seq_len, hidden_dim] where
    hidden_dim = current_heads * head_dim.
    """
    logger.info(
        "Split inference: model=%s, layer=%d, seq_len=%d, dtype=%s, data_bytes=%d",
        body.model or "(default)",
        body.hidden.split_layer,
        body.hidden.seq_len,
        body.hidden.dtype,
        body.hidden.data_bytes,
    )

    # Decode and validate hidden states
    hidden_states = None
    if body.hidden.data_b64:
        raw = base64.b64decode(body.hidden.data_b64)
        # Verify checksum if provided
        if body.hidden.checksum:
            computed = hashlib.md5(raw).hexdigest()[:16]
            if computed != body.hidden.checksum:
                logger.warning(
                    "Checksum mismatch for split inference: expected %s, got %s",
                    body.hidden.checksum,
                    computed,
                )
        # Validate size
        if len(raw) != body.hidden.data_bytes:
            logger.warning(
                "Data bytes mismatch: expected %d, got %d",
                body.hidden.data_bytes,
                len(raw),
            )

        # hidden_dim = current_heads * head_dim
        hidden_dim = body.hidden.current_heads * body.hidden.head_dim
        expected_bytes = (
            body.hidden.batch_size
            * body.hidden.seq_len
            * hidden_dim
            * (2 if body.hidden.dtype == "float16" else 4)
        )
        if len(raw) != expected_bytes and len(raw) != body.hidden.data_bytes:
            logger.warning(
                "Hidden states size mismatch: got %d bytes, expected %d for shape [%d,%d,%d] %s",
                len(raw),
                expected_bytes,
                body.hidden.batch_size,
                body.hidden.seq_len,
                hidden_dim,
                body.hidden.dtype,
            )

    # Decode prompt tokens if present
    if body.context.prompt_tokens_b64:
        try:
            token_bytes = base64.b64decode(body.context.prompt_tokens_b64)
            # token IDs are int32
        except Exception:
            logger.warning("Failed to decode prompt_tokens_b64")

    # Stream tokens from cloud model
    async def event_stream():
        try:
            async for chunk in model_manager.generate_from_hidden(
                model=body.model,
                seq_len=body.hidden.seq_len,
                temperature=body.context.temperature,
                top_p=body.context.top_p,
                max_new_tokens=body.context.max_new_tokens,
            ):
                yield chunk
        except Exception as exc:
            logger.exception("Split inference streaming failed: %s", exc)
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


# ── Feature Upload ──────────────────────────────────────────────


@router.post("/api/v1/infer/features")
async def feature_upload(request: Request, body: FeatureUploadRequest):
    """Accept visual features from an edge device for cloud-side analysis.

    The edge extracts patch features (e.g. from a ViT encoder) and sends
    them as base64-encoded float32 arrays.  The cloud can run a larger
    multi-modal model on these features.

    If ``raw_bytes_b64`` is provided instead of ``features_b64``, the
    cloud receives the original image bytes.
    """
    logger.info(
        "Feature upload: model=%s, shape=%s, features_b64_len=%d",
        body.model or "(default)",
        body.media.feature_shape,
        len(body.media.features_b64),
    )

    # Decode features
    if body.media.features_b64:
        try:
            import numpy as np

            raw = base64.b64decode(body.media.features_b64)
            features = np.frombuffer(raw, dtype=np.float32)
            if body.media.feature_shape:
                features = features.reshape(body.media.feature_shape)
            logger.debug("Decoded feature tensor: shape=%s", features.shape)
        except Exception as exc:
            logger.warning("Failed to decode features_b64: %s", exc)
    elif body.media.raw_bytes_b64:
        try:
            raw = base64.b64decode(body.media.raw_bytes_b64)
            logger.debug("Decoded raw image: %d bytes", len(raw))
        except Exception as exc:
            logger.warning("Failed to decode raw_bytes_b64: %s", exc)

    # Stream tokens
    async def event_stream():
        try:
            async for chunk in model_manager.generate_from_features(
                prompt=body.prompt,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                yield chunk
        except Exception as exc:
            logger.exception("Feature upload streaming failed: %s", exc)
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
