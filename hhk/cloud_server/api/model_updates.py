"""`GET /api/v1/edge/model-updates` — Model and rule update endpoint.

Edge devices periodically query this endpoint to check for available
LoRA weight updates and decision-rule changes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from config import config
from storage.models import ModelUpdateResponse, LoRAUpdate, RuleUpdate

logger = logging.getLogger("cloud_server.api.model_updates")

router = APIRouter(tags=["model_updates"])


@router.get("/api/v1/edge/model-updates", response_model=ModelUpdateResponse)
async def model_updates(
    device_id: str = Query(..., description="Edge device identifier"),
    current_lora_version: int = Query(0, description="Currently installed LoRA version"),
    current_rule_version: int = Query(0, description="Currently active rule version"),
):
    """Return available LoRA and rule updates for a device.

    The edge sends its current versions.  The cloud returns any updates
    that the edge has not yet applied.
    """
    logger.debug(
        "Model update check: device=%s, lora_ver=%d, rule_ver=%d",
        device_id,
        current_lora_version,
        current_rule_version,
    )

    result = ModelUpdateResponse()

    # ── LoRA updates ──
    if config.latest_lora_version > current_lora_version:
        result.lora_updates.append(
            LoRAUpdate(
                update_id=f"lora-quality-v{config.latest_lora_version}",
                url=f"{config.lora_base_url}/quality_v{config.latest_lora_version}.gguf",
                scale=1.0,
                version=config.latest_lora_version,
                checksum="a1b2c3d4",
            )
        )

    # ── Rule updates ──
    # In production this would query a rule database.  For now we serve a
    # static set of rules that can be extended.
    all_rules: list[RuleUpdate] = [
        RuleUpdate(
            rule_id="anomaly_threshold_v2",
            rule_type="threshold",
            target="anomaly_score",
            value={"min": 0.85, "max": 1.0},
            version=2,
        ),
        RuleUpdate(
            rule_id="confidence_threshold_v1",
            rule_type="threshold",
            target="cloud_review_confidence_threshold",
            value={"min": 0.78},
            version=1,
        ),
        RuleUpdate(
            rule_id="network_override_v1",
            rule_type="override",
            target="inference_mode",
            value={"mode": "local_only", "rtt_threshold_ms": 500},
            version=1,
        ),
    ]

    for rule in all_rules:
        if rule.version > current_rule_version:
            result.rule_updates.append(rule)

    return result
