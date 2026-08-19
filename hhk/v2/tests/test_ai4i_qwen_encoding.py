import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge-io-protocol"))

from ai4i_qwen_encoding import encode_ai4i_qwen_review  # noqa: E402
from edge_inference import build_prompt, load_grammar, parse_short_decision  # noqa: E402


def online_task() -> dict:
    return json.loads((ROOT / "data" / "standard_inputs" / "ai4i_milling_inference_v1.json")
                      .read_text(encoding="utf-8"))[0]


def test_compact_ai4i_prompt_uses_only_the_frozen_eight_features():
    task = online_task()
    prompt = encode_ai4i_qwen_review(task)
    assert prompt == build_prompt("ai4i_qwen_review", task)
    assert "AI4I_QWEN_8D_V1" in prompt
    assert "T=" in prompt and all(f"{field}=" in prompt for field in ("A", "P", "R", "Q", "W"))
    assert "ground_truth" not in prompt and "readings" not in prompt


@pytest.mark.parametrize("mutation", [
    lambda task: task["data"].update({"ground_truth": {"machine_failure": 1}}),
    lambda task: task["data"].update({"readings": {"Torque [Nm]": 40.0}}),
    lambda task: task["data"].update({"features": task["data"]["features"][:7]}),
    lambda task: task["data"]["features"].__setitem__(slice(0, 3), [1, 1, 0]),
    lambda task: task["data"]["features"].__setitem__(3, float("nan")),
])
def test_ai4i_prompt_rejects_non_online_or_invalid_features(mutation):
    task = copy.deepcopy(online_task())
    mutation(task)
    with pytest.raises(ValueError):
        encode_ai4i_qwen_review(task)


def test_ai4i_review_uses_binary_grammar_and_parser():
    grammar = load_grammar("ai4i_qwen_review")
    assert grammar is not None and "[01]" in grammar
    assert parse_short_decision("1", ai4i_review=True) == (1, None)
    assert parse_short_decision("2", ai4i_review=True) == (1, "UNKNOWN_CLASS")
