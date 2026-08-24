# AI4I Qwen Review Input v1

`AI4I_QWEN_8D_V1` is the compact input for the optional Qwen review expert.
It is derived only from the frozen online `ai4i_8d_v1` features contract.
The tabular AI4I MoE remains the primary local classifier and does not
tokenize this record or call Qwen.

The wire form is:

```text
AI4I_QWEN_8D_V1. AI4I failure review; 0=no failure pattern, 1=possible failure. T=type(H/L/M); A/P/R/Q/W=z*10. Output one digit only.
T=M A=-4 P=-3 R=+1 Q=+2 W=-17
```

Field order is fixed: `T`, `A`, `P`, `R`, `Q`, `W`.

| Field | Frozen source | Encoding |
| --- | --- | --- |
| `T` | Type one-hot `[H,L,M]` | exactly one of `H`, `L`, `M` |
| `A` | Air temperature z-score | z-score x 10, decimal ROUND_HALF_UP integer |
| `P` | Process temperature z-score | same |
| `R` | Rotational speed z-score | same |
| `Q` | Torque z-score | same |
| `W` | Tool wear z-score | same |

The encoder rejects a non-8D vector, invalid Type one-hot values, non-finite
values, a schema/route mismatch, and any `ground_truth` or `readings` field.
It therefore cannot leak labels or duplicate raw training fields into the
online Qwen request. The Qwen result is only an uncalibrated review
observation. It must not produce `failure_probability`, `confidence`, a
business priority, or a P0 safety decision.

The review route is explicit: call
`edge_inference.infer_ai4i_qwen_review(task)`. Calling the ordinary
`edge_inference.infer(task)` does not replace the tabular MoE route.

Measure actual token counts with the running model tokenizer, rather than
characters:

```bash
.venv-audit/bin/python tools/measure_ai4i_qwen_prompt_tokens.py
```

The frozen online corpus can be pre-encoded for offline review or deployment
fixtures without exposing training labels:

```bash
.venv-audit/bin/python data/build_ai4i_qwen_review_inputs.py
```
