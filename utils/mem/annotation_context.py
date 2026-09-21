"""Canonical Schema-v5 annotation vLLM context defaults (population + participant).

Aligned with final PICS v3 / g5e50p30 synthesis experiments (16384).
Override ``max_model_len`` only for exceptional debugging — not the default path.

Budget rule for every annotation LLM call:

  input_tokens + reserved_output_tokens + safety_margin_tokens <= max_model_len

Never truncate reference or candidate program source; reduce batch size instead.
An oversized singleton must fail and be logged.
"""

from __future__ import annotations

# Canonical default for Schema-v5 annotation vLLM servers (pop + person).
ANNOTATION_VLLM_MAX_MODEL_LEN = 16384

# Completion budgets (also used as reserved_output in the packing inequality).
POPULATION_DEFAULT_MAX_TOKENS = 8192
PARTICIPANT_DEFAULT_MAX_TOKENS = 2048

# Soft packing caps (still subject to the token inequality above).
POPULATION_DEFAULT_BATCH_SIZE = 2
PARTICIPANT_DEFAULT_MAX_CANDIDATES_PER_BATCH = 5

# Safety margin inside the context window.
ANNOTATION_SAFETY_MARGIN_TOKENS = 512
