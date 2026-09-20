#!/usr/bin/env python3
"""Bounded CPU exact-tokenizer prompt-budget audit for PICS v3 (no GPU/Slurm).

Audits G.1-style prompts under structure_aware_v3 + 14000/5000 for all 15
datasets. G.2 transfer uses preliminary source programs for preflight only;
regenerate after final pics_v3 Occurrence-EB.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

# Soften broken openai/aiohttp combos in some login-node envs.
if "aiohttp" not in sys.modules:
    aio = types.ModuleType("aiohttp")
    aio.SocketTimeoutError = TimeoutError
    aio.ServerTimeoutError = TimeoutError
    sys.modules["aiohttp"] = aio

from utils.teh.g2_paired_packing import pad_parent_to_max_chars
from utils.teh.pics_v3 import (
    HARD_PROMPT_TOKEN_CAP,
    LLM_MAX_TOKENS,
    MAX_PARENT_CHARS,
    SAMPLE_SIZE,
    VLLM_MAX_MODEL_LEN,
)
from utils.teh.prompt_snapshots import prompt_contract_scope
from utils.teh.prompt_units import OUTPUT_RESERVE, qwen_user_prompt_token_count
from utils.teh.t_pics_gated_transfer import default_seed_path

OUT = (
    REPO
    / "analysis_2026Sep/Sep20_V2/others/debug/pics_v3_prompt_budget_audit.csv"
)
DATASETS = [
    "1peterson2021using",
    "2plonsky2018when",
    "3frey2017cct",
    "4wulff2018description",
    "5speekenbrink2008learning",
    "7hilbig2014generalized",
    "10frey2017risk",
    "11enkavi2019recentprobes",
    "12badham2017deficits",
    "mixed_gambles",
    "bergert_nosofsky_2007",
    "guan_2020_stopping",
    "steyvers_2009_bandit",
    "13schulz2020finding",
    "14kool2016when",
]
FOCUS = {
    "12badham2017deficits",
    "14kool2016when",
    "10frey2017risk",
    "5speekenbrink2008learning",
    "13schulz2020finding",
    "3frey2017cct",
}


def _pids(alias: str) -> List[int]:
    import teh as teh_mod
    from utils.teh.teh_datasets import emnlp_ordinal_range

    start, end = emnlp_ordinal_range(alias)
    return teh_mod.resolve_participants_for_scope(
        dataset=alias,
        repo_root=REPO,
        participant_scope="range",
        single_participant_id=0,
        range_start_ordinal=start,
        range_end_ordinal=end,
        all_max_participants=None,
        participant_ordinals=None,
        filter_mixed_gambles=False,
        split_ratio=0.6,
        split_seed=0,
        psych_dataset_split="train",
    )


def _audit_g1(alias: str) -> List[Dict[str, Any]]:
    import teh as teh_mod

    seed = (REPO / default_seed_path(alias)).read_text(encoding="utf-8")
    prompts_dir = REPO / "prompts" / "teh" / "cursor_designed"
    # Prefer run prompts when available; else cursor-designed single-file.
    infer_candidates = [
        REPO
        / f"generated_outputs/psych101_train/teh/{alias}/t_pics_g1_sa40_v2",
    ]
    infer_text = None
    run_prompts = None
    # Prefer clean cursor-designed Enkavi prompt; preliminary job 258518 is contaminated.
    if alias == "11enkavi2019recentprobes":
        cursor = prompts_dir / f"{alias}.txt"
        if cursor.is_file():
            infer_text = cursor.read_text(encoding="utf-8")
    if infer_text is None:
        infer_candidates = [
            REPO
            / f"generated_outputs/psych101_train/teh/{alias}/t_pics_g1_sa40_v2",
        ]
        for base in infer_candidates:
            jobs = (
                sorted(base.glob("job_*/prompts/infer_single_choice.txt"))
                if base.is_dir()
                else []
            )
            # Skip known contaminated Enkavi job if any remain.
            jobs = [j for j in jobs if "258518" not in str(j)]
            if jobs:
                infer_text = jobs[-1].read_text(encoding="utf-8")
                run_prompts = jobs[-1].parent
                break
    if infer_text is None:
        cursor = prompts_dir / f"{alias}.txt"
        infer_text = (
            cursor.read_text(encoding="utf-8")
            if cursor.is_file()
            else "Write def choose(problem, history): return 0.5\n"
        )
        run_prompts = None

    pids = _pids(alias)
    # Use a small pooled subset for speed: first 8 participants (or all if fewer).
    sample_pids = pids[: min(8, len(pids))]
    kw = dict(
        split_ratio=0.6,
        split_seed=0,
        psych_dataset_split="train",
        filter_mixed_gambles=alias == "mixed_gambles",
        limited_data_protocol="structure_aware_v3",
        limited_train_val=40,
    )
    buf = io.StringIO()
    with prompt_contract_scope(True), redirect_stdout(buf):
        pooled_train = teh_mod._collect_pooled_split_trials_for_participants(
            alias, sample_pids, split="train", **kw
        )
        pooled_val = teh_mod._collect_pooled_split_trials_for_participants(
            alias, sample_pids, split="val", **kw
        )

    rows = []
    phases = [
        ("g1_iter0_seed", [seed], 1),
        ("g1_up_to_8_parents", [pad_parent_to_max_chars(seed, MAX_PARENT_CHARS)] * SAMPLE_SIZE, SAMPLE_SIZE),
        ("g3_explore", [seed], 1),
        ("participant_evolution", [seed, seed], 2),
    ]
    for phase, parents, n_req in phases:
        with prompt_contract_scope(True), redirect_stdout(buf):
            prompt, diag, steps = teh_mod._truncate_psych_prompt_to_budget(
                base_prompt=infer_text,
                train_trials=pooled_train,
                train_trials_source=pooled_train,
                val_trials=pooled_val,
                val_trials_source=pooled_val,
                extra_prompt_trials_label="Validation observations",
                parent_programs=list(parents),
                parent_context_builder=lambda prompt_parent_programs, **_k: (
                    teh_mod._build_parent_context_for_prompt(
                        prompt_parent_programs=prompt_parent_programs,
                        num_parents=len(prompt_parent_programs),
                        dataset=alias,
                        fitness_metric="loglik",
                        parent_train_accuracies=None,
                        parent_train_mses=None,
                        parent_val_logliks=None,
                        parent_overall_logliks=None,
                        cpc18_official_mse=False,
                    )
                ),
                parent_context_kwargs={},
                code_template_suffix="",
                candidate_output_rules="\nProvide only the choose() function.\n",
                dataset=alias,
                dataset_type=alias,
                hard_prompt_token_cap=HARD_PROMPT_TOKEN_CAP,
                prompt_token_estimator="qwen_chat",
                max_prompt_train_trials=60,
                max_prompt_trials_per_problem=5,
                prompt_train_trials_seed=0,
                max_parent_chars=MAX_PARENT_CHARS,
                refinement_val_observations=False,
                pre_capped_train=False,
                pre_capped_val=False,
                runtime_contract="",
                freeze_examples=False,
            )
            tokens = qwen_user_prompt_token_count(prompt)
        n_train_after = int(diag.get("train_trials_after") or 0)
        n_val_after = int(diag.get("val_trials_after") or 0)
        n_ex = n_train_after + n_val_after
        n_avail = len(pooled_train) + len(pooled_val)
        parents_after = int(diag.get("parents_after") or len(parents))
        removed_ex = n_ex < min(60, n_avail)
        parent_clipped = any(
            len(p) > MAX_PARENT_CHARS for p in parents
        ) or any("# truncated; keep concise" in (p or "") for p in parents)
        # After truncation, check prompt parent copies.
        rows.append(
            {
                "dataset": alias,
                "phase": phase,
                "focus": alias in FOCUS,
                "examples_available": n_avail,
                "examples_included": n_ex,
                "parents_requested": n_req,
                "parents_retained": parents_after,
                "final_chat_template_tokens": tokens,
                "input_plus_1024": tokens + LLM_MAX_TOKENS,
                "fits_14000": tokens <= HARD_PROMPT_TOKEN_CAP,
                "fits_16384": tokens + LLM_MAX_TOKENS <= VLLM_MAX_MODEL_LEN,
                "examples_removed": removed_ex,
                "parent_copies_compacted_at_5000": bool(parent_clipped),
                "trim_steps": ",".join(steps) if steps else "none",
                "probe_in_set_in_prompt": "probe_in_set" in prompt,
            }
        )
    return rows


def main() -> int:
    qwen_user_prompt_token_count("warmup")
    all_rows: List[Dict[str, Any]] = []
    for alias in DATASETS:
        print(f"[audit] {alias}", flush=True)
        all_rows.extend(_audit_g1(alias))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys()) if all_rows else []
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {OUT} n={len(all_rows)}")
    fails = [r for r in all_rows if not r["fits_14000"] or not r["fits_16384"]]
    if fails:
        print("OVER_BUDGET:")
        for r in fails:
            print(
                f"  {r['dataset']} {r['phase']}: tokens={r['final_chat_template_tokens']} "
                f"ex={r['examples_included']}/{r['examples_available']} "
                f"parents={r['parents_retained']}/{r['parents_requested']} "
                f"steps={r['trim_steps']}"
            )
        return 1
    enkavi = [r for r in all_rows if r["dataset"] == "11enkavi2019recentprobes"]
    assert enkavi and not any(r["probe_in_set_in_prompt"] for r in enkavi)
    print("ALL_PHASES_FIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
