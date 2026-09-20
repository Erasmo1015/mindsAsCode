"""G.2 paired target-example packing with the real Qwen tokenizer.

Reproduces all 15 audited control/transfer pairs for iteration 0, mixed-size
parents, and maximum permitted parent context. Does not launch jobs.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.teh.g2_paired_packing import (
    PairedPackingFitError,
    SOURCE_SUFFIX_MARKERS,
    assert_no_test_in_trials,
    fits_input_and_context,
    freeze_g2_target_examples,
    origin_splits_in_trials,
    pad_parent_to_max_chars,
    prompt_contains_source_suffix,
)
from utils.teh.prompt_snapshots import prompt_contract_scope
from utils.teh.prompt_units import QWEN_INPUT_CEILING, OUTPUT_RESERVE, VLLM_CONTEXT
from utils.teh.pics_v3 import SOURCE_YAML as PICS_V3_SOURCE_YAML
from utils.teh.t_pics_gated_transfer import default_seed_path, load_frozen_transfer_config

CAP = QWEN_INPUT_CEILING
MAX_PARENT_CHARS = 5000
SAMPLE_SIZE = 8
MAX_PROMPT_TRIALS = 60
SPLIT_SEED = 0
FREEZE_SEED = SPLIT_SEED + 60_000 + 1
PRELIMINARY_V2_CAP = 14_000
PRELIMINARY_V2_MAX_PARENT_CHARS = 3_500
BEFORE_CSV = (
    REPO
    / "analysis_2026Sep/Sep20_V2/others/debug/g2_control_transfer_prompt_budget.csv"
)
AFTER_CSV = (
    REPO
    / "analysis_2026Sep/Sep20_V3/others/debug/g2_paired_packing_before_after_pics_v3.csv"
)

LABELS = {
    "10frey2017risk": "Frey Risk",
    "11enkavi2019recentprobes": "Enkavi",
    "12badham2017deficits": "Badham",
    "13schulz2020finding": "Schulz",
    "14kool2016when": "Kool",
    "1peterson2021using": "Choice13k",
    "2plonsky2018when": "CPC18",
    "3frey2017cct": "Frey CCT",
    "4wulff2018description": "Wulff",
    "5speekenbrink2008learning": "Speekenbrink",
    "7hilbig2014generalized": "Hilbig",
    "bergert_nosofsky_2007": "Bergert",
    "guan_2020_stopping": "Guan",
    "mixed_gambles": "Mixed Gambles",
    "steyvers_2009_bandit": "Steyvers",
}


def test_fits_input_and_context_uses_live_output_reserve() -> None:
    """Raising output_reserve must tighten the vLLM fit check (OE-style)."""
    assert fits_input_and_context(14_000, input_ceiling=CAP, output_reserve=OUTPUT_RESERVE)
    assert not fits_input_and_context(14_000, input_ceiling=CAP, output_reserve=4096)
    assert fits_input_and_context(
        12_288, input_ceiling=12_288, output_reserve=4096
    )
    assert not fits_input_and_context(
        12_289, input_ceiling=12_288, output_reserve=4096
    )
    assert 12_288 + 4096 == VLLM_CONTEXT
    assert CAP == 14_000
    assert VLLM_CONTEXT == 16_384


def _cfg():
    assert PICS_V3_SOURCE_YAML.is_file(), f"missing {PICS_V3_SOURCE_YAML}"
    return load_frozen_transfer_config(PICS_V3_SOURCE_YAML)


def _targets() -> List[str]:
    return sorted(_cfg().targets)


@pytest.fixture(scope="module")
def qwen_ready():
    from utils.teh.prompt_units import qwen_user_prompt_token_count

    n = qwen_user_prompt_token_count("")
    assert n > 0
    return n


def _load_pair(target: str) -> Dict[str, Any]:
    from utils.prompt_flags import single_code_template_prompt_suffix
    from utils.teh.explore_source_prompt import build_rank1_explore_prompt_suffix
    from utils.teh.prompt_sanitize import CANDIDATE_OUTPUT_RULES
    import teh as teh_mod

    cfg = _cfg()
    entry = cfg.targets[target]
    source = entry.selected_source
    tgt_run = cfg.source_runs[target]
    run_dir = Path(tgt_run.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    prompts_dir = run_dir / "prompts"
    infer = (prompts_dir / "infer_single_choice.txt").read_text(encoding="utf-8")
    template_path = prompts_dir / "single_code_template.txt"
    code_template = ""
    if template_path.is_file():
        from utils.prompt_flags import load_single_code_template

        code_template = load_single_code_template(str(template_path))
    code_template_suffix = single_code_template_prompt_suffix(code_template)
    contract_path = prompts_dir / "runtime_contract.txt"
    runtime_contract = (
        contract_path.read_text(encoding="utf-8").strip() if contract_path.is_file() else ""
    )
    rank1 = Path(entry.rank1_program)
    if not rank1.is_absolute():
        rank1 = REPO / rank1
    source_code = rank1.read_text(encoding="utf-8")
    suffix = build_rank1_explore_prompt_suffix(
        source_dataset=source,
        program_path=str(rank1),
        split_seed=SPLIT_SEED,
        psych_dataset_split="train",
        split_ratio=0.6,
        limited_data_protocol="structure_aware_v3",
        limited_train_val=40,
        filter_mixed_gambles=source == "mixed_gambles",
        require_source_examples=True,
    )
    results = json.loads(
        (run_dir / "global_phase" / "results.json").read_text(encoding="utf-8")
    )
    pids = [int(x) for x in results["participant_ids"]]
    kw = dict(
        split_ratio=0.6,
        split_seed=SPLIT_SEED,
        psych_dataset_split="train",
        filter_mixed_gambles=target == "mixed_gambles",
        limited_data_protocol="structure_aware_v3",
        limited_train_val=40,
    )
    buf = io.StringIO()
    with prompt_contract_scope(True), redirect_stdout(buf):
        pooled_train = teh_mod._collect_pooled_split_trials_for_participants(
            target, pids, split="train", **kw
        )
        pooled_val = teh_mod._collect_pooled_split_trials_for_participants(
            target, pids, split="val", **kw
        )
    seed_path = REPO / default_seed_path(target)
    seed_code = seed_path.read_text(encoding="utf-8")
    freeze = freeze_g2_target_examples(
        dataset=target,
        pooled_train=pooled_train,
        pooled_val=pooled_val,
        infer_text=infer,
        source_suffix=suffix,
        runtime_contract=runtime_contract,
        seed_code=seed_code,
        code_template_suffix=code_template_suffix,
        candidate_output_rules=f"\n{CANDIDATE_OUTPUT_RULES}\n",
        max_parent_chars=MAX_PARENT_CHARS,
        max_prompt_train_trials=MAX_PROMPT_TRIALS,
        prompt_train_trials_seed=FREEZE_SEED,
        hard_prompt_token_cap=CAP,
        sample_size=SAMPLE_SIZE,
    )
    from utils.teh.g2_paired_packing import materialize_frozen_trials

    frozen_train, frozen_val = materialize_frozen_trials(
        freeze, pooled_train=pooled_train, pooled_val=pooled_val
    )
    return {
        "target": target,
        "source": source,
        "infer": infer,
        "suffix": suffix,
        "source_code": source_code,
        "runtime_contract": runtime_contract,
        "code_template_suffix": code_template_suffix,
        "candidate_output_rules": f"\n{CANDIDATE_OUTPUT_RULES}\n",
        "seed_code": seed_code,
        "freeze": freeze,
        "frozen_train": frozen_train,
        "frozen_val": frozen_val,
        "prompts_dir": prompts_dir,
    }


_PAIR_CACHE: Dict[str, Dict[str, Any]] = {}


def pair_bundle(target: str) -> Dict[str, Any]:
    if target not in _PAIR_CACHE:
        _PAIR_CACHE[target] = _load_pair(target)
    return _PAIR_CACHE[target]


def _parent_builder(dataset: str):
    import teh as teh_mod

    def _builder(*, prompt_parent_programs: List[str], **_k: Any) -> str:
        return teh_mod._build_parent_context_for_prompt(
            prompt_parent_programs=prompt_parent_programs,
            num_parents=len(prompt_parent_programs),
            dataset=dataset,
            fitness_metric="loglik",
            parent_train_accuracies=None,
            parent_train_mses=None,
            parent_val_logliks=None,
            parent_overall_logliks=None,
            cpc18_official_mse=False,
        )

    return _builder


def pack_arm(
    bundle: Dict[str, Any],
    *,
    arm: str,
    parents: Sequence[str],
) -> Dict[str, Any]:
    import teh as teh_mod

    suffix = bundle["suffix"] if arm == "transfer" else None
    base = bundle["infer"].rstrip()
    if suffix:
        base = f"{base}\n\n{suffix.strip()}\n"
    dataset = bundle["target"]
    ppc = max(1, int(bundle["freeze"].get("paired_parent_count") or SAMPLE_SIZE))
    parents = list(parents)[:ppc]
    buf = io.StringIO()
    with prompt_contract_scope(True), redirect_stdout(buf):
        prompt, diag, steps = teh_mod._truncate_psych_prompt_to_budget(
            base_prompt=base,
            train_trials=bundle["frozen_train"],
            train_trials_source=bundle["frozen_train"],
            val_trials=bundle["frozen_val"],
            val_trials_source=bundle["frozen_val"],
            extra_prompt_trials_label="Validation observations",
            parent_programs=list(parents),
            parent_context_builder=_parent_builder(dataset),
            parent_context_kwargs={},
            code_template_suffix=bundle["code_template_suffix"],
            candidate_output_rules=bundle["candidate_output_rules"],
            dataset=dataset,
            dataset_type=dataset,
            hard_prompt_token_cap=CAP,
            prompt_token_estimator="qwen_chat",
            max_prompt_train_trials=MAX_PROMPT_TRIALS,
            max_prompt_trials_per_problem=5,
            prompt_train_trials_seed=FREEZE_SEED,
            max_parent_chars=MAX_PARENT_CHARS,
            refinement_val_observations=False,
            pre_capped_train=True,
            pre_capped_val=True,
            runtime_contract=bundle["runtime_contract"],
            freeze_examples=True,
        )
        tokens = teh_mod.estimate_tokens(prompt)
    freeze = bundle["freeze"]
    example_ids = list(freeze["example_ids"])
    used = list(bundle["frozen_train"]) + list(bundle["frozen_val"])
    assert_no_test_in_trials(used, label=f"{dataset}/{arm}")
    return {
        "arm": arm,
        "prompt": prompt,
        "tokens": int(tokens),
        "steps": list(steps),
        "diag": diag,
        "example_ids": example_ids,
        "n_examples": int(freeze["n_examples_included"]),
        "n_available": int(freeze["n_selected_available"]),
        "parents_before": int(diag.get("parents_before") or len(parents)),
        "parents_after": int(diag.get("parents_after") or 0),
        "origin_splits": origin_splits_in_trials(used),
        "source_suffix_present": prompt_contains_source_suffix(prompt),
        "has_test": "test" in origin_splits_in_trials(used),
        "fits": fits_input_and_context(tokens, input_ceiling=CAP),
        "trial_cap_steps": [s for s in steps if "trials_cap" in s or s.startswith("per_problem_cap")],
    }


def mixed_parents(seed_code: str) -> List[str]:
    short = seed_code
    long = pad_parent_to_max_chars(seed_code, MAX_PARENT_CHARS)
    return [short, long, short, long, short, long, short, long]


def max_parents(seed_code: str) -> List[str]:
    long = pad_parent_to_max_chars(seed_code, MAX_PARENT_CHARS)
    return [long] * SAMPLE_SIZE


def _assert_pair(bundle: Dict[str, Any], ctrl: Dict[str, Any], xfer: Dict[str, Any]) -> None:
    assert ctrl["example_ids"] == xfer["example_ids"] == bundle["freeze"]["example_ids"]
    assert ctrl["n_examples"] == xfer["n_examples"] == bundle["freeze"]["n_examples_included"]
    assert int(bundle["freeze"].get("paired_parent_count") or 0) >= 1
    assert ctrl["parents_after"] == xfer["parents_after"]
    assert ctrl["tokens"] <= CAP
    assert xfer["tokens"] <= CAP
    assert ctrl["tokens"] + OUTPUT_RESERVE <= VLLM_CONTEXT
    assert xfer["tokens"] + OUTPUT_RESERVE <= VLLM_CONTEXT
    assert ctrl["fits"] and xfer["fits"]
    assert not ctrl["trial_cap_steps"]
    assert not xfer["trial_cap_steps"]
    assert not ctrl["has_test"] and not xfer["has_test"]
    assert ctrl["source_suffix_present"] is False
    assert xfer["source_suffix_present"] is True
    for marker in SOURCE_SUFFIX_MARKERS[:2]:
        assert marker not in ctrl["prompt"]
        assert marker in xfer["prompt"]
    source_code = bundle["source_code"].strip()
    assert source_code not in ctrl["prompt"]
    unique = next(
        (ln.strip() for ln in source_code.splitlines() if len(ln.strip()) > 24),
        source_code[:80],
    )
    assert unique in xfer["prompt"]
    assert "SOURCE example from source train+validation only" in xfer["prompt"]
    assert "(no trials available)" not in xfer["prompt"]
    assert "test" not in (ctrl["origin_splits"] + xfer["origin_splits"])


_RESULTS: List[Dict[str, Any]] = []


@pytest.mark.parametrize("target", _targets())
def test_g2_paired_packing_all_scenarios(target: str, qwen_ready: int) -> None:
    del qwen_ready
    bundle = pair_bundle(target)
    freeze = bundle["freeze"]
    assert freeze["transfer_required_tokens"] <= CAP
    seed = bundle["seed_code"]
    scenarios = {
        "iter0": [seed],
        "mixed": mixed_parents(seed),
        "max_parents": max_parents(seed),
    }
    row: Dict[str, Any] = {
        "target": target,
        "target_label": LABELS.get(target, target),
        "source": bundle["source"],
        "n_available": freeze["n_selected_available"],
        "n_included": freeze["n_examples_included"],
        "required_tokens": freeze["transfer_required_tokens"],
        "freeze_tokens": freeze["transfer_packed_tokens_at_freeze"],
        "suffix_tokens": freeze["source_suffix_tokens"],
    }
    for name, parents in scenarios.items():
        ctrl = pack_arm(bundle, arm="control", parents=parents)
        xfer = pack_arm(bundle, arm="transfer", parents=parents)
        _assert_pair(bundle, ctrl, xfer)
        row[f"{name}_ctrl_tokens"] = ctrl["tokens"]
        row[f"{name}_xfer_tokens"] = xfer["tokens"]
        row[f"{name}_ctrl_n"] = ctrl["n_examples"]
        row[f"{name}_xfer_n"] = xfer["n_examples"]
        row[f"{name}_same_ids"] = ctrl["example_ids"] == xfer["example_ids"]
        row[f"{name}_ctrl_parents"] = ctrl["parents_after"]
        row[f"{name}_xfer_parents"] = xfer["parents_after"]
        row[f"{name}_parent_retention_differed"] = (
            ctrl["parents_after"] != xfer["parents_after"]
        )
    assert row["iter0_ctrl_n"] == row["mixed_ctrl_n"] == row["max_parents_ctrl_n"]
    assert row["iter0_xfer_n"] == row["mixed_xfer_n"] == row["max_parents_xfer_n"]
    if target in {"steyvers_2009_bandit", "5speekenbrink2008learning"}:
        assert row["iter0_xfer_tokens"] <= CAP
        assert row["mixed_xfer_tokens"] <= CAP
        assert row["max_parents_xfer_tokens"] <= CAP
    _RESULTS.append(row)


def test_g2_required_block_fails_clearly(qwen_ready: int) -> None:
    del qwen_ready
    with pytest.raises(PairedPackingFitError):
        freeze_g2_target_examples(
            dataset="7hilbig2014generalized",
            pooled_train=[],
            pooled_val=[],
            infer_text="TASK " + " ".join(f"tok{i}" for i in range(80_000)),
            source_suffix="## Cross-task transfer context\nSOURCE-dataset context only\n",
            runtime_contract="CONTRACT",
            seed_code="def choose(problem, history):\n    return 0.5\n",
            code_template_suffix="TEMPLATE",
            candidate_output_rules="RULES",
            max_parent_chars=MAX_PARENT_CHARS,
            max_prompt_train_trials=60,
            prompt_train_trials_seed=FREEZE_SEED,
            hard_prompt_token_cap=CAP,
            sample_size=SAMPLE_SIZE,
        )


@pytest.fixture(scope="session", autouse=True)
def _write_before_after_table() -> None:
    yield
    if len(_RESULTS) < 15:
        return
    before_by_target: Dict[str, Dict[str, str]] = {}
    if BEFORE_CSV.is_file():
        with BEFORE_CSV.open(newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                before_by_target[rec["target"]] = rec
    AFTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target",
        "source",
        "before_iter0_ctrl_n",
        "before_iter0_xfer_n",
        "before_iter0_same_ids",
        "before_iter0_ctrl_tokens",
        "before_iter0_xfer_tokens",
        "after_iter0_n",
        "after_iter0_same_ids",
        "after_iter0_ctrl_tokens",
        "after_iter0_xfer_tokens",
        "after_mixed_same_ids",
        "after_max_same_ids",
        "after_max_ctrl_tokens",
        "after_max_xfer_tokens",
        "after_max_ctrl_parents",
        "after_max_xfer_parents",
        "steyvers_or_speekenbrink_over_14k",
    ]
    with AFTER_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(_RESULTS, key=lambda r: r["target"]):
            b = before_by_target.get(row["target"], {})
            over = int(row["iter0_xfer_tokens"]) > CAP or int(
                row["max_parents_xfer_tokens"]
            ) > CAP
            w.writerow(
                {
                    "target": row["target"],
                    "source": row["source"],
                    "before_iter0_ctrl_n": b.get("iter0_ctrl_n_ex"),
                    "before_iter0_xfer_n": b.get("iter0_xfer_n_ex"),
                    "before_iter0_same_ids": b.get("iter0_same_ids"),
                    "before_iter0_ctrl_tokens": b.get("iter0_ctrl_tokens"),
                    "before_iter0_xfer_tokens": b.get("iter0_xfer_tokens"),
                    "after_iter0_n": row["iter0_ctrl_n"],
                    "after_iter0_same_ids": row["iter0_same_ids"],
                    "after_iter0_ctrl_tokens": row["iter0_ctrl_tokens"],
                    "after_iter0_xfer_tokens": row["iter0_xfer_tokens"],
                    "after_mixed_same_ids": row["mixed_same_ids"],
                    "after_max_same_ids": row["max_parents_same_ids"],
                    "after_max_ctrl_tokens": row["max_parents_ctrl_tokens"],
                    "after_max_xfer_tokens": row["max_parents_xfer_tokens"],
                    "after_max_ctrl_parents": row["max_parents_ctrl_parents"],
                    "after_max_xfer_parents": row["max_parents_xfer_parents"],
                    "steyvers_or_speekenbrink_over_14k": over
                    if row["target"]
                    in {"steyvers_2009_bandit", "5speekenbrink2008learning"}
                    else False,
                }
            )
