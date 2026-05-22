#!/usr/bin/env python3
"""
Compute participant-level BIR, structure score S, and RBU for Psych-101 datasets + mixed_gambles.

Runs all configured datasets in one invocation (no per-dataset CLI). For each dataset and
participant ordinal in the configured range:

1. Load train trials (HF psych_dataset_split + within-participant train split).
2. BIR = fraction of repeated problem groups with both actions 0 and 1.
3. LLM structure score (combined call) -> analysis/Structure_score_all.txt.
4. S = sum of clipped numeric evidence values (summary/text fields ignored); S clipped to [0, 1].
5. RBU = clip(BIR - S, 0, 1).

Outputs:
  {output_root}/run_YYYYMMDD_HHMMSS/<dataset>/Structure_score_all.txt
  {output_root}/run_YYYYMMDD_HHMMSS/<dataset>/rbu_table.csv
  {combined_output_root}/run_YYYYMMDD_HHMMSS/rbu_all.csv

Example:
  python analysis/code/psych-101/rbu.py --mode local --model_name Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.code.choices13k.bir import compute_bir
from data_modules.mixed_gambles import DEFAULT_CSV_PATH, load_mixed_gambles_trials
from data_modules.psych101_binary import (
    DEFAULT_PSYCH_DATASET_SPLIT,
    format_trials_for_prompt,
    get_psych101_binary_experiment,
    normalize_psych101_dataset_alias,
    normalize_psych_dataset_split,
    split_psych_experiment,
)
from utils.rbu import (
    StructureScoreParseError,
    clip01,
    count_tokens_approx,
    parse_all_participant_structure_scores,
)
from utils.teh.participant_ids import load_valid_participant_ids
from utils.teh.teh_datasets import MIXED_GAMBLES, is_mixed_gambles_dataset

log = logging.getLogger(__name__)

_STRUCTURE_SCORE_ALL_FILENAME = "Structure_score_all.txt"
_PREPARED_INSTRUCTION_HEADER = "\n\n## Dataset-specific scoring instruction (prepared once for this run)\n"
_STRUCTURE_TOKEN_ESTIMATE_SLACK = 1.12

# (dataset alias, inclusive ordinal start, inclusive ordinal end)
RBU_DATASET_ORDINAL_RANGES: Tuple[Tuple[str, int, int], ...] = (
    ("1peterson2021using", 0, 49),
    ("2plonsky2018when", 0, 49),
    ("3frey2017cct", 0, 49),
    ("4wulff2018description", 0, 49),
    ("5speekenbrink2008learning", 0, 22),
    ("6sadeghiyeh2020temporal", 0, 49),
    ("7hilbig2014generalized", 0, 49),
    ("8flesch2018comparing", 0, 49),
    (MIXED_GAMBLES, 0, 49),
)

# Template_evo text_profile prompts exist for choice13k / cpc18 / mixed_gambles only.
_RBU_PROMPT_ALIAS: Dict[str, str] = {
    "1peterson2021using": "choice13k",
    "2plonsky2018when": "cpc18",
    "4wulff2018description": "choice13k",
    MIXED_GAMBLES: "mixed_gambles",
}
_DEFAULT_RBU_PROMPT_ALIAS = "cpc18"

RBU_TABLE_FIELDS = ("dataset", "participant_ordinal", "BIR", "S", "RBU")


@dataclass(frozen=True)
class ParticipantRbuRow:
    dataset: str
    participant_ordinal: int
    participant_id: int
    bir: float
    structure_score: float
    rbu: float

    def as_csv_dict(self) -> Dict[str, str]:
        return {
            "dataset": self.dataset,
            "participant_ordinal": str(self.participant_ordinal),
            "BIR": f"{self.bir:.4f}",
            "S": f"{self.structure_score:.4f}",
            "RBU": f"{self.rbu:.4f}",
        }


def _effective_psych_dataset_split(dataset: str, psych_dataset_split: str) -> str:
    if is_mixed_gambles_dataset(dataset):
        return DEFAULT_PSYCH_DATASET_SPLIT
    return normalize_psych_dataset_split(psych_dataset_split)


def _rbu_prompt_alias(dataset: str) -> str:
    alias = normalize_psych101_dataset_alias(dataset)
    return _RBU_PROMPT_ALIAS.get(alias, _DEFAULT_RBU_PROMPT_ALIAS)


def _default_prepare_instruction_path(dataset: str) -> Path:
    prompt_ds = _rbu_prompt_alias(dataset)
    p = REPO_ROOT / "prompts" / "Template_evo" / prompt_ds / "text_profile" / "prepare_instruction.txt"
    if not p.is_file():
        raise FileNotFoundError(
            f"RBU prepare_instruction.txt missing for dataset={dataset!r} "
            f"(prompt alias {prompt_ds!r}): {p}"
        )
    return p


def _default_use_instruction_path(dataset: str) -> Path:
    prompt_ds = _rbu_prompt_alias(dataset)
    p = REPO_ROOT / "prompts" / "Template_evo" / prompt_ds / "text_profile" / "use_instruction.txt"
    if not p.is_file():
        raise FileNotFoundError(
            f"RBU use_instruction.txt missing for dataset={dataset!r} "
            f"(prompt alias {prompt_ds!r}): {p}"
        )
    return p


def _train_trials_for_participant(
    dataset: str,
    participant_id: int,
    *,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> List[Dict[str, Any]]:
    """Train trials for one participant (TEH / Psych-101 within-participant split)."""
    if is_mixed_gambles_dataset(dataset):
        train_trials, _, _, _ = load_mixed_gambles_trials(
            participant_id,
            csv_path=mixed_gambles_csv,
            filter_gain_loss_only=filter_mixed_gambles,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        return train_trials
    alias = normalize_psych101_dataset_alias(dataset)
    exp = get_psych101_binary_experiment(
        alias,
        int(participant_id),
        split=psych_dataset_split,
        local_dataset=local_dataset,
    )
    train_trials, _, _, _ = split_psych_experiment(
        exp, split_ratio=split_ratio, split_seed=split_seed
    )
    return train_trials


def _resolve_participants_by_ordinals(
    dataset: str,
    *,
    range_start_ordinal: int,
    range_end_ordinal: int,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> Tuple[List[int], Dict[int, int]]:
    """
    Return (raw participant ids, ordinal -> raw id) for inclusive ordinal range.
    """
    valid = load_valid_participant_ids(
        dataset,
        REPO_ROOT,
        filter_mixed_gambles=filter_mixed_gambles,
        split_ratio=split_ratio,
        split_seed=split_seed,
        psych_dataset_split=_effective_psych_dataset_split(dataset, psych_dataset_split),
        local_dataset=local_dataset,
        mixed_gambles_csv=mixed_gambles_csv,
        auto_prepare=True,
    )
    if range_start_ordinal < 0 or range_end_ordinal >= len(valid) or range_start_ordinal > range_end_ordinal:
        raise ValueError(
            f"Invalid ordinal range [{range_start_ordinal}, {range_end_ordinal}] for {dataset!r} "
            f"(valid list length {len(valid)})."
        )
    raw_ids = [int(valid[o]) for o in range(range_start_ordinal, range_end_ordinal + 1)]
    ordinal_map = {o: int(valid[o]) for o in range(range_start_ordinal, range_end_ordinal + 1)}
    return raw_ids, ordinal_map


def _structure_score_s_from_components(components: Mapping[str, float]) -> float:
    """S = clip01(sum of clipped evidence components); ignores non-numeric fields like summary."""
    if not components:
        raise StructureScoreParseError("empty evidence components; cannot compute S")
    return clip01(float(sum(components.values())))


def _compute_rbu(bir: float, structure_score: float) -> float:
    return clip01(float(bir) - float(structure_score))


def _format_train_trials_block(
    participant_ids: Sequence[int],
    train_by_pid: Dict[int, List[Dict[str, Any]]],
    trials_per_participant: int,
) -> str:
    parts: List[str] = []
    for pid in sorted(int(x) for x in participant_ids):
        tr = train_by_pid.get(pid, [])
        trials = tr[:trials_per_participant] if trials_per_participant > 0 else []
        parts.append(
            f"\n\n## Participant {pid} — training trials (training split only; no test trials)\n"
        )
        parts.append(format_trials_for_prompt(trials, max_trials=len(trials)))
    return "".join(parts)


def _llm_write_run_instruction(
    *,
    client: OpenAI,
    model_name: str,
    prepare_instruction_path: Path,
    run_dir: Path,
    max_tokens: int = 4096,
) -> Path:
    """One LLM call; writes ``run_dir/instruction.txt``."""
    prompt = prepare_instruction_path.read_text(encoding="utf-8")
    out_path = run_dir / "instruction.txt"
    log.info("Writing dataset instruction via LLM -> %s", out_path)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError(f"RBU instruction LLM returned empty content (expected {out_path})")
    out_path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
    return out_path


def _llm_write_all_participant_structure_scores(
    *,
    client: OpenAI,
    model_name: str,
    use_instruction_path: Path,
    run_instruction_path: Path,
    participant_ids: Sequence[int],
    participant_train_trials: Dict[int, List[Dict[str, Any]]],
    dataset_dir: Path,
    structure_prompt_max_tokens: int,
    max_response_tokens: int,
    model_context_tokens: int = 32768,
    token_estimate_slack: float = _STRUCTURE_TOKEN_ESTIMATE_SLACK,
) -> Tuple[str, Path, int, int]:
    """
    One combined structure-scoring LLM call.

    Returns ``(raw_text, out_path, trials_per_participant_used, estimated_prompt_tokens)``.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    use_txt = use_instruction_path.read_text(encoding="utf-8")
    run_txt = run_instruction_path.read_text(encoding="utf-8")
    prefix = use_txt + _PREPARED_INSTRUCTION_HEADER + run_txt

    pids = sorted(int(x) for x in participant_ids)
    train_by: Dict[int, List[Dict[str, Any]]] = {
        pid: list(participant_train_trials[pid]) for pid in pids
    }
    if not pids:
        raise ValueError("structure scoring: empty participant_ids")

    min_full = min(len(train_by[pid]) for pid in pids)
    slack = float(token_estimate_slack)
    if slack < 1.0:
        raise ValueError("token_estimate_slack must be >= 1.0")

    def _inflate(n: int) -> int:
        return int(math.ceil(float(n) * slack))

    completion_reserve = int(max_response_tokens) + 64
    if int(model_context_tokens) <= completion_reserve:
        raise ValueError(
            f"model_context_tokens={model_context_tokens} must exceed max_response_tokens + 64"
        )
    prompt_cap = min(int(structure_prompt_max_tokens), int(model_context_tokens) - completion_reserve)
    if prompt_cap < 1:
        raise RuntimeError(f"RBU structure prompt cap is {prompt_cap} (must be >= 1)")

    if _inflate(count_tokens_approx(prefix)) > prompt_cap:
        raise RuntimeError(
            f"RBU structure prompt: instruction prefix exceeds cap ({prompt_cap} tokens, slack={slack})"
        )

    def _total_tokens_for_k(k: int) -> int:
        body = _format_train_trials_block(pids, train_by, k) if k > 0 else ""
        return count_tokens_approx(prefix + body)

    lo, hi = 0, min_full
    best_k = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _inflate(_total_tokens_for_k(mid)) <= prompt_cap:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1

    final_body = _format_train_trials_block(pids, train_by, best_k) if best_k > 0 else ""
    user_content = prefix + final_body
    est_tokens = count_tokens_approx(user_content)
    est_inflated = _inflate(est_tokens)
    max_resp_eff = min(int(max_response_tokens), int(model_context_tokens) - est_inflated - 64)
    if max_resp_eff < 256:
        raise RuntimeError(
            "RBU structure prompt leaves insufficient room for completion "
            f"(est_prompt={est_tokens}, inflated={est_inflated}, context={model_context_tokens})"
        )

    log.info(
        "[%s] structure LLM: participants=%d trials_each=%d est_prompt_tokens=%d max_completion=%d",
        dataset_dir.name,
        len(pids),
        best_k,
        est_tokens,
        max_resp_eff,
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.2,
        top_p=0.95,
        max_tokens=max_resp_eff,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("RBU combined structure-score LLM returned empty output")
    out_path = dataset_dir / _STRUCTURE_SCORE_ALL_FILENAME
    out_path.write_text(raw + ("\n" if not raw.endswith("\n") else ""), encoding="utf-8")
    return raw, out_path, best_k, est_tokens


def _parse_structure_scores(
    raw_text: str,
    *,
    expected_participant_ids: Tuple[int, ...],
) -> Dict[int, Tuple[float, Dict[str, float]]]:
    """Parse combined JSON; S is sum of clipped evidence (not mean)."""
    parsed_mean = parse_all_participant_structure_scores(
        raw_text,
        expected_participant_ids=expected_participant_ids,
    )
    out: Dict[int, Tuple[float, Dict[str, float]]] = {}
    for pid, (_, comps) in parsed_mean.items():
        s_val = _structure_score_s_from_components(comps)
        out[int(pid)] = (s_val, dict(comps))
    return out


def _write_rbu_table(path: Path, rows: Sequence[ParticipantRbuRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(RBU_TABLE_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow(row.as_csv_dict())


def _compute_bir_for_participants(
    dataset: str,
    raw_ids: Sequence[int],
    ordinal_by_raw: Dict[int, int],
    *,
    psych_dataset_split: str,
    split_ratio: float,
    split_seed: int,
    local_dataset: Optional[str],
    mixed_gambles_csv: str,
    filter_mixed_gambles: bool,
) -> Dict[int, float]:
    bir_by_raw: Dict[int, float] = {}
    for raw_id in tqdm(raw_ids, desc=f"BIR {dataset}", unit="participant"):
        ordinal = ordinal_by_raw[int(raw_id)]
        log.info("[%s] ordinal %d (raw id %d): loading train trials", dataset, ordinal, raw_id)
        train_trials = _train_trials_for_participant(
            dataset,
            int(raw_id),
            psych_dataset_split=psych_dataset_split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            local_dataset=local_dataset,
            mixed_gambles_csv=mixed_gambles_csv,
            filter_mixed_gambles=filter_mixed_gambles,
        )
        bir_val, n_groups, n_incon = compute_bir(train_trials)
        bir_by_raw[int(raw_id)] = float(bir_val)
        log.info(
            "[%s] ordinal %d: BIR=%.4f (%d/%d inconsistent groups, %d train trials)",
            dataset,
            ordinal,
            bir_val,
            n_incon,
            n_groups,
            len(train_trials),
        )
    return bir_by_raw


def process_dataset(
    dataset: str,
    range_start_ordinal: int,
    range_end_ordinal: int,
    *,
    run_dir: Path,
    client: Optional[OpenAI],
    args: argparse.Namespace,
) -> List[ParticipantRbuRow]:
    """Compute BIR / S / RBU for one dataset; write per-dataset artifacts under ``run_dir / dataset``."""
    dataset = normalize_psych101_dataset_alias(dataset)
    psych_split = _effective_psych_dataset_split(dataset, args.psych_dataset_split)
    dataset_dir = run_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "=== Dataset %s | psych_dataset_split=%s | ordinals [%d, %d] ===",
        dataset,
        psych_split,
        range_start_ordinal,
        range_end_ordinal,
    )

    raw_ids, ordinal_map = _resolve_participants_by_ordinals(
        dataset,
        range_start_ordinal=range_start_ordinal,
        range_end_ordinal=range_end_ordinal,
        psych_dataset_split=psych_split,
        split_ratio=float(args.split_ratio),
        split_seed=int(args.split_seed),
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        filter_mixed_gambles=bool(args.filter_mixed_gambles),
    )
    raw_to_ordinal = {raw: ord_ for ord_, raw in ordinal_map.items()}

    bir_by_raw = _compute_bir_for_participants(
        dataset,
        raw_ids,
        raw_to_ordinal,
        psych_dataset_split=psych_split,
        split_ratio=float(args.split_ratio),
        split_seed=int(args.split_seed),
        local_dataset=args.local_dataset,
        mixed_gambles_csv=args.mixed_gambles_csv,
        filter_mixed_gambles=bool(args.filter_mixed_gambles),
    )

    structure_by_raw: Dict[int, float] = {}
    if args.skip_structure_llm:
        log.warning("[%s] --skip_structure_llm: S=0, RBU=BIR for all participants", dataset)
        structure_by_raw = {int(pid): 0.0 for pid in raw_ids}
    else:
        if client is None:
            raise RuntimeError("OpenAI client required for structure scoring (omit --skip_structure_llm)")
        train_by_raw: Dict[int, List[Dict[str, Any]]] = {}
        for raw_id in raw_ids:
            train_by_raw[int(raw_id)] = _train_trials_for_participant(
                dataset,
                int(raw_id),
                psych_dataset_split=psych_split,
                split_ratio=float(args.split_ratio),
                split_seed=int(args.split_seed),
                local_dataset=args.local_dataset,
                mixed_gambles_csv=args.mixed_gambles_csv,
                filter_mixed_gambles=bool(args.filter_mixed_gambles),
            )

        prepare_path = (
            Path(args.prepare_instruction_path)
            if args.prepare_instruction_path
            else _default_prepare_instruction_path(dataset)
        )
        use_path = (
            Path(args.use_instruction_path)
            if args.use_instruction_path
            else _default_use_instruction_path(dataset)
        )
        log.info(
            "[%s] RBU prompts: prepare=%s use=%s (alias=%s)",
            dataset,
            prepare_path,
            use_path,
            _rbu_prompt_alias(dataset),
        )

        _llm_write_run_instruction(
            client=client,
            model_name=args.model_name,
            prepare_instruction_path=prepare_path,
            run_dir=dataset_dir,
            max_tokens=int(args.instruction_max_tokens),
        )

        n_part = len(raw_ids)
        max_resp_toks = min(32768, max(4096, 768 * max(1, n_part)))
        raw_scores, score_path, trials_k, _ = _llm_write_all_participant_structure_scores(
            client=client,
            model_name=args.model_name,
            use_instruction_path=use_path,
            run_instruction_path=dataset_dir / "instruction.txt",
            participant_ids=raw_ids,
            participant_train_trials=train_by_raw,
            dataset_dir=dataset_dir,
            structure_prompt_max_tokens=int(args.structure_prompt_max_tokens),
            max_response_tokens=max_resp_toks,
            model_context_tokens=int(args.structure_model_context_tokens),
        )
        log.info("[%s] wrote %s (trials/participant=%d)", dataset, score_path, trials_k)

        try:
            parsed = _parse_structure_scores(
                raw_scores,
                expected_participant_ids=tuple(sorted(int(x) for x in raw_ids)),
            )
        except StructureScoreParseError as exc:
            raise RuntimeError(
                f"[{dataset}] structure score parse failed: {exc}\n"
                f"raw preview:\n{raw_scores[:2400]!r}"
            ) from exc
        structure_by_raw = {pid: float(s_v) for pid, (s_v, _) in parsed.items()}

    rows: List[ParticipantRbuRow] = []
    for raw_id in sorted(raw_ids):
        ordinal = raw_to_ordinal[int(raw_id)]
        bir = float(bir_by_raw[int(raw_id)])
        s_val = float(structure_by_raw.get(int(raw_id), 0.0))
        rbu_val = _compute_rbu(bir, s_val)
        log.info(
            "[%s] ordinal %d (id %d): BIR=%.4f S=%.4f RBU=%.4f",
            dataset,
            ordinal,
            raw_id,
            bir,
            s_val,
            rbu_val,
        )
        rows.append(
            ParticipantRbuRow(
                dataset=dataset,
                participant_ordinal=ordinal,
                participant_id=int(raw_id),
                bir=bir,
                structure_score=s_val,
                rbu=rbu_val,
            )
        )

    table_path = dataset_dir / "rbu_table.csv"
    _write_rbu_table(table_path, rows)
    log.info("[%s] wrote %s (%d rows)", dataset, table_path, len(rows))
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute BIR, structure score S, and RBU for all Psych-101 TEH datasets "
            "(plus mixed_gambles) in one run."
        )
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="generated_outputs/analysis/rbu",
        help="Root for per-dataset outputs: {output_root}/run_<timestamp>/<dataset>/",
    )
    p.add_argument(
        "--combined_output_root",
        type=str,
        default="generated_outputs/psych-101/rbu",
        help="Root for combined rbu_all.csv: {combined_output_root}/run_<timestamp>/rbu_all.csv",
    )
    p.add_argument(
        "--psych_dataset_split",
        type=str,
        default=DEFAULT_PSYCH_DATASET_SPLIT,
        choices=["train", "test"],
        help="Psych-101 HF corpus selector (mixed_gambles ignores this). Default: train.",
    )
    p.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Within-participant train fraction for BIR/structure trials (TEH default 0.8).",
    )
    p.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="RNG seed for within-participant train/val/test split (TEH default 42).",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        metavar="ALIAS",
        help="Optional subset of dataset aliases to run (default: all configured datasets).",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        help="LLM model for structure scoring.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "local"],
        help="LLM mode: default uses OpenAI API; local routes to a vLLM server.",
    )
    p.add_argument(
        "--llm_server_url",
        type=str,
        default=os.getenv("VLLM_LOCAL_URL", "http://localhost:8000/v1"),
        help="Base URL for local vLLM when --mode local.",
    )
    p.add_argument(
        "--llm_api_key",
        type=str,
        default=os.getenv("VLLM_LOCAL_API_KEY", "EMPTY"),
        help="API key for local vLLM when --mode local.",
    )
    p.add_argument(
        "--structure_prompt_max_tokens",
        type=int,
        default=24000,
        metavar="N",
        help="Cap for combined structure-scoring prompt (inflated token estimate).",
    )
    p.add_argument(
        "--structure_model_context_tokens",
        type=int,
        default=32768,
        metavar="N",
        help="Model context window for structure-scoring calls.",
    )
    p.add_argument(
        "--instruction_max_tokens",
        type=int,
        default=4096,
        help="Max completion tokens for dataset instruction LLM call.",
    )
    p.add_argument(
        "--prepare_instruction_path",
        type=str,
        default=None,
        help="Override prepare_instruction.txt for all datasets (default: per-dataset prompt alias).",
    )
    p.add_argument(
        "--use_instruction_path",
        type=str,
        default=None,
        help="Override use_instruction.txt for all datasets (default: per-dataset prompt alias).",
    )
    p.add_argument(
        "--skip_structure_llm",
        action="store_true",
        help="Skip structure-score LLM; set S=0 and RBU=BIR (BIR-only dry run).",
    )
    p.add_argument(
        "--filter_mixed_gambles",
        action="store_true",
        default=False,
        help="For mixed_gambles: gain_loss trials only and matching valid_participant_ids JSON.",
    )
    p.add_argument(
        "--local_dataset",
        type=str,
        default=None,
        help="Optional local HuggingFace dataset path for Psych-101 loading.",
    )
    p.add_argument(
        "--mixed_gambles_csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="CSV path for mixed_gambles trials.",
    )
    p.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Run timestamp folder name (default: current time YYYYMMDD_HHMMSS).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging (-v INFO, -vv DEBUG).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not (0.0 < float(args.split_ratio) < 1.0):
        raise SystemExit(f"--split_ratio must be in (0, 1), got {args.split_ratio}")

    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (REPO_ROOT / args.output_root / f"run_{ts}").resolve()
    combined_dir = (REPO_ROOT / args.combined_output_root / f"run_{ts}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    client: Optional[OpenAI] = None
    if not args.skip_structure_llm:
        client_kwargs: Dict[str, Any] = {}
        if args.mode == "local":
            client_kwargs = {"base_url": args.llm_server_url, "api_key": args.llm_api_key}
        client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()

    selected = set(normalize_psych101_dataset_alias(d) for d in (args.datasets or []))
    specs = [
        (ds, lo, hi)
        for ds, lo, hi in RBU_DATASET_ORDINAL_RANGES
        if not selected or normalize_psych101_dataset_alias(ds) in selected
    ]
    if not specs:
        raise SystemExit("No datasets selected; check --datasets aliases.")

    log.info("RBU run %s | datasets=%d | output=%s", ts, len(specs), run_dir)
    log.info("Combined CSV -> %s/rbu_all.csv", combined_dir)

    all_rows: List[ParticipantRbuRow] = []
    for dataset, lo, hi in specs:
        try:
            rows = process_dataset(
                dataset,
                lo,
                hi,
                run_dir=run_dir,
                client=client,
                args=args,
            )
            all_rows.extend(rows)
        except Exception:
            log.exception("Failed dataset %s", dataset)
            raise

    combined_path = combined_dir / "rbu_all.csv"
    _write_rbu_table(combined_path, all_rows)
    log.info("Wrote combined table (%d rows) -> %s", len(all_rows), combined_path)

    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "psych_dataset_split": args.psych_dataset_split,
                "split_ratio": float(args.split_ratio),
                "split_seed": int(args.split_seed),
                "model_name": args.model_name,
                "mode": args.mode,
                "skip_structure_llm": bool(args.skip_structure_llm),
                "datasets": [{"dataset": ds, "ordinal_start": lo, "ordinal_end": hi} for ds, lo, hi in specs],
                "n_rows": len(all_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Per-dataset outputs: {run_dir}")
    print(f"Combined CSV: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
