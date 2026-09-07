"""
Kool, Cushman & Gershman (2016) — Daw-structure two-step task (Psych-101 exp2).

=============================================================================
EXECUTION SOURCE vs VALIDATION
=============================================================================
- Execution: Psych-101 ``kool2016when/exp2.csv`` (TEH alias ``14kool2016when``).
- Validation/reference only: ``wkool/tradeoffs`` ``data/daw paradigm/groupdata.mat``
  (setup copies it under ``datasets/external/kool2016_exp2/``).

=============================================================================
HISTORY / RESET SEMANTICS (audited before finalizing loader)
=============================================================================
Evidence:
1. Psych-101 instruction: spaceship→planet mappings "won't change during the game";
   alien treasure "will change slowly during the game"; goal over "the next 125 days".
   No per-day or per-block reset language in the transcript.
2. Author task (space_daw): continuous main game after practice; MB/MF analyses use
   prior-trial stay/common/reward (make_raw_data.m ``prev*`` fields).
3. Author pipeline: 150 logged rows with practice flag → drop practice → 125 trials;
   Psych-101 transcripts already contain 125 presented days (practice removed).
4. Timeouts: "You do not respond in time for this day" — no stage-1/2 choice.
   Author marks ``missed``; stay analyses skip missed|prevmissed.

Loader policy (documented, deliberate):
- History DOES NOT reset each day. Stage-1/2 learning carries across days within a
  participant session (one continuous history).
- There are NO mid-session block resets in Psych-101 exp2 / Daw paradigm main game.
- Prior-day stage-2 rewards ARE behaviorally available and remain in history.
- Timeout days emit NO choose() trials and add NO history entries when the
  participant never completes stage 1 (early timeout).
- If stage 1 is completed but stage 2 times out: emit the stage-1 decision only;
  after the choice, append stage-1 history with the observed planet (no reward);
  do not emit a stage-2 trial. Author ``missed=1`` for the day; fingerprint uses -1.
- Practice trials are already absent from Psych-101 (125 days); no extra drop.

=============================================================================
SPLIT SEMANTICS
=============================================================================
- Unit: presented day, never mid-day (stage-1 and stage-2 of a complete day stay
  together; stage-2-timeout days contribute only stage-1).
- Split is CONTIGUOUS in time over **usable days** (days with ≥1 emitted decision),
  early→train / middle→val / late→test. Not a random shuffle.
- Why usable-day cuts (not raw 1..125 calendar cuts): a few subjects time out on
  all late calendar days, which would empty val/test under a fixed 100/13/12 cut
  while still having many early usable days. Cutting on sorted usable days keeps
  chronological prior history valid and yields nonempty partitions when ≥3 usable
  days exist.
- When evaluating day d, history includes all completed stage decisions from
  earlier usable days < d (including earlier days assigned to other splits).

=============================================================================
ACTION CODING (raw Psych-101 → internal 0/1)
=============================================================================
Letters are participant-specific remappings of spaceships/aliens.

Stage 1 (spaceship choice):
  "You are presented with spaceships X and Y. You press <<Z>>."
  option_keys = [X, Y]  # presentation order
  action = 0 if Z == X else 1 if Z == Y
  choose() returns P(action=1) = P(second presented spaceship).

Stage 2 (alien choice), after planet is observed:
  "You see alien A and alien B. You press <<C>>."
  option_keys = [A, B]
  action = 0 if C == A else 1 if C == B
  choose() returns P(action=1) = P(second presented alien).

Same choose(problem, history) program handles both stages via problem["stage"].

=============================================================================
LEAKAGE / OBSERVABILITY
=============================================================================
Stage 1 problem MUST NOT contain: current planet/transition, current aliens,
current treasure, future days.
Stage 2 problem MAY contain: current planet, current aliens, same-day stage-1
action/spaceship. MUST NOT contain: current treasure / future.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_modules.psych101_binary import ParsedTrial, PsychBlock, PsychExperiment

DATASET_ALIAS = "14kool2016when"
EXPERIMENT_ID = "kool2016when/exp2.csv"
DISPLAY_NAME = "Kool et al. (2016) Daw two-step (exp2)"
OUTPUT_TYPE = "bernoulli"
SCHEMA_TYPE = "kool_twostep"
PARSER_ID = "kool_twostep"
N_PRESENTED_DAYS = 125
SPLIT_UNIT = "presented_day_contiguous"
DEFAULT_VALIDATION_MAT = "datasets/external/kool2016_exp2/groupdata.mat"

TASK_DESCRIPTION = (
    "Daw-structure two-step task (spaceship → planet → alien treasure). "
    "Each day has two binary decisions (stage 1 then stage 2) handled by the same "
    "choose(problem, history) program. History carries across days (no per-day reset). "
    "choose returns P(action=1) for the second presented option at that stage."
)

_RE_INSTR = re.compile(
    r"spaceships\s+([A-Z])\s+or\s+([A-Z]).*?"
    r"planets\s+([A-Z])\s+or\s+([A-Z]).*?"
    r"Planet\s+([A-Z])\s+has aliens\s+([A-Z])\s+and\s+([A-Z]).*?"
    r"[Pp]lanet\s+([A-Z])\s+has aliens\s+([A-Z])\s+and\s+([A-Z])",
    re.S,
)
_RE_DAY = re.compile(
    r"You are presented with spaceships\s+([A-Z])\s+and\s+([A-Z])\.\s*"
    r"You press <<([A-Z])>>\.\s*"
    r"You end up on planet\s+([A-Z])\.\s*"
    r"You see alien\s+([A-Z])\s+and alien\s+([A-Z])\.\s*"
    r"You press <<([A-Z])>>\.\s*"
    r"You find\s+(\d+)\s+pieces?\s+of space treasure\.",
)
_RE_TIMEOUT = re.compile(
    r"You are presented with spaceships\s+([A-Z])\s+and\s+([A-Z])\.\s*"
    r"You do not respond in time for this day\.",
)
_RE_STAGE1_THEN_TIMEOUT = re.compile(
    r"You are presented with spaceships\s+([A-Z])\s+and\s+([A-Z])\.\s*"
    r"You press <<([A-Z])>>\.\s*"
    r"You end up on planet\s+([A-Z])\.\s*"
    r"You see alien\s+([A-Z])\s+and alien\s+([A-Z])\.\s*"
    r"You do not respond in time for this day\.",
)
_RE_CHUNK = re.compile(
    r"You are presented with spaceships.*?(?=You are presented with spaceships|$)",
    re.S,
)


def raw_stage1_press_to_action(press: str, option_keys: Sequence[str]) -> int:
    keys = list(option_keys)
    if len(keys) != 2:
        raise ValueError(f"stage1 option_keys must have length 2, got {keys}")
    p = str(press)
    if p == keys[0]:
        return 0
    if p == keys[1]:
        return 1
    raise ValueError(f"stage1 press {p!r} not in option_keys {keys}")


def raw_stage2_press_to_action(press: str, option_keys: Sequence[str]) -> int:
    return raw_stage1_press_to_action(press, option_keys)


def parse_instruction_labels(instruction: str) -> Dict[str, Any]:
    m = _RE_INSTR.search(instruction)
    if not m:
        raise ValueError("kool exp2: failed to parse spaceship/planet/alien labels")
    return {
        "spaceship_labels": [m.group(1), m.group(2)],
        "planet_labels": [m.group(3), m.group(4)],
        "planet_aliens": {
            m.group(5): [m.group(6), m.group(7)],
            m.group(8): [m.group(9), m.group(10)],
        },
    }


def _parse_day_chunks(text: str) -> List[Dict[str, Any]]:
    chunks = _RE_CHUNK.findall(text)
    if len(chunks) != N_PRESENTED_DAYS:
        raise ValueError(
            f"kool exp2: expected {N_PRESENTED_DAYS} presented days, got {len(chunks)}"
        )
    days: List[Dict[str, Any]] = []
    for presented_day, chunk in enumerate(chunks, start=1):
        chunk = chunk.strip()
        m = _RE_DAY.search(chunk)
        if m:
            s1, s2, press1, planet, a1, a2, press2, win = m.groups()
            stage1_keys = [s1, s2]
            stage2_keys = [a1, a2]
            days.append(
                {
                    "kind": "complete",
                    "presented_day": presented_day,
                    "stage1_option_keys": stage1_keys,
                    "stage1_action": raw_stage1_press_to_action(press1, stage1_keys),
                    "spaceship": press1,
                    "planet": planet,
                    "stage2_option_keys": stage2_keys,
                    "stage2_action": raw_stage2_press_to_action(press2, stage2_keys),
                    "alien": press2,
                    "reward": int(win),
                }
            )
            continue
        # Stage-1 completed, planet/aliens shown, then stage-2 timeout.
        m2 = _RE_STAGE1_THEN_TIMEOUT.search(chunk)
        if m2:
            s1, s2, press1, planet, a1, a2 = m2.groups()
            stage1_keys = [s1, s2]
            days.append(
                {
                    "kind": "timeout_stage2",
                    "presented_day": presented_day,
                    "stage1_option_keys": stage1_keys,
                    "stage1_action": raw_stage1_press_to_action(press1, stage1_keys),
                    "spaceship": press1,
                    "planet": planet,
                    "stage2_option_keys": [a1, a2],
                }
            )
            continue
        tm = _RE_TIMEOUT.search(chunk)
        if tm:
            days.append(
                {
                    "kind": "timeout",
                    "presented_day": presented_day,
                    "stage1_option_keys": [tm.group(1), tm.group(2)],
                }
            )
            continue
        raise ValueError(
            f"kool exp2: unparseable day chunk at presented_day={presented_day}: "
            f"{chunk[:160]!r}"
        )
    return days


def parse_kool2016_exp2_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    """
    Parse one Psych-101 row.

    Blocks: one PsychBlock per complete day (timeouts omitted as blocks).
    Each block has two ParsedTrial entries (stage1, stage2) for auditability.
    Use build_kool_decision_trials() / split_kool2016_exp2_experiment() for
    evaluator trials (continuous history + contiguous day split).
    """
    text = str(row["text"])
    m0 = _RE_CHUNK.search(text)
    instruction = (
        text[: m0.start()].strip()
        if m0
        else text.split("You are presented")[0].strip()
    )
    labels = parse_instruction_labels(instruction)
    days = _parse_day_chunks(text)

    blocks: List[PsychBlock] = []
    for day in days:
        if day["kind"] == "timeout":
            continue
        if day["kind"] not in ("complete", "timeout_stage2"):
            continue
        stage1 = ParsedTrial(
            action=int(day["stage1_action"]),
            feedback={"planet": day["planet"]},
            problem_fields={
                "stage": 1,
                "presented_day": day["presented_day"],
                "option_keys": list(day["stage1_option_keys"]),
                "spaceship_options": list(day["stage1_option_keys"]),
            },
        )
        trials: List[ParsedTrial] = [stage1]
        if day["kind"] == "complete":
            stage2 = ParsedTrial(
                action=int(day["stage2_action"]),
                feedback=int(day["reward"]),
                problem_fields={
                    "stage": 2,
                    "presented_day": day["presented_day"],
                    "planet": day["planet"],
                    "option_keys": list(day["stage2_option_keys"]),
                    "alien_options": list(day["stage2_option_keys"]),
                    "stage1_action": int(day["stage1_action"]),
                    "stage1_option_keys": list(day["stage1_option_keys"]),
                    "spaceship": day["spaceship"],
                },
            )
            trials.append(stage2)
        static = {
            "schema_type": SCHEMA_TYPE,
            "presented_day": day["presented_day"],
            "has_feedback": True,
            "history_scope": "session_continuous",
            "split_unit": SPLIT_UNIT,
            "day_kind": day["kind"],
            **labels,
        }
        blocks.append(
            PsychBlock(
                trials=trials,
                option_keys=list(day["stage1_option_keys"]),
                problem_static=static,
                schema_type=SCHEMA_TYPE,
            )
        )

    if len(blocks) < 3:
        raise ValueError(
            f"kool exp2: need >=3 complete days for TEH split; got {len(blocks)}"
        )

    return PsychExperiment(
        instruction=instruction,
        blocks=blocks,
        dataset_alias=dataset_alias,
        schema_type=SCHEMA_TYPE,
    )


def _stage1_problem(
    day: Dict[str, Any],
    *,
    dataset_alias: str,
    experiment_id: str,
    labels: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_alias": dataset_alias,
        "experiment_id": experiment_id,
        "schema_type": SCHEMA_TYPE,
        "stage": 1,
        "presented_day": int(day["presented_day"]),
        "option_keys": list(day["stage1_option_keys"]),
        "spaceship_options": list(day["stage1_option_keys"]),
        "has_feedback": True,
        "history_scope": "session_continuous",
        "raw_key_coding": "letter_presentation_order",
        "internal_action_coding": "0=first_presented_key,1=second_presented_key",
        "spaceship_labels": list(labels["spaceship_labels"]),
        "planet_labels": list(labels["planet_labels"]),
        "planet_aliens": {k: list(v) for k, v in labels["planet_aliens"].items()},
    }


def _stage2_problem(
    day: Dict[str, Any],
    *,
    dataset_alias: str,
    experiment_id: str,
    labels: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_alias": dataset_alias,
        "experiment_id": experiment_id,
        "schema_type": SCHEMA_TYPE,
        "stage": 2,
        "presented_day": int(day["presented_day"]),
        "planet": day["planet"],
        "option_keys": list(day["stage2_option_keys"]),
        "alien_options": list(day["stage2_option_keys"]),
        "stage1_action": int(day["stage1_action"]),
        "stage1_option_keys": list(day["stage1_option_keys"]),
        "spaceship": day["spaceship"],
        "has_feedback": True,
        "history_scope": "session_continuous",
        "raw_key_coding": "letter_presentation_order",
        "internal_action_coding": "0=first_presented_key,1=second_presented_key",
        "spaceship_labels": list(labels["spaceship_labels"]),
        "planet_labels": list(labels["planet_labels"]),
        "planet_aliens": {k: list(v) for k, v in labels["planet_aliens"].items()},
    }


def _append_stage1_history(history: List[Dict[str, Any]], day: Dict[str, Any]) -> None:
    history.append(
        {
            "stage": 1,
            "action": int(day["stage1_action"]),
            "option_keys": list(day["stage1_option_keys"]),
            "spaceship": day["spaceship"],
            "planet": day["planet"],
        }
    )


def _append_stage2_history(history: List[Dict[str, Any]], day: Dict[str, Any]) -> None:
    history.append(
        {
            "stage": 2,
            "action": int(day["stage2_action"]),
            "option_keys": list(day["stage2_option_keys"]),
            "alien": day["alien"],
            "planet": day["planet"],
            "reward": int(day["reward"]),
            "feedback": int(day["reward"]),
        }
    )


def build_kool_decision_trials(
    exp: PsychExperiment,
    *,
    days: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Emit stage-1/stage-2 decision records with continuous session history."""
    alias = exp.dataset_alias
    exp_id = EXPERIMENT_ID
    labels = parse_instruction_labels(exp.instruction)

    if days is None:
        days_list: List[Dict[str, Any]] = []
        for block in exp.blocks:
            kind = block.problem_static.get("day_kind", "complete")
            t1 = block.trials[0]
            planet = (
                t1.feedback["planet"]
                if isinstance(t1.feedback, dict)
                else block.trials[-1].problem_fields.get("planet")
            )
            day_rec: Dict[str, Any] = {
                "kind": kind,
                "presented_day": int(block.problem_static["presented_day"]),
                "stage1_option_keys": list(t1.problem_fields["option_keys"]),
                "stage1_action": int(t1.action),
                "spaceship": t1.problem_fields["option_keys"][t1.action],
                "planet": planet,
            }
            if kind == "complete" and len(block.trials) == 2:
                t2 = block.trials[1]
                day_rec.update(
                    {
                        "stage2_option_keys": list(t2.problem_fields["option_keys"]),
                        "stage2_action": int(t2.action),
                        "alien": t2.problem_fields["option_keys"][t2.action],
                        "reward": int(t2.feedback),
                    }
                )
            elif kind == "timeout_stage2":
                # aliens may be absent when reconstructing; optional
                pass
            days_list.append(day_rec)
        days = days_list

    trials: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    for day in days:
        kind = day.get("kind")
        if kind == "timeout":
            continue
        if kind not in ("complete", "timeout_stage2"):
            continue
        p1 = _stage1_problem(
            day, dataset_alias=alias, experiment_id=exp_id, labels=labels
        )
        trials.append(
            {
                "problem": p1,
                "history": [dict(h) for h in history],
                "options": list(day["stage1_option_keys"]),
                "action": int(day["stage1_action"]),
            }
        )
        # Planet observed after stage-1 (even if stage-2 later times out).
        _append_stage1_history(history, day)
        if kind == "timeout_stage2":
            continue
        p2 = _stage2_problem(
            day, dataset_alias=alias, experiment_id=exp_id, labels=labels
        )
        trials.append(
            {
                "problem": p2,
                "history": [dict(h) for h in history],
                "options": list(day["stage2_option_keys"]),
                "action": int(day["stage2_action"]),
            }
        )
        _append_stage2_history(history, day)
    return trials


def _contiguous_day_bounds(n_days: int, split_ratio: float) -> Tuple[int, int, int]:
    if n_days < 3:
        raise ValueError(f"need >=3 presented days to split, got {n_days}")
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio}")
    n_train = int(n_days * split_ratio)
    n_train = max(1, min(n_train, n_days - 2))
    n_rem = n_days - n_train
    n_val = (n_rem + 1) // 2
    n_test = n_rem - n_val
    if n_val < 1:
        n_val = 1
        n_test = max(1, n_rem - 1)
        n_train = n_days - n_val - n_test
    if n_test < 1:
        n_test = 1
        n_val = max(1, n_rem - 1)
        n_train = n_days - n_val - n_test
    assert n_train + n_val + n_test == n_days
    return n_train, n_val, n_test


def split_kool_trials_contiguous(
    trials: Sequence[Dict[str, Any]],
    *,
    n_presented_days: int = N_PRESENTED_DAYS,
    split_ratio: float = 0.8,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Contiguous split over days that emit ≥1 decision (chronological).

    ``n_presented_days`` is retained for documentation/API parity with the 125-day
    calendar; the cut is taken on the sorted usable presented_day ids so subjects
    with late-only timeouts still get nonempty val/test when they have ≥3 usable
    days. ``split_seed`` is unused (no shuffle — required for history validity).
    """
    del split_seed, n_presented_days
    by_day: Dict[int, List[Dict[str, Any]]] = {}
    for t in trials:
        d = int(t["problem"]["presented_day"])
        by_day.setdefault(d, []).append(t)
    usable_days = sorted(by_day)
    n_train, n_val, _n_test = _contiguous_day_bounds(len(usable_days), split_ratio)
    train_days = set(usable_days[:n_train])
    val_days = set(usable_days[n_train : n_train + n_val])
    test_days = set(usable_days[n_train + n_val :])

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for d in usable_days:
        if d in train_days:
            dest = train
        elif d in val_days:
            dest = val
        else:
            dest = test
        day_trials = by_day[d]
        if len(day_trials) not in (1, 2):
            raise ValueError(
                f"day {d}: expected 1 or 2 decision trials, got {len(day_trials)}"
            )
        day_trials = sorted(day_trials, key=lambda x: int(x["problem"]["stage"]))
        if int(day_trials[0]["problem"]["stage"]) != 1:
            raise ValueError(f"day {d}: first trial must be stage1")
        if len(day_trials) == 2 and int(day_trials[1]["problem"]["stage"]) != 2:
            raise ValueError(f"day {d}: second trial must be stage2")
        dest.extend(day_trials)

    if not train or not val or not test:
        raise ValueError(
            f"kool contiguous split produced empty partition "
            f"(train={len(train)}, val={len(val)}, test={len(test)})"
        )
    options = list(train[0]["options"])
    return train, val, test, options


def split_kool2016_exp2_experiment(
    exp: PsychExperiment,
    *,
    split_ratio: float = 0.8,
    split_seed: int = 42,
    text: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    if text is not None:
        days = _parse_day_chunks(text)
        trials = build_kool_decision_trials(exp, days=days)
    else:
        trials = build_kool_decision_trials(exp, days=None)
    return split_kool_trials_contiguous(
        trials,
        n_presented_days=N_PRESENTED_DAYS,
        split_ratio=split_ratio,
        split_seed=split_seed,
    )


def resolve_validation_mat(path: Optional[str | Path] = None) -> Path:
    return Path(path) if path is not None else Path(DEFAULT_VALIDATION_MAT)


def load_groupdata_subjects(mat_path: Optional[str | Path] = None):
    import scipy.io as sio

    path = resolve_validation_mat(mat_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Kool validation MAT not found at {path}. "
            "Run: python scripts/setup_external_kool2016_exp2.py"
        )
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return mat["groupdata"].subdata


def fingerprint_win_timeout(days: Sequence[Dict[str, Any]]) -> Tuple[int, ...]:
    fp: List[int] = []
    for d in days:
        if d["kind"] in ("timeout", "timeout_stage2"):
            fp.append(-1)
        elif d["kind"] == "complete":
            fp.append(int(d["reward"]))
        else:
            fp.append(-2)
    return tuple(fp)


def author_fingerprint(sub) -> Tuple[int, ...]:
    w = np.asarray(sub.win).astype(int)
    m = np.asarray(sub.missed).astype(int)
    out = w.copy()
    out[m == 1] = -1
    return tuple(out.tolist())


def match_psych_to_author_index(
    days: Sequence[Dict[str, Any]],
    *,
    mat_path: Optional[str | Path] = None,
) -> int:
    target = fingerprint_win_timeout(days)
    subs = load_groupdata_subjects(mat_path)
    matches = [i for i, s in enumerate(subs) if author_fingerprint(s) == target]
    if len(matches) != 1:
        raise ValueError(f"expected unique author match, got {matches}")
    return matches[0]


def assert_psych_matches_groupdata(
    text: str,
    *,
    mat_path: Optional[str | Path] = None,
    author_index: Optional[int] = None,
) -> int:
    """
    Validate Psych-101 against author groupdata.mat:
    day count, timeout/missed, reward/win, choice1/choice2/state2, ordering.
    """
    days = _parse_day_chunks(text)
    if len(days) != N_PRESENTED_DAYS:
        raise AssertionError(f"expected {N_PRESENTED_DAYS} days, got {len(days)}")
    ai = (
        int(author_index)
        if author_index is not None
        else match_psych_to_author_index(days, mat_path=mat_path)
    )
    sub = load_groupdata_subjects(mat_path)[ai]
    if int(sub.N) != N_PRESENTED_DAYS:
        raise AssertionError(f"author N={sub.N}")

    ship_to_c1: Dict[str, List[int]] = defaultdict(list)
    alien_to_c2: Dict[str, List[int]] = defaultdict(list)
    planet_to_s2: Dict[str, List[int]] = defaultdict(list)
    for di, day in enumerate(days):
        if day["kind"] == "timeout":
            if int(sub.missed[di]) != 1:
                raise AssertionError(f"day {di}: psych timeout but author not missed")
            continue
        if day["kind"] == "timeout_stage2":
            if int(sub.missed[di]) != 1:
                raise AssertionError(
                    f"day {di}: psych stage2-timeout but author not missed"
                )
            ship_to_c1[day["spaceship"]].append(int(sub.choice1[di]))
            planet_to_s2[day["planet"]].append(int(sub.state2[di]))
            continue
        if day["kind"] != "complete":
            raise AssertionError(f"day {di}: unexpected kind {day['kind']}")
        if int(sub.missed[di]) != 0:
            raise AssertionError(f"day {di}: psych complete but author missed")
        ship_to_c1[day["spaceship"]].append(int(sub.choice1[di]))
        alien_to_c2[day["alien"]].append(int(sub.choice2[di]))
        planet_to_s2[day["planet"]].append(int(sub.state2[di]))
        if int(day["reward"]) != int(sub.win[di]):
            raise AssertionError(f"day {di}: reward/win mismatch")

    def mode_map(d: Dict[str, List[int]]) -> Dict[str, int]:
        return {k: Counter(v).most_common(1)[0][0] for k, v in d.items()}

    s2c = mode_map(ship_to_c1)
    a2c = mode_map(alien_to_c2)
    p2s = mode_map(planet_to_s2)

    for di, day in enumerate(days):
        if day["kind"] == "timeout":
            continue
        if day["kind"] in ("complete", "timeout_stage2"):
            if s2c[day["spaceship"]] != int(sub.choice1[di]):
                raise AssertionError(f"day {di}: choice1 mismatch")
            if p2s[day["planet"]] != int(sub.state2[di]):
                raise AssertionError(f"day {di}: state2/planet mismatch")
        if day["kind"] == "complete":
            if a2c[day["alien"]] != int(sub.choice2[di]):
                raise AssertionError(f"day {di}: choice2 mismatch")
    return ai


def assert_no_stage_leakage(trials: Sequence[Dict[str, Any]]) -> None:
    """Explicit leakage assertions for stage-1/2 problem+history."""
    banned_stage1 = {
        "planet",
        "alien",
        "alien_options",
        "reward",
        "treasure",
        "win",
        "feedback",
        "stage2_action",
        "common",
        "ps1a1",
        "ps1a2",
        "ps2a1",
        "ps2a2",
    }
    expected_hist = 0
    i = 0
    while i < len(trials):
        t = trials[i]
        p = t["problem"]
        stage = int(p["stage"])
        day = int(p["presented_day"])
        if stage != 1:
            raise AssertionError(f"trial {i}: expected stage1 at day boundary")
        for k in banned_stage1:
            if k in p:
                raise AssertionError(f"trial {i}: stage1 problem has {k}")
        if len(t["history"]) != expected_hist:
            raise AssertionError(
                f"day {day} stage1: history len {len(t['history'])} != {expected_hist}"
            )
        # Peek whether stage2 follows same day.
        has_stage2 = (
            i + 1 < len(trials)
            and int(trials[i + 1]["problem"]["stage"]) == 2
            and int(trials[i + 1]["problem"]["presented_day"]) == day
        )
        # After stage1 choice, planet enters history (+1).
        expected_after_s1 = expected_hist + 1
        if has_stage2:
            t2 = trials[i + 1]
            p2 = t2["problem"]
            if "planet" not in p2:
                raise AssertionError(f"trial {i+1}: stage2 missing planet")
            for k in ("reward", "treasure", "win"):
                if k in p2:
                    raise AssertionError(f"trial {i+1}: stage2 problem has {k}")
            if len(t2["history"]) != expected_after_s1:
                raise AssertionError(
                    f"day {day} stage2: history len {len(t2['history'])} "
                    f"!= {expected_after_s1}"
                )
            last = t2["history"][-1]
            if int(last.get("stage", 0)) != 1 or last.get("planet") != p2["planet"]:
                raise AssertionError(
                    f"trial {i+1}: stage2 history must end with same-day stage1+planet"
                )
            if "reward" in last or "feedback" in last:
                raise AssertionError(
                    f"trial {i+1}: pre-stage2 history must not include current reward"
                )
            expected_hist = expected_after_s1 + 1  # + stage2 reward entry
            i += 2
        else:
            # Stage2 timeout: only stage1 trial; planet remains in history, no reward.
            expected_hist = expected_after_s1
            i += 1


def assert_split_keeps_days_together(
    train: Sequence[Dict[str, Any]],
    val: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
) -> None:
    def days(trials: Sequence[Dict[str, Any]]):
        return {int(t["problem"]["presented_day"]) for t in trials}

    td, vd, sd = days(train), days(val), days(test)
    if td & vd or td & sd or vd & sd:
        raise AssertionError("day overlap across splits")
    for trials in (train, val, test):
        by: Dict[int, List[int]] = defaultdict(list)
        for t in trials:
            by[int(t["problem"]["presented_day"])].append(int(t["problem"]["stage"]))
        for d, stages in by.items():
            stages_sorted = sorted(stages)
            if stages_sorted not in ([1], [1, 2]):
                raise AssertionError(f"day {d} stages {stages} invalid")
    if max(td) >= min(vd) or max(vd) >= min(sd):
        raise AssertionError(
            f"split not contiguous: train={min(td)}..{max(td)} "
            f"val={min(vd)}..{max(vd)} test={min(sd)}..{max(sd)}"
        )
