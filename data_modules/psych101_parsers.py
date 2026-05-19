"""
Dataset-specific Psych-101 NL parsers -> PsychExperiment (see psych101_binary.py).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from data_modules.choice13k import (
    Gamble,
    _convert_to_experiment as _choice13k_convert_to_experiment,
    _extract_gamble_info,
    _extract_has_feedback,
)

from data_modules.psych101_binary import ParsedTrial, PsychBlock, PsychExperiment

_RE_PRESS_KEY = re.compile(r"You press <<([A-Z])>>", re.I)
_RE_RECEIVE = re.compile(r"You receive (-?\d+\.?\d*) points", re.I)
_RE_GAIN_LOSE = re.compile(r"and (gain|lose) (-?\d+\.?\d*) points", re.I)
_RE_OUTCOME_PROB = re.compile(
    r"(-?\d+\.?\d*)\s+points\s+with\s+(\d+\.?\d*)%\s+probability", re.I
)
_RE_LOTTERY_LINE = re.compile(r"Lottery\s+([A-Z])\s+offers\s+(.+?)\.\s*$", re.I | re.M)
_RE_WULFF_BLOCK = re.compile(
    r"Lottery\s+([A-Z])\s+offers\s+(.+?)\.\s*\n"
    r"Lottery\s+([A-Z])\s+offers\s+(.+?)\.\s*\n"
    r"You press <<([A-Z])>>\.\s*",
    re.I | re.S,
)
_RE_WEATHER_TRIAL = re.compile(
    r"You are seeing the following:\s*([^.]+)\.\s*"
    r"You press <<([A-Z])>>\.\s*"
    r"You are (correct|wrong), the weather is (?:indeed )?(rainy|fine)\w*\.?",
    re.I,
)
_RE_PRODUCT_TRIAL = re.compile(
    r"Product\s+([A-Z])\s+ratings:\s*\[([^\]]+)\]\.\s*"
    r"Product\s+([A-Z])\s+ratings:\s*\[([^\]]+)\]\.\s*"
    r"You press <<([A-Z])>>\.",
    re.I,
)
_RE_GAME_HEADER = re.compile(
    r"Game\s+(\d+)\.\s*There are\s+(\d+)\s+trials", re.I
)
_RE_MACHINE_LABEL = re.compile(r"labeled\s+([A-Z])\s+and\s+([A-Z])", re.I)
_RE_INSTRUCTED = re.compile(
    r"You are instructed to press ([A-Z]) and get (-?\d+) points\.?", re.I
)
_RE_SLOT_PRESS = re.compile(
    r"You press <<([A-Z])>> and get (-?\d+) points\.?", re.I
)
_RE_ROUND_HEADER = re.compile(r"Round\s+(\d+):\s*", re.I)
_RE_ROUND_GAIN = re.compile(
    r"You will be awarded\s+(-?\d+)\s+points for turning over a gain card", re.I
)
_RE_ROUND_LOSS = re.compile(
    r"You will lose\s+(-?\d+)\s+points for turning over a loss card", re.I
)
_RE_ROUND_NLOSS = re.compile(r"There are\s+(\d+)\s+loss cards in this round", re.I)
_RE_FREY_CCT_KEYS = re.compile(
    r"Press\s+([A-Z])\s+to turn a card over,\s+or\s+([A-Z])\s+to stop the round",
    re.I,
)
_RE_CCT_PRESS = re.compile(
    r"You press <<([A-Z])>>\s+and\s+"
    r"(?:turn over a (gain|loss) card|(?:stop the round and )?claim your payout)",
    re.I,
)
_RE_CCT_SCORE = re.compile(r"Your current score is\s+(-?\d+)", re.I)
_RE_TREE_TRIAL = re.compile(
    r"You get a tree with level (\d+) of leafiness and level (\d+) of branchiness "
    r"in the (North|South) garden\.\s*"
    r"You press <<([A-Z])>>(?: and get (-?\d+) points)?",
    re.I,
)
_N_CCT_CARDS = 32


def _parse_gamble_from_offers(offers: str) -> Gamble:
    matches = _RE_OUTCOME_PROB.findall(offers)
    if not matches:
        rewards = [float(r) for r in re.findall(r"(-?\d+\.?\d*)\s+points", offers)]
        if len(rewards) == 1:
            return Gamble(probs=[1.0], rewards=rewards)
        raise ValueError(f"Could not parse lottery offers: {offers!r}")
    rewards, probs = [], []
    for reward, prob in matches:
        rewards.append(float(reward))
        probs.append(float(prob) / 100.0)
    if not math.isclose(sum(probs), 1.0, rel_tol=1e-3):
        # single-outcome lotteries sometimes omit second outcome
        if len(probs) == 1:
            probs = [1.0]
        else:
            raise ValueError(f"Probs sum to {sum(probs)}, offers={offers!r}")
    return Gamble(probs=probs, rewards=rewards)


def _feedback_from_press_tail(tail: str) -> Optional[float]:
    m = _RE_RECEIVE.search(tail)
    if m:
        return float(m.group(1))
    m = _RE_GAIN_LOSE.search(tail)
    if m:
        sign, val = m.group(1).lower(), float(m.group(2))
        return val if sign == "gain" else -val
    return None


def _extract_press_trials(
    text: str, option_keys: List[str], *, require_known_key: bool = True
) -> List[ParsedTrial]:
    trials: List[ParsedTrial] = []
    for m in _RE_PRESS_KEY.finditer(text):
        key = m.group(1).upper()
        if require_known_key and key not in option_keys:
            continue
        if key not in option_keys:
            option_keys = list(option_keys) + [key]
        action = option_keys.index(key)
        tail = text[m.end() : m.end() + 200]
        fb = _feedback_from_press_tail(tail)
        trials.append(ParsedTrial(action=action, feedback=fb, problem_fields={}))
    return trials


def _experiment_from_choice13k_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    exp = _choice13k_convert_to_experiment(row)
    blocks = []
    for bi, block in enumerate(exp.blocks):
        static = {
            "schema_type": "A",
            "gamble_A": {"probs": block.gamble_A.probs, "rewards": block.gamble_A.rewards},
            "gamble_B": {"probs": block.gamble_B.probs, "rewards": block.gamble_B.rewards},
            "option_keys": list(block.option_keys),
            "has_feedback": block.has_feedback,
            "block_index": bi,
        }
        trials = [
            ParsedTrial(action=t.action, feedback=t.feedback, problem_fields={})
            for t in block.trials
        ]
        blocks.append(
            PsychBlock(
                trials=trials,
                option_keys=list(block.option_keys),
                problem_static=static,
                schema_type="A",
            )
        )
    return PsychExperiment(
        instruction=exp.instruction,
        blocks=blocks,
        dataset_alias=dataset_alias,
        schema_type="A",
    )


def _experiment_from_plonsky_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    data = dict(row)
    text = data["text"]
    text = text.replace("\n\n\n\nOption", "\n\nOption").replace("\n\n\nOption", "\n\nOption")
    instruction = text.split("\n\nOption")[0]
    trials_str = text[len(instruction) :].lstrip()
    blocks: List[PsychBlock] = []
    for bi, block_str in enumerate(trials_str.split("\n\n")):
        if "Option " not in block_str:
            continue
        gamble_info = block_str.split("\nYou press")[0]
        gamble_A, gamble_B, option_keys = _extract_gamble_info(gamble_info)
        press_text = block_str[len(gamble_info) :]
        parsed = _extract_press_trials(press_text, list(option_keys))
        if not parsed:
            continue
        static = {
            "schema_type": "A",
            "gamble_A": {"probs": gamble_A.probs, "rewards": gamble_A.rewards},
            "gamble_B": {"probs": gamble_B.probs, "rewards": gamble_B.rewards},
            "option_keys": list(option_keys),
            "has_feedback": _extract_has_feedback(block_str),
            "block_index": bi,
        }
        blocks.append(
            PsychBlock(
                trials=parsed,
                option_keys=list(option_keys),
                problem_static=static,
                schema_type="A",
            )
        )
    if not blocks:
        raise ValueError("plonsky parser produced no blocks")
    return PsychExperiment(
        instruction=instruction.strip(),
        blocks=blocks,
        dataset_alias=dataset_alias,
        schema_type="A",
    )


def parse_wulff_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    text = row["text"]
    m0 = re.search(r"Lottery\s+[A-Z]\s+offers", text, re.I)
    instruction = text[: m0.start()].strip() if m0 else text.split("\n\n")[0]
    blocks: List[PsychBlock] = []
    for bi, m in enumerate(_RE_WULFF_BLOCK.finditer(text)):
        k1, off1, k2, off2, press = (
            m.group(1).upper(),
            m.group(2),
            m.group(3).upper(),
            m.group(4),
            m.group(5).upper(),
        )
        option_keys = [k1, k2]
        g1 = _parse_gamble_from_offers(off1)
        g2 = _parse_gamble_from_offers(off2)
        action = option_keys.index(press)
        static = {
            "schema_type": "A",
            "gamble_A": {"probs": g1.probs, "rewards": g1.rewards},
            "gamble_B": {"probs": g2.probs, "rewards": g2.rewards},
            "option_keys": option_keys,
            "has_feedback": False,
            "block_index": bi,
        }
        blocks.append(
            PsychBlock(
                trials=[ParsedTrial(action=action, feedback=None, problem_fields={})],
                option_keys=option_keys,
                problem_static=static,
                schema_type="A",
            )
        )
    if not blocks:
        raise ValueError("wulff parser produced no blocks")
    return PsychExperiment(
        instruction=instruction,
        blocks=blocks,
        dataset_alias=dataset_alias,
        schema_type="A",
    )


def parse_speekenbrink_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    text = row["text"]
    m0 = re.search(r"You are seeing the following:", text, re.I)
    instruction = text[: m0.start()].strip() if m0 else text.split("\n\n")[0]
    m_rain = re.search(r"rainy weather \(by pressing ([A-Z])\)", text, re.I)
    m_fine = re.search(r"fine weather \(by pressing ([A-Z])\)", text, re.I)
    if not (m_rain and m_fine):
        raise ValueError("speekenbrink: could not find rainy/fine press keys in instructions")
    option_keys = [m_rain.group(1).upper(), m_fine.group(1).upper()]
    trials: List[ParsedTrial] = []
    for m in _RE_WEATHER_TRIAL.finditer(text):
        cards_str, key, correct_word, weather = (
            m.group(1),
            m.group(2).upper(),
            m.group(3).lower(),
            m.group(4).lower(),
        )
        cards = [int(c) for c in re.findall(r"card\s+(\d+)", cards_str, re.I)]
        if key not in option_keys:
            raise ValueError(f"speekenbrink: unexpected press key {key} not in {option_keys}")
        action = option_keys.index(key)
        was_correct = correct_word == "correct"
        trials.append(
            ParsedTrial(
                action=action,
                feedback=1.0 if was_correct else 0.0,
                problem_fields={
                    "cards": cards,
                    "weather_outcome": weather,
                    "was_correct": was_correct,
                },
            )
        )
    if not trials:
        raise ValueError("speekenbrink: no trials parsed")
    static = {
        "schema_type": "B",
        "option_keys": option_keys,
        "features": {"task": "weather_prediction"},
    }
    block = PsychBlock(
        trials=trials,
        option_keys=option_keys,
        problem_static=static,
        schema_type="B",
    )
    return PsychExperiment(
        instruction=instruction,
        blocks=[block],
        dataset_alias=dataset_alias,
        schema_type="B",
    )


def parse_hilbig_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    text = row["text"]
    m0 = re.search(r"Product\s+[A-Z]\s+ratings:", text, re.I)
    instruction = text[: m0.start()].strip() if m0 else text.split("\n\n")[0]
    trials: List[ParsedTrial] = []
    option_keys: Optional[List[str]] = None

    def _ratings_vec(s: str) -> List[int]:
        return [int(x) for x in s.split()]

    for m in _RE_PRODUCT_TRIAL.finditer(text):
        k_a, r_a, k_b, r_b, press = (
            m.group(1).upper(),
            _ratings_vec(m.group(2)),
            m.group(3).upper(),
            _ratings_vec(m.group(4)),
            m.group(5).upper(),
        )
        if option_keys is None:
            option_keys = [k_a, k_b]
        if press not in option_keys:
            raise ValueError(f"hilbig: press {press} not in {option_keys}")
        action = option_keys.index(press)
        trials.append(
            ParsedTrial(
                action=action,
                feedback=None,
                problem_fields={
                    "option_A_features": {"key": k_a},
                    "option_B_features": {"key": k_b},
                    "ratings_A": r_a,
                    "ratings_B": r_b,
                },
            )
        )
    if not option_keys or not trials:
        raise ValueError(f"hilbig: invalid parse keys={option_keys} n={len(trials)}")
    static = {"schema_type": "B", "option_keys": option_keys}
    return PsychExperiment(
        instruction=instruction,
        blocks=[
            PsychBlock(
                trials=trials,
                option_keys=option_keys,
                problem_static=static,
                schema_type="B",
            )
        ],
        dataset_alias=dataset_alias,
        schema_type="B",
    )


def parse_sadeghiyeh_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    text = row["text"]
    m0 = re.search(r"Game\s+\d+\.", text, re.I)
    instruction = text[: m0.start()].strip() if m0 else text.split("\n\n")[0]
    blocks: List[PsychBlock] = []
    game_parts = re.split(r"(?=Game\s+\d+\.)", text[m0.start() :] if m0 else text)
    for part in game_parts:
        if not part.strip() or not re.match(r"Game\s+\d+", part, re.I):
            continue
        gh = _RE_GAME_HEADER.search(part)
        if not gh:
            continue
        game_id = int(gh.group(1))
        n_trials_game = int(gh.group(2))
        ml = _RE_MACHINE_LABEL.search(part)
        if not ml:
            ml = _RE_MACHINE_LABEL.search(text)
        if not ml:
            raise ValueError("sadeghiyeh: no machine labels")
        option_keys = [ml.group(1).upper(), ml.group(2).upper()]
        trials: List[ParsedTrial] = []
        trial_idx = 0
        for im in _RE_INSTRUCTED.finditer(part):
            key, pts = im.group(1).upper(), int(im.group(2))
            trials.append(
                ParsedTrial(
                    action=option_keys.index(key),
                    feedback=float(pts),
                    problem_fields={
                        "trial_index": trial_idx,
                        "phase": "instructed",
                        "machine_options": list(option_keys),
                        "payoff": float(pts),
                    },
                )
            )
            trial_idx += 1
        for pm in _RE_SLOT_PRESS.finditer(part):
            key, pts = pm.group(1).upper(), int(pm.group(2))
            trials.append(
                ParsedTrial(
                    action=option_keys.index(key),
                    feedback=float(pts),
                    problem_fields={
                        "trial_index": trial_idx,
                        "phase": "free",
                        "machine_options": list(option_keys),
                        "payoff": float(pts),
                    },
                )
            )
            trial_idx += 1
        if not trials:
            continue
        static = {
            "schema_type": "C",
            "option_keys": option_keys,
            "game_id": game_id,
            "n_trials_game": n_trials_game,
            "machine_options": list(option_keys),
        }
        blocks.append(
            PsychBlock(
                trials=trials,
                option_keys=option_keys,
                problem_static=static,
                schema_type="C",
            )
        )
    if not blocks:
        raise ValueError("sadeghiyeh parser produced no blocks")
    return PsychExperiment(
        instruction=instruction,
        blocks=blocks,
        dataset_alias=dataset_alias,
        schema_type="C",
    )


def _infer_frey_cct_option_keys(instruction: str, text: str) -> List[str]:
    """Return [flip_key, stop_key] from instruction or press-line semantics."""
    m = _RE_FREY_CCT_KEYS.search(instruction)
    if m:
        return [m.group(1).upper(), m.group(2).upper()]
    flip_key: Optional[str] = None
    stop_key: Optional[str] = None
    for pm in _RE_CCT_PRESS.finditer(text):
        key = pm.group(1).upper()
        if pm.group(2):
            flip_key = key
        else:
            stop_key = key
    if flip_key and stop_key:
        return [flip_key, stop_key]
    seen = sorted({k.upper() for k in re.findall(r"You press <<([A-Z])>>", text)})
    if len(seen) >= 2:
        return [seen[0], seen[1]]
    if len(seen) == 1:
        return [seen[0], seen[0]]
    return ["E", "C"]


def _frey_cct_press_is_stop(pm: re.Match[str], stop_key: str) -> bool:
    if pm.group(1).upper() == stop_key:
        return True
    return pm.group(2) is None


def parse_frey_cct_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    text = row["text"]
    m0 = re.search(r"Round\s+\d+:", text, re.I)
    instruction = text[: m0.start()].strip() if m0 else text.split("\n\n")[0]
    blocks: List[PsychBlock] = []
    option_keys = _infer_frey_cct_option_keys(instruction, text)
    flip_key, stop_key = option_keys[0], option_keys[1]
    round_chunks = re.split(r"(?=Round\s+\d+:\s*)", text[m0.start() :] if m0 else text)
    for chunk in round_chunks:
        if not re.match(r"Round\s+\d+", chunk, re.I):
            continue
        rh = _RE_ROUND_HEADER.match(chunk)
        if not rh:
            continue
        round_id = int(rh.group(1))
        mg = _RE_ROUND_GAIN.search(chunk)
        ml = _RE_ROUND_LOSS.search(chunk)
        mn = _RE_ROUND_NLOSS.search(chunk)
        if not (mg and ml and mn):
            continue
        gain_amount = int(mg.group(1))
        loss_amount = int(ml.group(1))
        n_loss_cards = int(mn.group(1))
        cards_flipped = 0
        current_score = 0
        trials: List[ParsedTrial] = []
        for pm in _RE_CCT_PRESS.finditer(chunk):
            key = pm.group(1).upper()
            action = option_keys.index(key)
            n_remaining = _N_CCT_CARDS - cards_flipped
            static_trial = {
                "schema_type": "D",
                "option_keys": option_keys,
                "round_id": round_id,
                "gain_amount": gain_amount,
                "loss_amount": loss_amount,
                "n_loss_cards": n_loss_cards,
                "n_cards_total": _N_CCT_CARDS,
                "cards_flipped": cards_flipped,
                "current_score": current_score,
                "n_cards_remaining": n_remaining,
            }
            trials.append(
                ParsedTrial(
                    action=action,
                    feedback=None,
                    problem_fields=dict(static_trial),
                )
            )
            if _frey_cct_press_is_stop(pm, stop_key):
                break
            if key != flip_key:
                break
            cards_flipped += 1
            sm = _RE_CCT_SCORE.search(chunk, pm.end())
            if sm:
                current_score = int(sm.group(1))
            if pm.group(2) and pm.group(2).lower() == "loss":
                break
        if not trials:
            continue
        blocks.append(
            PsychBlock(
                trials=trials,
                option_keys=option_keys,
                problem_static={
                    "schema_type": "D",
                    "option_keys": option_keys,
                    "round_id": round_id,
                    "gain_amount": gain_amount,
                    "loss_amount": loss_amount,
                    "n_loss_cards": n_loss_cards,
                    "n_cards_total": _N_CCT_CARDS,
                },
                schema_type="D",
            )
        )
    if not blocks:
        raise ValueError("frey CCT parser produced no blocks")
    return PsychExperiment(
        instruction=instruction,
        blocks=blocks,
        dataset_alias=dataset_alias,
        schema_type="D",
    )


def parse_flesch_row(row: Dict[str, Any], dataset_alias: str) -> PsychExperiment:
    text = row["text"]
    m0 = re.search(r"You get a tree with", text, re.I)
    instruction = text[: m0.start()].strip() if m0 else text.split("\n\n")[0]
    m_acc = re.search(r"accept to plant the tree by pressing ([A-Z])", text, re.I)
    m_rej = re.search(r"reject to plant it by pressing ([A-Z])", text, re.I)
    if not (m_acc and m_rej):
        raise ValueError("flesch: could not find accept/reject keys in instructions")
    option_keys = [m_rej.group(1).upper(), m_acc.group(1).upper()]
    testing_idx = text.lower().find("testing phase")
    trials: List[ParsedTrial] = []
    for m in _RE_TREE_TRIAL.finditer(text):
        leaf, branch, garden, key = (
            int(m.group(1)),
            int(m.group(2)),
            m.group(3),
            m.group(4).upper(),
        )
        payoff = float(m.group(5)) if m.group(5) else None
        if key not in option_keys:
            raise ValueError(f"flesch: unexpected key {key} not in {option_keys}")
        phase = "testing" if (testing_idx >= 0 and m.start() >= testing_idx) else "training"
        trials.append(
            ParsedTrial(
                action=option_keys.index(key),
                feedback=payoff,
                problem_fields={
                    "tree_features": {"leafiness": leaf, "branchiness": branch},
                    "garden": garden,
                    "phase": phase,
                },
            )
        )
    if not trials:
        raise ValueError("flesch: no trials parsed")
    # Accept=T, Reject=N typical; ensure stable order [first_key, second_key]
    static = {"schema_type": "B", "option_keys": option_keys}
    return PsychExperiment(
        instruction=instruction,
        blocks=[
            PsychBlock(
                trials=trials,
                option_keys=option_keys,
                problem_static=static,
                schema_type="B",
            )
        ],
        dataset_alias=dataset_alias,
        schema_type="B",
    )


PARSER_DISPATCH = {
    "choice13k": _experiment_from_choice13k_row,
    "option_delivers_extended": _experiment_from_plonsky_row,
    "lottery_offers": parse_wulff_row,
    "weather_cards": parse_speekenbrink_row,
    "product_ratings": parse_hilbig_row,
    "slot_machine_bandit": parse_sadeghiyeh_row,
    "columbia_card_task": parse_frey_cct_row,
    "tree_accept_reject": parse_flesch_row,
}
