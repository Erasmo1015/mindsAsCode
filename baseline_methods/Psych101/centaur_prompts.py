"""
Centaur Psych-101-style prompt prefixes for TEH datasets (incl. five new tasks).

Display keys for suffix scoring are aligned with internal action indices 0..K-1.
For bandits whose Psych-101 / raw transcripts use 1-indexed presses, display keys
are \"1\"..\"K\" (action a <-> key str(a+1)). Never put future rewards, cue
validities, cond labels, or out-of-scope history into the prefix.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def centaur_display_keys(problem: Dict[str, Any]) -> List[str]:
    """Ordered Centaur press tokens, index-aligned with internal actions."""
    alias = str(problem.get("dataset_alias") or "")
    schema = str(problem.get("schema_type") or "")
    if schema == "categorical_bandit" or alias == "13schulz2020finding":
        n = int(problem.get("n_arms") or len(problem.get("option_keys") or []) or 8)
        return [str(i + 1) for i in range(n)]
    if alias == "steyvers_2009_bandit" or schema == "steyvers_bandit":
        n = int(problem.get("n_arms") or len(problem.get("option_keys") or []) or 4)
        return [str(i + 1) for i in range(n)]
    if alias == "guan_2020_stopping" or schema == "guan_stopping":
        return ["0", "1"]
    if alias == "bergert_nosofsky_2007" or schema == "bergert_pairwise":
        return ["0", "1"]
    keys = problem.get("option_keys") or []
    return [str(k) for k in keys]


def _action_display_key(problem: Dict[str, Any], action: int) -> str:
    keys = centaur_display_keys(problem)
    if 0 <= int(action) < len(keys):
        return keys[int(action)]
    return "?"


def _cues_line(cues: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(cues.keys(), key=lambda x: (len(str(x)), str(x))):
        parts.append(f"{k}={int(cues[k])}")
    return ", ".join(parts)


def build_bergert_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    """Independent pairwise cue choice; history is always empty (no leakage)."""
    cur = trials[trial_index]["problem"]
    hist = trials[trial_index].get("history") or []
    if hist:
        raise ValueError("bergert Centaur prefix expects empty history (no cross-trial carryover)")
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    opt_a = cur.get("option_A") or {}
    opt_b = cur.get("option_B") or {}
    cues_a = opt_a.get("cues") or {}
    cues_b = opt_b.get("cues") or {}
    # action 0 -> option_B, action 1 -> option_A (TEH coding)
    parts.append(
        "You must choose between two alternatives using six binary cues "
        "(cue validities are not shown).\n"
        f"Option <<0>> is option_B (id={opt_b.get('alternative_id', '?')}): {_cues_line(cues_b)}.\n"
        f"Option <<1>> is option_A (id={opt_a.get('alternative_id', '?')}): {_cues_line(cues_a)}."
    )
    parts.append("You press ")
    return "\n\n".join(parts)


def build_guan_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    """Within-problem continue/stop; only values_observed (no future values)."""
    cur = trials[trial_index]["problem"]
    hist = trials[trial_index].get("history") or []
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    env = cur.get("environment", "?")
    L = cur.get("sequence_length", "?")
    pos = cur.get("position", "?")
    observed = list(cur.get("values_observed") or [])
    # Leakage checks: observed length must equal position; history only continues.
    if isinstance(pos, int) and len(observed) != int(pos):
        raise ValueError(
            f"guan leakage check failed: len(values_observed)={len(observed)} != position={pos}"
        )
    for h in hist:
        if int(h.get("action", -1)) != 0:
            raise ValueError("guan history must contain continue (action=0) only")
    current = observed[-1] if observed else "?"
    parts.append(
        f"Environment: {env}. Sequence length: {L}. Position: {pos}.\n"
        f"Values observed so far: {observed}. Current value: {current}.\n"
        "Press <<0>> to continue or <<1>> to stop."
    )
    parts.append("You press ")
    return "\n\n".join(parts)


def build_steyvers_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    """4-arm bandit; history resets per game; display keys 1..4."""
    cur = trials[trial_index]["problem"]
    hist = trials[trial_index].get("history") or []
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    game = cur.get("game", "?")
    trial = cur.get("trial", "?")
    if not hist:
        parts.append(
            f"You are playing a four-armed bandit game (game {game}). "
            "On each trial press one of <<1>>, <<2>>, <<3>>, or <<4>> and observe a 0/1 reward."
        )
    else:
        parts.append(f"Game {game}, continuing within the same game.")
    for h in hist:
        key = _action_display_key(cur, int(h["action"]))
        reward = h.get("reward", h.get("feedback", "?"))
        parts.append(f"You press <<{key}>> and get {reward} points.")
    parts.append(f"Trial {trial}. You press ")
    return "\n\n".join(parts)


def build_schulz_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    """8-arm bandit; history within-round only; display keys 1..8; no cond labels."""
    cur = trials[trial_index]["problem"]
    hist = trials[trial_index].get("history") or []
    # Reject experimenter condition leakage if present on problem.
    banned = {"cond", "rcond", "structure", "srs", "rsr"}
    for k in cur:
        if str(k).lower() in banned:
            raise ValueError(f"schulz Centaur prefix forbids condition field {k!r}")
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    round_id = cur.get("round", "?")
    trial = cur.get("trial", "?")
    if not hist:
        parts.append(
            f"You are playing round {round_id}: choose among options "
            "<<1>> .. <<8>> and observe points. Options reset each round."
        )
    else:
        parts.append(f"You are playing round {round_id}:")
    for h in hist:
        key = _action_display_key(cur, int(h["action"]))
        reward = h.get("reward", h.get("feedback", "?"))
        parts.append(f"You press <<{key}>> and get {reward} points.")
    parts.append(f"Trial {trial}. You press ")
    return "\n\n".join(parts)


def build_kool_prefix(
    trials: List[Dict[str, Any]], trial_index: int, *, instruction: str = ""
) -> str:
    """Two-step spaceship/alien; continuous session history; letter keys."""
    cur = trials[trial_index]["problem"]
    hist = trials[trial_index].get("history") or []
    parts: List[str] = []
    if instruction.strip():
        parts.append(instruction.strip()[:2000])
    for h in hist:
        stage = int(h.get("stage", 0))
        keys = [str(k) for k in (h.get("option_keys") or [])]
        act = int(h["action"])
        letter = keys[act] if 0 <= act < len(keys) else "?"
        if stage == 1:
            if len(keys) >= 2:
                parts.append(f"You are presented with spaceships {keys[0]} and {keys[1]}.")
            parts.append(f"You press <<{letter}>>.")
            planet = h.get("planet")
            if planet is not None:
                parts.append(f"You arrive at planet {planet}.")
        elif stage == 2:
            if len(keys) >= 2:
                parts.append(f"You see aliens {keys[0]} and {keys[1]}.")
            reward = h.get("reward", h.get("feedback", "?"))
            parts.append(f"You press <<{letter}>> and get {reward} points.")
        else:
            parts.append(f"You press <<{letter}>>.")
    stage = int(cur.get("stage", 1))
    keys = [str(k) for k in (cur.get("option_keys") or [])]
    if stage == 1:
        if len(keys) >= 2:
            parts.append(f"You are presented with spaceships {keys[0]} and {keys[1]}.")
    else:
        # Planet already revealed by today's stage-1 history entry; only show aliens.
        if len(keys) >= 2:
            parts.append(f"You see aliens {keys[0]} and {keys[1]}.")
    parts.append("You press ")
    return "\n\n".join(parts)


def try_build_extended_centaur_prefix(
    trials: List[Dict[str, Any]],
    trial_index: int,
    *,
    instruction: str = "",
) -> Optional[str]:
    """Return prefix for extended schemas, or None to fall back to legacy builders."""
    problem = trials[trial_index]["problem"]
    alias = str(problem.get("dataset_alias") or "")
    schema = str(problem.get("schema_type") or "")
    if schema == "bergert_pairwise" or alias == "bergert_nosofsky_2007":
        return build_bergert_prefix(trials, trial_index, instruction=instruction)
    if schema == "guan_stopping" or alias == "guan_2020_stopping":
        return build_guan_prefix(trials, trial_index, instruction=instruction)
    # Steyvers uses schema_type categorical_bandit with game/trial (not round).
    if alias == "steyvers_2009_bandit" or (
        schema == "categorical_bandit" and "game" in problem and "round" not in problem
    ):
        return build_steyvers_prefix(trials, trial_index, instruction=instruction)
    if schema == "categorical_bandit" or alias == "13schulz2020finding":
        return build_schulz_prefix(trials, trial_index, instruction=instruction)
    if schema == "kool_twostep" or alias == "14kool2016when":
        return build_kool_prefix(trials, trial_index, instruction=instruction)
    return None


def assert_prefix_no_leakage(
    prefix: str,
    *,
    forbidden_substrings: Sequence[str] = (),
) -> None:
    low = prefix.lower()
    for s in forbidden_substrings:
        if s.lower() in low:
            raise ValueError(f"Centaur prefix leakage: found {s!r}")
