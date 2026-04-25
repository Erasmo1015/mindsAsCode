"""
CPC18 Track II (Individual Behavior) dataset loader.

Track II focuses on individual-level modeling:
- All predictions conditioned on a single participant
- Problems used in testing are familiar (already observed during training)
- Task is prediction/completion, not generalization to new problems
- Each participant has 25 trials per problem (5 blocks × 5 trials)

Data sources:
- Training: datasets/cpc18/raw-comp-set-data-Track-2.csv (trial-level with real actions/feedback)
- Testing: datasets/cpc18/Data-to-predict-Track-2.csv (block-level B-choice rates for MSE)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, NamedTuple, Set
from pathlib import Path


class Problem(NamedTuple):
    """CPC18 problem definition."""
    problem_id: int
    Ha: float
    pHa: float
    La: float
    LotShapeA: str
    LotNumA: int
    Hb: float
    pHb: float
    Lb: float
    LotShapeB: str
    LotNumB: int
    Amb: int  # 1 if ambiguous, 0 otherwise
    Corr: int  # -1, 0, or 1


class Trial(NamedTuple):
    """Single trial within a problem."""
    action: int  # 0 = A (L), 1 = B (R)
    feedback: float  # Actual payoff received (None if no feedback)
    block_id: int  # Which block (1-5) this trial belongs to
    trial_num: int  # Trial number within problem (1-25)


class ParticipantData(NamedTuple):
    """Data for a single participant."""
    participant_id: int
    problems: List[Problem]
    trials: Dict[int, List[Trial]]  # problem_id -> list of trials (ordered)
    test_targets: Dict[int, np.ndarray]  # problem_id -> observed B-rates for 5 blocks (for MSE)


def load_cpc18_track2_data(data_path: str = "datasets/cpc18", participant_id: int = 0) -> ParticipantData:
    """
    Load CPC18 Track II data for a specific participant.
    
    Uses actual Track II files:
    - raw-comp-set-data-Track-2.csv: Trial-level training data with real actions/feedback
    - Data-to-predict-Track-2.csv: Block-level test targets for MSE computation
    
    Args:
        data_path: Path to directory containing CPC18 Track II data files
        participant_id: Participant index (0-indexed, maps to SubjID in CSV)
    
    Returns:
        ParticipantData containing problems, trials, and test targets
    """
    data_path = Path(data_path)
    
    # Load training data (trial-level)
    raw_data_file = data_path / "raw-comp-set-data-Track-2.csv"
    if not raw_data_file.exists():
        raise FileNotFoundError(f"Could not find Track II raw data file: {raw_data_file}")
    
    # Load test targets (block-level)
    test_targets_file = data_path / "Data-to-predict-Track-2.csv"
    if not test_targets_file.exists():
        raise FileNotFoundError(f"Could not find Track II test targets file: {test_targets_file}")
    
    # Read training data
    df_raw = pd.read_csv(raw_data_file)
    
    # Read test targets
    df_targets = pd.read_csv(test_targets_file)
    
    # Get unique SubjIDs and map participant_id to SubjID
    unique_subj_ids = sorted(df_raw['SubjID'].unique())
    if participant_id >= len(unique_subj_ids):
        raise ValueError(f"participant_id {participant_id} is out of range. Available: 0-{len(unique_subj_ids)-1}")
    
    subj_id = unique_subj_ids[participant_id]
    
    # Filter data for this participant
    df_participant = df_raw[df_raw['SubjID'] == subj_id].copy()
    df_targets_participant = df_targets[df_targets['SubjID'] == subj_id].copy()
    
    if len(df_participant) == 0:
        raise ValueError(f"No data found for participant_id {participant_id} (SubjID {subj_id})")
    
    # Extract unique problems for this participant
    unique_game_ids = sorted(df_participant['GameID'].unique())
    
    problems = []
    trials_dict = {}
    test_targets_dict = {}
    
    # Process each problem
    for game_id in unique_game_ids:
        df_problem = df_participant[df_participant['GameID'] == game_id].copy()
        
        # Sort by Trial to ensure correct order
        df_problem = df_problem.sort_values('Trial').reset_index(drop=True)
        
        # Get problem parameters from first row (same for all trials)
        first_row = df_problem.iloc[0]
        
        problem = Problem(
            problem_id=int(game_id),
            Ha=float(first_row['Ha']),
            pHa=float(first_row['pHa']),
            La=float(first_row['La']),
            LotShapeA=str(first_row['LotShapeA']),
            LotNumA=int(first_row['LotNumA']),
            Hb=float(first_row['Hb']),
            pHb=float(first_row['pHb']),
            Lb=float(first_row['Lb']),
            LotShapeB=str(first_row['LotShapeB']),
            LotNumB=int(first_row['LotNumB']),
            Amb=int(first_row['Amb']),
            Corr=int(first_row['Corr']),
        )
        problems.append(problem)
        
        # Extract real trials (ordered by Trial number)
        trials = []
        for _, row in df_problem.iterrows():
            # Button: "L" = 0 (A), "R" = 1 (B)
            button = str(row['Button']).upper()
            action = 1 if button == 'R' else 0
            
            # Feedback: Payoff if Feedback==1, None if Feedback==0
            feedback_val = float(row['Feedback'])
            if feedback_val == 1:
                feedback = float(row['Payoff'])
            else:
                feedback = None
            
            block_id = int(row['block'])
            trial_num = int(row['Trial'])
            
            trials.append(Trial(
                action=action,
                feedback=feedback,
                block_id=block_id,
                trial_num=trial_num,
            ))
        
        trials_dict[problem.problem_id] = trials
        
        # Extract test targets for this problem (if available)
        df_target_problem = df_targets_participant[df_targets_participant['GameID'] == game_id]
        if len(df_target_problem) > 0:
            target_row = df_target_problem.iloc[0]
            test_targets_dict[problem.problem_id] = np.array([
                float(target_row['B.1']),
                float(target_row['B.2']),
                float(target_row['B.3']),
                float(target_row['B.4']),
                float(target_row['B.5']),
            ])
    
    return ParticipantData(
        participant_id=participant_id,
        problems=problems,
        trials=trials_dict,
        test_targets=test_targets_dict,
    )


def _problem_to_dict(problem: Problem) -> Dict[str, Any]:
    return {
        "Ha": problem.Ha,
        "pHa": problem.pHa,
        "La": problem.La,
        "LotShapeA": problem.LotShapeA,
        "LotNumA": problem.LotNumA,
        "Hb": problem.Hb,
        "pHb": problem.pHb,
        "Lb": problem.Lb,
        "LotShapeB": problem.LotShapeB,
        "LotNumB": problem.LotNumB,
        "Amb": problem.Amb,
        "Corr": problem.Corr,
    }


def _indexed_trials_for_problems(
    participant_data: ParticipantData, problem_ids: Set[int]
) -> List[Dict[str, Any]]:
    """Chronological trial records for given problems, with per-problem history. Order: problems in participant_data order."""
    out: List[Dict[str, Any]] = []
    for problem in participant_data.problems:
        if problem.problem_id not in problem_ids:
            continue
        problem_trials = participant_data.trials[problem.problem_id]
        problem_dict = _problem_to_dict(problem)
        history_accum: List[Dict[str, Any]] = []
        for trial in problem_trials:
            out.append(
                {
                    "problem": problem_dict.copy(),
                    "history": list(history_accum),
                    "action": trial.action,
                    "problem_id": problem.problem_id,
                    "block_id": trial.block_id,
                }
            )
            history_accum.append({"action": trial.action, "feedback": trial.feedback})
    return out


def _first_k_trials_in_problem(
    participant_data: ParticipantData, problem: Problem, k: int
) -> List[Dict[str, Any]]:
    """First k trials of a single problem in order (k <= len)."""
    problem_trials = participant_data.trials[problem.problem_id]
    k = min(k, len(problem_trials))
    problem_dict = _problem_to_dict(problem)
    out: List[Dict[str, Any]] = []
    history_accum: List[Dict[str, Any]] = []
    for j in range(k):
        trial = problem_trials[j]
        out.append(
            {
                "problem": problem_dict.copy(),
                "history": list(history_accum),
                "action": trial.action,
                "problem_id": problem.problem_id,
                "block_id": trial.block_id,
            }
        )
        history_accum.append({"action": trial.action, "feedback": trial.feedback})
    return out


def _tail_trials_in_problem(
    participant_data: ParticipantData, problem: Problem, start_index: int
) -> List[Dict[str, Any]]:
    """Trials from start_index to end, history includes all previous trials in the full problem (same as if unsplit)."""
    problem_trials = participant_data.trials[problem.problem_id]
    n = len(problem_trials)
    if start_index >= n or start_index < 0:
        return []
    problem_dict = _problem_to_dict(problem)
    out: List[Dict[str, Any]] = []
    # History from trials before start_index
    history_accum: List[Dict[str, Any]] = []
    for j in range(start_index):
        t0 = problem_trials[j]
        history_accum.append({"action": t0.action, "feedback": t0.feedback})
    for j in range(start_index, n):
        trial = problem_trials[j]
        out.append(
            {
                "problem": problem_dict.copy(),
                "history": list(history_accum),
                "action": trial.action,
                "problem_id": problem.problem_id,
                "block_id": trial.block_id,
            }
        )
        history_accum.append({"action": trial.action, "feedback": trial.feedback})
    return out


def _split_cpc18_holdout(
    participant_data: ParticipantData,
    split_ratio: float,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Per-participant train/test: assign whole problems to train or test (random, reproducible);
    if only one problem, split by trial index within the problem. Disjoint trial sets; history is valid
    (test trials see true prior outcomes in that problem, including the train part of the same problem when applicable).
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0,1), got {split_ratio!r}")
    n_p = len(participant_data.problems)
    if n_p == 0:
        return [], []
    rng = np.random.default_rng(int(split_seed))

    if n_p >= 2:
        ids = [p.problem_id for p in participant_data.problems]
        sh = list(ids)
        rng.shuffle(sh)
        n_tr = int(round(n_p * float(split_ratio)))
        n_tr = max(1, n_tr)
        n_tr = min(n_tr, n_p - 1)
        train_ids: Set[int] = set(sh[:n_tr])
        test_ids: Set[int] = set(sh[n_tr:])
    else:
        # Single problem: chronological trial split
        prob0 = participant_data.problems[0]
        tlist = participant_data.trials[prob0.problem_id]
        n_t = len(tlist)
        if n_t < 2:
            return _indexed_trials_for_problems(participant_data, {prob0.problem_id}), []
        n_tr2 = int(round(n_t * float(split_ratio)))
        n_tr2 = max(1, n_tr2)
        n_tr2 = min(n_tr2, n_t - 1)
        train = _first_k_trials_in_problem(participant_data, prob0, n_tr2)
        test = _tail_trials_in_problem(participant_data, prob0, n_tr2)
        return train, test

    train_trials = _indexed_trials_for_problems(participant_data, train_ids)
    test_trials = _indexed_trials_for_problems(participant_data, test_ids)
    return train_trials, test_trials


def split_cpc18_trials(
    participant_data: ParticipantData,
    train_ratio: float = 0.8,
    *,
    cpc18_official_mse: bool = False,
    split_ratio: float = 0.9,
    split_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, np.ndarray]]:
    """
    Prepare CPC18 Track II trials.

    cpc18_official_mse=True: official protocol — all trial-level data for both train and test
    (duplicate lists with the same per-trial semantics as before), plus block MSE against
    Data-to-predict-Track-2.csv. train_ratio is ignored (kept for API compatibility).

    cpc18_official_mse=False (default): per-participant train/test by random partition of
    **problems** (or trial split if only one problem), ratio split_ratio and split_seed, aligned
    with other TE datasets. Third return is an empty dict (no official block targets for the split
    line); use log-likelihood or accuracy on held-out trials.

    Returns:
        (train_trials, test_trials, test_observed_blocks) — test_observed_blocks is empty
        when cpc18_official_mse is False.
    """
    if cpc18_official_mse:
        train_trials: List[Dict[str, Any]] = []
        test_trials: List[Dict[str, Any]] = []

        for problem in participant_data.problems:
            problem_trials = participant_data.trials[problem.problem_id]
            problem_dict = _problem_to_dict(problem)
            history_accum: List[Dict[str, Any]] = []
            for trial in problem_trials:
                train_trials.append(
                    {
                        "problem": problem_dict.copy(),
                        "history": list(history_accum),
                        "action": trial.action,
                        "problem_id": problem.problem_id,
                        "block_id": trial.block_id,
                    }
                )
                history_accum.append(
                    {
                        "action": trial.action,
                        "feedback": trial.feedback,
                    }
                )

            test_history: List[Dict[str, Any]] = []
            for trial in problem_trials:
                test_trials.append(
                    {
                        "problem": problem_dict.copy(),
                        "history": list(test_history),
                        "action": trial.action,
                        "problem_id": problem.problem_id,
                        "block_id": trial.block_id,
                    }
                )
                test_history.append(
                    {
                        "action": trial.action,
                        "feedback": trial.feedback,
                    }
                )

        test_observed_blocks = participant_data.test_targets.copy()
        return train_trials, test_trials, test_observed_blocks

    t_tr, t_te = _split_cpc18_holdout(participant_data, split_ratio=split_ratio, split_seed=split_seed)
    return t_tr, t_te, {}
