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
from typing import List, Dict, Any, Tuple, NamedTuple
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


def split_cpc18_trials(participant_data: ParticipantData, train_ratio: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, np.ndarray]]:
    """
    Prepare CPC18 Track II trials for training and evaluation.
    
    IMPORTANT: CPC18 Track II does NOT use a train/test split.
    - ALL trials from raw-comp-set-data-Track-2.csv are used for training (parameter evolution)
    - ALL trials are also used for generating predictions (aggregated to block-level for MSE)
    - Block-level B-choice rates from Data-to-predict-Track-2.csv are used ONLY for MSE evaluation
    
    This matches the official CPC18 Track II protocol:
    - No artificial 80:20 trial split
    - Training uses all trial-level data to build histories
    - Evaluation computes block-level MSE against observed rates
    
    Args:
        participant_data: ParticipantData object
        train_ratio: IGNORED for CPC18 (kept for API compatibility). All trials are used.
    
    Returns:
        Tuple of (train_trials, test_trials, test_observed_blocks)
        - train_trials: ALL trials (used for parameter evolution and auxiliary accuracy)
        - test_trials: ALL trials (used for generating predictions for block-level MSE)
        - test_observed_blocks: Dict mapping problem_id to observed B-rates (5 blocks) from Data-to-predict-Track-2.csv
    """
    train_trials = []
    test_trials = []
    
    for problem in participant_data.problems:
        problem_trials = participant_data.trials[problem.problem_id]
        
        # Convert problem to dict format compatible with template program
        problem_dict = {
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
        
        # CPC18 Track II: Use ALL trials (no split)
        # Build history sequentially (as in Choice13k, accumulating actions and feedback)
        history_accum = []
        
        # Process ALL trials for training (used for parameter evolution)
        for trial in problem_trials:
            train_trials.append({
                "problem": problem_dict.copy(),
                "history": list(history_accum),  # History up to (but not including) this trial
                "action": trial.action,
                "problem_id": problem.problem_id,
                "block_id": trial.block_id,
            })
            # Add this trial to history for next trial
            history_accum.append({
                "action": trial.action,
                "feedback": trial.feedback,
            })
        
        # Process ALL trials for test (used for generating predictions for block-level MSE)
        # Note: test_trials are NOT held-out - they're the same trials, used for predictions
        test_history = []
        for trial in problem_trials:
            test_trials.append({
                "problem": problem_dict.copy(),
                "history": list(test_history),  # History up to (but not including) this trial
                "action": trial.action,
                "problem_id": problem.problem_id,
                "block_id": trial.block_id,
            })
            # Add this trial to history for next trial
            test_history.append({
                "action": trial.action,
                "feedback": trial.feedback,
            })
    
    # Get test targets for MSE computation (from Data-to-predict-Track-2.csv)
    # These are block-level B-choice rates, NOT trial-level data
    test_observed_blocks = participant_data.test_targets.copy()
    
    return train_trials, test_trials, test_observed_blocks
