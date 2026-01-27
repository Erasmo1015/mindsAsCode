#!/usr/bin/env python3
"""
Save Choice13k data to local files for easier inspection and debugging.
"""
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_modules.choice13k import get_choice13k_experiments

def save_choice13k_participant(participant_id: int, output_dir: str = "analysis/data/choice13k"):
    """
    Save a specific participant's Choice13k data to JSON files.
    
    Args:
        participant_id: Participant ID (0-indexed)
        output_dir: Directory to save the data
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load experiments
    print(f"Loading Choice13k data for participant {participant_id}...")
    experiments = get_choice13k_experiments(n_participants=participant_id + 1)
    
    if participant_id >= len(experiments):
        print(f"Error: Participant {participant_id} not found. Only {len(experiments)} participants available.")
        return
    
    experiment = experiments[participant_id]
    
    # Convert to JSON-serializable format
    def gamble_to_dict(gamble):
        return {
            "probs": gamble.probs,
            "rewards": gamble.rewards
        }
    
    def trial_to_dict(trial):
        return {
            "action": trial.action,
            "feedback": trial.feedback,
            "history": trial.history
        }
    
    def block_to_dict(block):
        return {
            "trials": [trial_to_dict(t) for t in block.trials],
            "gamble_A": gamble_to_dict(block.gamble_A),
            "gamble_B": gamble_to_dict(block.gamble_B),
            "has_feedback": block.has_feedback,
            "option_keys": block.option_keys,
            "gamble_info_text": block.gamble_info_text
        }
    
    experiment_dict = {
        "participant_id": participant_id,
        "instruction": experiment.instruction,
        "blocks": [block_to_dict(b) for b in experiment.blocks],
        "num_blocks": len(experiment.blocks),
        "total_trials": sum(len(b.trials) for b in experiment.blocks)
    }
    
    # Save full experiment data
    output_file = output_path / f"participant_{participant_id}_full.json"
    with open(output_file, 'w') as f:
        json.dump(experiment_dict, f, indent=2)
    print(f"Saved full experiment data to: {output_file}")
    
    # Save a human-readable summary
    summary_file = output_path / f"participant_{participant_id}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Choice13k Participant {participant_id} Data Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Instruction:\n{experiment.instruction}\n\n")
        f.write(f"Number of blocks: {len(experiment.blocks)}\n")
        f.write(f"Total trials: {sum(len(b.trials) for b in experiment.blocks)}\n\n")
        
        for block_idx, block in enumerate(experiment.blocks):
            f.write(f"\n{'='*80}\n")
            f.write(f"Block {block_idx + 1}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Has feedback: {block.has_feedback}\n")
            f.write(f"Option keys: {block.option_keys}\n\n")
            f.write(f"Gamble A: probs={block.gamble_A.probs}, rewards={block.gamble_A.rewards}\n")
            f.write(f"Gamble B: probs={block.gamble_B.probs}, rewards={block.gamble_B.rewards}\n\n")
            f.write(f"Gamble info text:\n{block.gamble_info_text}\n\n")
            f.write(f"Trials ({len(block.trials)}):\n")
            for trial_idx, trial in enumerate(block.trials):
                f.write(f"  Trial {trial_idx + 1}: action={trial.action}, feedback={trial.feedback}\n")
                f.write(f"    History: {trial.history[:100]}...\n" if len(trial.history) > 100 else f"    History: {trial.history}\n")
    
    print(f"Saved summary to: {summary_file}")
    
    # Save trials in the format used by Template_evo (for easy inspection)
    trials_file = output_path / f"participant_{participant_id}_trials.json"
    all_trials = []
    for block_idx, block in enumerate(experiment.blocks):
        for trial in block.trials:
            trial_data = {
                "block_idx": block_idx,
                "problem": {
                    "gamble_A": gamble_to_dict(block.gamble_A),
                    "gamble_B": gamble_to_dict(block.gamble_B),
                    "option_keys": block.option_keys,
                    "has_feedback": block.has_feedback
                },
                "action": trial.action,
                "feedback": trial.feedback,
                "history": trial.history
            }
            all_trials.append(trial_data)
    
    with open(trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"Saved trials data to: {trials_file}")
    
    print(f"\nTotal trials: {len(all_trials)}")
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Save Choice13k participant data to local files")
    parser.add_argument(
        "--participant_id",
        type=int,
        default=0,
        help="Participant ID to save (0-indexed)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/data/choice13k",
        help="Output directory for saved data"
    )
    args = parser.parse_args()
    
    save_choice13k_participant(args.participant_id, args.output_dir)

