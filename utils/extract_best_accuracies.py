#!/usr/bin/env python3
"""
Extract best train and test accuracies from old ROTE_evo experiment results.

This script scans through participant directories and finds the best train_acc
and best test_acc across all iterations for each participant.
"""

import json
import csv
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional


def find_best_accuracies(participant_dir: Path) -> Tuple[float, float, Optional[str], Optional[str]]:
    """
    Find best train and test accuracies for a participant.
    
    Returns:
        (best_train_acc, best_test_acc, best_train_program_id, best_test_program_id)
    """
    best_train_acc = 0.0
    best_test_acc = 0.0
    best_train_program_id = None
    best_test_program_id = None
    
    # Check baseline first
    baseline_file = participant_dir / "baseline_results.json"
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
            baseline_train = baseline.get("train_accuracy", 0.0)
            baseline_test = baseline.get("test_accuracy", 0.0)
            if baseline_train > best_train_acc:
                best_train_acc = baseline_train
                best_train_program_id = "baseline"
            if baseline_test > best_test_acc:
                best_test_acc = baseline_test
                best_test_program_id = "baseline"
    
    # Check all iterations
    iteration_dirs = sorted(participant_dir.glob("iteration_*"))
    for iter_dir in iteration_dirs:
        metrics_file = iter_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        iteration = metrics.get("iteration", -1)
        
        # Check best_train_acc and best_test_acc from iteration summary
        iter_best_train = metrics.get("best_train_acc")
        iter_best_test = metrics.get("best_test_acc")
        
        if iter_best_train is not None and iter_best_train > best_train_acc:
            best_train_acc = iter_best_train
            best_train_program_id = f"iteration_{iteration}_best"
        
        if iter_best_test is not None and iter_best_test > best_test_acc:
            best_test_acc = iter_best_test
            best_test_program_id = f"iteration_{iteration}_best"
        
        # Also check all candidates in case best wasn't selected as parent
        candidate_results = metrics.get("candidate_results", [])
        for candidate in candidate_results:
            if not candidate.get("valid", False):
                continue
            train_acc = candidate.get("train_acc", 0.0)
            test_acc = candidate.get("test_acc", 0.0)
            candidate_idx = candidate.get("idx", -1)
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_program_id = f"iteration_{iteration}_candidate_{candidate_idx}"
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_program_id = f"iteration_{iteration}_candidate_{candidate_idx}"
    
    return best_train_acc, best_test_acc, best_train_program_id, best_test_program_id


def extract_all_participants(run_dir: Path) -> List[Dict]:
    """Extract best accuracies for all participants in a run."""
    results = []
    
    participant_dirs = sorted(run_dir.glob("participant_*"))
    print(f"Found {len(participant_dirs)} participant directories")
    
    for part_dir in participant_dirs:
        # Extract participant ID from directory name
        try:
            participant_id = int(part_dir.name.split("_")[1])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse participant ID from {part_dir.name}")
            continue
        
        print(f"Processing participant {participant_id}...")
        best_train, best_test, train_prog_id, test_prog_id = find_best_accuracies(part_dir)
        
        results.append({
            "participant_id": participant_id,
            "train_acc": best_train,
            "test_acc": best_test,
            "best_train_program_id": train_prog_id,
            "best_test_program_id": test_prog_id,
        })
    
    return sorted(results, key=lambda x: x["participant_id"])


def main():
    parser = argparse.ArgumentParser(description="Extract best accuracies from old ROTE_evo experiments")
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the run directory (e.g., generated_outputs/choice13k_ROTE_evo_non_strict/run_251230_225001)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: run_dir/participants_summary.csv)",
    )
    parser.add_argument(
        "--include_program_ids",
        action="store_true",
        help="Include program_id columns in output (default: False, only participant_id, train_acc, test_acc)",
    )
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist")
        return
    
    print(f"Analyzing run directory: {run_dir}")
    results = extract_all_participants(run_dir)
    
    if not results:
        print("No participants found!")
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "participants_summary.csv"
    
    # Write CSV
    fieldnames = ["participant_id", "train_acc", "test_acc"]
    if args.include_program_ids:
        fieldnames.extend(["best_train_program_id", "best_test_program_id"])
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "participant_id": result["participant_id"],
                "train_acc": result["train_acc"],
                "test_acc": result["test_acc"],
            }
            if args.include_program_ids:
                row["best_train_program_id"] = result["best_train_program_id"]
                row["best_test_program_id"] = result["best_test_program_id"]
            writer.writerow(row)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Processed {len(results)} participants")
    print(f"\nSummary statistics:")
    train_accs = [r["train_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    print(f"  Train acc: mean={sum(train_accs)/len(train_accs):.4f}, max={max(train_accs):.4f}, min={min(train_accs):.4f}")
    print(f"  Test acc:  mean={sum(test_accs)/len(test_accs):.4f}, max={max(test_accs):.4f}, min={min(test_accs):.4f}")


if __name__ == "__main__":
    main()

