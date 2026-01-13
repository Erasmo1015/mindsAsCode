"""
Utility script to collect best programs from ROTE output and organize them for Template_evo.

This script:
1. Reads ROTE output JSON files (epoch_X_agent_types.json)
2. Extracts the best program for each agent type
3. Organizes programs by problem config (num_blocks, num_walls)
4. Saves them in a structure that Template_evo can use as seed programs

Usage:
    python utils/collect_template_program.py --exp_folder generated_outputs/gridworld/run_260107_145709
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_hand_designed_mapping() -> Dict[int, str]:
    """
    Get mapping from agent_id to hand-designed program name.
    
    Returns:
        Dictionary mapping agent_id to program filename (without .txt extension)
    """
    hand_designed_dir = Path("generated_outputs/hand_designed")
    if not hand_designed_dir.exists():
        print(f"Warning: {hand_designed_dir} does not exist. Cannot create agent_id mapping.")
        return {}
    
    files = sorted([f for f in os.listdir(hand_designed_dir) if f.endswith('.txt')])
    mapping = {i: f.replace('.txt', '') for i, f in enumerate(files)}
    return mapping


def extract_problem_config_from_results(results_csv_path: Path) -> Optional[Tuple[int, int]]:
    """
    Extract num_blocks and num_walls from results.csv file.
    
    Args:
        results_csv_path: Path to results.csv file
        
    Returns:
        Tuple of (num_blocks, num_walls) or None if not found
    """
    if not results_csv_path.exists():
        return None
    
    try:
        with open(results_csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None
            # Read first data line (skip header)
            parts = lines[1].strip().split(',')
            # Find num_blocks and num_walls columns
            header = lines[0].strip().split(',')
            try:
                num_blocks_idx = header.index('num_blocks')
                num_walls_idx = header.index('num_walls')
                num_blocks = int(parts[num_blocks_idx])
                num_walls = int(parts[num_walls_idx])
                return (num_blocks, num_walls)
            except (ValueError, IndexError):
                return None
    except Exception as e:
        print(f"Error reading results.csv: {e}")
        return None


def collect_best_programs_from_epoch(
    exp_folder: Path,
    epoch: int = 0,
    output_base_dir: Path = Path("persona_code_example/gridworld")
) -> Dict[int, Dict]:
    """
    Collect best programs from a specific epoch.
    
    Args:
        exp_folder: Path to ROTE experiment folder (e.g., generated_outputs/gridworld/run_XXX)
        epoch: Epoch number (default: 0)
        output_base_dir: Base directory for output programs
        
    Returns:
        Dictionary mapping agent_id to program info:
        {
            agent_id: {
                'program_path': Path to source program,
                'weight': float,
                'hypothesis_id': int,
                'output_path': Path to saved program
            }
        }
    """
    epoch_dir = exp_folder / f"epoch_{epoch}"
    json_file = epoch_dir / f"epoch_{epoch}_agent_types.json"
    
    if not json_file.exists():
        print(f"Error: {json_file} does not exist!")
        return {}
    
    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract problem config from results.csv
    results_csv = exp_folder / "results.csv"
    problem_config = extract_problem_config_from_results(results_csv)
    if problem_config is None:
        print(f"Warning: Could not extract problem config from {results_csv}")
        print("Using default: num_blocks=3, num_walls=1")
        problem_config = (3, 1)
    
    num_blocks, num_walls = problem_config
    print(f"Detected problem config: num_blocks={num_blocks}, num_walls={num_walls}")
    
    # Get agent_id to hand-designed program mapping
    agent_mapping = get_hand_designed_mapping()
    
    # Create output directory structure
    problem_dir = output_base_dir / f"num_blocks{num_blocks}_num_walls{num_walls}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    
    collected_programs = {}
    agent_types = data.get('agent_types', {})
    
    print(f"\nCollecting best programs from epoch {epoch}...")
    print(f"Found {len(agent_types)} agent types")
    
    for agent_key, agent_data in agent_types.items():
        agent_id = agent_data.get('agent_id')
        best_program = agent_data.get('best_program', {})
        
        if not best_program:
            print(f"Warning: No best_program found for {agent_key}")
            continue
        
        # Get program path (relative to epoch_dir)
        program_path_rel = best_program.get('program_path', '')
        if not program_path_rel:
            print(f"Warning: No program_path in best_program for {agent_key}")
            continue
        
        # Remove epoch_X/ prefix if present (program_path might be like "epoch_0/hyp_4/good/program.py")
        if program_path_rel.startswith(f"epoch_{epoch}/"):
            program_path_rel = program_path_rel[len(f"epoch_{epoch}/"):]
        
        # Resolve full path
        source_program_path = epoch_dir / program_path_rel
        
        if not source_program_path.exists():
            print(f"Warning: Program file does not exist: {source_program_path}")
            continue
        
        # Create output filename
        # Option 1: Use agent_id
        output_filename = f"agent_{agent_id}.py"
        # Option 2: Use hand-designed program name if available
        if agent_id in agent_mapping:
            hand_designed_name = agent_mapping[agent_id]
            output_filename = f"{hand_designed_name}_agent{agent_id}.py"
        
        output_program_path = problem_dir / output_filename
        
        # Copy program file
        try:
            shutil.copy2(source_program_path, output_program_path)
            print(f"  Agent {agent_id}: {source_program_path.name} -> {output_filename} (weight: {best_program.get('weight', 0):.4f})")
            
            collected_programs[agent_id] = {
                'program_path': source_program_path,
                'weight': best_program.get('weight', 0.0),
                'hypothesis_id': best_program.get('hypothesis_id', -1),
                'output_path': output_program_path,
                'hand_designed_name': agent_mapping.get(agent_id, 'unknown')
            }
        except Exception as e:
            print(f"Error copying program for agent {agent_id}: {e}")
    
    print(f"\n✓ Collected {len(collected_programs)} programs to {problem_dir}")
    return collected_programs


def main():
    parser = argparse.ArgumentParser(
        description="Collect best programs from ROTE output for Template_evo"
    )
    parser.add_argument(
        "--exp_folder",
        type=str,
        required=True,
        help="Path to ROTE experiment folder (e.g., generated_outputs/gridworld/run_260107_145709)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch number to collect from (default: 0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="persona_code_example/gridworld",
        help="Base output directory for collected programs (default: persona_code_example/gridworld)"
    )
    parser.add_argument(
        "--all_epochs",
        action="store_true",
        help="Collect from all epochs found in the experiment folder"
    )
    
    args = parser.parse_args()
    
    exp_folder = Path(args.exp_folder)
    if not exp_folder.exists():
        print(f"Error: Experiment folder does not exist: {exp_folder}")
        return
    
    output_base_dir = Path(args.output_dir)
    
    if args.all_epochs:
        # Find all epochs
        epochs = []
        for item in exp_folder.iterdir():
            if item.is_dir() and item.name.startswith('epoch_'):
                try:
                    epoch_num = int(item.name.split('_')[1])
                    epochs.append(epoch_num)
                except ValueError:
                    continue
        
        epochs = sorted(epochs)
        print(f"Found {len(epochs)} epochs: {epochs}")
        
        for epoch in epochs:
            print(f"\n{'='*80}")
            print(f"Processing epoch {epoch}")
            print(f"{'='*80}")
            collect_best_programs_from_epoch(exp_folder, epoch, output_base_dir)
    else:
        # Collect from single epoch
        collect_best_programs_from_epoch(exp_folder, args.epoch, output_base_dir)
    
    print(f"\n{'='*80}")
    print("Collection complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

