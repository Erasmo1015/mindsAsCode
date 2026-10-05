"""Continue a PICS v4 run by replaying committed candidates before new generation.

The ordinary scorer and elite update code reconstruct state. Completed population
and participant stages use the existing native skip logic. No test score selects
parents. Incomplete candidate batches are regenerated with their original index.
"""
from pathlib import Path
import json
import math
import os
import threading


def install_saved_candidate_replay(teh, root):
    root = Path(root).resolve()
    stages = {}
    records = {}
    replayed = set()
    lock = threading.Lock()
    for participant in (root / 'selected').glob('participant_*'):
        if not participant.is_dir():
            continue
        for stage in [participant / 'explore_phase', *participant.glob('iteration_*')]:
            metrics_file = stage / 'metrics.json'
            if not metrics_file.is_file():
                continue
            metrics = json.loads(metrics_file.read_text())
            count = int(metrics.get('n_candidates', metrics.get('explore_candidates_requested', 0)))
            if count <= 0 or len(metrics['candidate_results']) != count:
                raise RuntimeError(f'Incomplete saved metrics: {metrics_file}')
            codes = [(stage / 'candidates' / f'candidate_{i}.py').read_text() for i in range(count)]
            stages[stage.resolve()] = (metrics, codes)
        trace = participant / 'mem_trace.jsonl'
        if trace.exists():
            records[trace.resolve()] = {key(row): row for row in
                                       map(json.loads, trace.read_text().splitlines())}

    original_variants = teh.generate_program_variants
    original_iteration = teh._generate_iteration_candidate_codes
    original_trace = teh.append_mem_trace_record

    def saved_stage(kwargs):
        stats = kwargs.get('prompt_stats_path')
        return Path(stats).parent.resolve() if stats is not None else None

    def note(stage):
        with lock:
            if stage not in replayed:
                replayed.add(stage)
                print(f'[resume] Replaying committed candidates, no LLM request: {stage}', flush=True)

    def variants(*args, **kwargs):
        stage = saved_stage(kwargs)
        if kwargs.get('phase') == 'explore' and stage in stages:
            metrics, codes = stages[stage]
            offset = int(kwargs.get('pics_lossless_candidate_offset', 0))
            count = int(kwargs['n_variants'])
            if offset + count > len(codes):
                raise RuntimeError(f'Changed exploration budget: {stage}')
            note(stage)
            return codes[offset:offset + count]
        if os.environ.get('PICS_RESUME_VERIFY_ONLY') == '1':
            raise ResumeVerified(f'Next unfinished generation stage: {stage}')
        return original_variants(*args, **kwargs)

    def iteration(**kwargs):
        stage = saved_stage(kwargs['variant_kwargs'])
        if stage in stages:
            metrics, codes = stages[stage]
            if len(codes) != int(kwargs['n_candidates']):
                raise RuntimeError(f'Changed candidate count: {stage}')
            sources = list(metrics['candidate_sources'])
            expected = ['fresh'] * int(kwargs['fresh_n_candidates'])
            expected += ['normal'] * (len(codes) - len(expected))
            if sources != expected:
                raise RuntimeError(f'Changed candidate schedule: {stage}')
            note(stage)
            return list(codes), sources
        if os.environ.get('PICS_RESUME_VERIFY_ONLY') == '1':
            raise ResumeVerified(f'Validated saved work through {stage.parent}; next unfinished stage {stage.name}')
        return original_iteration(**kwargs)

    def trace(path, row):
        old = records.get(Path(path).resolve(), {}).get(key(row)) if path is not None else None
        if old is not None:
            for field in ('code_sha256', 'candidate_id', 'reference_id', 'reference_type',
                          'runtime_valid', 'survived_elite_truncation', 'prompted_parent_ids',
                          'selected_parents', 'best_selected_parent_id', 'train_loglik',
                          'val_loglik', 'selection_score', 'delta_f'):
                if field in old or field in row:
                    a, b = old.get(field), row.get(field)
                    equal = equivalent(a, b)
                    if not equal:
                        raise RuntimeError(f'Resume reconstruction mismatch {path} {key(row)} {field}: {a!r} != {b!r}')
            return  # Saved trace already contains this committed event.
        return original_trace(path, row)

    teh.generate_program_variants = variants
    teh._generate_iteration_candidate_codes = iteration
    teh.append_mem_trace_record = trace
    print(f'[resume] Found {len(stages)} committed participant stages under {root}', flush=True)
    return stages


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a == b


def key(row):
    return (row.get('record_type'), row.get('phase'), row.get('iteration'), row.get('candidate_id'))


class ResumeVerified(RuntimeError):
    pass


if __name__ == '__main__':
    import sys
    import teh
    output = sys.argv[sys.argv.index('--output_dir') + 1]
    if '--pics_v4' not in sys.argv:
        raise RuntimeError('Saved candidate replay requires --pics_v4')
    install_saved_candidate_replay(teh, output)
    teh.main()
