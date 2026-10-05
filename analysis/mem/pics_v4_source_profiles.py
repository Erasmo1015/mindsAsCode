"""Frozen uniform-v8 post hoc population construct profiles; CPU only.

No fitness/test score enters any profile, resampling, or source selection.
Historical annotations are never loaded; explicit official shard names only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.mem.pics_v4_population_annotation import AUTHORITATIVE_JOBS, OUTPUT_REL, inventory_populations
from analysis.mem.pop_v4_source_selection.run_source_selection_v4 import cosine, fit_occurrence_eb
from utils.mem.population_transition_finalize_v5 import annotation_row_valid, finalize_annotation_jsonl, finalize_population_row
from utils.mem.trace import record_contains_test_metrics
from utils.mem.pics_v4_frozen import check_frozen

CONSTRUCTS = ('history', 'value', 'probability_used', 'feedback', 'learning')
SHARDS = (
    'pics_v4_popann_pack1', 'pics_v4_popann_enkavi_311666_20261005',
    'pics_v4_popann_pack2_312023_20261004', 'pics_v4_popann_pack3',
    'pics_v4_popann_pack4', 'pics_v4_popann_pack5_312024_20261004',
    'pics_v4_popann_badham_312024_20261005', 'pics_v4_popann_kool_retry_311935_20261004',
    'pics_v4_popann_steyvers_311671_20261005', 'pics_v4_popann_pack7',
    'pics_v4_popann_peterson_311672_20261004', 'pics_v4_popann_pack8',
    'pics_v4_popann_schulz_311673_20261004',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).is_file() else []


def fit_profiles(panel):
    """Use the unchanged historical equations with an explicit v5 vocabulary."""
    if panel.empty or any(c not in panel for c in CONSTRUCTS):
        raise ValueError('Empty/incomplete occurrence panel')
    if not panel[list(CONSTRUCTS)].isin([0, 1]).all().all():
        raise ValueError('Construct presence must be binary')
    if any(record_contains_test_metrics({'selection_score': None, **r}) for r in panel.to_dict('records')):
        raise ValueError('Passive test fields refused')
    return fit_occurrence_eb(panel, motifs=CONSTRUCTS)


def select_sources(profiles):
    datasets = sorted(profiles.dataset)
    vectors = {r.dataset: np.asarray([getattr(r, c) for c in CONSTRUCTS]) for r in profiles.itertuples()}
    matrix = pd.DataFrame([[cosine(vectors[a], vectors[b]) for b in datasets] for a in datasets], index=datasets, columns=datasets)
    rows = []
    for target in datasets:
        ranks = sorted([(float(matrix.loc[target, s]), s) for s in datasets if s != target], key=lambda x: (-x[0], x[1]))
        if len(ranks) < 2:
            raise ValueError('At least three source profiles required')
        rows.append(dict(target_id=target, selected_source_id=ranks[0][1], cosine=ranks[0][0], runner_up_source_id=ranks[1][1], runner_up_cosine=ranks[1][0], cosine_margin=ranks[0][0]-ranks[1][0]))
    return matrix, pd.DataFrame(rows)


def verify_manifest(repo):
    live = inventory_populations(repo)
    if live['n_ready_datasets'] != 15 or len(live['programs']) != 1500:
        raise ValueError(f'Official population integrity failed: {live["datasets"]}')
    stored = json.loads((repo/OUTPUT_REL/'program_manifest.json').read_text())
    def index(programs):
        keyed = {p['resume_key']: p for p in programs}
        if len(keyed) != 1500 or len(programs) != 1500:
            raise ValueError('Missing/duplicate canonical population identities')
        return keyed
    if index(stored['programs']) != index(live['programs']):
        raise ValueError('Manifest differs from SHA-verified official live inventory')
    if Counter(p['dataset'] for p in live['programs']) != Counter({d: 100 for _, names in AUTHORITATIVE_JOBS for d in names}):
        raise ValueError('Not exactly 100 events per dataset')
    return live


def load_join(repo, manifest):
    expected = {p['resume_key']: p for p in manifest['programs']}
    annotations, failures, paths = {}, {}, []
    for shard in SHARDS:
        root = repo/OUTPUT_REL/'annotations'/shard
        path = root/'annotations_population_v5.jsonl'
        if not path.is_file():
            raise ValueError(f'Missing official annotation shard: {path}')
        paths.append(path)
        for r in jsonl(path):
            key = r.get('resume_key')
            if key not in expected or key in annotations:
                raise ValueError(f'Unmatched/duplicate annotation: {key}')
            p = expected[key]
            for field in ('dataset', 'run_id', 'iteration', 'code_sha256', 'parent_code_sha256', 'code_path', 'parent_code_path', 'parent_id', 'source', 'reference_kind', 'reference_type'):
                if r.get(field) != p.get(field):
                    raise ValueError(f'Annotation provenance mismatch: {key}: {field}')
            if r.get('candidate_id') != p['candidate_id'] or r.get('schema_version') != 5:
                raise ValueError(f'Candidate/schema mismatch: {key}')
            if record_contains_test_metrics(r):
                raise ValueError(f'Passive test fields refused: {key}')
            ok, error = annotation_row_valid(r)
            if not ok:
                raise ValueError(f'Invalid stored annotation {key}: {error}')
            annotations[key] = r
        for r in jsonl(root/'annotation_failures.jsonl'):
            key = r.get('resume_key')
            if key not in expected or key in failures:
                raise ValueError(f'Unmatched/duplicate failure identity: {key}')
            failures[key] = r
    missing = set(expected)-set(annotations)-set(failures)
    if missing:
        raise ValueError(f'Missing joins without a recorded failure: {sorted(missing)}')
    return annotations, failures, paths


def build_master(manifest, annotations, failures):
    rows = []
    for p in manifest['programs']:
        key = p['resume_key']; a = annotations.get(key)
        row = {k: p[k] for k in ('dataset', 'run_id', 'source_job_id', 'iteration', 'program_id', 'parent_id', 'resume_key', 'source', 'code_sha256', 'parent_code_sha256', 'code_path', 'parent_code_path', 'runtime_valid')}
        unresolved = bool(a and (a.get('exclude_from_construct_effect_fitting') or a.get('nmc_adjudication_status') == 'unresolved' or a.get('semantic_resolution_status') == 'nmc_needs_adjudication' or 'unresolved' in (a.get('transition_by_construct') or {}).values()))
        reason = 'annotation_failure' if a is None else 'unresolved_semantic' if unresolved else ''
        row.update(annotation_joined=a is not None, exclusion_reason=reason, primary_eligible=not reason and p['runtime_valid'] is True, sensitivity_eligible=not reason)
        state = (a or {}).get('candidate_motif_state', [])
        if any(c not in CONSTRUCTS for c in state):
            raise ValueError(f'Unknown constructs: {key}')
        for c in CONSTRUCTS:
            row[c] = int(c in state) if a is not None else None
        row['raw_llm_annotation'] = json.dumps((a or {}).get('raw_llm_annotation'))
        row['semantic_corrections'] = json.dumps((a or {}).get('semantic_corrections', []))
        rows.append(row)
    return pd.DataFrame(rows)


def boot_panel(panel, seed):
    """Exact schema-v5 audited within-dataset (iteration,parent_id) cluster draw."""
    rng = np.random.default_rng(seed)
    pieces = []
    for dataset, g in panel.groupby('dataset', sort=True):
        keys = g.apply(lambda r: f'{dataset}|{int(r.iteration)}|{r.parent_id}', axis=1)
        clusters = keys.unique()
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        pieces.append(pd.concat([g[keys == c] for c in chosen], ignore_index=True))
    return pd.concat(pieces, ignore_index=True)


def _bootstrap_one(args):
    records, b = args
    panel = boot_panel(pd.DataFrame(records), 20260920+b*9973)
    profiles, _, _, _ = fit_profiles(panel)
    _, selected = select_sources(profiles)
    return selected.set_index('target_id').selected_source_id.to_dict()


def stability(panel, selected, output, workers):
    counts = {d: Counter() for d in selected.target_id}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for b, mapping in enumerate(pool.map(_bootstrap_one, ((panel.to_dict('records'), b) for b in range(1000)), chunksize=10), 1):
            for target, source in mapping.items():
                counts[target][source] += 1
            if b % 100 == 0:
                print(f'{output.name}: bootstrap {b}/1000', flush=True)
    # Historical LOIO means leave one ITERATION out; no dataset is held out.
    details = []
    original = selected.set_index('target_id').selected_source_id.to_dict()
    for it in range(1, 11):
        profiles, _, _, _ = fit_profiles(panel[panel.iteration != it])
        _, s = select_sources(profiles)
        for r in s.to_dict('records'):
            details.append(dict(held_out_iteration=it, flipped=r['selected_source_id'] != original[r['target_id']], **r))
    loio = pd.DataFrame(details)
    summary = selected.copy()
    summary['bootstrap_winner_frequency'] = [counts[d][original[d]]/1000 for d in summary.target_id]
    summary['bootstrap_mode'] = [sorted(counts[d], key=lambda s: (-counts[d][s], s))[0] for d in summary.target_id]
    summary['loio_iteration_flips'] = [int(loio[loio.target_id == d].flipped.sum()) for d in summary.target_id]
    summary['loio_iteration_retention'] = 1-summary.loio_iteration_flips/10
    summary['unstable'] = (summary.bootstrap_winner_frequency < .70) | (summary.cosine_margin < .02) | (summary.loio_iteration_flips > 0)
    summary.to_csv(output/'stability.csv', index=False)
    loio.to_csv(output/'leave_one_iteration_out.csv', index=False)
    dump(output/'bootstrap_winners.json', {d: dict(c) for d, c in counts.items()})
    dump(output/'leave_one_dataset_out.json', dict(status='blocked_definition_mismatch', reason='Audited project LOIO drops one iteration globally and refits every dataset profile. It does not hold out a dataset and has no held-out dataset-profile construction. A leave-one-dataset-out implementation would require a new scientific definition. Primary deterministic fit, bootstrap and historical leave-one-iteration-out are unaffected.'))
    return summary


def figures(profiles, matrix, selected, stable, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9})
    for name, data, labels, cmap in (
        ('construct_profiles', profiles.set_index('dataset')[list(CONSTRUCTS)].values, ['History', 'Value', 'Probability', 'Feedback', 'Learning'], 'viridis'),
        ('cosine_similarity', matrix.values, list(matrix.columns), 'viridis'),
    ):
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(data, vmin=0, vmax=1, aspect='auto', cmap=cmap)
        ax.set_yticks(range(len(profiles)), profiles.dataset)
        ax.set_xticks(range(len(labels)), labels, rotation=65 if len(labels)>5 else 0, ha='right' if len(labels)>5 else 'center')
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output/f'{name}.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(stable.target_id, stable.bootstrap_winner_frequency)
    ax.axvline(.7, color='grey', linestyle='--'); ax.set_xlim(0, 1)
    ax.set_xlabel('Bootstrap frequency of primary winner (1,000 cluster draws)' if output.name=='primary_runtime_valid' else 'Bootstrap frequency of sensitivity winner (1,000 cluster draws)')
    fig.tight_layout(); fig.savefig(output/'bootstrap_stability.pdf'); plt.close(fig)


def run(repo, output, workers):
    config = check_frozen(repo)
    if output.exists():
        raise ValueError('Immutable run output exists; use a new versioned output directory')
    live = verify_manifest(repo)
    annotations, failures, paths = load_join(repo, live)
    output.mkdir(parents=True)
    dump(output/'frozen_config.json', config)
    before = {str(p.relative_to(repo)): sha(p) for p in paths}
    finalization = {}
    for path in paths:
        first = finalize_annotation_jsonl(path, repo)
        saved = sha(path)
        second = finalize_annotation_jsonl(path, repo)
        if not first['schema_ok'] or not second['schema_ok'] or second['rewritten'] or sha(path) != saved:
            raise ValueError(f'Finalizer validation/idempotence failed: {path}')
        finalization[str(path.relative_to(repo))] = dict(first=first, second=second)
    dump(output/'postprocessing.json', finalization)
    annotations, failures, paths = load_join(repo, live)
    frame = build_master(live, annotations, failures)
    frame.to_csv(output/'population_master.csv', index=False)
    counts = []
    for dataset, g in frame.groupby('dataset', sort=True):
        counts.append(dict(dataset=dataset, canonical_events=len(g), runtime_valid=int(g.runtime_valid.sum()), runtime_invalid=int((~g.runtime_valid).sum()), annotation_failures=int((g.exclusion_reason=='annotation_failure').sum()), unresolved_semantic=int((g.exclusion_reason=='unresolved_semantic').sum()), primary_denominator=int(g.primary_eligible.sum()), sensitivity_denominator=int(g.sensitivity_eligible.sum()), duplicate_source_sha_events=len(g)-g.code_sha256.nunique()))
    pd.DataFrame(counts).to_csv(output/'cohort_counts.csv', index=False)
    dump(output/'integrity.json', dict(canonical_events=1500, unique_resume_keys=frame.resume_key.nunique(), unmatched=0, missing_unexplained=0, sha_mismatches=0, duplicate_candidate_identities=0, duplicate_source_sha_events=1500-frame.code_sha256.nunique(), annotation_failures=failures, lineage=dict(AUTHORITATIVE_JOBS), resume_executions={'311666':'312215', '311671':'312213'}, ignored_historical_shards=['pics_v4_popann_a40/job_311208', 'pics_v4_popann_h100/job_311200']))
    maps = {}; reproducibility = {}
    for name, mask in [('primary_runtime_valid', frame.primary_eligible), ('sensitivity_all_generated', frame.sensitivity_eligible)]:
        out = output/name; out.mkdir()
        cols = ['dataset', 'iteration', 'parent_id', 'resume_key', *CONSTRUCTS]
        panel = frame.loc[mask, cols].copy()
        for c in CONSTRUCTS:
            panel[c] = panel[c].astype(int)
        if panel.dataset.nunique() != 15:
            raise ValueError('Every dataset must have a complete profile')
        panel.to_csv(out/'occurrence_panel.csv', index=False)
        profiles, eb, model, meta = fit_profiles(panel)
        repeat, repeat_eb, repeat_model, _ = fit_profiles(panel)
        for a, b in [(profiles, repeat), (eb, repeat_eb), (model, repeat_model)]:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        reproducibility[name] = 'two deterministic fits bitwise identical'
        profiles.to_csv(out/'profiles.csv', index=False); eb.to_csv(out/'eb_estimates.csv', index=False); model.to_csv(out/'model.csv', index=False)
        dump(out/'estimator.json', meta)
        matrix, selected = select_sources(profiles)
        matrix.to_csv(out/'similarity_matrix.csv', index_label='target_id')
        selected.to_csv(out/'selected_sources.csv', index=False)
        maps[name] = selected.set_index('target_id').selected_source_id.to_dict()
        dump(out/'source_map.json', dict(interpretation='post hoc construct-profile/source-selection analysis; never used by completed target-only experiments', sources=maps[name]))
        stable = stability(panel, selected, out, workers)
        figures(profiles, matrix, selected, stable, out)
    differences = {d: dict(primary=s, sensitivity=maps['sensitivity_all_generated'][d]) for d, s in maps['primary_runtime_valid'].items() if s != maps['sensitivity_all_generated'][d]}
    dump(output/'map_comparison.json', dict(n_differences=len(differences), differences=differences, primary_policy_unchanged=True))
    dump(output/'reproducibility.json', reproducibility)
    inputs = {str(p.relative_to(repo)): sha(p) for p in paths}
    inputs[str(Path(OUTPUT_REL)/'program_manifest.json')] = sha(repo/OUTPUT_REL/'program_manifest.json')
    for shard in SHARDS:
        p = repo/OUTPUT_REL/'annotations'/shard/'annotation_failures.jsonl'
        if p.is_file(): inputs[str(p.relative_to(repo))] = sha(p)
    versions = {}
    from importlib.metadata import version
    for package in ('numpy', 'pandas', 'matplotlib', 'statsmodels', 'PyYAML', 'scipy'):
        versions[package] = version(package)
    dump(output/'run_freeze.json', dict(date=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(), command=[sys.executable, *sys.argv], python=sys.version, platform=platform.platform(), packages=versions, seed=20260920, bootstrap_B=1000, slurm_job_id=os.environ.get('SLURM_JOB_ID'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), annotation_hashes_before_finalization=before, input_hashes=inputs, frozen_configuration_hash=sha(repo/'analysis/config/mem/pics_v4_uniform_v8_frozen.json'), code_hashes=config['code_hashes'], output_hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}))
    print(f'Completed frozen analysis: {output}; map differences: {len(differences)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.repo.absolute(), args.output.absolute(), args.workers)
