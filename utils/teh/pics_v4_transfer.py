"""Identity adapter for the existing PICS population coordinator (no search code).

Only evaluated, completed uniform-v8 target-only artifacts are source programs.
The approved Occurrence-EB map is immutable; diagnostic selectors are excluded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MAP = REPO / 'analysis_2026Sep/Codex/Oct5/others/population_source_profiles/run_v1/primary_runtime_valid/source_map.json'
MAP_SHA = 'f3b043f6a82c1fc89bb52b476903afe9f48b66c2f51809c57c818a9b0a7dcef6'
ARTIFACTS = REPO / 'analysis/config/pics_v4_official_sources.json'
ARTIFACTS_SHA = '741d26f21bcdb0b71920e9633f49b1a8f28933c16cada1e2e1facefdbff7068f'
KINDS = {m: 'pics_v4_' + m for m in ('target_only', 'transfer_based_only', 'official_gate')}
_identity: dict[str, Any] | None = None
_suffix = ''


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)).encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def official_inputs(config_path=None):
    from utils.teh.pics_aamas_v0 import CANONICAL_DATASETS
    if config_path is not None and Path(config_path).resolve() != MAP.resolve():
        raise RuntimeError('PICS v4 accepts only the approved primary source map')
    if sha(MAP) != MAP_SHA or sha(ARTIFACTS) != ARTIFACTS_SHA:
        raise RuntimeError('PICS v4 approved source map/artifact manifest SHA mismatch')
    sources = read(MAP)['sources']
    if set(sources) != set(CANONICAL_DATASETS) or any(t == s or s not in sources for t, s in sources.items()):
        raise RuntimeError('PICS v4 source-map coverage/self-exclusion mismatch')
    return sources, read(ARTIFACTS)


def target_run(target: str) -> Path:
    _, artifacts = official_inputs()
    for rel, expected_hash in artifacts['target_inputs_sha256'][target].items():
        if sha(REPO / rel) != expected_hash:
            raise RuntimeError('frozen target-only input SHA mismatch')
    return REPO / f'generated_outputs/psych101_train/teh/{target}/pics_v4_target_only/job_{artifacts["target_jobs"][target]}'


def verify_completed(arm: Path, *, target: str, kind: str, allow_kind_relocation: bool = False) -> dict:
    """Require completion, live evaluated bytes and rank-0 agreement before use."""
    from utils.teh.pics_v4 import assert_pics_v4_resume
    root = arm.parent.parent
    if f'/{target}/{kind}/' not in str(root):
        if not (allow_kind_relocation and f'/{target}/' in str(root)):
            raise RuntimeError('population target/KIND path mismatch')
    expected = current_identity()
    if kind == KINDS['target_only']:
        # Frozen target-only records predate transfer identity. Validate their
        # original policy without introducing transfer metadata.
        saved, saved_suffix = current_identity(), _suffix
        configure_identity(None)
        try:
            assert_pics_v4_resume(root)
        finally:
            configure_identity(saved, saved_suffix)
    else:
        if expected is None and not allow_kind_relocation:
            raise RuntimeError('missing expected transfer identity')
        if expected is not None:
            mode = next(m for m, k in KINDS.items() if k == kind)
            assert_pics_v4_resume(root, expected_identity=dict(expected, mode=mode, kind=kind))
    prov, complete = read(arm / 'POPULATION_PROVENANCE.json'), read(arm / 'STAGE_COMPLETE.json')
    for field, expected in (('dataset', target), ('kind', kind), ('runtime_valid', True), ('global_iters', 10), ('n_candidates', 10)):
        if prov.get(field) != expected or complete.get(field) != expected:
            raise RuntimeError(f'population completion mismatch: {field}')
    best = arm / 'global_phase/best_program.py'
    recorded_rank1 = Path(prov['rank1_program']).resolve()
    if allow_kind_relocation:
        if recorded_rank1 != Path(complete['rank1_program']).resolve() or sha(recorded_rank1) != prov['rank1_sha256']:
            raise RuntimeError('population rank-1 path mismatch')
    elif recorded_rank1 != best.resolve() or Path(complete['rank1_program']).resolve() != best.resolve():
        raise RuntimeError('population rank-1 path mismatch')
    if sha(best) != prov['rank1_sha256'] or sha(best) != complete['rank1_sha256']:
        raise RuntimeError('population live rank-1 SHA mismatch')
    results = read(arm / 'global_phase/results.json')
    pool = arm / 'global_phase/global_elite_pool'
    elite = read(pool / 'pool_manifest.json')
    if isinstance(elite, dict):
        elite = elite.get('programs', elite.get('entries', []))
    rank0 = elite[0]
    if sha(pool / rank0['filename']) != sha(best) or rank0['program_id'] != results['pool_best_program_id']:
        raise RuntimeError('population rank-0 identity/bytes mismatch')
    if results.get('evolution_selection_score') != 'train_val' or prov.get('test_used_for_selection') is not False:
        raise RuntimeError('population selection must use train+validation only')
    score = results['pool_best_selection_score']
    if score != prov['pooled_train_val_score'] or score != complete['pooled_train_val_score'] or score != rank0['global_fitness']:
        raise RuntimeError('population train+validation selection-score mismatch')
    if kind != KINDS['target_only']:
        expected = current_identity()
        mode = next(m for m, k in KINDS.items() if k == kind)
        if expected is None:
            if prov.get('run_identity') != complete.get('run_identity') or not prov.get('run_identity'):
                raise RuntimeError('missing expected transfer identity')
            expected = prov['run_identity']
        else:
            expected = dict(expected, mode=mode, kind=kind)
            assert_identity(prov.get('run_identity'), expected)
            assert_identity(complete.get('run_identity'), expected)
        for field, value in expected['policies'].items():
            if prov.get(field) != value:
                raise RuntimeError(f'completed population method-policy mismatch: {field}')
        if prov.get('track_mode') != mode or complete.get('track_mode') != mode or prov.get('arm_role') != arm.name or complete.get('arm_role') != arm.name:
            raise RuntimeError('completed population mode/arm mismatch')
        if complete.get('rank1_program_id') != results['pool_best_program_id'] or prov.get('rank1_program_id') != results['pool_best_program_id']:
            raise RuntimeError('completed population evaluated candidate identity mismatch')
        hashes = prov.get('panel_bank_sha256')
        if not hashes or hashes != complete.get('panel_bank_sha256'):
            raise RuntimeError('completed population missing panel-bank attestation')
        for rel, expected_hash in hashes.items():
            if sha(arm / rel) != expected_hash:
                raise RuntimeError('completed population panel-bank SHA mismatch')
    return prov


def resolve_source(target: str, *, config_path=None) -> dict:
    from utils.teh.prompt_sanitize import sanitize_evolution_candidate_code
    sources, artifacts = official_inputs(config_path)
    source = sources[target]
    frozen = artifacts['sources'][source]
    if sha(REPO / 'utils/teh/prompt_sanitize.py') != artifacts['sanitizer_sha256']:
        raise RuntimeError('frozen source sanitizer SHA mismatch')
    for rel, expected_hash in frozen['input_sha256'].items():
        if sha(REPO / rel) != expected_hash:
            raise RuntimeError(f'official source provenance/lineage input SHA mismatch: {rel}')
    best = REPO / frozen['source_program_path']
    arm = best.parent.parent
    prov = verify_completed(arm, target=source, kind=KINDS['target_only'])
    raw = REPO / frozen['candidate_path']
    if sha(best) != frozen['source_program_sha256'] or sha(raw) != frozen['canonical_candidate_or_seed_sha256']:
        raise RuntimeError('official source raw/evaluated SHA mismatch')
    if read(best.parent / 'results.json')['pool_best_program_id'] != frozen['source_program_id']:
        raise RuntimeError('official source candidate identity mismatch')
    if prov['pooled_train_val_score'] != frozen['selection_score']:
        raise RuntimeError('official source frozen selection-score mismatch')
    transformed = sanitize_evolution_candidate_code(raw.read_text(), required_markers=('def choose(',))
    if transformed.encode() != best.read_bytes():
        raise RuntimeError('raw-to-evaluated sanitizer transformation failed (AST equivalence is insufficient)')
    return {
        'source_dataset': source, 'source_program_id': frozen['source_program_id'],
        'source_program_path': str(best.resolve()), 'source_program_sha256': sha(best),
        'raw_program_path': str(raw.resolve()), 'raw_program_sha256': sha(raw),
        'raw_candidate_identity': frozen['candidate_identity'], 'raw_resume_key': frozen['candidate_resume_key'],
        'lineage': frozen['lineage'], 'selection_score': prov['pooled_train_val_score'],
        'selection_score_type': 'count_pooled_train_val_mean_loglik', 'test_used_for_selection': False,
        'sanitizer': 'utils.teh.prompt_sanitize.sanitize_evolution_candidate_code',
        'sanitizer_sha256': sha(REPO / 'utils/teh/prompt_sanitize.py'), 'exact_transformed_bytes_equal': True,
        'completion_sha256': sha(arm / 'STAGE_COMPLETE.json'),
        'provenance_sha256': sha(arm / 'POPULATION_PROVENANCE.json'),
    }


def load_config(config_path=None):
    from utils.teh.t_pics_gated_transfer import FrozenTransferConfig, SelectedSourceEntry, SourcePopulationRun
    sources, _ = official_inputs(config_path)
    records = {s: resolve_source(t, config_path=config_path) for t, s in sources.items()}
    runs = {s: SourcePopulationRun(s, str(r['lineage']['canonical_population_job_id']), Path(r['source_program_path']).parents[3], Path(r['source_program_path'])) for s, r in records.items()}
    entries = {t: SelectedSourceEntry(t, s, s, runs[s].run_dir, runs[s].job_id, runs[s].rank1_program) for t, s in sources.items()}
    return FrozenTransferConfig(MAP, 'occurrence_eb_uniform_v8_primary', sources, entries, runs)


def configure_identity(identity=None, suffix=''):
    global _identity, _suffix
    _identity = identity
    _suffix = suffix


def current_identity():
    return _identity

def identity_fields():
    return {'run_identity': _identity} if _identity is not None else {}


def source_suffix():
    if _identity is None:
        raise RuntimeError('PICS v4 transfer identity is not configured')
    return _suffix


def observed_cohort(target, participant_ids, *, local_dataset=None):
    from baseline_methods.MLE import trials_for_participant
    from utils.teh.prompt_snapshots import stamp_prompt_participant_id
    from data_modules.mixed_gambles import DEFAULT_CSV_PATH
    train, val = [], []
    for pid in participant_ids:
        tr, va, test = trials_for_participant(target, int(pid), split_ratio=.6, split_seed=0, filter_mixed_gambles=False, psych_dataset_split='train', local_dataset=local_dataset, mixed_gambles_csv=DEFAULT_CSV_PATH, limited_data_protocol='structure_aware_v3', limited_train_val=40)
        del test
        train.extend(stamp_prompt_participant_id(t, int(pid)) for t in tr)
        val.extend(stamp_prompt_participant_id(t, int(pid)) for t in va)
    return train, val


def prepare_identity(args, *, suffix: str, observed_trials=None) -> dict:
    from utils.teh.pics_v4 import pics_v4_marker_payload
    mode, target = args.pics_aamas_v0_track_mode, args.dataset
    if getattr(args, 'filter_mixed_gambles', False):
        raise RuntimeError('PICS v4 frozen target cohort requires filter_mixed_gambles=false')
    for field in ('global_prompt_source_program', 'global_prompt_source_dataset', 'explore_prompt_source_program', 'explore_prompt_source_dataset', 't_pics_source', 't_pics_gated_source', 'initial_pool_programs', 'initial_pool_dir', 't_pics_reuse_gate_pool', 't_pics_gated_independent', 'max_observed_trials_per_participant'):
        if getattr(args, field, None):
            raise RuntimeError(f'PICS v4 transfer method override forbidden: {field}')
    for field, expected in (('sample_size', 8), ('elite_pool_size', 50), ('fresh_n_candidates', 10), ('n_iterations', 10), ('explore_candidates', 50), ('explore_population_top_k', 1), ('early_stop_iters', -1), ('n_eval_seeds', 1), ('max_error_prompt_chars', 0), ('sample_parents', True), ('sampled_parents_decay', True), ('fitness_metric', 'loglik'), ('split_mode', 'within_participant'), ('error_feedback_mode', 'legacy')):
        if getattr(args, field, expected) != expected:
            raise RuntimeError(f'PICS v4 frozen method argument mismatch: {field}')
    if getattr(args, 'pics_v3_elite_failover', False):
        raise RuntimeError('PICS v4 transfer forbids elite failover')
    if float(getattr(args, 'split_ratio', .6)) != .6 or float(getattr(args, 'mdl_lambda', 0)) != 0 or getattr(args, 'refinement_phase', False):
        raise RuntimeError('PICS v4 split/MDL/refinement policy mismatch')
    if mode not in ('transfer_based_only', 'official_gate'):
        raise RuntimeError('transfer identity requires a transfer/gate mode')
    for field, expected in (('split_seed', 0), ('pics_run_seed', 0), ('global_iters', 10), ('n_candidates', 10), ('hard_prompt_token_cap', 15360), ('llm_max_tokens', 1024), ('max_parent_chars', 5000), ('limited_data_protocol', 'structure_aware_v3'), ('limited_train_val', 40), ('evolution_selection_score', 'train_val')):
        if getattr(args, field) != expected:
            raise RuntimeError(f'PICS v4 frozen method argument mismatch: {field}')
    target_root = target_run(target)
    prov = verify_completed(target_root / 'target_population/control', target=target, kind=KINDS['target_only'])
    for field in ('range_start_ordinal', 'range_end_ordinal'):
        if getattr(args, field, prov[field]) != prov[field]:
            raise RuntimeError(f'PICS v4 frozen target participant range mismatch: {field}')
    train, val = observed_trials if observed_trials is not None else observed_cohort(target, prov['participant_ids'], local_dataset=getattr(args, 'local_dataset', None))
    from utils.teh.t_pics_gated_transfer import default_seed_path
    seed_path = REPO / default_seed_path(target)
    if getattr(args, 'seed_path', None) and sha(Path(args.seed_path)) != sha(seed_path):
        raise RuntimeError('PICS v4 neutral seed override forbidden')
    identity = {
        'schema': 'pics_v4_transfer_identity_v1', 'mode': mode, 'kind': KINDS[mode], 'target': target,
        'source': resolve_source(target, config_path=args.t_pics_source_config),
        'source_map_path': str(MAP.resolve()), 'source_map_sha256': MAP_SHA,
        'source_artifacts_sha256': ARTIFACTS_SHA,
        'base_prompt_sha256': sha(target_root / 'prompts/infer_single_choice.txt'),
        'source_suffix_sha256': hashlib.sha256(suffix.encode()).hexdigest(),
        'cohort': {f: prov[f] for f in ('participant_ids', 'sa40_fingerprint', 'observed_train_fingerprint', 'observed_val_fingerprint', 'range_start_ordinal', 'range_end_ordinal')},
        'target_train_cohort_sha256': digest(train), 'target_val_cohort_sha256': digest(val),
        'policies': pics_v4_marker_payload(),
        'seed_program_raw_sha256': sha(seed_path),
        'schedule': {'population_iters': 10, 'candidates': 10, 'exploration': 50, 'participant_iters': 10, 'sample_size': 8, 'elite_pool_cap': 50, 'n_eval_seeds': 1, 'mdl_lambda': 0, 'refinement': False, 'early_stop': False, 'failover': False, 'input_tokens': 15360, 'output_tokens': 1024, 'context_tokens': 16384, 'parent_chars': 5000},
        'method_code_sha256': {p: sha(REPO / p) for p in ('teh.py', 'utils/teh/pics_v4_transfer.py', 'utils/teh/pics_v4.py', 'utils/teh/pics_v4_panels.py', 'utils/teh/pics_aamas_v0.py', 'utils/teh/explore_source_prompt.py', 'utils/teh/prompt_sanitize.py', 'utils/teh/prompt_snapshots.py', 'utils/teh/aamas_v0_lossless_trials.py', 'utils/teh/elite_sha.py', 'utils/teh/t_pics_gated_transfer.py', 'utils/teh/teh_runtime.py', 'utils/teh/pics_v4_dry_run.py', 'utils/teh_transfer/prompts.py', 'utils/teh/limited_data_protocol.py', 'utils/teh/limited_data_registry.py', 'analysis/config/teh_datasets.yaml', 'prompts/teh/additional_prompt/pics_v4_uniform_additional_prompt_v8.txt', 'cluster/v0/ours/main/aamas_v0/_fill.sh', 'cluster/v0/ours/main/aamas_v0/job_track_h100.sh')},
    }
    from utils.teh.pics_v4_recovery import method_identity_hashes
    identity['method_code_sha256'] = method_identity_hashes(identity['method_code_sha256'], mode=mode)
    configure_identity(identity, suffix)
    return identity


def assert_identity(found: dict, expected=None):
    expected = expected or current_identity()
    if expected is None or found != expected:
        raise RuntimeError('PICS v4 transfer provenance/resume identity mismatch')


def assert_arm_resume(arm: Path, *, role: str):
    """Reject partial/completed arm mismatches before the coordinator mutates files."""
    identity = current_identity()
    if identity is None:
        raise RuntimeError('missing transfer run identity')
    path = arm / 'RUN_IDENTITY.json'
    expected = {'run_identity': identity, 'arm': role, 'suffix_sha256': identity['source_suffix_sha256'] if role == 'transfer' else hashlib.sha256(b'').hexdigest()}
    if path.exists():
        if read(path) != expected:
            raise RuntimeError('PICS v4 arm resume/source identity mismatch')
    elif arm.exists() and any(arm.iterdir()):
        raise RuntimeError('PICS v4 partial population has no bound arm identity')
    bank_attestation = arm / 'global_phase/pics_v4_panel_banks/ATTESTATION.json'
    bank_files = list((arm / 'global_phase/pics_v4_panel_banks').glob('*.json'))
    if bank_files:
        if not bank_attestation.is_file():
            raise RuntimeError('partial population has unattested panel banks')
        for rel, expected_hash in read(bank_attestation).items():
            if sha(bank_attestation.parent / rel) != expected_hash:
                raise RuntimeError('partial population panel-bank SHA mismatch')
    complete = arm / 'STAGE_COMPLETE.json' 
    if complete.exists():
        prov = verify_completed(arm, target=identity['target'], kind=identity['kind'])
        assert_identity(prov.get('run_identity'))
        assert_identity(read(complete).get('run_identity'))
    return expected


def initialize_arm(arm: Path, *, role: str):
    expected = assert_arm_resume(arm, role=role)
    arm.mkdir(parents=True, exist_ok=True)
    path = arm / 'RUN_IDENTITY.json'
    if not path.exists():
        path.write_text(json.dumps(expected, indent=2) + '\n')


def assert_participant_resume(path: Path, participant_id: int):
    identity = current_identity()
    if identity is None:
        return None
    if int(participant_id) not in identity['cohort']['participant_ids']:
        raise RuntimeError('participant outside frozen transfer cohort')
    expected = {'run_identity': identity, 'participant_id': int(participant_id)}
    marker = Path(path) / 'RUN_IDENTITY.json'
    if marker.exists():
        if read(marker) != expected:
            raise RuntimeError('participant mode/source resume identity mismatch')
    elif Path(path).exists() and any(Path(path).iterdir()):
        raise RuntimeError('participant partial run is missing transfer identity')
    return expected


def initialize_participant(path: Path, participant_id: int):
    expected = assert_participant_resume(path, participant_id)
    if expected is None:
        return
    Path(path).mkdir(parents=True, exist_ok=True)
    marker = Path(path) / 'RUN_IDENTITY.json'
    if not marker.exists():
        marker.write_text(json.dumps(expected, indent=2) + '\n')


def validate_materialization(target_arm, transfer_arm, target, output_dir):
    """Called by the existing materializer before evaluation or writes."""
    from utils.teh.pics_v4 import assert_pics_v4_resume, assert_pics_v4_output
    identity = current_identity()
    if identity is None or identity['mode'] != 'official_gate':
        raise RuntimeError('missing official-gate run identity')
    assert_pics_v4_output(output_dir)
    assert_pics_v4_resume(output_dir)
    if target_arm.resolve() != (target_run(target) / 'target_population/control').resolve():
        raise RuntimeError('official gate requires the frozen official target-only population')
    left = verify_completed(target_arm, target=target, kind=KINDS['target_only'])
    right = verify_completed(transfer_arm, target=target, kind=KINDS['transfer_based_only'])
    bound = right.get('run_identity')
    expected = dict(identity, mode='transfer_based_only', kind=KINDS['transfer_based_only'])
    assert_identity(bound, expected)
    assert_identity(read(transfer_arm / 'STAGE_COMPLETE.json').get('run_identity'), expected)
    if read(transfer_arm / 'RUN_IDENTITY.json') != {'run_identity': expected, 'arm': 'transfer', 'suffix_sha256': expected['source_suffix_sha256']}:
        raise RuntimeError('materialized transfer arm identity mismatch')
    for field in ('participant_ids', 'range_start_ordinal', 'range_end_ordinal', 'prompt_policy', 'trial_prompt_policy', 'prompt_mode', 'rendered_prompt_sha256', 'method_version', 'sa40_fingerprint', 'observed_train_fingerprint', 'observed_val_fingerprint', 'split_seed', 'pics_run_seed', 'additional_prompt_sha256', 'search_rng', 'elite_policy', 'limited_data_protocol', 'limited_train_val', 'model_name', 'hard_prompt_token_cap', 'llm_max_tokens', 'max_parent_chars', 'vllm_max_model_len'):
        if left.get(field) != right.get(field):
            raise RuntimeError(f'official gate target observed-cohort/method mismatch: {field}')
    return left, right
