"""Fail-closed integration checks for the existing three-mode architecture."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from utils.teh import pics_v4_transfer as adapter
from utils.teh.pics_v4 import configure_pics_v4, pics_v4_marker_payload, write_pics_v4_marker, assert_pics_v4_resume
from utils.teh.pics_aamas_v0 import decide_aamas_v0_gate, materialize_official_gate
from utils.teh.aamas_v0_lossless_trials import set_pics_run_seeds


@pytest.fixture(autouse=True)
def reset():
    configure_pics_v4(True)
    adapter.configure_identity(None)
    set_pics_run_seeds(run_seed=0, split_seed=0)
    yield
    adapter.configure_identity(None)
    configure_pics_v4(False)


def identity(mode='transfer_based_only'):
    return {'mode': mode, 'kind': adapter.KINDS[mode], 'target': 'target',
        'source': {'source_dataset': 'source', 'source_program_id': 'global_iteration_1_candidate_1', 'source_program_path': '/official/best_program.py', 'raw_program_sha256': 'raw', 'source_program_sha256': 'evaluated'},
        'source_map_sha256': adapter.MAP_SHA, 'source_suffix_sha256': hashlib.sha256(b'source context').hexdigest(),
        'base_prompt_sha256': 'base', 'policies': pics_v4_marker_payload(), 'cohort': {'participant_ids': [0]}}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize('field', ['mode', 'kind', 'target', 'source_map_sha256', 'base_prompt_sha256', 'source_suffix_sha256', 'policies', 'cohort'])
def test_resume_rejects_identity_change_before_write(tmp_path, field):
    expected = identity()
    adapter.configure_identity(expected)
    root = tmp_path / 'target/pics_v4_transfer_based_only/job_1'
    write_pics_v4_marker(root)
    before = (root / 'TRIAL_PROMPT_POLICY.json').read_bytes()
    changed = copy.deepcopy(expected)
    changed[field] = 'wrong'
    payload = pics_v4_marker_payload()
    payload['run_identity'] = changed
    dump(root / 'TRIAL_PROMPT_POLICY.json', payload)
    damaged = (root / 'TRIAL_PROMPT_POLICY.json').read_bytes()
    with pytest.raises(RuntimeError, match='mismatch'):
        write_pics_v4_marker(root)
    assert (root / 'TRIAL_PROMPT_POLICY.json').read_bytes() == damaged != before


@pytest.mark.parametrize('field', ['source_dataset', 'source_program_id', 'source_program_path', 'raw_program_sha256', 'source_program_sha256'])
def test_source_identity_resume_mismatch(tmp_path, field):
    expected = identity()
    adapter.configure_identity(expected)
    root = tmp_path / 'target/pics_v4_transfer_based_only/job_1'
    write_pics_v4_marker(root)
    payload = adapter.read(root / 'TRIAL_PROMPT_POLICY.json')
    payload['run_identity']['source'][field] = 'wrong'
    dump(root / 'TRIAL_PROMPT_POLICY.json', payload)
    with pytest.raises(RuntimeError):
        assert_pics_v4_resume(root)


def test_target_only_marker_payload_unchanged_and_cross_mode_rejected(tmp_path):
    root = tmp_path / 'target/pics_v4_target_only/job_1'
    write_pics_v4_marker(root)
    assert adapter.read(root / 'TRIAL_PROMPT_POLICY.json') == pics_v4_marker_payload()
    adapter.configure_identity(identity())
    with pytest.raises(RuntimeError):
        assert_pics_v4_resume(root)
    wrong = tmp_path / 'other/pics_v4_transfer_based_only/job_1'
    with pytest.raises(RuntimeError):
        write_pics_v4_marker(wrong)
    assert not wrong.exists()


def test_target_marker_cannot_accept_transfer_fields(tmp_path):
    root = tmp_path / 'target/pics_v4_target_only/job_1'
    write_pics_v4_marker(root)
    payload = adapter.read(root / 'TRIAL_PROMPT_POLICY.json')
    payload['source_dataset'] = 'source'
    dump(root / 'TRIAL_PROMPT_POLICY.json', payload)
    with pytest.raises(RuntimeError):
        assert_pics_v4_resume(root)


def test_partial_arm_and_bank_tampering_rejected(tmp_path):
    adapter.configure_identity(identity())
    arm = tmp_path / 'target/pics_v4_transfer_based_only/job_1/target_population/transfer'
    dump(arm / 'global_phase/results.json', {})
    with pytest.raises(RuntimeError, match='no bound'):
        adapter.initialize_arm(arm, role='transfer')
    (arm / 'global_phase/results.json').unlink()
    (arm / 'global_phase').rmdir()
    adapter.initialize_arm(arm, role='transfer')
    bank = arm / 'global_phase/pics_v4_panel_banks/population_fresh.json'
    dump(bank, {'conditioning': identity()})
    with pytest.raises(RuntimeError, match='unattested'):
        adapter.assert_arm_resume(arm, role='transfer')
    dump(bank.parent / 'ATTESTATION.json', {bank.name: adapter.sha(bank)})
    adapter.assert_arm_resume(arm, role='transfer')
    dump(bank, {'wrong': True})
    with pytest.raises(RuntimeError, match='SHA mismatch'):
        adapter.assert_arm_resume(arm, role='transfer')


def test_only_approved_map_and_all_15_evaluated_artifacts():
    if not adapter.ARTIFACTS.exists() or not adapter.target_run('14kool2016when').exists():
        pytest.skip('official offline bank not installed')
    sources, _ = adapter.official_inputs()
    assert len(sources) == 15
    with pytest.raises(RuntimeError, match='only the approved'):
        adapter.official_inputs(Path('fitness_source_map.json'))
    for target, source in sources.items():
        record = adapter.resolve_source(target)
        assert record['source_dataset'] == source != target
        assert record['exact_transformed_bytes_equal'] is True
        assert record['test_used_for_selection'] is False
    assert adapter.resolve_source('7hilbig2014generalized')['source_program_id'] == 'global_baseline'


@pytest.mark.parametrize('left,right,winner', [(-1., -1., 'target_only'), (-1., -.9, 'transfer'), (-1., -1.1, 'target_only'), (-1., -1. + 1e-13, 'transfer')])
def test_unchanged_exact_gate(left, right, winner):
    result = decide_aamas_v0_gate(target_only_score=left, transfer_score=right)
    assert result['selected_arm'] == winner
    assert result['test_used_for_gate'] is False


def test_source_conditioned_bank_uses_actual_suffix_and_fingerprint(tmp_path, monkeypatch):
    import teh
    from utils.teh.pics_v4_panels import clear_pics_v4_panel_state, registered_panel_bank
    from utils.teh.pics_aamas_v0 import configure_pics_aamas_v0_prompt
    configure_pics_aamas_v0_prompt(True)
    adapter.configure_identity(identity())
    prompt_dir = tmp_path / 'prompts'
    prompt_dir.mkdir()
    (prompt_dir / 'infer_single_choice.txt').write_text('Target instruction\n')
    (prompt_dir / 'single_code_template.txt').write_text('def choose(problem, history):\n    pass\n')
    # Token-count fixture tests the actual factory's wrapper/fingerprint binding;
    # full actual Qwen renders are separately exercised by the all-15 preflight.
    from utils.teh import pics_v4_panels as panels
    monkeypatch.setattr(panels, 'qwen_user_prompt_token_count', lambda p: len(p))
    trials = [{'_prompt_participant_id': 0, '_ldp': {'unit_id': str(i)}, 'problem': {'dataset_alias': '3frey2017cct', 'position': i}, 'history': [], 'action': 0} for i in range(20)]
    fingerprints, sizes = [], []
    for suffix in (None, 'SOURCE PROGRAM\n' * 40):
        clear_pics_v4_panel_state()
        teh._ensure_pics_v4_population_fresh_bank(dataset='3frey2017cct', trials=trials, seed_code='def choose(problem, history):\n    return 0.5', n_slots=10, run_prompts_dir=str(prompt_dir), output_dir=None, fitness_metric='loglik', master_seed=0, cap=2500, prompt_suffix=suffix)
        bank = registered_panel_bank(dataset='3frey2017cct', phase='population', participant_id=None, name='fresh')
        fingerprints.append(bank.fingerprint)
        sizes.append(len(bank.panels[0]))
    assert fingerprints[0] != fingerprints[1]
    assert sizes[1] < sizes[0]
    clear_pics_v4_panel_state()
    configure_pics_aamas_v0_prompt(False)


def test_materializer_validates_before_evaluation_and_preserves_gate(tmp_path, monkeypatch):
    expected = identity('official_gate')
    adapter.configure_identity(expected)
    target_root = tmp_path / 'target/pics_v4_target_only/job_1'
    transfer_root = tmp_path / 'target/pics_v4_transfer_based_only/job_2'
    out = tmp_path / 'target/pics_v4_official_gate/job_3'
    target_arm, transfer_arm = target_root / 'target_population/control', transfer_root / 'target_population/transfer'
    for arm, kind in ((target_arm, 'pics_v4_target_only'), (transfer_arm, 'pics_v4_transfer_based_only')):
        (arm / 'global_phase/global_elite_pool').mkdir(parents=True)
        dump(arm / 'POPULATION_PROVENANCE.json', {'dataset': 'target', 'kind': kind, 'rank1_program': str(arm / 'global_phase/best_program.py'), 'prompt_policy': 'structured_snapshot_sparse_history_v1', 'rank1_sha256': 'rank1'})
    calls = []
    def denied(*args):
        raise RuntimeError('completion/source identity mismatch')
    monkeypatch.setattr(adapter, 'validate_materialization', denied)
    with pytest.raises(RuntimeError, match='completion/source'):
        materialize_official_gate(target_only_run=target_root, transfer_run=transfer_root, output_dir=out, target_dataset='target', evaluate_score=lambda *args: calls.append(args))
    assert not calls and not out.exists()
    monkeypatch.setattr(adapter, 'validate_materialization', lambda *args: (adapter.read(target_arm / 'POPULATION_PROVENANCE.json'), adapter.read(transfer_arm / 'POPULATION_PROVENANCE.json')))
    result = materialize_official_gate(target_only_run=target_root, transfer_run=transfer_root, output_dir=out, target_dataset='target', evaluate_score=lambda *args: -1.)
    assert result['selected_arm'] == 'target_only'
    assert result['run_identity'] == expected
    assert (out / 'selected/retained_global_elite_pool').resolve() == (target_arm / 'global_phase/global_elite_pool').resolve()


def materialization_fixture(tmp_path, monkeypatch):
    """Real validator and materializer, with small complete train-val fixtures."""
    from utils.teh.pics_v4 import PICS_V4_TRIAL_POLICY
    target_root = tmp_path / 'target/pics_v4_target_only/job_1'
    transfer_root = tmp_path / 'target/pics_v4_transfer_based_only/job_2'
    out = tmp_path / 'target/pics_v4_official_gate/job_3'
    expected = identity('official_gate')
    transfer_identity = dict(expected, mode='transfer_based_only', kind='pics_v4_transfer_based_only')
    provs = []
    for root, mode, role, bound in ((target_root, 'target_only', 'control', None), (transfer_root, 'transfer_based_only', 'transfer', transfer_identity)):
        arm = root / 'target_population' / role
        best = arm / 'global_phase/best_program.py'
        best.parent.mkdir(parents=True)
        best.write_text('def choose(problem, history):\n    return 0.5\n')
        pool = best.parent / 'global_elite_pool'
        pool.mkdir()
        (pool / '000_global_baseline.py').write_bytes(best.read_bytes())
        dump(pool / 'pool_manifest.json', {'programs': [{'rank': 0, 'program_id': 'global_baseline', 'filename': '000_global_baseline.py', 'global_fitness': -1.}]})
        dump(best.parent / 'results.json', {'pool_best_program_id': 'global_baseline', 'pool_best_selection_score': -1., 'evolution_selection_score': 'train_val'})
        marker = pics_v4_marker_payload()
        if bound:
            marker['run_identity'] = bound
        dump(root / 'TRIAL_PROMPT_POLICY.json', marker)
        prov = dict(pics_v4_marker_payload(), dataset='target', kind=adapter.KINDS[mode], track_mode=mode, arm_role=role,
            global_iters=10, n_candidates=10, runtime_valid=True, rank1_program=str(best.resolve()), rank1_sha256=adapter.sha(best), rank1_program_id='global_baseline', pooled_train_val_score=-1., test_used_for_selection=False,
            participant_ids=[0], range_start_ordinal=0, range_end_ordinal=0, prompt_mode='pics_v4_registered', rendered_prompt_sha256='base', sa40_fingerprint='sa40', observed_train_fingerprint='train', observed_val_fingerprint='val', limited_data_protocol='structure_aware_v3', limited_train_val=40, model_name='Qwen/Qwen2.5-Coder-32B-Instruct', llm_max_tokens=1024, max_parent_chars=5000, vllm_max_model_len=16384)
        prov['prompt_policy'] = prov['trial_prompt_policy'] = PICS_V4_TRIAL_POLICY
        if bound:
            prov['run_identity'] = bound
            dump(arm / 'RUN_IDENTITY.json', {'run_identity': bound, 'arm': role, 'suffix_sha256': bound['source_suffix_sha256']})
            bank = best.parent / 'pics_v4_panel_banks/population_fresh.json'
            dump(bank, {'fixture': True})
            dump(bank.parent / 'ATTESTATION.json', {bank.name: adapter.sha(bank)})
            prov['panel_bank_sha256'] = {str(bank.relative_to(arm)): adapter.sha(bank)}
        dump(arm / 'POPULATION_PROVENANCE.json', prov)
        dump(arm / 'STAGE_COMPLETE.json', prov)
        provs.append(prov)
    monkeypatch.setattr(adapter, 'target_run', lambda target: target_root)
    adapter.configure_identity(expected)
    return target_root, transfer_root, out, provs


@pytest.mark.parametrize('damage', ['missing_completion', 'rank1_bytes', 'source_identity', 'test_selection', 'cohort', 'panel_bytes', 'policy', 'program_id'])
def test_real_materializer_fails_closed_before_callback(tmp_path, monkeypatch, damage):
    target, transfer, out, provs = materialization_fixture(tmp_path, monkeypatch)
    arm = transfer / 'target_population/transfer'
    if damage == 'missing_completion':
        (arm / 'STAGE_COMPLETE.json').unlink()
    elif damage == 'rank1_bytes':
        (arm / 'global_phase/best_program.py').write_text('def choose(problem, history):\n    return 0.4\n')
    elif damage == 'panel_bytes':
        dump(arm / 'global_phase/pics_v4_panel_banks/population_fresh.json', {'tampered': True})
    else:
        changed = adapter.read(arm / 'POPULATION_PROVENANCE.json')
        if damage == 'source_identity':
            changed['run_identity']['source']['source_program_sha256'] = 'wrong'
        elif damage == 'test_selection':
            changed['test_used_for_selection'] = True
        elif damage == 'cohort':
            changed['observed_train_fingerprint'] = 'wrong'
        elif damage == 'policy':
            changed['panel_policy'] = 'old'
        else:
            changed['rank1_program_id'] = 'wrong'
        dump(arm / 'POPULATION_PROVENANCE.json', changed)
    calls = []
    with pytest.raises((RuntimeError, FileNotFoundError)):
        materialize_official_gate(target_only_run=target, transfer_run=transfer, output_dir=out, target_dataset='target', evaluate_score=lambda *args: calls.append(args))
    assert not calls and not out.exists()


def test_real_materializer_tie_retains_target_without_rewriting_inputs(tmp_path, monkeypatch):
    target, transfer, out, _ = materialization_fixture(tmp_path, monkeypatch)
    hashes = {p: adapter.sha(p) for root in (target, transfer) for p in root.rglob('*') if p.is_file()}
    result = materialize_official_gate(target_only_run=target, transfer_run=transfer, output_dir=out, target_dataset='target', evaluate_score=lambda *args: -1.)
    assert result['selected_arm'] == 'target_only'
    assert result['test_used_for_gate'] is False
    assert all(adapter.sha(p) == h for p, h in hashes.items())


def test_participant_identity_rejects_copied_target_results(tmp_path):
    expected = identity()
    expected['cohort']['participant_ids'] = [0]
    adapter.configure_identity(expected)
    path = tmp_path / 'selected/participant_0'
    dump(path / 'results.json', {'target_only_fixture': True})
    with pytest.raises(RuntimeError, match='missing transfer identity'):
        adapter.initialize_participant(path, 0)
    (path / 'results.json').unlink()
    adapter.initialize_participant(path, 0)
    adapter.assert_participant_resume(path, 0)
    dump(path / 'RUN_IDENTITY.json', {'run_identity': identity('official_gate'), 'participant_id': 0})
    with pytest.raises(RuntimeError, match='mode/source'):
        adapter.assert_participant_resume(path, 0)


def test_v4_source_example_uses_sparse_snapshot_and_never_test(monkeypatch):
    from utils.teh.explore_source_prompt import _load_source_example_trials
    from utils.teh_transfer.prompts import one_example_trial_text
    from utils.teh import participant_ids
    from baseline_methods import MLE
    monkeypatch.setattr(participant_ids, 'load_valid_participant_ids', lambda *a, **k: [123])
    trial = {'problem': {'dataset_alias': '3frey2017cct', 'schema_type': 'cct'}, 'history': [{'action': 0, 'feedback': 1} for _ in range(30)], 'action': 1}
    monkeypatch.setattr(MLE, 'trials_for_participant', lambda *a, **k: ([trial], [], [{'passive_test_secret': 'NEVER_INCLUDE'}]))
    trials = _load_source_example_trials('3frey2017cct', split_seed=0, psych_dataset_split='train', limited_data_protocol='structure_aware_v3', limited_train_val=40, require=True)
    text = one_example_trial_text(trials, seed=0)
    assert 'participant=123' in text
    assert 'displayed_history_entries=8' in text
    assert 'actual_runtime_history_length=30' in text
    assert 'NEVER_INCLUDE' not in text


def test_historical_source_example_serializer_unchanged():
    from utils.teh_transfer.prompts import one_example_trial_text
    from data_modules.psych101_binary import format_trials_for_prompt
    trial = {'problem': {'dataset_alias': '3frey2017cct', 'schema_type': 'cct'}, 'history': [], 'action': 1}
    configure_pics_v4(False)
    assert one_example_trial_text([trial], seed=0) == format_trials_for_prompt([trial], max_trials=1)


def test_frozen_mixed_gambles_cohort_does_not_enable_filter(monkeypatch):
    from baseline_methods import MLE
    seen = []
    def trials(*args, **kwargs):
        seen.append(kwargs['filter_mixed_gambles'])
        return [], [], []
    monkeypatch.setattr(MLE, 'trials_for_participant', trials)
    assert adapter.observed_cohort('mixed_gambles', [101]) == ([], [])
    assert seen == [False]


def test_frozen_mixed_gambles_filter_override_rejected():
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match='filter_mixed_gambles=false'):
        adapter.prepare_identity(SimpleNamespace(dataset='mixed_gambles', pics_aamas_v0_track_mode='transfer_based_only', filter_mixed_gambles=True), suffix='')
