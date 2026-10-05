"""CPU preflight of the existing v4 builders; never invokes a model or scheduler.

Production files and caches are read-only. Temporary local Arrow derivatives
avoid Hugging Face builder locks. Explicit reports belong in the dated analysis
tree; without --report-dir no persistent file is written.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from utils.teh import pics_v4_transfer as transfer


def install_read_only_guard(report_dir=None):
    allowed = [Path('/tmp')]
    if report_dir:
        root = (transfer.REPO / 'analysis_2026Sep/Codex/Oct5/others').resolve()
        destination = Path(report_dir).resolve()
        if root not in destination.parents:
            raise RuntimeError('reports must be in the dated auxiliary analysis tree')
        allowed.append(destination)

    def writable(p):
        if isinstance(p, int) or str(p) == '/dev/null':
            return True
        path = Path(p).resolve()
        return any(path == root or root in path.parents for root in allowed)

    def guard(event, args):
        if event == 'open':
            path, mode, flags = args
            writing = (mode and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writing and not writable(path):
                raise PermissionError(f'preflight denied production/cache write: {path}')
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.rename', 'os.symlink', 'os.link'):
            path = args[0]
            if event in ('os.remove', 'os.rmdir') and len(args) > 1 and isinstance(args[1], int) and args[1] >= 0:
                path = Path(os.readlink(f'/proc/self/fd/{args[1]}')) / str(path)
            if not writable(path) or (event in ('os.rename', 'os.symlink', 'os.link') and not writable(args[1])):
                raise PermissionError(f'preflight denied mutation: {event}: {path}')
        if event in ('socket.connect', 'subprocess.Popen', 'os.system'):
            raise PermissionError(f'preflight denied external execution: {event}')
    sys.addaudithook(guard)


def local_snapshot(directory):
    import datasets
    datasets.disable_progress_bars()
    from data_modules.psych101_binary import experiment_id_for_alias, is_psych101_dataset
    datasets.disable_caching()
    cache = Path('/careAIDrive/zichang/cache/huggingface/datasets/marcelbinz___psych-101/default/0.0.0/611565c66395e2787cd7e3305149bb75dc138024')
    files = sorted(cache.glob('psych-101-train-*.arrow'))
    if len(files) != 2:
        raise RuntimeError('preflight needs both existing offline Psych-101 train shards')
    data = datasets.concatenate_datasets([datasets.Dataset.from_file(str(p)) for p in files])
    aliases = transfer.official_inputs()[0]
    wanted = [str(experiment_id_for_alias(t)) for t in aliases if is_psych101_dataset(t)]
    indices = [i for i, e in enumerate(data['experiment']) if any(str(e).startswith(w) if w.endswith('/') else str(e) == w for w in wanted)]
    destination = str(Path(directory) / 'Psych-101')
    datasets.DatasetDict({'train': data.select(indices)}).save_to_disk(destination)
    return destination, {str(p): transfer.sha(p) for p in files}


def method_args(target, mode, local_dataset):
    _, artifacts = transfer.official_inputs()
    root = transfer.target_run(target)
    prov = transfer.read(root / 'target_population/control/POPULATION_PROVENANCE.json')
    return SimpleNamespace(dataset=target, pics_aamas_v0_track_mode=mode,
        t_pics_source_config=str(transfer.MAP), split_seed=0, pics_run_seed=0,
        global_iters=10, n_candidates=10, hard_prompt_token_cap=15360,
        llm_max_tokens=1024, max_parent_chars=5000, limited_data_protocol='structure_aware_v3',
        limited_train_val=40, evolution_selection_score='train_val', local_dataset=local_dataset,
        range_start_ordinal=prov['range_start_ordinal'], range_end_ordinal=prov['range_end_ordinal'])


class RenderClient:
    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.prompts.append(kwargs['messages'][0]['content'])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='def choose(problem, history):\n    return min(0.99, max(0.01, float(0.5)))\n'))])


def check_target(target, mode, local_dataset, *, output_root=None, transfer_run=None):
    import teh
    from utils.teh.explore_source_prompt import build_rank1_explore_prompt_suffix
    from utils.teh.pics_v4_panels import clear_pics_v4_panel_state, registered_panel_bank
    from utils.teh.pics_v4 import assert_pics_v4_resume
    from utils.teh.prompt_units import qwen_user_prompt_token_count
    from utils.teh.teh_runtime import setup_teh_run_prompts
    transfer.configure_identity(None)
    record = transfer.resolve_source(target)
    suffix = build_rank1_explore_prompt_suffix(source_dataset=record['source_dataset'], program_path=record['source_program_path'], split_seed=0, psych_dataset_split='train', best_loglik=None, split_ratio=.6, limited_data_protocol='structure_aware_v3', limited_train_val=40, require_source_examples=True, local_dataset=local_dataset, filter_mixed_gambles=record['source_dataset'] == 'mixed_gambles')
    args = method_args(target, mode, local_dataset)
    prov = transfer.read(transfer.target_run(target) / 'target_population/control/POPULATION_PROVENANCE.json')
    tr, va = transfer.observed_cohort(target, prov['participant_ids'], local_dataset=local_dataset)
    identity = transfer.prepare_identity(args, suffix=suffix, observed_trials=(tr, va))
    if output_root:
        assert_pics_v4_resume(output_root)
        for role in ('control', 'transfer'):
            transfer.assert_arm_resume(Path(output_root) / 'target_population' / role, role=role)
    if transfer_run:
        if mode != 'official_gate' or not output_root:
            raise RuntimeError('materialization preflight requires official_gate and --output-root')
        transfer.validate_materialization(transfer.target_run(target) / 'target_population/control', Path(transfer_run) / 'target_population/transfer', target, Path(output_root))
    root = transfer.target_run(target)
    seed = transfer.REPO / 'persona_code_example/te_vanilla/choices13k.py'
    seed_code = teh.load_seed_program(str(seed))
    rendered, counts, banks = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='pics_v4_dry_prompts_') as temporary:
        prompts = setup_teh_run_prompts(Path(temporary), target, seed, client=None, use_llm=False, prefer_auto_llm_prompt=False, require_auto_llm_prompt=False, split_seed=0, limited_data_protocol='structure_aware_v3', limited_train_val=40, pics_aamas_v0_prompt=True, local_dataset=local_dataset, filter_mixed_gambles=False)
        if transfer.sha(prompts / 'infer_single_choice.txt') != identity['base_prompt_sha256']:
            raise RuntimeError('current registered uniform-v8 prompt differs from frozen target-only prompt')
        # Actual future transfer iteration-1 elites do not exist. Use the frozen
        # target bank's initial unique elite solely to exercise the exact mean
        # parent envelope; no fitness/source selector is fitted here.
        old_bank = transfer.read(root / 'target_population/control/global_phase/pics_v4_panel_banks/population_parent_conditioned.json')
        elites = teh._pics_v4_elite_from_snapshot(old_bank['unique_initial_elite'])
        for arm in (('transfer',) if mode == 'transfer_based_only' else ('control', 'transfer')):
            clear_pics_v4_panel_state()
            actual_suffix = suffix if arm == 'transfer' else None
            common = dict(dataset=target, trials=tr + va, n_slots=10, run_prompts_dir=str(prompts), output_dir=None, fitness_metric='loglik', master_seed=0, cap=15360, prompt_suffix=actual_suffix)
            teh._ensure_pics_v4_population_fresh_bank(seed_code=seed_code, **common)
            teh._ensure_pics_v4_population_parent_bank(elite_parents=elites, sample_size=int(old_bank['sample_size']), max_parent_chars=5000, **common)
            for role, parents in (('fresh', [seed_code]), ('parent', [elites[0][0]])):
                client = RenderClient()
                codes = teh.generate_program_variants(client=client, model_name='Qwen/Qwen2.5-Coder-32B-Instruct', parent_programs=parents, train_trials=tr, extra_prompt_trials=va, n_variants=10, max_tokens=1024, dataset=target, fitness_metric='loglik', max_workers=1, hard_prompt_token_cap=15360, prompt_token_estimator='qwen_chat', run_prompts_dir=str(prompts), phase='global_evolution', iteration=1 if role == 'fresh' else 2, prompt_suffix=actual_suffix, g2_arm=arm, max_parent_chars=5000, pics_run_seed=0, pics_lossless_generation_role=role, max_error_prompt_chars=0, error_feedback_mode='legacy', llm_decoding_seed_base=60001)
                if len(client.prompts) != 10 or not all(codes):
                    raise RuntimeError('actual candidate renderer did not produce all ten absolute slots')
                tokens = [qwen_user_prompt_token_count(p) for p in client.prompts]
                if max(tokens) > 15360 or max(tokens) + 1024 > 16384:
                    raise RuntimeError('Qwen chat-template context budget exceeded')
                for p in client.prompts:
                    if ('## Cross-task transfer context' in p) != (arm == 'transfer'):
                        raise RuntimeError('source injection/control isolation failed')
                    if arm == 'transfer' and Path(record['source_program_path']).read_text().strip() not in p:
                        raise RuntimeError('evaluated source program is absent from actual prompt')
                key = f'{arm}_{role}'
                counts[key] = tokens
                rendered[key] = client.prompts[0]
                bank = registered_panel_bank(dataset=target, phase='population', participant_id=None, name='fresh' if role == 'fresh' else 'parent_conditioned')
                banks[key] = {'fingerprint': bank.fingerprint, 'n_stream': len(bank.stream), 'slot_sizes': [len(p) for p in bank.panels]}
    return {'target': target, 'mode': mode, 'run_identity': identity, 'input_token_counts': counts, 'banks': banks, 'maximum_input_tokens': max(max(v) for v in counts.values()), 'materialization': 'verified_completed_inputs' if transfer_run else 'awaiting_future_completed_transfer' if mode == 'official_gate' else 'not_applicable', 'parent_envelope_fixture': 'frozen target-only initial unique elite; future transfer elites unknown'}, rendered


def verify_report(path, *, mode, target=None, transfer_run=None, output_root=None):
    """Lightweight launch check of CPU-produced certificates; no rerendering."""
    path = Path(path)
    report = transfer.read(path)
    sources, _ = transfer.official_inputs()
    import importlib.metadata
    for package, version in report['package_versions'].items():
        actual = sys.version if package == 'python' else importlib.metadata.version(package)
        if actual != version:
            raise RuntimeError(f'CPU preflight package version changed: {package}')
    if report.get('source_map_sha256') != transfer.MAP_SHA or report.get('model_calls') != 0:
        raise RuntimeError('invalid CPU dry-render certificate')
    rows = {r['target']: r for r in report['rows']}
    wanted = [target] if target else sorted(sources)
    for name in wanted:
        row = rows[name]
        bound = row['run_identity']
        if row['mode'] != mode or bound['kind'] != transfer.KINDS[mode] or bound['source_artifacts_sha256'] != transfer.ARTIFACTS_SHA:
            raise RuntimeError('CPU preflight mode/artifact certificate mismatch')
        if bound['source'] != transfer.resolve_source(name):
            raise RuntimeError('CPU preflight evaluated source identity changed')
        transfer.target_run(name)
        for rel, expected_hash in bound['method_code_sha256'].items():
            if transfer.sha(transfer.REPO / rel) != expected_hash:
                raise RuntimeError('method code changed after CPU preflight')
        if row['maximum_input_tokens'] > 15360:
            raise RuntimeError('CPU preflight token limit mismatch')
        if transfer_run:
            if not target or mode != 'official_gate' or not output_root:
                raise RuntimeError('materialization certificate needs a single gate target/output')
            transfer.configure_identity(bound)
            transfer.validate_materialization(transfer.target_run(name) / 'target_population/control', Path(transfer_run) / 'target_population/transfer', name, Path(output_root))
    for rel, expected_hash in report['render_sha256'].items():
        if transfer.sha(path.parent / rel) != expected_hash:
            raise RuntimeError('CPU render certificate output SHA mismatch')
    for rel, expected_hash in report['input_hashes'].items():
        if transfer.sha(Path(rel)) != expected_hash:
            raise RuntimeError('CPU preflight dataset input SHA mismatch')
    print(json.dumps({'status': 'CPU_certificate_verified', 'mode': mode, 'targets': wanted, 'model_calls': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('transfer_based_only', 'official_gate'), default='transfer_based_only')
    parser.add_argument('--target')
    parser.add_argument('--output-root')
    parser.add_argument('--transfer-run')
    parser.add_argument('--report-dir')
    parser.add_argument('--verify-report', help='Verify an existing CPU certificate without rendering or model calls')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HOME='/careAIDrive/zichang/cache/huggingface')
    sys.dont_write_bytecode = True
    if args.verify_report:
        from utils.teh.pics_v4 import configure_pics_v4
        from utils.teh.aamas_v0_lossless_trials import set_pics_run_seeds
        configure_pics_v4(True)
        set_pics_run_seeds(run_seed=0, split_seed=0)
        verify_report(args.verify_report, mode=args.mode, target=args.target, transfer_run=args.transfer_run, output_root=args.output_root)
        return
    from utils.teh import prompt_units
    snapshot = prompt_units._local_qwen_snapshot()
    if snapshot is None:
        raise RuntimeError('existing Qwen tokenizer snapshot is required')
    prompt_units._QWEN_TOKENIZER = prompt_units._LocalQwenTokenizer(snapshot)
    if args.report_dir:
        destination = Path(args.report_dir).resolve()
        allowed_root = (transfer.REPO / 'analysis_2026Sep/Codex/Oct5/others').resolve()
        if allowed_root not in destination.parents:
            raise RuntimeError('report directory must be under the dated auxiliary tree')
        destination.mkdir(parents=True, exist_ok=True)
    install_read_only_guard(args.report_dir)
    from utils.teh.pics_v4 import configure_pics_v4
    from utils.teh.pics_aamas_v0 import configure_pics_aamas_v0_prompt
    from utils.teh.aamas_v0_lossless_trials import set_pics_run_seeds
    configure_pics_v4(True)
    configure_pics_aamas_v0_prompt(True)
    set_pics_run_seeds(run_seed=0, split_seed=0)
    targets = [args.target] if args.target else sorted(transfer.official_inputs()[0])
    rows = []
    render_hashes = {}
    with tempfile.TemporaryDirectory(prefix='pics_v4_read_only_data_') as temporary:
        with contextlib.redirect_stdout(io.StringIO()):
            local, input_hashes = local_snapshot(temporary)
        for target in targets:
            with contextlib.redirect_stdout(io.StringIO()):
                result, renders = check_target(target, args.mode, local, output_root=args.output_root, transfer_run=args.transfer_run)
            rows.append(result)
            print(json.dumps({'target': target, 'mode': args.mode, 'maximum_input_tokens': result['maximum_input_tokens'], 'status': 'passed'}), flush=True)
            if args.report_dir:
                out = Path(args.report_dir)
                out.mkdir(parents=True, exist_ok=True)
                for key, prompt in renders.items():
                    path = out / f'{target}_{key}.txt'
                    path.write_text(prompt)
                    render_hashes[path.name] = transfer.sha(path)
    import importlib.metadata
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'datasets', 'transformers', 'tokenizers', 'huggingface_hub')}
    versions['python'] = sys.version
    code_hashes = rows[0]['run_identity']['method_code_sha256']
    if any(r['run_identity']['method_code_sha256'] != code_hashes for r in rows) or any(transfer.sha(transfer.REPO / p) != h for p, h in code_hashes.items()):
        raise RuntimeError('method code changed during preflight; repeat on a fixed code version')
    report = {'package_versions': versions, 'render_sha256': render_hashes, 'tokenizer_input_sha256': {str(p): transfer.sha(p) for p in Path(snapshot).glob('*') if p.is_file()}, 'schema': 'pics_v4_three_mode_preflight_v1', 'jobs_submitted': 0, 'model_calls': 0, 'source_map_sha256': transfer.MAP_SHA, 'rows': rows, 'input_hashes': input_hashes, 'maximum_input_tokens': max(r['maximum_input_tokens'] for r in rows)}
    if args.report_dir:
        out = Path(args.report_dir)
        (out / 'PREFLIGHT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'targets_checked': len(rows), 'maximum_input_tokens': report['maximum_input_tokens'], 'jobs_submitted': 0, 'model_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
