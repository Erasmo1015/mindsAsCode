"""Explicit selected-ledger plumbing around the unchanged uniform-v8 builder.

Hooks are scoped to one call and restored afterwards. Historical entrypoints,
scientific rules, and their immutable hashes remain unchanged.
"""
from contextlib import contextmanager
from collections import Counter
import json
from pathlib import Path

from utils.mem.pics_v4_selected_scope import SelectedScope, digest, strict_rows
from utils.mem.pics_v4_uniform_v8_reference import REFERENCE_POLICY, PicsV4ReferenceError

LEDGER_PINS = (
    ('selected_track_ledger.json', 'd8bf646b988509eeca98727b0062b96083474ea9458516f1e204e7a29a3afaba'),
    ('selected_track_ledger_rest5.json', 'd51410e144ab479110ad83ff469396d54d1b57497db8ef98b1c77309699bebe2'),
)
PROVENANCE_SHA = '79fd839b85f0ca3838eb4f1a441ce8c673187c07cffae53d196f7c5dbf867dcf'
DATASETS = ('1peterson2021using', '2plonsky2018when', '3frey2017cct',
            '4wulff2018description', 'mixed_gambles', '7hilbig2014generalized',
            'bergert_nosofsky_2007', '11enkavi2019recentprobes',
            '5speekenbrink2008learning', '12badham2017deficits', '10frey2017risk',
            'guan_2020_stopping', 'steyvers_2009_bandit', '13schulz2020finding',
            '14kool2016when')
SLICES = {'exploration': ('explore', 'explore'),
          'evolution_normal': ('evolution', 'normal'),
          'evolution_fresh': ('evolution', 'fresh')}


def validate_union(ledgers):
    """Reject partial, overlapping, conflicting, or non-selected identities."""
    datasets, events, identities, roots = {}, set(), set(), set()
    for path, pin, ledger in ledgers:
        if ledger['official_files']['provenance.json']['sha256'] != PROVENANCE_SHA:
            raise PicsV4ReferenceError('selected snapshot mismatch')
        for meta in ledger['datasets']:
            ds = meta['dataset']
            if ds in datasets or ds not in DATASETS or not meta['annotation_ready']:
                raise PicsV4ReferenceError('overlapping/nonofficial/uncertified dataset')
            root = Path(meta['selected_root']).resolve()
            if meta['chosen_track'] not in ('target_only', 'transfer_based_only'):
                raise PicsV4ReferenceError('unsupported selected track')
            if root.name != 'selected' or root.parent.parent.name != 'pics_v4_' + meta['chosen_track'] or root.parent.name != meta['canonical_output_job']:
                raise PicsV4ReferenceError('selected root/track/job conflict')
            if root in roots:
                raise PicsV4ReferenceError('overlapping selected root')
            roots.add(root)
            for row in meta['transitions']:
                key = tuple(row['event_key'])
                if key in events or key[0] != ds:
                    raise PicsV4ReferenceError('duplicate/conflicting canonical event')
                events.add(key)
                if row['status'] == 'eligible':
                    full = tuple(row['uniform_v8_resume_key'])
                    if full in identities:
                        raise PicsV4ReferenceError('duplicate canonical SHA identity')
                    identities.add(full)
            datasets[ds] = (path, pin, ledger, meta)
    if set(datasets) != set(DATASETS):
        raise PicsV4ReferenceError('selected union must cover exactly all 15 datasets')
    return datasets


def load_union(repo):
    base = Path(repo) / 'analysis_2026Sep/Codex/Oct6/others/selected_participant_provenance'
    ledgers = []
    for name, pin in LEDGER_PINS:
        path = base / name
        data = path.read_bytes()
        if digest(data) != pin:
            raise PicsV4ReferenceError('selected ledger SHA mismatch')
        ledgers.append((path, pin, json.loads(data)))
    return validate_union(ledgers)


@contextmanager
def selected_builder_context(repo, scopes, traces):
    """Narrow build hooks; selected resolver retains canonical SHA join logic."""
    from analysis.mem import build_dataset as builder
    from utils.mem import pics_v4_participant_join as join
    old_guard, old_resolver = builder._guard_official_participant_build, join.resolve_uniform_v8_reference
    expected = {Path(p).resolve() for p in traces}
    root = Path(repo).resolve()

    def guard(run_dir, trace_files, reference_policy):
        if Path(run_dir).resolve() != root:
            return old_guard(run_dir, trace_files, reference_policy)
        if reference_policy != REFERENCE_POLICY or {Path(p).resolve() for p in trace_files} != expected:
            raise PicsV4ReferenceError('selected build scope/trace list mismatch')

    def resolver(record, *, participant_dir, **kwargs):
        parent = Path(participant_dir).resolve().parent
        scope = scopes.get(parent)
        if scope is None:
            raise PicsV4ReferenceError('losing/non-selected build root refused')
        return scope.resolve(record, participant_dir=participant_dir, **kwargs)

    builder._guard_official_participant_build = guard
    join.resolve_uniform_v8_reference = resolver
    try:
        yield builder
    finally:
        builder._guard_official_participant_build = old_guard
        join.resolve_uniform_v8_reference = old_resolver


def build_selected_slice(repo, slice_name):
    """Canonical complete selected-scope build; no file written before success."""
    from utils.mem.pics_v4_frozen import check_frozen
    from utils.mem.pics_v4_participant_join import V8AnnotationIndex
    from utils.mem.participant_semantic_postprocess_v5 import postprocess_participant_annotation
    check_frozen(Path(repo))
    union = load_union(repo)
    phase, source = SLICES[slice_name]
    scopes, trace_files, annotations, row_flow = {}, [], [], []
    correction_counts = Counter()
    for dataset in DATASETS:
        path, pin, ledger, meta = union[dataset]
        committed = [r for r in meta['transitions'] if (r['phase'], r['source']) == (phase, source)]
        eligible = [r for r in committed if r['status'] == 'eligible']
        flow = {'dataset': dataset, 'committed': len(committed), 'pre_annotation_excluded': len(committed)-len(eligible),
                'exclusions_by_reason': dict(Counter(r.get('exclusion_reason') or r['status'] for r in committed if r['status'] != 'eligible')),
                'verified_annotations': len(eligible), 'empty_eligible_slice': not eligible}
        row_flow.append(flow)
        # Empty Badham exploration is represented by the ledger, not a fake annotation.
        if not eligible:
            if dataset != '12badham2017deficits' or slice_name != 'exploration':
                raise PicsV4ReferenceError('unexpected empty certified slice')
            continue
        scope = SelectedScope(path, pin, dataset, phase, source)
        scopes[scope.root] = scope
        trace_files.extend(Path(t['path']) for t in meta['traces'])
        output = Path(ledger['annotation_output_base']) / PROVENANCE_SHA / dataset / slice_name
        ann_path = output / 'annotations_v5.jsonl'
        if scope.completed(ann_path) != set(scope.keys):
            raise PicsV4ReferenceError('incomplete selected annotation shard')
        if strict_rows(output / 'annotation_failures.jsonl'):
            raise PicsV4ReferenceError('recorded annotation failures')
        for ann in strict_rows(ann_path):
            canonical = tuple(ann['uniform_v8_resume_key'])
            certified = scope.keys[canonical]
            ref = Path(certified['reference_artifact']['path']).read_text()
            cand = Path(certified['candidate_artifact']['path']).read_text()
            # Existing deterministic postprocessor only; original raw journal stays preserved.
            original_labels = {**ann, **ann['raw_llm_annotation']}
            fixed, corrections, _ = postprocess_participant_annotation(original_labels, reference_code=ref, candidate_code=cand, dataset=dataset)
            fixed['raw_llm_annotation'] = ann['raw_llm_annotation']
            for correction in corrections:
                correction_counts[correction.rule_id] += 1
            annotations.append(fixed)
    index = V8AnnotationIndex(annotations)
    audit = {}
    with selected_builder_context(repo, scopes, trace_files) as builder:
        rows, exclusions = builder.build_rows_v5(run_dir=Path(repo), annotations=index,
            reference_policy=REFERENCE_POLICY, phase=phase, source=source,
            trace_files=trace_files, join_audit=audit)
    if len(rows) != len(annotations):
        raise PicsV4ReferenceError('selected build lost a canonical eligible event')
    for row in rows:
        key = tuple(json.loads(row['uniform_v8_resume_key']))
        ann = index[key]
        row.update({field: json.dumps(ann[field]) if isinstance(ann[field], (list, dict)) else ann[field]
                    for field in ('official_gate_selected_key', 'official_gate_provenance_sha256',
                                  'chosen_track', 'selected_ledger_sha256', 'selected_execution_lineage')})
    for flow in row_flow:
        selected = [r for r in rows if r['dataset'] == flow['dataset']]
        unresolved = sum(bool(r['exclude_from_construct_effect_fitting']) for r in selected)
        flow.update(joined=len(selected), fit_eligible=len(selected)-unresolved,
                    excluded_from_construct_fitting=unresolved,
                    unresolved_nmc=sum(r['nmc_adjudication_status']=='unresolved' for r in selected),
                    participants=len({(r['dataset'], r['run_id'], r['participant_id']) for r in selected}))
    return rows, {'slice': slice_name, 'row_flow': row_flow, 'join': audit,
                  'builder_exclusions': dict(exclusions), 'deterministic_corrections': dict(correction_counts)}
