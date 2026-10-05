"""Recovery changes are tested on isolated fixtures, never live experiment files."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from resume_pics_v4 import CandidateReplay, install_saved_candidate_replay
from utils.teh.pics_v4_recovery import participant_directories, approved_code_hashes


def fixture_engine(calls):
    engine = SimpleNamespace()
    def parallel(n, one, **kwargs):
        return [one(i) for i in range(n)]
    def variants(**kw):
        def one(i):
            index = kw.get('pics_lossless_candidate_offset', 0) + i
            calls.append((index, kw.get('llm_decoding_seed_base', 0) + i, kw.get('parents')))
            return f'def choose(problem, history):\n    return {0.1 + index * 0.01}\n'
        return engine._parallel_generate_children(kw['n_variants'], one)
    def iteration(**kw):
        fresh = kw['fresh_n_candidates']; total = kw['n_candidates']; common = dict(kw['variant_kwargs'])
        codes = engine.generate_program_variants(**dict(common, n_variants=total, pics_lossless_candidate_offset=0))
        return codes, ['fresh'] * fresh + ['normal'] * (total - fresh)
    def trace(path, row):
        with Path(path).open('a') as stream:stream.write(json.dumps(row) + '\n')
    engine.generate_program_variants=variants;engine._parallel_generate_children=parallel
    engine._generate_iteration_candidate_codes=iteration;engine.append_mem_trace_record=trace
    return engine


@pytest.mark.parametrize('phase', ['population', 'explore', 'evolution'])
def test_partial_batch_only_missing_candidates_at_original_slots(tmp_path, phase):
    owner = tmp_path/'target_population/transfer/global_phase' if phase=='population' else tmp_path/'selected/participant_0'
    stage = owner/('explore_phase' if phase=='explore' else 'iteration_2');stage.mkdir(parents=True)
    codes=stage/'candidates';codes.mkdir(); saved=codes/'candidate_1.py';saved.write_text('saved exact bytes\n')
    before=(saved.read_bytes(),saved.stat().st_mtime_ns); calls=[];e=fixture_engine(calls);replay=CandidateReplay(e,tmp_path);replay.install()
    try:
        out=e.generate_program_variants(n_variants=4,prompt_stats_path=stage/'prompt_stats.json',phase=phase,llm_decoding_seed_base=300,parents=['frozen-parent'])
        assert out[1]=='saved exact bytes\n';assert [c[0] for c in calls]==[0,2,3]
        assert [c[1] for c in calls]==[300,302,303]
        assert (saved.read_bytes(),saved.stat().st_mtime_ns)==before
        calls.clear(); assert e.generate_program_variants(n_variants=4,prompt_stats_path=stage/'prompt_stats.json',phase=phase,llm_decoding_seed_base=300,parents=['frozen-parent'])==out
        assert not calls
    finally:replay.restore()


def test_discovery_skips_csv_files_and_bad_directory_ids(tmp_path):
    for name in ['participant_0','participant_12','participant_bad','participant_01']:(tmp_path/name).mkdir()
    (tmp_path/'participant_details_loglik.csv').write_text('report')
    (tmp_path/'participant_4').write_text('a file')
    assert {p.name for p in participant_directories(tmp_path)}=={'participant_0','participant_12'}


def test_committed_metrics_candidates_and_banks_are_not_overwritten(tmp_path):
    s=tmp_path/'selected/participant_0/iteration_1';(s/'candidates').mkdir(parents=True)
    c=s/'candidates/candidate_0.py';c.write_text('saved')
    metrics={'n_candidates':1,'candidate_sources':['fresh'],'candidate_results':[{'selection_score':-0.5}]};p=s/'metrics.json';p.write_text(json.dumps(metrics))
    bank=tmp_path/'selected/participant_0/pics_v4_panel_banks/fresh.json';bank.parent.mkdir();bank.write_text('{"cursor":5}')
    before={f:(f.read_bytes(),f.stat().st_mtime_ns)for f in [c,p,bank]};e=fixture_engine([]);r=CandidateReplay(e,tmp_path);r.install()
    try:
        c.write_text('saved');p.write_text(json.dumps(metrics,indent=2));bank.write_text('{"cursor":5}')
        assert before=={f:(f.read_bytes(),f.stat().st_mtime_ns)for f in before}
        for f,bad in [(c,'changed'),(p,'{"n_candidates":2}'),(bank,'{"cursor":6}')]:
            with pytest.raises(RuntimeError,match='mismatch'):f.write_text(bad)
        with pytest.raises(RuntimeError,match='schedule'):
            e._generate_iteration_candidate_codes(n_candidates=1,fresh_n_candidates=0,variant_kwargs={'prompt_stats_path':s/'prompt_stats.json'})
    finally:r.restore()


def test_committed_and_partial_trace_events_deduplicate_and_fail_closed(tmp_path):
    p=tmp_path/'selected/participant_0';p.mkdir(parents=True);trace=p/'mem_trace.jsonl'
    row={'record_type':'candidate','phase':'evolution','iteration':2,'candidate_id':'candidate_0','selection_score':-0.5,'selected_parents':['parent']}
    trace.write_text(json.dumps(row)+'\n');before=(trace.read_bytes(),trace.stat().st_mtime_ns);e=fixture_engine([]);r=CandidateReplay(e,tmp_path);r.install()
    try:
        e.append_mem_trace_record(trace,row);assert (trace.read_bytes(),trace.stat().st_mtime_ns)==before
        with pytest.raises(RuntimeError,match='mismatch'):e.append_mem_trace_record(trace,dict(row,selection_score=-0.4))
        new=dict(row,candidate_id='candidate_1');e.append_mem_trace_record(trace,new);e.append_mem_trace_record(trace,new)
        assert len(trace.read_text().splitlines())==2
    finally:r.restore()


def test_population_metrics_do_not_require_participant_candidate_results(tmp_path):
    stage=tmp_path/'target_population/transfer/global_phase/iteration_1';(stage/'candidates').mkdir(parents=True)
    (stage/'candidates/candidate_0.py').write_text('saved')
    (stage/'metrics.json').write_text(json.dumps({'n_candidates':1,'candidate_sources':['fresh']}))
    r=CandidateReplay(fixture_engine([]),tmp_path);assert stage.resolve() in r.stages


def test_transfer_resume_with_summary_csv(tmp_path):
    from utils.teh import pics_v4_transfer as adapter
    from utils.teh.pics_v4 import configure_pics_v4,write_pics_v4_marker,assert_pics_v4_resume
    from utils.teh_psych.test_scripts.test_pics_v4_transfer import identity
    configure_pics_v4(True);adapter.configure_identity(identity())
    root=tmp_path/'target/pics_v4_transfer_based_only/job_1';write_pics_v4_marker(root)
    p=root/'selected/participant_0';adapter.initialize_participant(p,0)
    (p.parent/'participant_details_loglik.csv').write_text('report')
    try:assert_pics_v4_resume(root)
    finally:adapter.configure_identity(None);configure_pics_v4(False)


def test_historical_phase_prefix_attestation_is_proved_not_ignored(tmp_path):
    import hashlib
    bank=tmp_path/'selected/participant_0/pics_v4_panel_banks';bank.mkdir(parents=True)
    original={'panel_bank_policy':'conditioning_aware_panel_banks_v2','continuation_policy':'within_block_carry_forward_wrap_fill_v2','parent_envelope_policy':'initial_unique_elite_mean_x_v1','exploration_parent_fingerprint':'explore'}
    explore=bank/'exploration_parent.json';explore.write_text('{}')
    old_index=(json.dumps(original,indent=2)+'\n').encode()
    (bank/'FINGERPRINTS.json').write_text(json.dumps(dict(original,evolution_fresh_fingerprint='fresh'),indent=2)+'\n')
    (bank/'evolution_fresh.json').write_text('{}')
    att=bank/'ATTESTATION.json';att.write_text(json.dumps({'exploration_parent.json':hashlib.sha256(explore.read_bytes()).hexdigest(),'FINGERPRINTS.json':hashlib.sha256(old_index).hexdigest()}))
    before=att.read_bytes();e=fixture_engine([]);r=CandidateReplay(e,tmp_path);r.install()
    try:
        actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in bank.glob('*.json')if p.name!='ATTESTATION.json'}
        att.write_text(json.dumps(actual));assert att.read_bytes()==before
        with pytest.raises(RuntimeError,match='mismatch'):att.write_text(json.dumps(dict(actual,exploration_parent='wrong')))
    finally:r.restore()
    explore.write_text('tampered')
    with pytest.raises(RuntimeError,match='SHA mismatch'):CandidateReplay(e,tmp_path)


def test_interrupted_atomic_trace_keeps_old_commits(tmp_path, monkeypatch):
    import os
    owner=tmp_path/'selected/participant_0';owner.mkdir(parents=True);path=owner/'mem_trace.jsonl'
    old={'record_type':'candidate','phase':'evolution','iteration':1,'candidate_id':'old','selection_score':None}
    path.write_text(json.dumps(old))  # Historical complete row interrupted before newline.
    before=path.read_bytes();e=fixture_engine([]);r=CandidateReplay(e,tmp_path);r.install()
    replace=os.replace
    def interrupted(*args):raise OSError('simulated preemption before commit')
    try:
        monkeypatch.setattr(os,'replace',interrupted)
        new=dict(old,candidate_id='new',selection_score=float('-inf'))
        with pytest.raises(OSError,match='preemption'):e.append_mem_trace_record(path,new)
        assert path.read_bytes()==before
        monkeypatch.setattr(os,'replace',replace);e.append_mem_trace_record(path,new)
        e.append_mem_trace_record(path,new)
        rows=[json.loads(l)for l in path.read_text().splitlines()]
        assert rows==[old,dict(new,selection_score=None)]
    finally:r.restore()
