from types import SimpleNamespace
import json
import pytest
from resume_pics_v4 import install_saved_candidate_replay


def setup_stage(tmp_path):
    root=tmp_path/'job_311671';part=root/'selected/participant_0';stage=part/'iteration_1'
    (stage/'candidates').mkdir(parents=True)
    (stage/'metrics.json').write_text(json.dumps({'n_candidates':2,'candidate_results':[{'idx':0},{'idx':1}],'candidate_sources':['fresh','normal']}))
    for i in range(2):(stage/'candidates'/f'candidate_{i}.py').write_text(f'def choose(p,h): return {i/2}')
    row={'record_type':'candidate','phase':'evolution','iteration':1,'candidate_id':'iteration_1_candidate_0','selection_score':-0.4}
    trace=part/'mem_trace.jsonl';trace.write_text(json.dumps(row)+'\n')
    calls=[]
    def live(*args,**kwargs):calls.append(kwargs);return 'live'
    module=SimpleNamespace(generate_program_variants=live,_generate_iteration_candidate_codes=live,append_mem_trace_record=live)
    install_saved_candidate_replay(module,root)
    return module,stage,trace,row,calls


def test_saved_candidates_skip_llm_and_preserve_order(tmp_path):
    module,stage,*_=setup_stage(tmp_path)
    codes,sources=module._generate_iteration_candidate_codes(n_candidates=2,fresh_n_candidates=1,variant_kwargs={'prompt_stats_path':stage/'prompt_stats.json'})
    assert len(codes)==2 and 'return 0.0' in codes[0] and sources==['fresh','normal']


def test_schedule_change_refused(tmp_path):
    module,stage,*_=setup_stage(tmp_path)
    with pytest.raises(RuntimeError,match='schedule'):
        module._generate_iteration_candidate_codes(n_candidates=2,fresh_n_candidates=2,variant_kwargs={'prompt_stats_path':stage/'prompt_stats.json'})


def test_trace_replay_deduplicates_and_rejects_changed_score(tmp_path):
    module,_,trace,row,calls=setup_stage(tmp_path)
    module.append_mem_trace_record(trace,row)
    assert calls==[]
    with pytest.raises(RuntimeError,match='mismatch'):
        module.append_mem_trace_record(trace,dict(row,selection_score=-0.3))


def test_unfinished_iteration_uses_live_generation(tmp_path):
    module,stage,_,_,calls=setup_stage(tmp_path)
    assert module._generate_iteration_candidate_codes(variant_kwargs={'prompt_stats_path':stage.parent/'iteration_2/prompt_stats.json'})=='live'
    assert len(calls)==1


def test_nested_score_roundoff_keeps_parent_identity_and_order():
    from resume_pics_v4 import equivalent
    assert equivalent([{'program_id':'a','val_loglik':-0.7661052516995354}], [{'program_id':'a','val_loglik':-0.7661052516995353}])
    assert not equivalent([{'program_id':'a','val_loglik':-0.76}], [{'program_id':'b','val_loglik':-0.76}])
