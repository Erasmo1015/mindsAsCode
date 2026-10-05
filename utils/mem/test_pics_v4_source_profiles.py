"""Uniform-v8 regression against the historical five-construct estimator."""
import numpy as np
import pandas as pd
import pytest
from analysis.mem import pics_v4_source_profiles as new
from analysis.mem.pop_v4_source_selection import run_source_selection_v4 as old
from utils.mem.pics_v4_frozen import check_frozen, guard_participant_frame


def fixture():
    rows = []
    for i, dataset in enumerate(['alpha', 'beta', 'gamma']):
        for j in range(12+i):
            rows.append(dict(dataset=dataset, iteration=j%10+1, parent_id='seed', **{c:int(j < int((12+i)*[0.0,0.4,0.9][i])) if k<4 else 0 for k,c in enumerate(new.CONSTRUCTS)}))
    return pd.DataFrame(rows)


def test_same_estimator_exact_fixture(monkeypatch):
    panel = fixture()
    monkeypatch.setattr(old, 'MOTIFS', new.CONSTRUCTS)
    historical = old.fit_occurrence_eb(panel)
    current = new.fit_profiles(panel)
    for a,b in zip(historical[:3], current[:3]):
        pd.testing.assert_frame_equal(a,b,check_exact=True)
    assert historical[3] == current[3]
    # Independent equation check guards the fixed fixture against shared regressions.
    profiles, eb, model, _ = current
    assert (model.tau2 > 0).any()
    assert (model.tau2 == 0).any()
    for c in new.CONSTRUCTS:
        g = eb[eb.motif==c]
        n = g.n_programs.to_numpy(); k = g.n_positive.to_numpy()
        p = (k+.5)/(n+1)
        y = np.log(p/(1-p)); v = 1/(n*p*(1-p))
        alpha = np.log(((k.sum()+.5)/(n.sum()+1))/(1-(k.sum()+.5)/(n.sum()+1)))
        tau = max(0, np.var(y,ddof=1)-np.mean(v))
        z = alpha+(tau/(tau+v))*(y-alpha)
        np.testing.assert_allclose(g.eb_prob,1/(1+np.exp(-z)),rtol=1e-15)


def test_exact_tie_self_excluded():
    profiles = pd.DataFrame([dict(dataset=d, **{c:.5 for c in new.CONSTRUCTS}) for d in ['zeta','alpha','beta']])
    _, selected = new.select_sources(profiles)
    assert selected.set_index('target_id').selected_source_id.to_dict() == {'alpha':'beta','beta':'alpha','zeta':'alpha'}


def test_historical_cluster_bootstrap_equal():
    p = fixture(); rng=np.random.default_rng(20260920); parts=[]
    for ds,g in p.groupby('dataset',sort=True):
        keys=g.apply(lambda r:f'{ds}|{int(r.iteration)}|{r.parent_id}',axis=1)
        chosen=rng.choice(keys.unique(),size=len(keys.unique()),replace=True)
        parts.append(pd.concat([g[keys==k] for k in chosen],ignore_index=True))
    pd.testing.assert_frame_equal(new.boot_panel(p,20260920),pd.concat(parts,ignore_index=True))


def test_frozen_config_tamper(tmp_path):
    p=tmp_path/'analysis/config/mem';p.mkdir(parents=True)
    (p/'pics_v4_uniform_v8_frozen.json').write_text('{}')
    with pytest.raises(ValueError,match='configuration changed'):
        check_frozen(tmp_path)


def test_passive_test_and_nonbinary_refused():
    p=fixture();p['test_loglik']=-1
    with pytest.raises(ValueError,match='Passive test'):
        new.fit_profiles(p)
    p=fixture();p.loc[0,'history']=2
    with pytest.raises(ValueError,match='binary'):
        new.fit_profiles(p)


def test_failed_annotation_retains_master_event():
    p=dict(dataset='alpha',run_id='job_1',source_job_id='1',iteration=1,program_id='candidate_0',parent_id='seed',resume_key='event',source='fresh',code_sha256='a',parent_code_sha256='b',code_path='a.py',parent_code_path='b.py',runtime_valid=True)
    master=new.build_master({'programs':[p]}, {}, {'event':{'error':'failed'}})
    assert len(master)==1
    assert master.iloc[0].exclusion_reason=='annotation_failure'
    assert not master.iloc[0].primary_eligible
    assert not master.iloc[0].sensitivity_eligible


@pytest.mark.parametrize('mode', ['missing','duplicate','unmatched','sha'])
def test_annotation_join_fails_closed(tmp_path, monkeypatch, mode):
    import json
    monkeypatch.setattr(new, 'SHARDS', ('official',))
    monkeypatch.setattr(new, 'annotation_row_valid', lambda r: (True,''))
    root=tmp_path/new.OUTPUT_REL/'annotations/official';root.mkdir(parents=True)
    program=dict(resume_key='key',dataset='alpha',run_id='job_1',iteration=1,
        code_sha256='a',parent_code_sha256='b',code_path='a.py',parent_code_path='b.py',
        parent_id='seed',source='fresh',reference_kind='seed_baseline',reference_type='seed_baseline',candidate_id='candidate_0')
    row=dict(program,schema_version=5)
    rows=[row]
    if mode=='missing': rows=[]
    if mode=='duplicate': rows=[row,row]
    if mode=='unmatched': row['resume_key']='other'
    if mode=='sha': row['code_sha256']='wrong'
    (root/'annotations_population_v5.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        new.load_join(tmp_path,{'programs':[program]})


def test_participant_models_closed():
    frame=pd.DataFrame({'reference_policy':['pics_v4_uniform_v8']})
    with pytest.raises(ValueError,match='final uniform-v8 coverage'):
        guard_participant_frame(frame,fitting=True)


def test_frozen_inputs_tamper(tmp_path, monkeypatch):
    import hashlib, json
    from utils.mem import pics_v4_frozen as lock
    p=tmp_path/'analysis/config/mem';p.mkdir(parents=True)
    config=json.dumps({'code_hashes':{}}).encode()
    (p/'pics_v4_uniform_v8_frozen.json').write_bytes(config)
    monkeypatch.setattr(lock,'CONFIG_SHA256',hashlib.sha256(config).hexdigest())
    frozen=json.dumps({'input_hashes':{'input.json':hashlib.sha256(b'original').hexdigest()}}).encode()
    (p/'pics_v4_uniform_v8_inputs_frozen.json').write_bytes(frozen)
    monkeypatch.setattr(lock,'INPUT_LOCK_SHA256',hashlib.sha256(frozen).hexdigest())
    (tmp_path/'input.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):
        lock.check_frozen(tmp_path)
