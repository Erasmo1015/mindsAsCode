"""Secondary uniform-v8 comparators, calling the exact historical schema-v5 code."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis.mem.pics_v4_source_profiles import load_join, verify_manifest, jsonl, sha
from utils.mem.trace import record_contains_test_metrics
from utils.mem.pics_v4_source_profile_extensions_frozen import check_extensions

HISTORY = 'analysis_2026Sep/Sep20_V3/others/source_selection_schema_v5/selector_robustness/run_selector_robustness.py'
BASE = 'analysis_2026Sep/Codex/Oct5/others/population_source_profiles'
_PANEL = _SUPPORT = _DATASETS = _COLS = None


@lru_cache(maxsize=1)
def historical():
    spec = importlib.util.spec_from_file_location('uniform_v8_historical_comparators', REPO/HISTORY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._patch()
    return module


def dump(path, value):
    # Historical MixedLM diagnostics can contain NaN; represent those explicitly as null.
    def clean(v):
        if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
        return v
    Path(path).write_text(json.dumps(clean(value),indent=2,sort_keys=True,allow_nan=False)+'\n')


def panel_from_official(repo):
    live = verify_manifest(repo)
    annotations, failures, _ = load_join(repo, live)
    master = pd.read_csv(repo/BASE/'run_v1/population_master.csv').set_index('resume_key')
    traces = {}; rows = []; input_hashes = {}
    for p in live['programs']:
        key = p['resume_key']
        if not bool(master.loc[key,'sensitivity_eligible']): continue
        a = annotations[key]
        trace_path = (repo/p['code_path']).parents[2]/'mem_trace.jsonl'
        if trace_path not in traces:
            records = [r for r in jsonl(trace_path) if r.get('record_type')=='candidate']
            index = {r['candidate_id']:r for r in records}
            if len(index)!=len(records): raise ValueError('Duplicate trace candidates')
            traces[trace_path] = index
            input_hashes[str(trace_path.relative_to(repo))] = sha(trace_path)
        r = traces[trace_path][p['program_id']]
        if record_contains_test_metrics(r) or r.get('evolution_selection_score')!='train_val':
            raise ValueError('Passive test/non-train-validation selection score refused')
        if r.get('phase')!='global_evolution' or int(r['iteration'])!=p['iteration'] or r['dataset']!=p['dataset'] or r['reference_id']!=p['parent_id']:
            raise ValueError('Trace event/reference provenance mismatch')
        valid = r.get('runtime_valid') is True
        candidate = r.get('selection_score'); reference = r.get('reference_score',r.get('reference_parent_score'))
        score_ok = valid and candidate is not None and reference is not None and np.isfinite(candidate) and np.isfinite(reference)
        delta = float(candidate)-float(reference) if score_ok else None
        if score_ok and r.get('delta_f') is not None and not np.isclose(delta,r['delta_f'],atol=1e-6,rtol=1e-6):
            raise ValueError('Inconsistent stored train+validation delta')
        exact = score_ok and r.get('reference_is_exact') is True and r.get('reference_is_proxy') is not True
        row = dict(dataset=p['dataset'],run_id=p['run_id'],iteration=p['iteration'],parent_id=p['parent_id'],program_id=p['program_id'],resume_key=key,runtime_valid=valid,reference_is_exact=exact,delta_f=delta if exact else None)
        for c in historical().MOT5: row[c]=int(c in a['candidate_motif_state'])
        row.update(historical().rss.transition_flags(a))
        rows.append(row)
    panel=pd.DataFrame(rows)
    if len(panel)!=1493 or int(panel.reference_is_exact.sum())!=1383:
        raise ValueError('Final reporting cohort/score coverage changed')
    return panel,input_hashes


def _init(path,support,datasets,cols):
    global _PANEL,_SUPPORT,_DATASETS,_COLS
    _PANEL=pd.read_csv(path);_SUPPORT=support;_DATASETS=datasets;_COLS=cols


def _one(b):
    h=historical(); boot=h.boot_panel(_PANEL,np.random.default_rng(20260920+b*9973))
    fitted=h.fit_all(boot,_SUPPORT,_DATASETS,_COLS)
    return {'replicate':b,'maps':{m:fitted[m].set_index('target_id').selected_source_id.to_dict() for m in ('fitness','combined')},'fit_diagnostics':fitted['fit_meta']}


def run(output,workers):
    check_extensions(REPO)
    if output.exists(): raise ValueError('Comparator output already exists')
    panel,trace_hashes=panel_from_official(REPO)
    output.mkdir(parents=True)
    panel.to_csv(output/'analysis_panel.csv',index=False)
    h=historical();datasets=sorted(panel.dataset.unique())
    exact=panel[panel.reference_is_exact & panel.delta_f.notna()]
    support=h.rss.fitness_support_audit(exact); supported=support['supported_transitions']
    if not supported: raise ValueError('No supported historical fitness dimensions')
    dump(output/'fitness_support.json',support)
    fitted=h.fit_all(panel,supported,datasets,supported)
    cols=[c for c in fitted['fit_sig'] if c not in ('dataset','dataset_label')]
    fitted=h.fit_all(panel,supported,datasets,cols)
    if not cols: raise ValueError('No fitted fitness dimensions')
    for name in ('occ_sig','fit_sig','comb_sig'):
        fitted[name].to_csv(output/f'{name}.csv',index=False)
    dump(output/'fit_diagnostics.json',fitted['fit_meta']);dump(output/'combined_definition.json',fitted['comb_meta'])
    counts={m:{d:Counter() for d in datasets} for m in ('fitness','combined')}
    diagnostics=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_init,initargs=(str(output/'analysis_panel.csv'),supported,datasets,cols)) as pool:
        for b,result in enumerate(pool.map(_one,range(200),chunksize=1),1):
            diagnostics.append(result)
            for m,mapping in result['maps'].items():
                for d,s in mapping.items(): counts[m][d][s]+=1
            if b%10==0:
                dump(output/'bootstrap_progress.json',{'completed':b,'total':200})
                print(f'Fitness/combined bootstrap {b}/200',flush=True)
    dump(output/'bootstrap_diagnostics.json',diagnostics)
    loio=[];loio_meta=[]
    for it in range(1,11):
        result=h.fit_all(panel[panel.iteration!=it],supported,datasets,cols)
        loio_meta.append(dict(held_out_iteration=it,fit_diagnostics=result['fit_meta']))
        for m in ('fitness','combined'):
            original=fitted[m].set_index('target_id').selected_source_id.to_dict()
            for r in result[m].to_dict('records'):
                loio.append(dict(method=m,held_out_iteration=it,flipped=r['selected_source_id']!=original[r['target_id']],**r))
        print(f'Historical LOIO {it}/10',flush=True)
    loio=pd.DataFrame(loio);loio.to_csv(output/'leave_one_iteration_out.csv',index=False);dump(output/'loio_fit_diagnostics.json',loio_meta)
    tables=[]
    primary=pd.read_csv(REPO/BASE/'run_v1/primary_runtime_valid/stability.csv')
    primary['method']='frozen_primary_occurrence';primary['bootstrap_B']=1000;primary['source_allowlist']='all_15'
    tables.append(primary)
    for m in ('fitness','combined'):
        table=fitted[m].copy();table['method']=m;table['bootstrap_B']=200;table['source_allowlist']='historical_six'
        table['bootstrap_winner_frequency']=[counts[m][r.target_id][r.selected_source_id]/200 for r in table.itertuples()]
        table['bootstrap_mode']=[sorted(counts[m][d],key=lambda s:(-counts[m][d][s],s))[0] for d in table.target_id]
        table['loio_iteration_flips']=[int(loio[(loio.method==m)&(loio.target_id==d)].flipped.sum()) for d in table.target_id]
        table['loio_iteration_retention']=1-table.loio_iteration_flips/10
        table['unstable']=(table.bootstrap_winner_frequency<.7)|(table.cosine_margin<.02)|(table.loio_iteration_flips>0)
        table.to_csv(output/f'{m}_stability.csv',index=False);tables.append(table)
        dump(output/f'{m}_bootstrap_winners.json',{d:dict(c) for d,c in counts[m].items()})
    joined=pd.concat(tables,ignore_index=True)
    columns=['target_id','method','selected_source_id','bootstrap_B','source_allowlist','bootstrap_winner_frequency','loio_iteration_flips','loio_iteration_retention','cosine_margin','unstable']
    joined[columns].to_csv(output/'selector_comparison.csv',index=False)
    summary=[]
    for m,g in joined.groupby('method'):
        summary.append(dict(method=m,unstable_targets=int(g.unstable.sum()),loio_flip_targets=int((g.loio_iteration_flips>0).sum()),bootstrap_below_70=int((g.bootstrap_winner_frequency<.7).sum()),minimum_bootstrap_frequency=float(g.bootstrap_winner_frequency.min()),median_bootstrap_frequency=float(g.bootstrap_winner_frequency.median()),tight_margin_targets=int((g.cosine_margin<.02).sum())))
    dump(output/'stability_summary.json',summary)
    dump(output/'run_freeze.json',dict(command=[sys.executable,*sys.argv],cpu_job=os.environ.get('SLURM_JOB_ID'),historical_implementation=HISTORY,trace_input_hashes=trace_hashes,primary_policy_unchanged=True,counts={'canonical_events':1500,'resolved_annotations':1493,'runtime_valid_resolved':1383},bootstrap_B=200,seed=20260920,historical_source_allowlist=list(h.rss.SIX),fitness_columns=cols,output_hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}))
    print('Secondary comparator analysis complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();run(args.output.absolute(),args.workers)
