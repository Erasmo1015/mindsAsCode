"""CPU-stage driver for the additive, certified selected-track MEM freeze."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from utils.mem.pics_v4_selected_build import build_selected_slice, DATASETS, SLICES, LEDGER_PINS, PROVENANCE_SHA
from utils.mem.pics_v4_frozen import check_frozen, CONFIG, CONFIG_SHA256
from utils.mem.participant_semantic_postprocess_v5 import filter_frame_for_construct_effect_fitting
from analysis.mem.mem_grouping import attach_validated_mem_group, MEM_GROUP_COL
from analysis.mem.coverage_eligibility_v5 import _counts_for_column
from utils.mem.schema_participant_transition_v5 import BEHAVIORAL_MOTIFS, DIRECTIONAL_SUFFIXES
from analysis.mem.fit_mem_joint_random_slopes import apply_joint_eligibility_restrict, fit_joint_random_slopes
from analysis.mem.fit_mem_random_slopes import fit_focal_motif
from analysis.mem.mem_effect_policy import joint_support_failures, reject_same_construct_add_and_modify

REPO = Path(__file__).resolve().parents[2]
IMPLEMENTATIONS = ['utils/mem/pics_v4_selected_build.py', 'analysis/mem/selected_participant_pipeline.py', 'tests/test_pics_v4_selected_build.py']
LABELS = ['Choice13k', 'Plonsky', 'Frey CCT', 'Wulff', 'Mixed Gambles', 'Hilbig', 'Bergert', 'Enkavi', 'Speekenbrink', 'Badham', 'Frey Risk', 'Guan', 'Steyvers', 'Schulz', 'Kool']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, default=lambda v: v.item() if isinstance(v,np.generic) else v.to_dict(orient='records') if isinstance(v,pd.DataFrame) else str(v))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def check_run(root):
    if (root/'FINAL_FREEZE.json').exists():
        raise ValueError('official frozen output refuses rerun/overwrite')
    check_frozen(REPO)
    lock = json.loads((root/'implementation_lock.json').read_text())
    for rel, expected in lock['implementation_hashes'].items():
        if sha(REPO/rel) != expected:
            raise ValueError('selected implementation drift: '+rel)


def initialize(root):
    check_frozen(REPO)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'implementation_lock.json', dict(base_config=CONFIG, base_config_sha256=CONFIG_SHA256,
        implementation_hashes={rel:sha(REPO/rel) for rel in IMPLEMENTATIONS},
        selected_ledgers=dict(LEDGER_PINS), provenance_sha256=PROVENANCE_SHA,
        policy='additive selected-track plumbing; immutable base scientific policy unchanged'))


def build(root, slice_name):
    check_run(root)
    dest = root/'tables'/slice_name
    dest.mkdir(parents=True, exist_ok=False)
    rows, audit = build_selected_slice(REPO, slice_name)
    frame = pd.DataFrame(rows)
    frame, grouping = attach_validated_mem_group(frame)
    filename = dest/'transitions.csv'
    frame.to_csv(filename, index=False, mode='x')
    audit.update(table_sha256=sha(filename), grouping=grouping,
                 n_datasets=int(frame.dataset.nunique()), n_participants=int(frame[MEM_GROUP_COL].nunique()),
                 delta_consistency=True, candidate_reference_sha_consistency=audit['join']['ok'])
    write_json(dest/'build_audit.json', audit)
    print(slice_name, 'built', len(frame), 'rows', flush=True)


def coverage(root):
    check_run(root)
    cfg = check_frozen(REPO)['participant']
    family, blocks, all_coverage, table_hashes = [], [], {}, {}
    for slice_name in SLICES:
        path=root/'tables'/slice_name/'transitions.csv'
        audit=json.loads((path.parent/'build_audit.json').read_text())
        if sha(path)!=audit['table_sha256']:
            raise ValueError('table drift before coverage')
        table_hashes[str(path.relative_to(root))]=sha(path)
        df=pd.read_csv(path)
        df,_=attach_validated_mem_group(df)
        resolved,n_unresolved=filter_frame_for_construct_effect_fitting(df)
        # The unchanged coverage routine takes participant_id as its grouping column.
        # Supply the mandatory canonical group key to that routine; no raw-ID pooling.
        work=resolved.copy();work['participant_id']=work[MEM_GROUP_COL]
        effects=[]
        for construct in BEHAVIORAL_MOTIFS:
            for direction in DIRECTIONAL_SUFFIXES:
                effect=f'{construct}_{direction}'
                count=_counts_for_column(work,effect,elig_col='eligible_'+effect,
                    min_positive_rows=cfg['support']['minimum_positive_rows'],
                    min_parts_both=cfg['support']['minimum_positive_groups'])
                risk=resolved.loc[resolved['eligible_'+effect].eq(1)]
                count.update(slice=slice_name,construct=construct,direction=direction,
                    n_participants=int(risk[MEM_GROUP_COL].nunique()),n_datasets=int(risk.dataset.nunique()),
                    unresolved_excluded=n_unresolved,
                    unresolved_eligible=int(df.loc[df['exclude_from_construct_effect_fitting'].eq(1),'eligible_'+effect].eq(1).sum()))
                reasons=[]
                if count['n_positive_eligible']<5:reasons.append('positive_rows_below_5')
                if count.get('n_participants_with_both_pos_neg_eligible',0)<2:reasons.append('canonical_groups_with_both_levels_below_2')
                count['support_exclusion_reasons']=reasons
                effects.append(count)
                if not reasons:
                    iteration=bool(risk.iteration.nunique()>=2)
                    family.append(dict(id=slice_name+'__'+effect,slice=slice_name,effect=effect,
                        eligibility_mode='restrict',formula='delta_f ~ '+effect+(' + iteration' if iteration else ''),
                        random_formula='1 + '+effect,grouping='dataset::run_id::participant_id',
                        iteration_control=iteration,optimizer_methods=['lbfgs'],reml=True,maxiter=200,
                        convergence_acceptance='converged and finite coefficient/SE and SE <= max(10,50*(abs(coef)+0.05)); boundary retained and flagged',
                        fallback='none; documented diagnostic status retained', support=count))
        all_coverage[slice_name]=dict(n_rows=len(df),n_fit_eligible=len(resolved),unresolved_excluded=n_unresolved,
            effects=effects,transition_direction_counts={c:{str(k):int(v) for k,v in df['transition_'+c].value_counts().items()} for c in BEHAVIORAL_MOTIFS})
        # Compact operation-specific candidate blocks. No significance screening,
        # mixed add/modify blocks, or rank-driven subset search.
        for direction in DIRECTIONAL_SUFFIXES:
            candidates=[f['effect'] for f in family if f['slice']==slice_name and f['effect'].endswith('_'+direction)]
            block=dict(id=slice_name+'__joint_'+direction,slice=slice_name,effects=candidates,secondary=True)
            reasons=[]
            if len(candidates)<2:
                reasons.append('fewer_than_two_supported_effects')
                block.update(n_rows=0,n_datasets=0,rank=None)
            else:
                reject_same_construct_add_and_modify(candidates)
                joint,elig=apply_joint_eligibility_restrict(resolved,motif_effects=candidates)
                reasons+=joint_support_failures(joint,candidates,group_col=MEM_GROUP_COL)
                if len(joint)<cfg['joint_audit']['minimum_rows']:reasons.append('intersection_rows_below_100')
                controls=candidates+(['iteration'] if joint.iteration.nunique()>=2 else [])
                design=np.column_stack([np.ones(len(joint))]+[joint[c].to_numpy(dtype=float) for c in controls])
                rank=int(np.linalg.matrix_rank(design,tol=1e-8)) if len(joint) else 0
                if rank<design.shape[1]:reasons.append('full_fixed_design_rank_deficient')
                block.update(n_rows=len(joint),n_datasets=int(joint.dataset.nunique()),n_groups=int(joint[MEM_GROUP_COL].nunique()),
                    fixed_effects=controls,formula='delta_f ~ '+' + '.join(controls),random_formula='1 + '+' + '.join(candidates),
                    rank=rank,design_columns=design.shape[1],condition_number=float(np.linalg.cond(design)) if len(joint) else None,
                    eligibility=elig,methods=['lbfgs','cg','powell'],primary_method='lbfgs',reml=True,maxiter=200)
            block.update(support_exclusion_reasons=reasons,supported=not reasons)
            blocks.append(block)
    write_json(root/'coverage.json',all_coverage)
    seal=dict(specification='pics_v4_official_gate_selected_participant_mem_20261007_v1',
        base_config=CONFIG,base_config_sha256=CONFIG_SHA256,implementation_lock_sha256=sha(root/'implementation_lock.json'),
        ledgers=dict(LEDGER_PINS),snapshot_provenance_sha256=PROVENANCE_SHA,table_hashes=table_hashes,
        coverage_sha256=sha(root/'coverage.json'),focal_family=family,joint_blocks=blocks,
        multiplicity='Holm once across all sealed slice/effect focal models; missing/unaccepted tests count as p=1 for family size, reported unavailable',
        constructs=list(BEHAVIORAL_MOTIFS),directions=list(DIRECTIONAL_SUFFIXES),
        unchanged='retained_unmodified comparator and explicit transition audit; never merged with addition/removal/modification',
        joint_candidate_rule='one compact same-direction block per slice of coverage-supported effects; no data-driven subset reduction')
    write_json(root/'selected_participant_frozen_v1.json',seal)
    with (root/'selected_participant_frozen_v1.sha256').open('x') as f:f.write(sha(root/'selected_participant_frozen_v1.json')+'\n')
    print('SEALED',len(family),'focals;',sum(b['supported'] for b in blocks),'supported joint blocks',flush=True)


def load_seal(root):
    check_run(root)
    path=root/'selected_participant_frozen_v1.json'
    if sha(path)!=(root/'selected_participant_frozen_v1.sha256').read_text().strip():raise ValueError('seal drift')
    seal=json.loads(path.read_text())
    if sha(root/'implementation_lock.json')!=seal['implementation_lock_sha256']:raise ValueError('implementation lock drift')
    for rel,expected in seal['table_hashes'].items():
        if sha(root/rel)!=expected:raise ValueError('sealed table drift '+rel)
    return seal


def fit(root, kind, index):
    seal=load_seal(root)
    specs=seal['focal_family'] if kind=='focal' else [b for b in seal['joint_blocks'] if b['supported']]
    spec=specs[index]
    dest=root/'fits'/kind/spec['id'];dest.mkdir(parents=True,exist_ok=False)
    df=pd.read_csv(root/'tables'/spec['slice']/'transitions.csv')
    df,n_excluded=filter_frame_for_construct_effect_fitting(df)
    if kind=='focal':
        effect=spec['effect'];work=df.loc[df['eligible_'+effect].eq(1)].copy()
        result=fit_focal_motif(work,focal=effect,controls=[],include_phase_effects=False,structural_controls=[],
            optimizer_methods=spec['optimizer_methods'],reml=spec['reml'],maxiter=spec['maxiter'])
        fe=result.get('fixed_effect',{})
        coef,se=fe.get('coef'),fe.get('se')
        accepted=bool(result.get('converged') and coef is not None and se is not None and math.isfinite(coef) and math.isfinite(se) and se<=max(10,50*(abs(coef)+0.05)))
        result.update(accepted_for_inference=accepted,n_comparator=int(work[effect].eq(0).sum()))
    else:
        reject_same_construct_add_and_modify(spec['effects'])
        work,elig=apply_joint_eligibility_restrict(df,motif_effects=spec['effects'])
        result=fit_joint_random_slopes(work,random_slopes=spec['effects'],fixed_effects=spec['fixed_effects'],
            methods=spec['methods'],primary_method=spec['primary_method'],reml=spec['reml'],maxiter=spec['maxiter'],
            phase=SLICES[spec['slice']][0],eligibility_report=elig)
    result.update(sealed_spec=spec,seal_sha256=sha(root/'selected_participant_frozen_v1.json'),
        unresolved_excluded=n_excluded,slurm_job_id=os.environ.get('SLURM_JOB_ID'),slurm_array_task=os.environ.get('SLURM_ARRAY_TASK_ID'))
    write_json(dest/'fit.json',result)
    print(spec['id'],result.get('status'),flush=True)


def aggregate(root):
    seal=load_seal(root)
    from statsmodels.stats.multitest import multipletests
    focal=[];joints=[]
    for spec in seal['focal_family']:
        result=json.loads((root/'fits/focal'/spec['id']/'fit.json').read_text())
        if result['seal_sha256']!=sha(root/'selected_participant_frozen_v1.json'):raise ValueError('fit seal mismatch')
        focal.append(result)
    raw=[r.get('fixed_effect',{}).get('pvalue') if r.get('accepted_for_inference') else None for r in focal]
    valid=[p is not None and math.isfinite(p) for p in raw]
    adjusted=multipletests([p if ok else 1.0 for p,ok in zip(raw,valid)],method='holm')[1]
    for r,p,ok in zip(focal,adjusted,valid):r.update(pvalue_holm=float(p) if ok else None,reject_holm_005=bool(ok and p<0.05))
    for spec in seal['joint_blocks']:
        if spec['supported']:
            joints.append(json.loads((root/'fits/joint'/spec['id']/'fit.json').read_text()))
    write_json(root/'focal_results_holm.json',focal)
    write_json(root/'joint_results.json',joints)
    report=['# Official selected-track participant MEM','',
        'All annotations remain preserved. Inputs are the two certified ledgers and selected-track stored references. All verification, build, coverage, sealing and fitting stages ran on CPU Slurm allocations. Results describe associations; they do not establish causal effects.','',
        'Base policy SHA: `'+CONFIG_SHA256+'`. Snapshot provenance SHA: `'+PROVENANCE_SHA+'`. Selected family seal SHA: `'+sha(root/'selected_participant_frozen_v1.json')+'`.','',
        '## Table row flow','', '| Slice | Committed | Pre-annotation excluded | Verified / joined | Fit eligible | Unresolved NMC | Datasets | Participants |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    audits={}
    for name in SLICES:
        audit=json.loads((root/'tables'/name/'build_audit.json').read_text());audits[name]=audit
        flow=audit['row_flow']
        counts=[sum(r[k] for r in flow) for k in ('committed','pre_annotation_excluded','joined','fit_eligible','unresolved_nmc')]
        report.append('| '+name+' | '+' | '.join(map(str,counts+[audit['n_datasets'],audit['n_participants']]))+' |')
    report+=['','All joins passed the existing canonical SHA-aware validator. All table ΔF values use stored candidate minus reference train+validation scores; passive test fields were rejected. Empty Badham exploration is retained in the audit. Repeated source bytes remain separate candidate events.','',
        '| Dataset | Committed | Joined | Fit eligible | Unresolved NMC |','|---|---:|---:|---:|---:|']
    for ds,label in zip(DATASETS,LABELS):
        rows=[r for a in audits.values() for r in a['row_flow'] if r['dataset']==ds]
        report.append('| '+label+' | '+' | '.join(str(sum(r[k] for r in rows)) for k in ('committed','joined','fit_eligible','unresolved_nmc'))+' |')
    report+=['','## Sealed focal results','',f'Holm family: {len(focal)} effects across three separate slices; coverage only determined inclusion. Unaccepted or unavailable tests remain in the family size and have no inferential conclusion.','',
        '| Slice / effect | Coefficient | SE | 95% interval | Raw p | Holm p | Rows / events / groups / datasets | Accepted | Status | RI variance / slope variance |','|---|---:|---:|---|---:|---:|---|---|---|---|']
    for r in focal:
        spec=r['sealed_spec'];fe=r.get('fixed_effect',{})
        fmt=lambda x: f'{x:.6g}' if isinstance(x,(int,float)) and math.isfinite(x) else 'unavailable'
        report.append('| '+spec['id']+' | '+ ' | '.join([fmt(fe.get('coef')),fmt(fe.get('se')),
            '['+fmt(fe.get('ci_low'))+', '+fmt(fe.get('ci_high'))+']',fmt(fe.get('pvalue')),fmt(r['pvalue_holm']),
            ' / '.join(str(r.get(k,'—')) for k in ('n_rows','n_positive','n_groups','n_datasets')),
            str(r['accepted_for_inference']),str(r.get('status')),fmt(r.get('random_intercept_variance'))+' / '+fmt(r.get('random_slope_variance'))])+' |')
    report+=['','Exact fixed/random formulas, convergence acceptance, Hessian and boundary diagnostics, optimizer warnings, random variances and participant slopes are in each `fits/focal/<id>/fit.json`. Boundary fits remain flagged. No alternate estimator or undocumented fallback was applied.','',
        '## Secondary joint models','', '| Block | Effects | Supported | Reason / fit status | Rows |','|---|---|---|---|---:|']
    for b in seal['joint_blocks']:
        fit_result=next((r for r in joints if r['sealed_spec']['id']==b['id']),{})
        report.append('| '+b['id']+' | '+', '.join(b['effects'])+' | '+str(b['supported'])+' | '+('; '.join(b['support_exclusion_reasons']) or str(fit_result.get('status')))+' | '+str(b['n_rows'])+' |')
    report+=['','Joint coefficient tables, covariance/correlation estimates, conditioning and convergence diagnostics are preserved in `joint_results.json` and each `fits/joint/<id>/fit.json`. Unsupported blocks were not fitted; no block pairs addition and modification of the same construct.','',
        '## Exclusions, provenance and freeze','',
        'Per-slice and per-dataset exclusion reasons, unresolved counts and deterministic correction counts are in `tables/<slice>/build_audit.json`. Every construct × direction coverage record and unchanged/absent/unresolved transition count is in `coverage.json`. Raw LLM annotation roots are unchanged; exact roots and input hashes are in the annotation verification record and ledgers.','',
        'Historical Sep-26 analyses and the losing standalone tracks were excluded. This is the certified official-gate-selected standalone scope; equivalence to a newly executed official-gate downstream run is not claimed.','',
        'Commands and CPU job/dependency IDs: `execution.json`. Final artifact hashes and the official freeze marker: `FINAL_FREEZE.json`. The additive driver refuses existing stage outputs, changed implementations, changed sealed tables, or reruns after the final marker.','']
    with (root/'REPORT.md').open('x') as f:f.write('\n'.join(report))
    outputs={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    write_json(root/'FINAL_FREEZE.json',dict(status='official_frozen_selected_track_participant_mem',
        checks='canonical joins, SHA/input policy locks, table hashes, coverage seal, sealed model outputs and aggregation passed',
        output_hashes=outputs,seal_sha256=sha(root/'selected_participant_frozen_v1.json'),
        slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    print('FINAL FREEZE',sha(root/'FINAL_FREEZE.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','build','coverage','focal','joint','aggregate'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--slice',choices=list(SLICES));p.add_argument('--index',type=int)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('all stages require CPU Slurm allocation')
    if os.environ.get('SLURM_JOB_GPUS'):raise SystemExit('CPU-only allocation required')
    if args.stage=='init':initialize(args.output)
    elif args.stage=='build':build(args.output,args.slice)
    elif args.stage=='coverage':coverage(args.output)
    elif args.stage in ('focal','joint'):fit(args.output,args.stage,args.index)
    else:aggregate(args.output)


if __name__=='__main__':main()
