"""Interrupted annotation fixtures only; no production labels or model calls."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.mem.pics_v4_selected_scope import (
    SelectedScope, RawCache, digest, encoded, strict_rows, repair_tail, PicsV4ReferenceError,
)
from utils.mem.pics_v4_uniform_v8_reference import uniform_v8_resume_key
from analysis.mem.annotate_selected_participants import install


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "pics_v4_transfer_based_only" / "job_42" / "selected"
    p = root / "participant_1"
    p.mkdir(parents=True)
    trace = p / "mem_trace.jsonl"
    trace.write_text('{"fixture":true}\n')
    reference = tmp_path / "reference.py"
    reference.write_text("def choose(problem, history):\n    return 0.5\n")
    official = tmp_path / "provenance.json"
    official.write_text('{"snapshot":"fixture"}\n')
    artifact = lambda path: {"path": str(path), "sha256": digest(path.read_bytes())}
    transitions = []
    for phase, source, ref_type in [("explore", "explore", "population_program"),
                                    ("evolution", "normal", "best_prompted_parent"),
                                    ("evolution", "fresh", "seed_baseline")]:
        for idx in range(2):
            candidate = p / f"{phase}_{source}_{idx}.py"
            candidate.write_bytes(reference.read_bytes())
            cid = f"candidate_{idx}"
            rid = "baseline" if source == "fresh" else "parent_0"
            rec = dict(record_type="candidate", dataset="mixed_gambles", run_id="selected", participant_id=1,
                       phase=phase, source=source, iteration=1, candidate_id=cid,
                       code_path=candidate.name, reference_id=rid, reference_type=ref_type,
                       runtime_valid=True, selection_score=-0.5, reference_score=-0.5, delta_f=0.0)
            key = list(uniform_v8_resume_key(dataset=rec["dataset"], job_id="job_42", run_id="selected",
                       participant_id=1, phase=phase, source=source, iteration=1, candidate_id=cid,
                       candidate_sha256=artifact(candidate)["sha256"], reference_id=rid,
                       reference_type=ref_type, reference_sha256=artifact(reference)["sha256"]))
            event = [str(rec[k]) for k in ["dataset", "run_id", "participant_id", "phase", "source", "iteration", "candidate_id"]]
            transitions.append(dict(status="eligible", phase=phase, source=source, stored_record=rec,
                       event_key=event, uniform_v8_resume_key=key,
                       official_gate_selected_key=[artifact(official)["sha256"], "fixture", "transfer_based_only", "43", "lineage"]+key,
                       reference_id=rid, reference_type=ref_type, candidate_artifact=artifact(candidate), reference_artifact=artifact(reference)))
    data = dict(official_files={"provenance.json":artifact(official)}, method_hashes={},
                annotation_output_base=str(tmp_path/"new_annotations"),
                datasets=[dict(dataset="mixed_gambles", annotation_ready=True, chosen_track="transfer_based_only",
                canonical_output_job="job_42", lineage=[{"execution":"43","output":"42"}], selected_root=str(root),
                traces=[artifact(trace)], resume_inputs=[artifact(trace),artifact(reference)], transitions=transitions)])
    ledger = tmp_path / "ledger.json"
    ledger.write_bytes(encoded(data))
    def scope(phase="explore",source="explore"):
        return SelectedScope(ledger,digest(ledger.read_bytes()),"mixed_gambles",phase,source)
    return SimpleNamespace(root=root, p=p, ledger=ledger, data=data, transitions=transitions, scope=scope, trace=trace, official=official)


def row_for(scope, key):
    row = dict(schema_version=5, annotation_kind="participant_program_motif_transition",
               prompt_version="fixture", reference_policy="pics_v4_uniform_v8",
               transition_identity_policy="pics_v4_uniform_v8_transition_sha256_v1",
               reference_motif_state=[],candidate_motif_state=[],added_motifs=[],modified_motifs=[],removed_motifs=[],
               uniform_v8_resume_key=list(key))
    for name, value in zip(["dataset","job_id","run_id","participant_id","phase","source","iteration","candidate_id",
                            "candidate_code_sha256","reference_id","reference_type","reference_code_sha256"],key):
        row[name]=value
    return scope.stamp(row)


def worker_fixture(monkeypatch, scope, output):
    import analysis.mem.build_dataset as builder
    monkeypatch.setattr(builder,"_guard_official_participant_build",builder._guard_official_participant_build)
    path = Path(__file__).resolve().parents[2]/"analysis/mem/annotate_edits.py"
    spec=importlib.util.spec_from_file_location("selected_fixture_worker",path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    install(worker,scope,output)
    return worker


def source_for(scope, fixture, idx=0):
    r=next(r for r in scope.events.values() if r["stored_record"]["candidate_id"]==f"candidate_{idx}")
    s=copy.deepcopy(r["stored_record"])
    v=scope.resolve(s,participant_dir=fixture.p)
    s.update(code=v["candidate_code"],_participant_dir=str(fixture.p),_candidate_sha256=v["candidate_sha256"],
             _reference_sha256=v["reference_sha256"],_job_id=v["job_id"],_resume_key=v["resume_key"])
    return s,v


def test_successful_rows_skipped_once_and_repeated_bytes_not_collapsed(fixture,tmp_path):
    scope=fixture.scope(); keys=list(scope.keys)
    assert keys[0][8]==keys[1][8] and keys[0]!=keys[1]
    path=tmp_path/"annotations.jsonl"
    path.write_text("".join(json.dumps(row_for(scope,k))+"\n" for k in keys))
    assert scope.completed(path)==set(keys)
    assert scope.completed(path)==set(keys)


def test_phase_and_normal_fresh_identities_cannot_collide(fixture):
    keys=[next(iter(fixture.scope(p,s).keys)) for p,s in [("explore","explore"),("evolution","normal"),("evolution","fresh")]]
    assert len(set(keys))==3


def test_non_selected_root_rejected(fixture):
    s=fixture.scope();rec=fixture.transitions[0]["stored_record"]
    with pytest.raises(PicsV4ReferenceError,match="non-selected"):
        s.resolve(rec,participant_dir=fixture.p.parent.parent.parent/"pics_v4_target_only/job_42/selected/participant_1")


@pytest.mark.parametrize("field,value",[("chosen_track","target_only"),("official_gate_provenance_sha256","f"*64),
                                       ("selected_ledger_sha256","e"*64),("selected_execution_lineage",[])])
def test_resume_scope_drift_refused(fixture,tmp_path,field,value):
    s=fixture.scope();row=row_for(s,next(iter(s.keys)));row[field]=value
    path=tmp_path/"a.jsonl";path.write_text(json.dumps(row)+"\n")
    with pytest.raises(PicsV4ReferenceError,match="mismatch"):s.completed(path)


def test_duplicate_completed_rows_fail(fixture,tmp_path):
    s=fixture.scope();line=json.dumps(row_for(s,next(iter(s.keys))))+"\n"
    path=tmp_path/"a.jsonl";path.write_text(line+line)
    with pytest.raises(PicsV4ReferenceError,match="duplicate"):s.completed(path)


@pytest.mark.parametrize("failure",[{"error":"LLM failed"},{"success":False},{"annotation_kind":"annotation_failure"}])
def test_failure_rows_never_count_as_success(fixture,tmp_path,failure):
    s=fixture.scope();row=row_for(s,next(iter(s.keys)));row.update(failure)
    path=tmp_path/"a.jsonl";path.write_text(json.dumps(row)+"\n")
    with pytest.raises(PicsV4ReferenceError,match="failure"):s.completed(path)


def test_wrong_slice_resume_refused(fixture,tmp_path):
    explore=fixture.scope();normal=fixture.scope("evolution","normal")
    row=row_for(explore,next(iter(explore.keys)))
    path=tmp_path/"a.jsonl";path.write_text(json.dumps(row)+"\n")
    with pytest.raises(PicsV4ReferenceError,match="outside"):normal.completed(path)


def test_official_snapshot_and_trace_drift_before_mutation(fixture):
    fixture.official.write_text("changed")
    with pytest.raises(PicsV4ReferenceError,match="snapshot drift"):fixture.scope()
    assert not (fixture.official.parent/"new_annotations").exists()


def test_candidate_sha_and_passive_test_refused(fixture):
    s=fixture.scope();rec=copy.deepcopy(fixture.transitions[0]["stored_record"])
    rec["test_loglik"]=-0.1
    with pytest.raises(PicsV4ReferenceError,match="passive"):s.resolve(rec,participant_dir=fixture.p)
    rec.pop("test_loglik");Path(fixture.transitions[0]["candidate_artifact"]["path"]).write_text("changed")
    with pytest.raises(PicsV4ReferenceError,match="SHA"):s.resolve(rec,participant_dir=fixture.p)


def test_raw_cache_preemption_and_collision_free_paths(tmp_path):
    request={"model":"fixture","messages":[{"role":"user","content":"same"}]}
    a=RawCache(tmp_path,["snapshot","target_only","normal","event1"])
    assert a.response(request,lambda:'{"fixture":1}')=='{"fixture":1}'
    # Restart before any finalized annotation was written: no second model call.
    a=RawCache(tmp_path,["snapshot","target_only","normal","event1"])
    assert a.response(request,lambda:pytest.fail("repeated LLM call"))=='{"fixture":1}'
    for binding in [["snapshot","transfer_based_only","normal","event1"],
                    ["snapshot","target_only","fresh","event1"],["snapshot","target_only","normal","event2"]]:
        assert RawCache(tmp_path,binding).response(request,lambda:"other")=="other"
    assert len(list(tmp_path.glob("*.json")))==4


def test_raw_cache_corruption_fails_closed(tmp_path):
    cache=RawCache(tmp_path,["event"]);cache.response({},lambda:"original")
    path=next(tmp_path.glob("*.json")); row=json.loads(path.read_bytes());row["raw"]="tampered";path.write_bytes(encoded(row))
    with pytest.raises(PicsV4ReferenceError,match="corruption"):cache.response({},lambda:pytest.fail("network"))


@pytest.mark.parametrize("tail",[b'{"truncated":',b'{"truncated":\n',b'{"complete_but_unterminated":true}'])
def test_interrupted_jsonl_append_explicit_quarantine_repair(fixture,tail):
    s=fixture.scope();path=Path(s.ledger["annotation_output_base"])/s.ledger["official_files"]["provenance.json"]["sha256"]/s.dataset/"exploration"/"annotations_v5.jsonl"
    path.parent.mkdir(parents=True)
    prefix=(json.dumps(row_for(s,next(iter(s.keys))))+"\n").encode();original=prefix+tail;path.write_bytes(original)
    with pytest.raises(PicsV4ReferenceError):s.completed(path)
    quarantine=repair_tail(path,s)
    assert quarantine.read_bytes()==original and path.read_bytes()==prefix
    assert len(s.completed(path))==1


def test_valid_tail_or_duplicate_prefix_not_repaired(fixture):
    s=fixture.scope();path=Path(s.ledger["annotation_output_base"])/s.ledger["official_files"]["provenance.json"]["sha256"]/s.dataset/"exploration"/"annotations_v5.jsonl"
    path.parent.mkdir(parents=True);line=(json.dumps(row_for(s,next(iter(s.keys))))+"\n").encode()
    path.write_bytes(line)
    with pytest.raises(PicsV4ReferenceError,match="valid completed"):repair_tail(path,s)
    path.write_bytes(line+line+b'{"incomplete":')
    with pytest.raises(PicsV4ReferenceError,match="duplicate"):repair_tail(path,s)
    assert path.read_bytes()==line+line+b'{"incomplete":'


def test_native_worker_write_resume_and_no_overwrite(fixture,monkeypatch,tmp_path):
    scope=fixture.scope();worker=worker_fixture(monkeypatch,scope,tmp_path)
    source,ref=source_for(scope,fixture)
    llm=dict(candidate_id=source["candidate_id"],schema_version=5,prompt_version="fixture",
             annotation_kind="participant_program_motif_transition",
             reference_motif_state=[],candidate_motif_state=[],modified_motifs=[],structural_operations=[],
             no_meaningful_change=True,evidence=[],confidence=0.9)
    path=tmp_path/"annotations_v5.jsonl";done=set()
    kwargs=dict(rows=[llm],batch=[source],key=("selected","mixed_gambles",1,"explore",1),ref_id=ref["reference_id"],
                ref_type=ref["reference_type"],completed=done,reference_code=ref["reference_code"],schema_version=5,prompt_version="fixture",model_name="fixture")
    assert worker._write_annotation_rows(path,**kwargs)==1
    before=path.read_bytes();assert worker._load_completed_keys(path,schema_version=5,reference_policy="pics_v4_uniform_v8")==done
    with pytest.raises(PicsV4ReferenceError,match="already committed"):worker._write_annotation_rows(path,**kwargs)
    assert path.read_bytes()==before


def test_native_batch_raw_response_replay_uses_same_validation(fixture,monkeypatch,tmp_path):
    scope=fixture.scope();worker=worker_fixture(monkeypatch,scope,tmp_path);source,ref=source_for(scope,fixture)
    payload=[dict(candidate_id=source["candidate_id"],schema_version=5,prompt_version="fixture",
                  reference_motif_state=[],candidate_motif_state=[],modified_motifs=[],structural_operations=[],
                  no_meaningful_change=True,evidence=[],confidence=0.9)]
    calls=[]
    def create(**kwargs):
        calls.append(kwargs);return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    kwargs=dict(model_name="fixture",reference_code=ref["reference_code"],batch=[source],use_guided_json=False,schema_version=5,prompt_version="fixture")
    result=worker._annotate_batch(client,**kwargs)
    assert not result[2],result[2]
    assert worker._annotate_batch(client,**kwargs)==result
    assert len(calls)==1
