"""Narrow selected-track scope adapter for the unchanged schema-v5 annotator.

This entry point does not select tracks, labels, constructs, or fitting policies.
It only consumes a SHA-certified provenance ledger and adds durable raw replay.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from utils.mem.pics_v4_selected_scope import (
    SelectedScope, RawCache, digest, encoded, repair_tail, PicsV4ReferenceError,
)

SLICES = {"exploration": ("explore", "explore"),
          "evolution_normal": ("evolution", "normal"),
          "evolution_fresh": ("evolution", "fresh")}


def install(worker, scope: SelectedScope, output: Path) -> None:
    """Adapt existing worker functions for this process only; historical files stay frozen."""
    import analysis.mem.build_dataset as builder
    original_guard = builder._guard_official_participant_build

    def guard(run_dir, trace_files, reference_policy):
        if Path(run_dir).resolve() != scope.root:
            return original_guard(run_dir, trace_files, reference_policy)
        if reference_policy != "pics_v4_uniform_v8":
            raise PicsV4ReferenceError("selected official scope requires frozen v8 references")
        expected = {Path(p["path"]).resolve() for p in scope.meta["traces"]}
        if {Path(p).resolve() for p in trace_files} != expected:
            raise PicsV4ReferenceError("selected trace inventory differs from certified ledger")

    builder._guard_official_participant_build = guard
    worker.resolve_uniform_v8_reference = scope.resolve
    loader = worker._load_grouped_candidates

    def selected_groups(*args, **kwargs):
        contexts, groups = loader(*args, **kwargs)
        result = {}
        for key, records in groups.items():
            # Explicit frozen slices; no normal/fresh pooling and no track override.
            rows = [r for r in records if (r.get("phase"), r.get("source")) == (scope.phase, scope.source)]
            if rows:
                result[key] = rows
        return contexts, result

    worker._load_grouped_candidates = selected_groups
    worker._load_completed_keys = lambda path, **kwargs: scope.completed(Path(path))
    enrich = worker._enrich_annotation_row
    worker._enrich_annotation_row = lambda *args, **kwargs: scope.stamp(enrich(*args, **kwargs))
    annotate = worker._annotate_batch

    def cached_batch(client, **kwargs):
        identities = [scope.event(r, Path(r["_participant_dir"]))["official_gate_selected_key"] for r in kwargs["batch"]]
        cache = RawCache(output / "raw_requests", [scope.ledger_sha, identities])

        def create(**request):
            def generate():
                response = client.chat.completions.create(**request)
                return response.choices[0].message.content or ""
            raw = cache.response(request, generate)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])

        proxy = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return annotate(proxy, **kwargs)

    worker._annotate_batch = cached_batch
    splits = worker.annotate_with_splits

    def collision_free_splits(client, **kwargs):
        ids = [scope.event(r, Path(r["_participant_dir"]))["official_gate_selected_key"] for r in kwargs["batch"]]
        kwargs["batch_tag"] = "selected_" + digest(encoded([scope.ledger_sha, ids]))
        return splits(client, **kwargs)

    worker.annotate_with_splits = collision_free_splits
    writer = worker._write_annotation_rows

    def durable_rows(path, **kwargs):
        for row in kwargs["rows"]:
            sources = [r for r in kwargs["batch"] if r["candidate_id"] == row["candidate_id"]]
            if len(sources) != 1 or tuple(sources[0]["_resume_key"]) in kwargs["completed"]:
                raise PicsV4ReferenceError("duplicate/already committed selected annotation")
        n = writer(path, **kwargs)
        for file in [path, kwargs.get("corrections_path"), kwargs.get("adjudication_path")]:
            if file is not None and Path(file).exists():
                fd = os.open(file, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        return n

    worker._write_annotation_rows = durable_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--ledger-sha256", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--slice", choices=SLICES, required=True)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--repair-tail", type=Path)
    p.add_argument("--llm-server-url", default="")
    p.add_argument("--model-name", default="")
    args = p.parse_args()
    phase, source = SLICES[args.slice]
    scope = SelectedScope(args.ledger, args.ledger_sha256, args.dataset, phase, source)
    output = Path(scope.ledger["annotation_output_base"]) / scope.ledger["official_files"]["provenance.json"]["sha256"] / args.dataset / args.slice
    if args.repair_tail:
        print({"quarantined_original": str(repair_tail(args.repair_tail, scope)), "annotation_started": False})
        return
    if args.verify_only:
        print({"dataset": args.dataset, "slice": args.slice, "eligible": len(scope.events), "output": str(output), "annotation_started": False})
        return
    if not args.llm_server_url or not args.model_name:
        p.error("annotation requires explicit --llm-server-url and --model-name")
    from analysis.mem import annotate_edits as worker
    install(worker, scope, output)
    import analysis.mem.build_dataset as builder
    builder._guard_official_participant_build(scope.root, worker._discover_trace_files(scope.root), "pics_v4_uniform_v8")
    sys.argv = ["annotate_edits", "--run_dir", str(scope.root), "--output_dir", str(output),
                "--schema_version", "5", "--reference_policy", "pics_v4_uniform_v8",
                "--phases", phase, "--include_explore", "--include_fresh",
                "--llm_server_url", args.llm_server_url, "--model_name", args.model_name]
    worker.main()


if __name__ == "__main__":
    main()
