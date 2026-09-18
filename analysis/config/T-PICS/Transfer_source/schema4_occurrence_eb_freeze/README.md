# Schema-v4 Occurrence-EB freeze manifest

Machine-readable freeze of the **ICLR source-selection** stack that produced:

`analysis/config/T-PICS/Transfer_source/occurrence_eb_schema4_iter10.yaml`

## Authoritative narrative

See **[Frozen schema-v4 annotation and Occurrence-EB source selection](../../Documentation.md#frozen-schema-v4-annotation-and-occurrence-eb-source-selection)** in `analysis/config/T-PICS/Documentation.md`.

## This directory

| File | Purpose |
| --- | --- |
| `freeze_manifest.json` | Every authoritative source-selection file: `path`, `sha256`, `role`, `git_commit` (if tracked), `runtime_required` |
| `README.md` | This pointer |

## Policy (do not violate)

1. Frozen schema-v4 source-selection files and artifacts **must remain unchanged** — they define the main ICLR method.
2. Future participant-program annotation or MEM work **must** use a **new versioned** directory/module and **new** output paths.
3. Reuse by **copy/import** into that new version — **never overwrite** these frozen artifacts.
4. Any taxonomy or model change needs a **new schema/model name** and must **not** silently regenerate `occurrence_eb_schema4_iter10.yaml`.

## Runtime vs provenance

- **Runtime-required:** the YAML map plus the fifteen frozen G.1 `global_phase/best_program.py` files it names (default gated T-PICS).
- **Provenance-only:** schema-v4 modules, PopAnnot/SourceSelect scripts, manifests, annotation jsonl, Occurrence-EB fit outputs, eval tables, reports.

Do not retune, reannotate, or resubmit jobs from this freeze.
