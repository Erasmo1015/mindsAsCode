"""
Optional dataset-prompt evolution (default off).

Light gen+eval scoring on fixed train/val for a small set of dev participants.
Preserves a frozen API/schema/sandbox contract while mutating behavioural guidance.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

FROZEN_BEGIN = "<<<FROZEN_BEGIN>>>"
FROZEN_END = "<<<FROZEN_END>>>"
EVOLVABLE_BEGIN = "<<<EVOLVABLE_BEGIN>>>"
EVOLVABLE_END = "<<<EVOLVABLE_END>>>"

TASK_KNOWLEDGE_HEADER = "## Task / domain knowledge (retained from prompt-generation input)"

MUTATION_ROLES = ("repair", "improve_modelling", "explore_strategy")

# Legacy markers used only as a last-resort split when API/requirements cannot be found.
_LEGACY_EVOLVABLE_MARKERS = (
    "Behavioral requirements:",
    "Generation requirements:",
    "Diversity requirements:",
    "Mutation guidance:",
)

_API_PATTERNS = (
    re.compile(r"def\s+choose\s*\(\s*problem\s*,\s*history\s*\)", re.I),
    re.compile(r"P\(action\s*=\s*1\)|return(?:ing)?\s+float|dict\[int,\s*float\]", re.I),
)

_SANDBOX_REQUIRED = (
    re.compile(r"no imports|Pure Python", re.I),
    re.compile(r"deterministic", re.I),
)

_LEAKAGE_PATTERNS = (
    re.compile(r"\btest\s+set\b", re.I),
    re.compile(r"\bhold[- ]?out\b", re.I),
    re.compile(r"\bleak(?:age|ing)?\b.*\btest\b", re.I),
    re.compile(r"use\s+the\s+test\s+trials", re.I),
)

_SANDBOX_WEAKENED = (
    re.compile(
        r"\b(?:you may|allowed to|feel free to)\s+import\b|\buse numpy\b|\bimport random\b",
        re.I,
    ),
    re.compile(
        r"\b(?:you may|allowed to|feel free to)\s+(?:sample|use randomness)\b|"
        r"\ballow(?:ed)?\s+randomness\b|"
        r"\bnon-deterministic\s+(?:code|programs?|behaviour|behavior)\b",
        re.I,
    ),
)

_REQUIREMENTS_START = re.compile(r"(?im)^Requirements:\s*$")
_SCHEMA_START = re.compile(
    r"(?im)^(?:#{1,3}\s*)?Runtime schema summary(?:\s*\([^)]*\))?\s*:?\s*$"
)
_SECTION_END_HINTS = (
    "Behavioral requirements:",
    "Generation requirements:",
    "Diversity requirements:",
    "Mutation guidance:",
    "Task description",
    "## Task",
    "### Task",
    TASK_KNOWLEDGE_HEADER,
    "Cognitive",
    "Modeling guidance",
    "Modelling guidance",
)


@dataclass
class PromptCandidate:
    prompt_id: str
    text: str
    role: str = "baseline"
    parent_id: Optional[str] = None
    mean_train_fitness: float = float("-inf")
    mean_val_fitness: float = float("-inf")
    executable_rate: float = 0.0
    rejected: bool = False
    reject_reason: Optional[str] = None
    error_signatures: List[str] = field(default_factory=list)
    strong_snippets: List[str] = field(default_factory=list)
    n_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def frozen_hash(frozen_text: str) -> str:
    return hashlib.sha256(frozen_text.encode("utf-8")).hexdigest()


def extract_marked_section(text: str, begin: str, end: str) -> Optional[str]:
    if begin not in text or end not in text:
        return None
    try:
        mid = text.split(begin, 1)[1]
        return mid.split(end, 1)[0].strip()
    except IndexError:
        return None


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda se: se[0])
    merged: List[Tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_s, prev_e = merged[-1]
        if start <= prev_e:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return merged


def _span_until_section(text: str, start: int) -> Tuple[int, int]:
    rest = text[start:]
    end = len(rest)
    for marker in _SECTION_END_HINTS:
        idx = rest.find(marker)
        if idx > 0:
            end = min(end, idx)
    # Also stop at a following top-level markdown heading (except the current line).
    for m in re.finditer(r"(?m)^## ", rest):
        if m.start() > 0:
            end = min(end, m.start())
            break
    return start, start + end


def _find_choose_api_span(text: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"def\s+choose\s*\(\s*problem\s*,\s*history\s*\)", text)
    if not m:
        return None
    start = m.start()
    rest = text[start:]
    doc = re.match(
        r"def\s+choose\s*\(\s*problem\s*,\s*history\s*\)\s*:?\s*\n\s*\"\"\"[\s\S]*?\"\"\"",
        rest,
    )
    if doc:
        return start, start + doc.end()
    # Docstring-less / short API note: take through return/P(action) lines.
    lines = rest.splitlines(keepends=True)
    end_rel = 0
    seen_return = False
    for i, line in enumerate(lines[:24]):
        if i > 0 and _REQUIREMENTS_START.match(line):
            break
        if i > 0 and any(line.startswith(h) for h in _SECTION_END_HINTS):
            break
        end_rel += len(line)
        if re.search(r"return\s*:", line, re.I) or "P(action" in line:
            seen_return = True
        if i > 0 and seen_return and line.strip() == "":
            break
    return start, start + max(end_rel, len(lines[0]) if lines else 0)


def _find_requirements_span(text: str) -> Optional[Tuple[int, int]]:
    m = _REQUIREMENTS_START.search(text)
    if not m:
        return None
    return _span_until_section(text, m.start())


def _find_schema_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in _SCHEMA_START.finditer(text):
        spans.append(_span_until_section(text, m.start()))
    return spans


def _default_evolvable_stub() -> str:
    return (
        "Behavioral requirements:\n"
        "- Prefer models that use observed problem/history structure.\n"
        "- Encourage diversity across candidates.\n"
    )


def split_frozen_evolvable(prompt: str) -> Tuple[str, str]:
    """
    Split into frozen contract vs evolvable guidance.

    Frozen: API (`choose`), return type, Requirements/sandbox, runtime schema.
    Evolvable: task/domain narrative, behavioural/cognitive/generation guidance.
    """
    frozen = extract_marked_section(prompt, FROZEN_BEGIN, FROZEN_END)
    evolvable = extract_marked_section(prompt, EVOLVABLE_BEGIN, EVOLVABLE_END)
    if frozen is not None and evolvable is not None:
        return frozen, evolvable

    spans: List[Tuple[int, int]] = []
    api = _find_choose_api_span(prompt)
    if api:
        spans.append(api)
    req = _find_requirements_span(prompt)
    if req:
        spans.append(req)
    spans.extend(_find_schema_spans(prompt))
    spans = _merge_spans(spans)

    if not spans:
        # Last resort: freeze everything before first behavioural/generation marker.
        split_at = None
        for marker in _LEGACY_EVOLVABLE_MARKERS:
            idx = prompt.find(marker)
            if idx >= 0 and (split_at is None or idx < split_at):
                split_at = idx
        if split_at is None:
            split_at = max(1, int(len(prompt) * 0.6))
        return prompt[:split_at].strip(), prompt[split_at:].strip()

    frozen_chunks = [prompt[s:e].strip() for s, e in spans if prompt[s:e].strip()]
    frozen_text = "\n\n".join(frozen_chunks)

    evolvable_parts: List[str] = []
    prev = 0
    for s, e in spans:
        if s > prev:
            chunk = prompt[prev:s].strip()
            if chunk:
                evolvable_parts.append(chunk)
        prev = e
    if prev < len(prompt):
        chunk = prompt[prev:].strip()
        if chunk:
            evolvable_parts.append(chunk)
    evolvable_text = "\n\n".join(evolvable_parts).strip()
    if not evolvable_text:
        evolvable_text = _default_evolvable_stub()
    return frozen_text, evolvable_text


def wrap_prompt_with_contract(prompt: str) -> str:
    if FROZEN_BEGIN in prompt and EVOLVABLE_BEGIN in prompt:
        return prompt
    frozen, evolvable = split_frozen_evolvable(prompt)
    if not evolvable:
        evolvable = _default_evolvable_stub()
    return (
        f"{FROZEN_BEGIN}\n{frozen}\n{FROZEN_END}\n\n"
        f"{EVOLVABLE_BEGIN}\n{evolvable}\n{EVOLVABLE_END}\n"
    )


def render_prompt_for_generation(prompt: str) -> str:
    """Strip contract markers for the program-generation LLM."""
    text = prompt
    for tag in (FROZEN_BEGIN, FROZEN_END, EVOLVABLE_BEGIN, EVOLVABLE_END):
        text = text.replace(tag, "")
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _knowledge_already_present(haystack: str, needle: str, *, min_chars: int = 40) -> bool:
    text = (needle or "").strip()
    if len(text) < min_chars:
        return bool(text) and text in haystack
    probe = re.sub(r"\s+", " ", text[:120]).strip().lower()
    compact = re.sub(r"\s+", " ", haystack).lower()
    return bool(probe) and probe in compact


def ensure_task_knowledge_in_prompt(
    prompt: str,
    *,
    task_description: str,
    instruction_excerpt: str,
    max_instruction_chars: int = 2000,
) -> str:
    """
    Ensure auto prompt-gen task knowledge is present and evolvable.

    Injects a task/domain block when missing so mutation can keep refining modelling
    guidance without losing facts from llm_input. Preserves marker-free prompts for
    the default non-evolution path (markers only if the input already had them).
    """
    desc = (task_description or "").strip()
    instr = (instruction_excerpt or "").strip()[: max(0, int(max_instruction_chars))]
    if not desc and not instr:
        return prompt

    had_markers = FROZEN_BEGIN in prompt and EVOLVABLE_BEGIN in prompt
    wrapped = wrap_prompt_with_contract(prompt)
    frozen, evolvable = split_frozen_evolvable(wrapped)
    rendered = render_prompt_for_generation(wrapped)

    need_desc = bool(desc) and not _knowledge_already_present(rendered, desc)
    need_instr = bool(instr) and not _knowledge_already_present(
        rendered, instr, min_chars=60
    )
    if not need_desc and not need_instr:
        return prompt if not had_markers else wrapped

    parts: List[str] = [TASK_KNOWLEDGE_HEADER, ""]
    if need_desc or (desc and not _knowledge_already_present(evolvable, desc)):
        if desc:
            parts.append(desc)
            parts.append("")
    if need_instr:
        parts.append("Instruction excerpt:")
        parts.append(instr)
        parts.append("")
    block = "\n".join(parts).strip()
    if TASK_KNOWLEDGE_HEADER in evolvable:
        new_evolvable = evolvable.rstrip() + "\n\n" + block + "\n"
    else:
        new_evolvable = block + "\n\n" + evolvable
    marked = (
        f"{FROZEN_BEGIN}\n{frozen}\n{FROZEN_END}\n\n"
        f"{EVOLVABLE_BEGIN}\n{new_evolvable.strip()}\n{EVOLVABLE_END}\n"
    )
    if had_markers:
        return marked
    # One-shot / unmarked baseline: keep marker-free text for PICS.
    return render_prompt_for_generation(marked)


def validate_child_prompt(
    child: str,
    *,
    parent_frozen_hash: str,
) -> Tuple[bool, Optional[str]]:
    wrapped = wrap_prompt_with_contract(child)
    frozen, _ = split_frozen_evolvable(wrapped)
    if frozen_hash(frozen) != parent_frozen_hash:
        return False, "frozen_hash_mismatch"
    rendered = render_prompt_for_generation(wrapped)
    if not any(p.search(rendered) for p in _API_PATTERNS):
        return False, "missing_api_or_return_type"
    if not all(p.search(frozen) for p in _SANDBOX_REQUIRED):
        # Also accept if present in full rendered text (markers may wrap oddly).
        if not all(p.search(rendered) for p in _SANDBOX_REQUIRED):
            return False, "sandbox_requirements_missing"
    if any(p.search(rendered) for p in _SANDBOX_WEAKENED):
        return False, "sandbox_weakened"
    if any(p.search(rendered) for p in _LEAKAGE_PATTERNS):
        return False, "test_leakage_language"
    return True, None


def build_mutation_user_message(
    *,
    parent_prompt: str,
    role: str,
    feedback: str,
) -> str:
    role_guidance = {
        "repair": (
            "Role: repair. Fix schema misuse, KeyErrors, and executability failures. "
            "Keep frozen sections identical."
        ),
        "improve_modelling": (
            "Role: improve_modelling. Strengthen behavioural modelling guidance using "
            "success/failure feedback. Keep frozen sections identical."
        ),
        "explore_strategy": (
            "Role: explore_strategy. Propose alternate modelling strategies in the "
            "evolvable section only. Keep frozen sections identical."
        ),
    }.get(role, "Role: mutate evolvable guidance only.")
    return (
        "You are revising a dataset evolution instruction prompt.\n"
        f"{role_guidance}\n\n"
        "Rules:\n"
        f"- Preserve {FROZEN_BEGIN}...{FROZEN_END} byte-for-byte (same content).\n"
        f"- Only edit content inside {EVOLVABLE_BEGIN}...{EVOLVABLE_END}.\n"
        "- Frozen content is only the API, return type, schema, sandbox, and leakage "
        "constraints. Task/domain facts and cognitive modelling guidance are evolvable.\n"
        "- Retain factual task/domain knowledge already present in the evolvable section; "
        "do not delete cue validities, payoffs, stage rules, or instruction facts.\n"
        "- Do not add imports, randomness, or test-set leakage language.\n"
        "- Output the full prompt with both marker pairs; no markdown fence.\n\n"
        f"## Parent prompt\n\n{parent_prompt}\n\n"
        f"## Compact feedback\n\n{feedback}\n"
    )


def select_beam(
    candidates: Sequence[PromptCandidate],
    *,
    beam_size: int = 3,
    always_retain_best: bool = True,
) -> List[PromptCandidate]:
    """Keep top-k by mean val fitness; length is tie-break only; reject flagged skipped."""
    valid = [c for c in candidates if not c.rejected]
    if not valid:
        return []

    def sort_key(c: PromptCandidate) -> Tuple[float, int]:
        # Higher val fitness better; shorter length only as weak tie-break.
        return (c.mean_val_fitness, -c.n_chars)

    ranked = sorted(valid, key=sort_key, reverse=True)
    beam = ranked[: max(1, beam_size)]
    if always_retain_best and ranked:
        best = ranked[0]
        if best.prompt_id not in {c.prompt_id for c in beam}:
            beam = [best] + beam[:-1]
    return beam


def aggregate_pics_style_prompt_scores(
    per_participant_scores: Sequence[Sequence[Tuple[float, float]]],
) -> Dict[str, float]:
    """
    PICS-aligned prompt scoring.

    For each development participant, take the best (train_ll, val_ll) among that
    participant's candidates (typically seed + generated programs) by val_ll, then
    average those best scores across participants.

    Do not mean over all generations.
    """
    best_trains: List[float] = []
    best_vals: List[float] = []
    for scores in per_participant_scores:
        usable = [
            (float(t), float(v))
            for t, v in scores
            if t == t and v == v  # not NaN
        ]
        if not usable:
            continue
        best_t, best_v = max(usable, key=lambda tv: tv[1])
        best_trains.append(best_t)
        best_vals.append(best_v)
    if not best_vals:
        return {
            "mean_train_fitness": float("-inf"),
            "mean_val_fitness": float("-inf"),
        }
    return {
        "mean_train_fitness": sum(best_trains) / len(best_trains),
        "mean_val_fitness": sum(best_vals) / len(best_vals),
    }


def compact_feedback_for_parent(parent: PromptCandidate) -> str:
    lines = [
        f"mean_train_fitness={parent.mean_train_fitness:.6f}",
        f"mean_val_fitness={parent.mean_val_fitness:.6f}",
        f"executable_rate={parent.executable_rate:.3f}",
    ]
    if parent.error_signatures:
        lines.append("top_errors: " + "; ".join(parent.error_signatures[:5]))
    if parent.strong_snippets:
        lines.append("strong_program_snippets:")
        for snip in parent.strong_snippets[:2]:
            lines.append(snip[:400])
    return "\n".join(lines)


def run_dataset_prompt_evolution(
    *,
    prompts_dir: Path,
    baseline_prompt: str,
    iterations: int,
    population: int = 3,
    children_per_round: int = 3,
    dev_participant_ids: Sequence[int],
    eval_candidates: int = 5,
    seed_program: str,
    mutate_fn: Callable[[str, str, str], str],
    score_fn: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Beam-search evolve dataset prompts.

    mutate_fn(parent_text, role, feedback) -> child_text
    score_fn(rendered_prompt_text) -> metrics dict with keys:
      mean_train_fitness, mean_val_fitness, executable_rate,
      error_signatures, strong_snippets
    """
    if iterations <= 0:
        return {
            "dataset_prompt_evolved": False,
            "evolution_iterations": 0,
            "best_prompt_id": None,
            "skipped": True,
        }

    evo_dir = prompts_dir / "dataset_prompt_evolution"
    evo_dir.mkdir(parents=True, exist_ok=True)
    log_path = evo_dir / "search_log.jsonl"

    wrapped0 = wrap_prompt_with_contract(baseline_prompt)
    frozen0, _ = split_frozen_evolvable(wrapped0)
    parent_fhash = frozen_hash(frozen0)

    def score_candidate(cid: str, text: str, role: str, parent_id: Optional[str]) -> PromptCandidate:
        cand = PromptCandidate(
            prompt_id=cid,
            text=text,
            role=role,
            parent_id=parent_id,
            n_chars=len(text),
        )
        ok, reason = validate_child_prompt(text, parent_frozen_hash=parent_fhash)
        if not ok:
            cand.rejected = True
            cand.reject_reason = reason
            return cand
        metrics = score_fn(render_prompt_for_generation(wrap_prompt_with_contract(text)))
        cand.mean_train_fitness = float(metrics.get("mean_train_fitness", float("-inf")))
        cand.mean_val_fitness = float(metrics.get("mean_val_fitness", float("-inf")))
        cand.executable_rate = float(metrics.get("executable_rate", 0.0))
        cand.error_signatures = list(metrics.get("error_signatures") or [])
        cand.strong_snippets = list(metrics.get("strong_snippets") or [])
        return cand

    baseline = score_candidate("p0_baseline", wrapped0, "baseline", None)
    (evo_dir / "p0_baseline.txt").write_text(baseline.text, encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(json.dumps({"event": "score", **baseline.to_dict()}) + "\n")

    beam = select_beam([baseline], beam_size=population)
    global_best = baseline
    prompt_counter = 1

    for round_i in range(1, iterations + 1):
        round_dir = evo_dir / f"round_{round_i}"
        round_dir.mkdir(parents=True, exist_ok=True)
        children: List[PromptCandidate] = []
        # One child per role from current top parent (or rotate parents if beam>1).
        parents_cycle = list(beam) if beam else [global_best]
        for j, role in enumerate(MUTATION_ROLES[: max(1, children_per_round)]):
            parent = parents_cycle[j % len(parents_cycle)]
            feedback = compact_feedback_for_parent(parent)
            child_text = mutate_fn(parent.text, role, feedback)
            cid = f"p{prompt_counter}_{role}"
            prompt_counter += 1
            child = score_candidate(cid, child_text, role, parent.prompt_id)
            children.append(child)
            (round_dir / f"{cid}.txt").write_text(child.text, encoding="utf-8")
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(json.dumps({"event": "score", "round": round_i, **child.to_dict()}) + "\n")

        pool = list(beam) + children
        beam = select_beam(pool, beam_size=population)
        for c in beam:
            if c.mean_val_fitness > global_best.mean_val_fitness:
                global_best = c
        # Always retain global best in beam.
        if global_best.prompt_id not in {c.prompt_id for c in beam}:
            beam = [global_best] + [c for c in beam if c.prompt_id != global_best.prompt_id]
            beam = beam[:population]

        (round_dir / "population.json").write_text(
            json.dumps(
                {
                    "round": round_i,
                    "beam": [c.to_dict() for c in beam],
                    "children": [c.to_dict() for c in children],
                    "global_best_id": global_best.prompt_id,
                    "dev_participant_ids": list(dev_participant_ids),
                    "eval_candidates": eval_candidates,
                    "seed_program_chars": len(seed_program),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # Persist best as the active infer prompt (without requiring markers for PICS).
    best_rendered = render_prompt_for_generation(global_best.text)
    infer_path = prompts_dir / "infer_single_choice.txt"
    infer_path.write_text(best_rendered, encoding="utf-8")
    (evo_dir / "best_prompt.txt").write_text(global_best.text, encoding="utf-8")
    (evo_dir / "best_prompt_id.txt").write_text(global_best.prompt_id + "\n", encoding="utf-8")

    result = {
        "dataset_prompt_evolved": True,
        "evolution_iterations": iterations,
        "best_prompt_id": global_best.prompt_id,
        "best_mean_val_fitness": global_best.mean_val_fitness,
        "best_mean_train_fitness": global_best.mean_train_fitness,
        "best_executable_rate": global_best.executable_rate,
        "dev_participant_ids": list(dev_participant_ids),
        "population": population,
        "children_per_round": children_per_round,
        "eval_candidates": eval_candidates,
    }
    (evo_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    meta_path = prompts_dir / "prompt_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    meta.update(
        {
            "dataset_prompt_evolved": True,
            "evolution_iterations": iterations,
            "best_prompt_id": global_best.prompt_id,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return result


def maybe_run_dataset_prompt_evolution(
    *,
    iterations: int,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """No-op when iterations == 0."""
    if int(iterations) <= 0:
        return None
    return run_dataset_prompt_evolution(iterations=int(iterations), **kwargs)
