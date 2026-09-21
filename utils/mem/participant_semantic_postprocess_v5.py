#!/usr/bin/env python3
"""Deterministic participant Schema-v5 semantic validation / postprocessing.

Participant-only. Does **not** touch population annotations or source-selection.

**Generic rules** (all datasets):
  SEED_BASELINE_REF_ABSENT, UNUSED_HISTORY_ABSENT, NMC_AST_PARAM_OR_STRUCTURE
  (clear NMC only — no auto construct attribution), NMC_NEEDS_ADJUDICATION,
  MODIFIED_RESTRICT_TO_INTERSECTION, REDERIVE_DIRECTIONS, logging.

**Bergert-only** (``dataset == bergert_nosofsky_2007`` strict gate):
  BERGERT_ACTION_MEANS_SOLE_EVIDENCE, BERGERT_UNSUPPORTED_PROB_FEEDBACK_LEARNING_ABSENT,
  BERGERT_CUE_WEIGHT_VALUE_MODIFIED (cue-weight/scoring param|op → value_modified).

Outside Bergert, ambiguous construct attribution after NMC clear is routed to
adjudication — never globally mapped to ``value_modified``.

**Production NMC_NEEDS_ADJUDICATION behavior** (no automatic focused LLM pass):
  Rows remain explicitly unresolved
  (``semantic_resolution_status=nmc_needs_adjudication``,
  ``nmc_adjudication_status=unresolved``,
  ``exclude_from_construct_effect_fitting=True``).
  They are not silently treated as NMC, modified, or unchanged; retained
  constructs without attributed modification get transition ``unresolved``.
  Focal/joint construct-transition fitters must drop these rows; coverage /
  audit retain them. A separate offline queue records adjudication payloads.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from utils.mem.schema_participant_transition_v5 import (
    BEHAVIORAL_MOTIFS,
    SCHEMA_VERSION,
    derive_directional_motifs,
    transition_type_for_construct,
)

ACTION_MEANS_FIELD = "action_means_option_A_when_1"
ACTION_MEANS_ALIASES = (
    ACTION_MEANS_FIELD,
    "action_means",
)

RULE_SEED_BASELINE_REF_ABSENT = "SEED_BASELINE_REF_ABSENT"
RULE_UNUSED_HISTORY_ABSENT = "UNUSED_HISTORY_ABSENT"
RULE_BERGERT_ACTION_MEANS_SOLE = "BERGERT_ACTION_MEANS_SOLE_EVIDENCE"
RULE_BERGERT_UNSUPPORTED_PFL = "BERGERT_UNSUPPORTED_PROB_FEEDBACK_LEARNING_ABSENT"
RULE_BERGERT_CUE_WEIGHT_VALUE_MOD = "BERGERT_CUE_WEIGHT_VALUE_MODIFIED"
RULE_NMC_AST_PARAM_OR_STRUCTURE = "NMC_AST_PARAM_OR_STRUCTURE"
RULE_NMC_NEEDS_ADJUDICATION = "NMC_NEEDS_ADJUDICATION"
RULE_MODIFIED_INTERSECT = "MODIFIED_RESTRICT_TO_INTERSECTION"
RULE_REDERIVE_DIRECTIONS = "REDERIVE_DIRECTIONS"

BERGERT_DATASET = "bergert_nosofsky_2007"

# Production resolution stamps (no automatic focused LLM adjudication pass).
SEMANTIC_POSTPROCESS_VERSION = "participant_semantic_v5_2026Sep22"
STATUS_RESOLVED = "resolved"
STATUS_NMC_NEEDS_ADJUDICATION = "nmc_needs_adjudication"
NMC_ADJUDICATION_UNRESOLVED = "unresolved"
NMC_ADJUDICATION_NOT_APPLICABLE = "not_applicable"
TRANSITION_UNRESOLVED = "unresolved"

RAW_LLM_PRESERVE_KEYS = (
    "reference_motif_state",
    "candidate_motif_state",
    "added_motifs",
    "removed_motifs",
    "modified_motifs",
    "structural_operations",
    "no_meaningful_change",
    "transition_by_construct",
    "evidence",
    "confidence",
)


def _is_bergert(dataset: str) -> bool:
    return str(dataset or "").strip() == BERGERT_DATASET


@dataclass
class SemanticCorrection:
    rule_id: str
    side: str  # reference | candidate | both | transition
    construct: Optional[str]
    original_labels: Dict[str, Any]
    corrected_labels: Dict[str, Any]
    code_evidence: List[str]
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _strip_comments_and_docstrings(src: str) -> str:
    try:
        tree = ast.parse(src or "")
    except SyntaxError:
        return src or ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return ast.unparse(tree) if hasattr(ast, "unparse") else (src or "")


def is_seed_baseline_constant_program(code: str) -> bool:
    """True for the official seed baseline: choose(...) body is effectively ``return 0.5``."""
    src = (code or "").strip()
    if not src:
        return False
    # Fast path
    compact = re.sub(r"\s+", "", src)
    if re.search(r"defchoose\([^)]*\):return0\.5\s*$", compact):
        return True
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "return 0.5" in src and src.count("return") == 1 and "option_" not in src
    funcs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(funcs) != 1:
        # allow module-level helpers only if the choose-like fn is constant
        choose_fns = [f for f in funcs if f.name in ("choose", "agent", "policy")]
        if len(choose_fns) != 1 and len(funcs) != 1:
            return False
        fn = choose_fns[0] if choose_fns else funcs[0]
    else:
        fn = funcs[0]
    body = [
        b
        for b in fn.body
        if not (
            isinstance(b, ast.Expr)
            and isinstance(getattr(b, "value", None), ast.Constant)
            and isinstance(b.value.value, str)
        )
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    val = body[0].value
    if isinstance(val, ast.Constant) and val.value == 0.5:
        return True
    if isinstance(val, ast.Num) and float(val.n) == 0.5:  # type: ignore[attr-defined]
        return True
    return False


def _assign_target_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.extend(_assign_target_names(elt))
    return names


def history_runtime_used(code: str, history_param: str = "history") -> bool:
    """True iff ``history`` (or an alias derived from it) is used in the body."""
    src = code or ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        body = src.split(":", 1)[1] if ":" in src else src
        if re.search(
            rf"\b{re.escape(history_param)}\s*[\[\.\(]", body
        ) or re.search(rf"\bfor\b[^\n]*\b{re.escape(history_param)}\b", body):
            return True
        return False

    used = False
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local_aliases: Set[str] = set()
        arg_names = {a.arg for a in node.args.args}
        if history_param in arg_names:
            local_aliases.add(history_param)

        class BodyUse(ast.NodeVisitor):
            def __init__(self) -> None:
                self.hit = False

            def visit_Assign(self, n: ast.Assign) -> None:
                self.visit(n.value)
                src_names = {
                    x.id
                    for x in ast.walk(n.value)
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)
                }
                if src_names & local_aliases:
                    for t in n.targets:
                        for name in _assign_target_names(t):
                            local_aliases.add(name)

            def visit_AnnAssign(self, n: ast.AnnAssign) -> None:
                if n.value is not None:
                    self.visit(n.value)

            def visit_Name(self, n: ast.Name) -> None:
                if isinstance(n.ctx, ast.Load) and n.id in local_aliases:
                    self.hit = True

            def visit_arguments(self, n: ast.arguments) -> None:
                return

        bu = BodyUse()
        for stmt in node.body:
            bu.visit(stmt)
        if bu.hit:
            used = True
    return used


def _code_mentions_action_means(code: str) -> bool:
    return any(a in (code or "") for a in ACTION_MEANS_ALIASES)


def _evidence_mentions_action_means(evidence: Sequence[str]) -> bool:
    blob = " ".join(evidence or []).lower()
    return any(a.lower() in blob for a in ACTION_MEANS_ALIASES)


def _has_value_without_action_means(code: str) -> bool:
    """Cue/option scoring / utility independent of action_means flag."""
    c = code or ""
    c_no_am = c
    for a in ACTION_MEANS_ALIASES:
        c_no_am = c_no_am.replace(a, "")
    patterns = [
        r"option_A",
        r"option_B",
        r"\bcues\b",
        r"score_[AB]",
        r"\butility\b",
        r"\bpayoff\b",
        r"weighted",
        r"sum\s*\(",
    ]
    return any(re.search(p, c_no_am) for p in patterns)


def _has_real_probability_use(code: str) -> bool:
    """True only when problem probability/likelihood/odds/uncertainty fields are read."""
    c = code or ""
    lines = [
        ln
        for ln in c.splitlines()
        if not any(a in ln for a in ACTION_MEANS_ALIASES)
    ]
    body = "\n".join(lines)
    # Require field access patterns — not local names like log_odds / logit alone.
    if re.search(
        r"problem\s*\[\s*['\"]prob|"
        r"problem\s*\.\s*get\s*\(\s*['\"]prob|"
        r"\[['\"]probability['\"]\]|"
        r"\[['\"]likelihood['\"]\]|"
        r"\[['\"]odds['\"]\]|"
        r"\[['\"]uncertainty['\"]\]|"
        r"get\(\s*['\"]probability['\"]|"
        r"get\(\s*['\"]likelihood['\"]|"
        r"get\(\s*['\"]p_win['\"]|"
        r"get\(\s*['\"]p_loss['\"]|"
        r"['\"]prob_['\"]|"
        r"option_[AB][^\n]{0,40}prob",
        body,
        re.I,
    ):
        return True
    return False


def _has_feedback_use(code: str) -> bool:
    c = code or ""
    lines = [
        ln
        for ln in c.splitlines()
        if not any(a in ln for a in ACTION_MEANS_ALIASES)
    ]
    body = "\n".join(lines)
    return bool(
        re.search(
            r"reward|correct(?:ness)?|outcome|feedback|success|failure",
            body,
            re.I,
        )
    )


def _has_learning_use(code: str) -> bool:
    c = code or ""
    return bool(
        re.search(
            r"\bupdate\b|running_?mean|posterior|bayes|q\s*\[|eligib|"
            r"\+=\s*.*(reward|outcome)|learn",
            c,
            re.I,
        )
    )


def construct_supported_without_action_means(construct: str, code: str) -> bool:
    if construct == "value":
        return _has_value_without_action_means(code)
    if construct == "probability_used":
        return _has_real_probability_use(code)
    if construct == "feedback":
        return _has_feedback_use(code)
    if construct == "learning":
        return _has_learning_use(code)
    if construct == "history":
        return history_runtime_used(code)
    return False


def action_means_sole_support(
    construct: str, code: str, evidence: Sequence[str]
) -> bool:
    """True when the construct label is not independently supported and AM is involved."""
    if construct not in (
        "value",
        "probability_used",
        "feedback",
        "learning",
    ):
        return False
    if construct_supported_without_action_means(construct, code):
        return False
    # Label present without independent support: strip if AM appears in code/evidence,
    # OR if there is simply no independent support (AM may be the model's implicit reason).
    if _code_mentions_action_means(code) or _evidence_mentions_action_means(evidence):
        return True
    # No AM mention and no independent support → still strip (unsupported claim)
    # but only under Bergert rule when AM is the known confounder; if no AM at all,
    # leave other validators to handle. Here: sole-AM rule requires AM involvement.
    return False


class _NormalizeAst(ast.NodeTransformer):
    """Normalize names for rename-invariant comparison; keep constants/ops."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        self.generic_visit(node)
        if node.id in ("True", "False", "None"):
            return node
        return ast.copy_location(ast.Name(id="V", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg="V", annotation=None), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.name = "F"
        node.decorator_list = []
        node.returns = None
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


def _ast_dump_normalized(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return None
    tree = _NormalizeAst().visit(tree)
    return ast.dump(tree, annotate_fields=True)


def _numeric_constants(code: str) -> List[float]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", code or "")[:50]]
    out: List[float] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(
            n.value, bool
        ):
            out.append(float(n.value))
        elif isinstance(n, ast.Num):  # type: ignore[attr-defined]
            out.append(float(n.n))  # type: ignore[attr-defined]
    return out


CounterLike = Dict[str, int]


def _operator_bag(code: str) -> CounterLike:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return {}
    bag: Dict[str, int] = {}
    for n in ast.walk(tree):
        key = type(n).__name__
        if key in (
            "Add",
            "Sub",
            "Mult",
            "Div",
            "Pow",
            "BitOr",
            "BitAnd",
            "If",
            "For",
            "While",
            "Compare",
            "BoolOp",
            "AugAssign",
            "Return",
            "Call",
        ):
            bag[key] = bag.get(key, 0) + 1
        if isinstance(n, ast.BinOp):
            op = type(n.op).__name__
            bag[f"op_{op}"] = bag.get(f"op_{op}", 0) + 1
        if isinstance(n, (ast.If, ast.For, ast.While)):
            bag["control"] = bag.get("control", 0) + 1
    return bag


@dataclass
class AstDiffResult:
    equivalent: bool
    numeric_diff: bool
    operator_diff: bool
    control_diff: bool
    structure_diff: bool
    notes: List[str] = field(default_factory=list)

    @property
    def meaningful_non_nmc(self) -> bool:
        return (
            self.numeric_diff
            or self.operator_diff
            or self.control_diff
            or self.structure_diff
        )


def compare_ast_semantics(ref_code: str, cand_code: str) -> AstDiffResult:
    d1 = _ast_dump_normalized(ref_code)
    d2 = _ast_dump_normalized(cand_code)
    nums_diff = _numeric_constants(ref_code) != _numeric_constants(cand_code)
    ops1, ops2 = _operator_bag(ref_code), _operator_bag(cand_code)
    op_diff = ops1 != ops2
    control_diff = ops1.get("control", 0) != ops2.get("control", 0) or any(
        ops1.get(k, 0) != ops2.get(k, 0) for k in ("If", "For", "While")
    )
    structure_diff = d1 is None or d2 is None or d1 != d2
    # If dumps equal after name-normalize, treat as rename/format equivalent
    # even if raw text differs — unless numbers somehow differ (shouldn't).
    if d1 is not None and d2 is not None and d1 == d2 and not nums_diff:
        return AstDiffResult(
            equivalent=True,
            numeric_diff=False,
            operator_diff=False,
            control_diff=False,
            structure_diff=False,
            notes=["normalized_ast_identical"],
        )
    notes = []
    if nums_diff:
        notes.append("numeric_constants_differ")
    if op_diff:
        notes.append("operator_or_call_bag_differs")
    if control_diff:
        notes.append("control_flow_differs")
    if structure_diff and not (nums_diff or op_diff or control_diff):
        notes.append("normalized_ast_structure_differs")
    return AstDiffResult(
        equivalent=False,
        numeric_diff=nums_diff,
        operator_diff=op_diff,
        control_diff=control_diff,
        structure_diff=structure_diff,
        notes=notes,
    )


def _labels_snapshot(ann: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reference_motif_state": list(ann.get("reference_motif_state") or []),
        "candidate_motif_state": list(ann.get("candidate_motif_state") or []),
        "added_motifs": list(ann.get("added_motifs") or []),
        "removed_motifs": list(ann.get("removed_motifs") or []),
        "modified_motifs": list(ann.get("modified_motifs") or []),
        "no_meaningful_change": bool(ann.get("no_meaningful_change")),
        "transition_by_construct": dict(ann.get("transition_by_construct") or {}),
    }


def _rederive(ann: Dict[str, Any]) -> None:
    ref = list(ann.get("reference_motif_state") or [])
    cand = list(ann.get("candidate_motif_state") or [])
    nmc = bool(ann.get("no_meaningful_change"))
    modified_in = list(ann.get("modified_motifs") or [])
    inter = set(ref) & set(cand)
    modified = [m for m in modified_in if m in inter]
    if nmc:
        modified = []
    added, removed, modified = derive_directional_motifs(ref, cand, modified)
    if nmc:
        added, removed, modified = [], [], []
    ann["added_motifs"] = list(added)
    ann["removed_motifs"] = list(removed)
    ann["modified_motifs"] = list(modified)
    tbc: Dict[str, str] = {}
    for m in BEHAVIORAL_MOTIFS:
        tbc[m] = transition_type_for_construct(
            m,
            reference_has=m in ref,
            candidate_has=m in cand,
            modified=m in modified,
            no_meaningful_change=nmc,
        )
    ann["transition_by_construct"] = tbc


def postprocess_participant_annotation(
    ann: Dict[str, Any],
    *,
    reference_code: str,
    candidate_code: str,
    dataset: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[SemanticCorrection], Dict[str, Any]]:
    """Apply deterministic semantic rules; return (corrected, corrections, meta).

    ``meta`` may include ``needs_nmc_adjudication`` with a focused repair payload.
    """
    out = copy.deepcopy(ann)
    corrections: List[SemanticCorrection] = []
    meta: Dict[str, Any] = {"needs_nmc_adjudication": None}
    ds = str(dataset or out.get("dataset") or "")
    evidence = list(out.get("evidence") or [])

    # --- 1) Seed baseline constant → reference all-absent ---
    if is_seed_baseline_constant_program(reference_code):
        before = _labels_snapshot(out)
        if out.get("reference_motif_state"):
            out["reference_motif_state"] = []
            corrections.append(
                SemanticCorrection(
                    rule_id=RULE_SEED_BASELINE_REF_ABSENT,
                    side="reference",
                    construct=None,
                    original_labels=before,
                    corrected_labels=_labels_snapshot(out),
                    code_evidence=[(reference_code or "")[:240]],
                    note="Exact seed-baseline constant program; all five constructs absent on reference.",
                )
            )

    # --- 2) Unused history on each side ---
    for side, code, key in (
        ("reference", reference_code, "reference_motif_state"),
        ("candidate", candidate_code, "candidate_motif_state"),
    ):
        state = list(out.get(key) or [])
        if "history" in state and not history_runtime_used(code):
            before = _labels_snapshot(out)
            out[key] = [m for m in state if m != "history"]
            corrections.append(
                SemanticCorrection(
                    rule_id=RULE_UNUSED_HISTORY_ABSENT,
                    side=side,
                    construct="history",
                    original_labels=before,
                    corrected_labels=_labels_snapshot(out),
                    code_evidence=[
                        f"history_runtime_used=false; signature-only or unused arg",
                        (code or "")[:240],
                    ],
                    note="Removed History; unused formal history / no alias use.",
                )
            )

    # --- 3) Bergert-only semantic rules (strict dataset gate) ---
    if _is_bergert(ds):
        # 3a) action_means sole-evidence strip (Value/P/F/L)
        for side, code, key in (
            ("reference", reference_code, "reference_motif_state"),
            ("candidate", candidate_code, "candidate_motif_state"),
        ):
            state = list(out.get(key) or [])
            for construct in (
                "value",
                "probability_used",
                "feedback",
                "learning",
            ):
                if construct not in state:
                    continue
                if not action_means_sole_support(construct, code, evidence):
                    continue
                if not (
                    _code_mentions_action_means(code)
                    or _evidence_mentions_action_means(evidence)
                ):
                    continue
                before = _labels_snapshot(out)
                out[key] = [m for m in state if m != construct]
                state = list(out[key])
                corrections.append(
                    SemanticCorrection(
                        rule_id=RULE_BERGERT_ACTION_MEANS_SOLE,
                        side=side,
                        construct=construct,
                        original_labels=before,
                        corrected_labels=_labels_snapshot(out),
                        code_evidence=[
                            f"construct={construct}",
                            f"action_means_in_code={_code_mentions_action_means(code)}",
                            f"independent_support=false",
                            f"dataset={ds}",
                            (code or "")[:320],
                        ],
                        note=(
                            f"Removed {construct}: evidence relies solely on "
                            f"{ACTION_MEANS_FIELD} / no independent implementation."
                        ),
                    )
                )

        # 3b) Bergert-only: P/F/L without independent support
        for side, code, key in (
            ("reference", reference_code, "reference_motif_state"),
            ("candidate", candidate_code, "candidate_motif_state"),
        ):
            state = list(out.get(key) or [])
            for construct in ("probability_used", "feedback", "learning"):
                if construct not in state:
                    continue
                if construct_supported_without_action_means(construct, code):
                    continue
                before = _labels_snapshot(out)
                out[key] = [m for m in state if m != construct]
                state = list(out[key])
                corrections.append(
                    SemanticCorrection(
                        rule_id=RULE_BERGERT_UNSUPPORTED_PFL,
                        side=side,
                        construct=construct,
                        original_labels=before,
                        corrected_labels=_labels_snapshot(out),
                        code_evidence=[
                            f"construct={construct}",
                            "independent_support=false",
                            f"dataset={ds}",
                            (code or "")[:320],
                        ],
                        note=(
                            f"Removed {construct}: no code evidence of actual "
                            "probability / realized feedback / learning update."
                        ),
                    )
                )

    # --- 4) NMC vs AST (generic: clear NMC only; never auto-map to value_modified) ---
    nmc = bool(out.get("no_meaningful_change"))
    diff = compare_ast_semantics(reference_code, candidate_code)
    if nmc and diff.meaningful_non_nmc:
        before = _labels_snapshot(out)
        out["no_meaningful_change"] = False
        ref = set(out.get("reference_motif_state") or [])
        cand = set(out.get("candidate_motif_state") or [])
        inter = ref & cand
        mod = list(out.get("modified_motifs") or [])
        structural = list(out.get("structural_operations") or [])
        if diff.numeric_diff and "parameter_change" not in structural:
            structural.append("parameter_change")
        out["modified_motifs"] = mod
        out["structural_operations"] = structural
        corrections.append(
            SemanticCorrection(
                rule_id=RULE_NMC_AST_PARAM_OR_STRUCTURE,
                side="both",
                construct=None,
                original_labels=before,
                corrected_labels=_labels_snapshot(out),
                code_evidence=list(diff.notes)
                + [
                    f"numeric_diff={diff.numeric_diff}",
                    f"operator_diff={diff.operator_diff}",
                    f"control_diff={diff.control_diff}",
                    "auto_value_modified=false",
                ],
                note=(
                    "Cleared NMC: normalized AST differs in parameters/ops/control/"
                    "update shape. Construct attribution deferred (generic path)."
                ),
            )
        )
        meta["needs_nmc_adjudication"] = {
            "question": (
                "Is the code change behaviorally/algebraically equivalent? "
                "If not, which already-present construct was modified?"
            ),
            "allowed_constructs": sorted(inter),
            "ast_diff": {
                "numeric_diff": diff.numeric_diff,
                "operator_diff": diff.operator_diff,
                "control_diff": diff.control_diff,
                "structure_diff": diff.structure_diff,
                "notes": list(diff.notes),
            },
            "reference_code_sha256": hashlib.sha256(
                (reference_code or "").encode()
            ).hexdigest(),
            "candidate_code_sha256": hashlib.sha256(
                (candidate_code or "").encode()
            ).hexdigest(),
        }
        corrections.append(
            SemanticCorrection(
                rule_id=RULE_NMC_NEEDS_ADJUDICATION,
                side="both",
                construct=None,
                original_labels=before,
                corrected_labels=_labels_snapshot(out),
                code_evidence=list(diff.notes),
                note="Queued for focused NMC adjudication (no auto construct assignment).",
            )
        )

    # --- 4b) Bergert-only: cue-weight/scoring param|op → value_modified ---
    if _is_bergert(ds):
        ref_set = set(out.get("reference_motif_state") or [])
        cand_set = set(out.get("candidate_motif_state") or [])
        inter_b = ref_set & cand_set
        if (
            "value" in inter_b
            and (diff.numeric_diff or diff.operator_diff)
            and (
                _has_value_without_action_means(reference_code)
                or _has_value_without_action_means(candidate_code)
            )
        ):
            mod = list(out.get("modified_motifs") or [])
            if "value" not in mod:
                before = _labels_snapshot(out)
                if out.get("no_meaningful_change"):
                    out["no_meaningful_change"] = False
                mod.append("value")
                out["modified_motifs"] = mod
                structural = list(out.get("structural_operations") or [])
                if diff.numeric_diff and "parameter_change" not in structural:
                    structural.append("parameter_change")
                    out["structural_operations"] = structural
                corrections.append(
                    SemanticCorrection(
                        rule_id=RULE_BERGERT_CUE_WEIGHT_VALUE_MOD,
                        side="both",
                        construct="value",
                        original_labels=before,
                        corrected_labels=_labels_snapshot(out),
                        code_evidence=[
                            f"dataset={ds}",
                            f"numeric_diff={diff.numeric_diff}",
                            f"operator_diff={diff.operator_diff}",
                            "cue_or_option_scoring_present=true",
                        ],
                        note=(
                            "Bergert cue-weight/scoring parameter or operator change "
                            "→ value_modified (dataset-gated)."
                        ),
                    )
                )
                if meta.get("needs_nmc_adjudication"):
                    meta["needs_nmc_adjudication"] = None
                    corrections = [
                        c
                        for c in corrections
                        if c.rule_id != RULE_NMC_NEEDS_ADJUDICATION
                    ]

    # --- 5) Restrict modified to intersection; rederive directions ---
    before = _labels_snapshot(out)
    ref = list(out.get("reference_motif_state") or [])
    cand = list(out.get("candidate_motif_state") or [])
    inter = set(ref) & set(cand)
    mod = [m for m in (out.get("modified_motifs") or []) if m in inter]
    if mod != list(out.get("modified_motifs") or []):
        out["modified_motifs"] = mod
        corrections.append(
            SemanticCorrection(
                rule_id=RULE_MODIFIED_INTERSECT,
                side="transition",
                construct=None,
                original_labels=before,
                corrected_labels=_labels_snapshot(out),
                code_evidence=[f"intersection={sorted(inter)}"],
                note="Dropped modified_motifs outside reference∩candidate.",
            )
        )
    # Ensure schema consistency when not NMC: need some signal
    if not out.get("no_meaningful_change"):
        _rederive(out)
        added = out.get("added_motifs") or []
        removed = out.get("removed_motifs") or []
        modified = out.get("modified_motifs") or []
        structural = out.get("structural_operations") or []
        if not (added or removed or modified or structural):
            # Presence inventories identical and empty mods after strip — treat as NMC
            # only if AST-equivalent; else keep structural parameter_change if nums differ
            if diff.equivalent or not diff.meaningful_non_nmc:
                out["no_meaningful_change"] = True
            elif diff.numeric_diff:
                out["structural_operations"] = list(structural) + (
                    [] if "parameter_change" in structural else ["parameter_change"]
                )
    _rederive(out)
    if corrections:
        # Final snapshot note for rederive (compact — only if directions changed)
        after = _labels_snapshot(out)
        if after != before:
            corrections.append(
                SemanticCorrection(
                    rule_id=RULE_REDERIVE_DIRECTIONS,
                    side="transition",
                    construct=None,
                    original_labels=before,
                    corrected_labels=after,
                    code_evidence=[],
                    note="Re-derived added/removed/modified/unchanged + transition_by_construct.",
                )
            )

    out["schema_version"] = int(out.get("schema_version") or SCHEMA_VERSION)
    out["semantic_postprocess_version"] = SEMANTIC_POSTPROCESS_VERSION
    _stamp_resolution_status(out, meta)
    return out, corrections, meta


def _stamp_resolution_status(out: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Mark NMC adjudication rows as explicitly unresolved (no silent NMC/mod/unchanged)."""
    if meta.get("needs_nmc_adjudication"):
        out["semantic_resolution_status"] = STATUS_NMC_NEEDS_ADJUDICATION
        out["nmc_adjudication_status"] = NMC_ADJUDICATION_UNRESOLVED
        out["exclude_from_construct_effect_fitting"] = True
        ref = set(out.get("reference_motif_state") or [])
        cand = set(out.get("candidate_motif_state") or [])
        mod = set(out.get("modified_motifs") or [])
        tbc = dict(out.get("transition_by_construct") or {})
        for m in BEHAVIORAL_MOTIFS:
            if m in ref and m in cand and m not in mod:
                # Ambiguous construct attribution — not NMC, not modified, not unchanged.
                tbc[m] = TRANSITION_UNRESOLVED
        out["transition_by_construct"] = tbc
        # Directional modified flags stay empty; do not invent construct mods.
        return
    out["semantic_resolution_status"] = STATUS_RESOLVED
    out["nmc_adjudication_status"] = NMC_ADJUDICATION_NOT_APPLICABLE
    out["exclude_from_construct_effect_fitting"] = False


def snapshot_raw_llm_annotation(ann: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy validated LLM fields before semantic correction."""
    raw: Dict[str, Any] = {}
    for k in RAW_LLM_PRESERVE_KEYS:
        if k in ann:
            raw[k] = copy.deepcopy(ann.get(k))
    return raw


def finalize_v5_annotation_for_write(
    validated_ann: Dict[str, Any],
    *,
    reference_code: str,
    candidate_code: str,
    dataset: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[SemanticCorrection], Dict[str, Any]]:
    """Production finalize: preserve raw LLM fields → postprocess → stamp status.

    Returns corrected annotation (with ``raw_llm_annotation``), corrections, meta.
    Does **not** run a focused LLM adjudication pass; unresolved NMC rows stay
    explicitly unresolved for offline queue / audit.
    """
    raw = snapshot_raw_llm_annotation(validated_ann)
    fixed, corrs, meta = postprocess_participant_annotation(
        validated_ann,
        reference_code=reference_code,
        candidate_code=candidate_code,
        dataset=dataset,
    )
    fixed["raw_llm_annotation"] = raw
    return fixed, corrs, meta


def row_excluded_from_construct_effect_fitting(ann_or_row: Dict[str, Any]) -> bool:
    """True when construct-transition effect fitters must drop this row."""
    if bool(ann_or_row.get("exclude_from_construct_effect_fitting")):
        return True
    status = str(ann_or_row.get("semantic_resolution_status") or "")
    if status == STATUS_NMC_NEEDS_ADJUDICATION:
        return True
    if str(ann_or_row.get("nmc_adjudication_status") or "") == NMC_ADJUDICATION_UNRESOLVED:
        return True
    return False


def filter_frame_for_construct_effect_fitting(df: Any) -> Tuple[Any, int]:
    """Drop unresolved-NMC rows from a MEM analysis frame.

    Returns ``(filtered_df, n_excluded)``. Legacy CSVs without status columns
    are left unchanged (n_excluded=0).
    """
    n_before = int(len(df))
    if "exclude_from_construct_effect_fitting" in df.columns:
        excl = (
            pd_to_int_flag(df["exclude_from_construct_effect_fitting"]) == 1
        )
        out = df.loc[~excl].copy()
        return out, n_before - int(len(out))
    if "semantic_resolution_status" in df.columns:
        status = df["semantic_resolution_status"].fillna("resolved").astype(str)
        out = df.loc[status != STATUS_NMC_NEEDS_ADJUDICATION].copy()
        return out, n_before - int(len(out))
    if "nmc_adjudication_status" in df.columns:
        status = df["nmc_adjudication_status"].fillna("not_applicable").astype(str)
        out = df.loc[status != NMC_ADJUDICATION_UNRESOLVED].copy()
        return out, n_before - int(len(out))
    return df, 0


def pd_to_int_flag(series: Any) -> Any:
    """Coerce a CSV column to 0/1 ints (null → 0)."""
    import pandas as pd  # local import keeps postprocess free of hard pandas dep at import for unit tests that don't need it

    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def apply_postprocess_jsonl(
    in_path: Path,
    *,
    out_path: Path,
    corrections_path: Path,
    adjudication_path: Path,
    code_loader,
) -> Dict[str, Any]:
    """Offline apply to an annotations_v5.jsonl.

    ``code_loader(ann) -> (reference_code, candidate_code)``.
    """
    n = 0
    n_corr_rows = 0
    rule_counts: Dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout, corrections_path.open("w", encoding="utf-8") as fcorr, adjudication_path.open(
        "w", encoding="utf-8"
    ) as fadj:
        for line in fin:
            if not line.strip():
                continue
            ann = json.loads(line)
            n += 1
            ref_c, cand_c = code_loader(ann)
            fixed, corrs, meta = postprocess_participant_annotation(
                ann,
                reference_code=ref_c,
                candidate_code=cand_c,
                dataset=ann.get("dataset"),
            )
            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            if corrs:
                n_corr_rows += 1
            for c in corrs:
                rule_counts[c.rule_id] = rule_counts.get(c.rule_id, 0) + 1
                rec = c.to_dict()
                rec["global_candidate_id"] = ann.get("global_candidate_id")
                rec["candidate_id"] = ann.get("candidate_id")
                rec["participant_id"] = ann.get("participant_id")
                rec["phase"] = ann.get("phase")
                rec["source"] = ann.get("source")
                fcorr.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if meta.get("needs_nmc_adjudication"):
                fadj.write(
                    json.dumps(
                        {
                            "global_candidate_id": ann.get("global_candidate_id"),
                            "candidate_id": ann.get("candidate_id"),
                            "payload": meta["needs_nmc_adjudication"],
                            "reference_motif_state": fixed.get("reference_motif_state"),
                            "candidate_motif_state": fixed.get("candidate_motif_state"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return {
        "n_rows": n,
        "n_rows_with_corrections": n_corr_rows,
        "rule_counts": rule_counts,
        "out_path": str(out_path),
        "corrections_path": str(corrections_path),
        "adjudication_path": str(adjudication_path),
    }
