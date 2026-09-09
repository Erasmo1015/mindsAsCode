"""MDL-inspired ranking overlay for TEH/PICS elite selection.

``mdl_score = n * selection_score - mdl_lambda * program_size``

``selection_score`` is the existing mean train or trial-weighted train+val
log-likelihood. ``n`` is the number of trials that underlie that mean.
``program_size`` is the Python AST node count after dropping module/function
docstrings (comments and formatting are not in the AST).

When ``mdl_lambda <= 0`` this module must not change ranking keys: callers keep
sorting on the original fitness / elite tuple index 1.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

EliteParent = Tuple[Any, ...]

_DOCSTRING_CONTAINERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)
_ELITE_MDL_IDX = 7


def normalize_mdl_lambda(value: Any) -> float:
    if value is None:
        return 0.0
    lam = float(value)
    if lam < 0.0:
        raise ValueError(f"mdl_lambda must be >= 0, got {value!r}")
    return lam


def mdl_enabled(mdl_lambda: Any) -> bool:
    return normalize_mdl_lambda(mdl_lambda) > 0.0


def _is_docstring_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    # Python 3.7 compatibility (ast.Str removed in 3.14).
    return type(node).__name__ == "Str" and isinstance(getattr(node, "s", None), str)


def _docstring_node_ids(tree: ast.AST) -> set:
    skip: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_CONTAINERS):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr):
            continue
        if _is_docstring_constant(first.value):
            skip.add(id(first))
            skip.add(id(first.value))
    return skip


def program_ast_size(code: str) -> int:
    """Count AST nodes, excluding module and function (not class) docstrings."""
    text = code if isinstance(code, str) else str(code or "")
    if not text.strip():
        return 0
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 10**9
    skip = _docstring_node_ids(tree)
    return sum(1 for node in ast.walk(tree) if id(node) not in skip)


def selection_used_val(
    evolution_selection_score: str,
    n_val: int,
    val_loglik: Any,
) -> bool:
    if str(evolution_selection_score).strip() != "train_val":
        return False
    if int(n_val) <= 0 or val_loglik is None:
        return False
    try:
        return math.isfinite(float(val_loglik))
    except (TypeError, ValueError):
        return False


def selection_trial_count(
    evolution_selection_score: str,
    n_train: int,
    n_val: int,
    *,
    val_used: bool,
) -> int:
    """Exact trial count underlying the mean selection_score."""
    n_tr = max(0, int(n_train))
    n_vl = max(0, int(n_val))
    if str(evolution_selection_score).strip() == "train_val" and val_used and n_vl > 0:
        return n_tr + n_vl
    return n_tr


def compute_mdl_score(
    selection_score: float,
    n: int,
    program_size: int,
    mdl_lambda: float,
) -> float:
    return float(n) * float(selection_score) - float(mdl_lambda) * float(program_size)


def compute_mdl_fields(
    *,
    selection_score: float,
    n: int,
    program_size: int,
    mdl_lambda: float,
) -> Dict[str, Any]:
    total = float(n) * float(selection_score)
    penalty = float(mdl_lambda) * float(program_size)
    return {
        "mdl_n": int(n),
        "mdl_total_loglik": total,
        "program_ast_size": int(program_size),
        "mdl_complexity_penalty": penalty,
        "mdl_score": total - penalty,
    }


def attach_mdl_fields(
    result: Dict[str, Any],
    *,
    code: str,
    selection_score: float,
    n: int,
    mdl_lambda: float,
    runtime_valid: bool,
) -> Dict[str, Any]:
    """Add MDL log fields when enabled. Does not change ``fitness`` / train_val_loglik."""
    if not mdl_enabled(mdl_lambda) or not runtime_valid:
        return result
    size = program_ast_size(code)
    fields = compute_mdl_fields(
        selection_score=float(selection_score),
        n=int(n),
        program_size=size,
        mdl_lambda=float(mdl_lambda),
    )
    result.update(fields)
    return result


def candidate_rank_key(
    result: Dict[str, Any],
    mdl_lambda: float = 0.0,
    *,
    score_key: str = "fitness",
) -> float:
    if mdl_enabled(mdl_lambda) and result.get("runtime_valid", True) is not False:
        score = result.get("mdl_score")
        if score is not None:
            return float(score)
    raw = result.get(score_key, result.get("fitness"))
    if raw is None:
        return float("-inf")
    return float(raw)


def sort_candidates(
    results: List[Dict[str, Any]],
    mdl_lambda: float = 0.0,
    *,
    score_key: str = "fitness",
) -> None:
    results.sort(
        key=lambda row: candidate_rank_key(row, mdl_lambda, score_key=score_key),
        reverse=True,
    )


def elite_rank_key(parent: Sequence[Any], mdl_lambda: float = 0.0) -> float:
    if mdl_enabled(mdl_lambda) and len(parent) > _ELITE_MDL_IDX and parent[_ELITE_MDL_IDX] is not None:
        return float(parent[_ELITE_MDL_IDX])
    return float(parent[1])


def sort_elites(elite_parents: List[EliteParent], mdl_lambda: float = 0.0) -> None:
    elite_parents.sort(key=lambda parent: elite_rank_key(parent, mdl_lambda), reverse=True)


def sort_elite_pairs(
    pairs: List[Tuple[EliteParent, Any]],
    mdl_lambda: float = 0.0,
) -> None:
    pairs.sort(key=lambda row: elite_rank_key(row[0], mdl_lambda), reverse=True)


def with_elite_mdl_score(
    parent: Sequence[Any],
    mdl_score: Optional[float],
    mdl_lambda: float,
) -> EliteParent:
    """Keep a 7-tuple when MDL is off; append mdl_score as index 7 when on."""
    core = tuple(parent[:7])
    if not mdl_enabled(mdl_lambda) or mdl_score is None:
        return core
    return core + (float(mdl_score),)


def elite_mdl_fields(parent: Sequence[Any], mdl_lambda: float = 0.0) -> Optional[Dict[str, Any]]:
    if not mdl_enabled(mdl_lambda) or len(parent) <= _ELITE_MDL_IDX:
        return None
    score = parent[_ELITE_MDL_IDX]
    if score is None:
        return None
    code = parent[0] or ""
    size = program_ast_size(str(code))
    selection = float(parent[1])
    n_est = None
    # Reconstruct n when possible: mdl_score = n * S - lam * size
    lam = float(mdl_lambda)
    denom = float(selection)
    if abs(denom) > 1e-15:
        n_est = (float(score) + lam * float(size)) / denom
    return {
        "mdl_score": float(score),
        "program_ast_size": int(size),
        "mdl_complexity_penalty": lam * float(size),
        "mdl_n_reconstructed": n_est,
    }


def apply_mdl_to_scored_elite(
    parent: Sequence[Any],
    *,
    selection_score: float,
    n: int,
    mdl_lambda: float,
    runtime_valid: bool = True,
) -> EliteParent:
    core = tuple(parent[:7])
    if not mdl_enabled(mdl_lambda) or not runtime_valid:
        return core
    size = program_ast_size(str(parent[0] or ""))
    score = compute_mdl_score(selection_score, n, size, mdl_lambda)
    return core + (score,)
