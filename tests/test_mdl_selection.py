"""MDL ranking overlay: formula, AST size, and lambda=0 identity."""

from __future__ import annotations

from utils.teh.mdl_selection import (
    attach_mdl_fields,
    candidate_rank_key,
    compute_mdl_score,
    elite_core,
    elite_rank_key,
    program_ast_size,
    selection_trial_count,
    selection_used_val,
    sort_candidates,
    sort_elites,
    with_elite_mdl_score,
)

SMALL = """def choose(problem, history):
    return 0.5
"""

SMALL_COMMENTS = """def choose(problem, history):
    # extra comment and trailing spaces   
    return 0.5
"""

SMALL_TABS = """def choose(problem, history):\n\n\treturn 0.5\n"""

FUNC_DOC = '''def choose(problem, history):
    """function docstring"""
    return 0.5
'''

MODULE_DOC = '''"""module docstring"""
def choose(problem, history):
    return 0.5
'''

CLASS_DOC = '''class Policy:
    """class docstring is counted"""
    def choose(self, problem, history):
        return 0.5
'''

CLASS_NO_DOC = '''class Policy:
    def choose(self, problem, history):
        return 0.5
'''

LARGER = """def choose(problem, history):
    x = 1
    y = 2
    z = x + y
    return 0.5
"""


def test_mdl_formula_n_times_score_minus_lambda_size():
    assert compute_mdl_score(-0.4, 10, 25, 1.0) == 10 * -0.4 - 1.0 * 25
    assert compute_mdl_score(-0.4, 10, 25, 0.0) == 10 * -0.4
    assert compute_mdl_score(-0.2, 3, 8, 2.5) == 3 * -0.2 - 2.5 * 8


def test_ast_size_ignores_comments_and_formatting():
    base = program_ast_size(SMALL)
    assert base > 0
    assert program_ast_size(SMALL_COMMENTS) == base
    assert program_ast_size(SMALL_TABS) == base
    assert program_ast_size(SMALL.replace("    ", "\t")) == base


def test_function_and_module_docstrings_excluded():
    base = program_ast_size(SMALL)
    assert program_ast_size(FUNC_DOC) == base
    assert program_ast_size(MODULE_DOC) == base


def test_class_docstring_is_counted():
    assert program_ast_size(CLASS_DOC) > program_ast_size(CLASS_NO_DOC)


def test_selection_trial_count_matches_score_fallback():
    assert selection_trial_count("train", 12, 4, val_used=False) == 12
    assert selection_trial_count("train_val", 12, 4, val_used=True) == 16
    assert selection_trial_count("train_val", 12, 4, val_used=False) == 12
    assert selection_used_val("train_val", 4, -0.3) is True
    assert selection_used_val("train_val", 0, -0.3) is False
    assert selection_used_val("train", 4, -0.3) is False


def test_lambda_zero_ranking_ignores_size_and_stored_mdl():
    small = {"fitness": -0.2, "mdl_score": 99.0, "runtime_valid": True, "code": SMALL}
    large = {"fitness": -0.1, "mdl_score": -99.0, "runtime_valid": True, "code": LARGER}
    rows = [small, large]
    sort_candidates(rows, 0.0)
    assert rows[0] is large
    assert candidate_rank_key(small, 0.0) == -0.2

    a = with_elite_mdl_score(("a", -0.2, None, "a", None, None, -0.2), 50.0, 0.0)
    b = with_elite_mdl_score(("b", -0.1, None, "b", None, None, -0.1), -50.0, 0.0)
    assert len(a) == 7
    elites = [a, b]
    sort_elites(elites, 0.0)
    assert elites[0][3] == "b"
    assert elite_rank_key(a, 0.0) == -0.2


def test_lambda_positive_prefers_higher_mdl_score():
    n = 10
    lam = 1.0
    small_size = program_ast_size(SMALL)
    large_size = program_ast_size(LARGER)
    assert large_size > small_size
    score = -0.3
    small = {
        "fitness": score,
        "code": SMALL,
        "runtime_valid": True,
        "selection_score": score,
    }
    large = {
        "fitness": score,
        "code": LARGER,
        "runtime_valid": True,
        "selection_score": score,
    }
    attach_mdl_fields(small, code=SMALL, selection_score=score, n=n, mdl_lambda=lam, runtime_valid=True)
    attach_mdl_fields(large, code=LARGER, selection_score=score, n=n, mdl_lambda=lam, runtime_valid=True)
    assert small["mdl_score"] == compute_mdl_score(score, n, small_size, lam)
    assert small["mdl_score"] > large["mdl_score"]
    rows = [large, small]
    sort_candidates(rows, lam)
    assert rows[0] is small

    elite_small = with_elite_mdl_score(
        (SMALL, score, None, "small", None, None, score),
        small["mdl_score"],
        lam,
    )
    elite_large = with_elite_mdl_score(
        (LARGER, score, None, "large", None, None, score),
        large["mdl_score"],
        lam,
    )
    elites = [elite_large, elite_small]
    sort_elites(elites, lam)
    assert elites[0][3] == "small"


def test_elite_core_unpacks_mdl_eight_tuple():
    """Regression: job 245398 died unpacking MDL 8-tuples as 7 values."""
    core = (SMALL, -0.2, 0.5, "prog", None, None, -0.2)
    parent = with_elite_mdl_score(core, -12.0, 1.0)
    assert len(parent) == 8
    try:
        _code, _fitness, _test_acc, _prog_id, _, _, _train_acc = parent
        raise AssertionError("8-tuple must not unpack into 7 names")
    except ValueError as exc:
        assert "too many values to unpack" in str(exc)
    code, fitness, test_acc, prog_id, _, _, train_acc = elite_core(parent)
    assert code == SMALL
    assert fitness == -0.2
    assert test_acc == 0.5
    assert prog_id == "prog"
    assert train_acc == -0.2
    assert elite_core(core) == core
    mse_code, mse_fit, mse_test, mse_id, train_mse, test_mse = elite_core(parent)[:6]
    assert mse_code == SMALL and mse_fit == -0.2 and mse_id == "prog"
    assert train_mse is None and test_mse is None


def test_invalid_candidates_do_not_get_mdl_fields():
    row = {"fitness": -1e9, "code": SMALL, "runtime_valid": False}
    attach_mdl_fields(row, code=SMALL, selection_score=-0.2, n=10, mdl_lambda=1.0, runtime_valid=False)
    assert "mdl_score" not in row


def test_cap_elite_preserving_program_ids_lambda_zero_unchanged():
    import teh

    elite = [
        ("code_a", 0.1, 0.0, "global_target", None, None, 0.1),
        ("code_b", 0.9, 0.0, "explore_candidate_0", None, None, 0.9),
        ("code_c", 0.8, 0.0, "explore_candidate_1", None, None, 0.8),
    ]
    val = [0.1, 0.9, 0.8]
    teh._cap_elite_preserving_program_ids(
        elite,
        val,
        elite_cap=2,
        pinned_ids=["global_target"],
        track_elite_val_loglik=True,
    )
    ids = [str(p[3]) for p in elite]
    assert "global_target" in ids
    assert len(elite) == 2
    assert ids[0] == "global_target" or "explore_candidate_0" in ids
