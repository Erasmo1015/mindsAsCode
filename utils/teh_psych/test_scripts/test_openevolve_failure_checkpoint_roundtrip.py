"""CPU-only OpenEvolve evaluator → database → best/parent → checkpoint round-trip.

Uses the audit checkout at reference_repos/openevolve_official_audit (the
imported reference_repos/openevolve clone is not on this host). No vLLM, no GPU.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_OE = REPO_ROOT / "reference_repos" / "openevolve_official_audit"
if not AUDIT_OE.is_dir():
    pytest.skip("OpenEvolve audit checkout missing", allow_module_level=True)

# Official config.py imports dacite at module load; do not pip-install it here.
if "dacite" not in sys.modules:
    _dacite = types.ModuleType("dacite")

    class _DaciteConfig:
        def __init__(self, **kwargs):
            pass

    def _from_dict(data_class, data, config=None):
        return data_class(**data)

    _dacite.Config = _DaciteConfig
    _dacite.from_dict = _from_dict
    sys.modules["dacite"] = _dacite

sys.path.insert(0, str(AUDIT_OE))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baseline_methods" / "Psych101"))

# A sibling test stubs `openevolve` for runner imports. Drop that stub so this
# file can load the official audit checkout.
for _name in list(sys.modules):
    if _name == "openevolve" or _name.startswith("openevolve."):
        _mod = sys.modules[_name]
        _file = getattr(_mod, "__file__", None) or ""
        if "openevolve_official_audit" not in str(_file):
            del sys.modules[_name]

from openevolve.config import Config, DatabaseConfig, EvaluatorConfig  # noqa: E402
from openevolve.database import Program, ProgramDatabase  # noqa: E402
from openevolve.evaluator import Evaluator  # noqa: E402
from openevolve.utils.format_utils import format_metrics_safe  # noqa: E402
from openevolve.utils.metrics_utils import get_fitness_score  # noqa: E402

from run_openevolve import (  # noqa: E402
    CHOICE13K_LOGLIK_EPS,
    FAILED_COMBINED_SCORE,
    _find_best_program_by_observed_loglik,
    _install_runtime_patches,
    _render_evaluator_py,
    _write_evolution_split_json,
    truncate_vanilla_messages,
)


SEED = "def choose(problem, history):\n    return 0.5\n"
INVALID = "def choose(problem, history):\n    return 2.0\n"
TRIALS = [
    {"problem": {"option_keys": ["A", "B"]}, "history": [], "action": 0},
    {"problem": {"option_keys": ["A", "B"]}, "history": [], "action": 1},
]


def _strict_json(obj) -> str:
    return json.dumps(obj, allow_nan=False)


def test_failed_combined_score_is_finite_and_below_clipped_floor():
    clipped_floor = math.log(CHOICE13K_LOGLIK_EPS)
    assert math.isfinite(FAILED_COMBINED_SCORE)
    assert FAILED_COMBINED_SCORE < clipped_floor
    _strict_json({"combined_score": FAILED_COMBINED_SCORE, "error": 0.0, "timeout": True})
    with pytest.raises(ValueError):
        _strict_json({"combined_score": float("-inf")})


def test_evaluator_database_checkpoint_prefers_seed_over_all_invalid_children(tmp_path: Path):
    evo_split = tmp_path / "trials_evolution_split.json"
    _write_evolution_split_json(evo_split, TRIALS[:1], TRIALS[1:])
    eval_path = tmp_path / "evaluator.py"
    eval_path.write_text(
        _render_evaluator_py(evo_split, split_ratio=0.6, categorical=False),
        encoding="utf-8",
    )

    _install_runtime_patches()
    ev_cfg = EvaluatorConfig()
    ev_cfg.timeout = 15
    ev_cfg.max_retries = 0
    ev_cfg.cascade_evaluation = False
    ev_cfg.use_llm_feedback = False
    ev_cfg.parallel_evaluations = 1
    evaluator = Evaluator(ev_cfg, str(eval_path), llm_ensemble=None, prompt_sampler=None)

    seed_metrics = asyncio.run(evaluator.evaluate_program(SEED, "seed"))
    invalid_metrics = asyncio.run(evaluator.evaluate_program(INVALID, "invalid"))

    assert math.isfinite(seed_metrics["combined_score"])
    assert seed_metrics["combined_score"] == pytest.approx(math.log(0.5), abs=1e-6)
    assert invalid_metrics["error"] == 0.0
    assert invalid_metrics["combined_score"] == FAILED_COMBINED_SCORE
    assert invalid_metrics["combined_score"] < seed_metrics["combined_score"]
    assert invalid_metrics["combined_score"] < math.log(CHOICE13K_LOGLIK_EPS)
    _strict_json(invalid_metrics)
    format_metrics_safe(invalid_metrics)
    format_metrics_safe(seed_metrics)

    prompt, _state = truncate_vanilla_messages(
        task_text="task",
        program_code=INVALID,
        metrics=invalid_metrics,
        trials_compact="y=0 split=train",
        max_prompt_tokens=14000,
        reserved_completion_tokens=256,
        model_context_len=16384,
        categorical=False,
    )
    assert "combined_score=-inf" not in prompt["user"]
    assert "combined_score=-21." in prompt["user"] or "combined_score=-21" in prompt["user"]

    db_cfg = DatabaseConfig()
    db_cfg.num_islands = 1
    db_cfg.population_size = 10
    db_cfg.archive_size = 10
    db_cfg.in_memory = True
    db = ProgramDatabase(db_cfg)

    seed = Program(id="seed", code=SEED, language="python", metrics=dict(seed_metrics), iteration_found=0)
    db.add(seed, iteration=0)
    assert db.best_program_id == "seed"

    for i in range(3):
        child = Program(
            id=f"bad{i}",
            code=INVALID,
            language="python",
            metrics=dict(invalid_metrics),
            parent_id="seed",
            iteration_found=i + 1,
        )
        db.add(child, iteration=i + 1)

    assert db.best_program_id == "seed"
    best = db.get_best_program()
    assert best is not None and best.id == "seed" and best.code == SEED
    assert get_fitness_score(invalid_metrics, db_cfg.feature_dimensions) < get_fitness_score(
        seed_metrics, db_cfg.feature_dimensions
    )
    assert not db._is_better(db.programs["bad0"], db.programs["seed"])
    parent, _insp = db.sample_from_island(island_id=0, num_inspirations=0)
    assert parent.id in db.programs

    ckpt = tmp_path / "checkpoint_1"
    db.save(str(ckpt), iteration=1)
    for prog_json in (ckpt / "programs").glob("*.json"):
        blob = prog_json.read_text(encoding="utf-8")
        assert "-Infinity" not in blob
        assert "Infinity" not in blob
        _strict_json(json.loads(blob))

    loaded = ProgramDatabase(db_cfg)
    loaded.load(str(ckpt))
    loaded_best = loaded.get_best_program()
    assert loaded_best is not None and loaded_best.id == "seed"
    assert loaded.best_program_id == "seed"
    for pid, prog in loaded.programs.items():
        score = prog.metrics.get("combined_score")
        assert math.isfinite(float(score))
        _strict_json(prog.to_dict())

    picked, picked_ll = _find_best_program_by_observed_loglik(ckpt)
    assert picked is not None
    data = json.loads(picked.read_text(encoding="utf-8"))
    assert data["code"] == SEED
    assert picked_ll == pytest.approx(seed_metrics["combined_score"], abs=1e-6)
    assert picked_ll > FAILED_COMBINED_SCORE
