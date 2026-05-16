#!/usr/bin/env python3
"""Run evolved_program.choose on test cases"""
import importlib.util
import json
import math
from pathlib import Path


def _load_choose() -> object:
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("evolved_program", here / "evolved_program.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evolved_program.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "choose"):
        raise AttributeError("evolved_program.py must define choose(problem, history)")
    return mod.choose


def _cases_path(here: Path) -> Path:
    for name in ("test_cases.json", "participant2_test_cases.json"):
        p = here / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No test cases JSON in {here}. Expected test_cases.json or participant2_test_cases.json"
    )


def _action_label(action: int) -> str:
    return "B" if int(action) == 1 else "A"


def _trial_loglik(p_b: float, observed_action: int) -> float:
    p_b = min(max(float(p_b), 1e-9), 1.0 - 1e-9)
    p_obs = p_b if int(observed_action) == 1 else (1.0 - p_b)
    return math.log(p_obs)


def main() -> None:
    here = Path(__file__).resolve().parent
    choose = _load_choose()
    cases = json.loads(_cases_path(here).read_text(encoding="utf-8"))

    logliks = []
    for case in cases:
        case_id = case.get("id", "?")
        p_b = float(choose(case["problem"], case.get("history", [])))
        action = case.get("observed_action")
        if action is None:
            print(f"Test {case_id}: P(B)={p_b:.2f}")
            continue
        ll = _trial_loglik(p_b, action)
        logliks.append(ll)
        print(
            f"Test {case_id}: observed action {_action_label(action)}, "
            f"P(B)={p_b:.4f}, log-likelihood={ll:.2f}"
        )

    if logliks:
        mean_ll = sum(logliks) / len(logliks)
        print(f"Mean log-likelihood ({len(logliks)} trials): {mean_ll:.2f}")


if __name__ == "__main__":
    main()
