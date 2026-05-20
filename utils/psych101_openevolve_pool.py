"""Pickle-safe helpers for Psych101 OpenEvolve process-pool workers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

WORKER_VANILLA: Dict[str, Any] = {}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_OPENVOLVE_PATH = _REPO_ROOT / "baseline_methods" / "Psych101" / "run_openevolve.py"


def get_worker_vanilla_ctx() -> Dict[str, Any]:
    return WORKER_VANILLA


def set_worker_vanilla_ctx(ctx: Dict[str, Any] | None) -> None:
    WORKER_VANILLA.clear()
    if ctx:
        WORKER_VANILLA.update(ctx)


def _load_run_openevolve_module():
    mod = sys.modules.get("psych101_run_openevolve")
    if mod is not None:
        return mod
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "_install_runtime_patches"):
        sys.modules["psych101_run_openevolve"] = main
        return main
    spec = importlib.util.spec_from_file_location("psych101_run_openevolve", _RUN_OPENVOLVE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load run_openevolve from {_RUN_OPENVOLVE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["psych101_run_openevolve"] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_patches_installed() -> None:
    mod = _load_run_openevolve_module()
    mod._install_runtime_patches()
