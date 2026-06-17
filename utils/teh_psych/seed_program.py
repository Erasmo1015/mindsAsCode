"""Default categorical seed program for teh_psych."""
from __future__ import annotations

from pathlib import Path

DEFAULT_CATEGORICAL_SEED_SOURCE = '''def choose(problem, history):
    options = problem["options"]
    p = 1.0 / len(options)
    return {option["action"]: p for option in options}
'''

REPO_DEFAULT_SEED_PATH = (
    Path(__file__).resolve().parents[2] / "persona_code_example" / "teh_psych" / "choices13k.py"
)


def resolve_seed_program_path(seed_path: str | None) -> Path:
    if seed_path:
        return Path(seed_path).expanduser().resolve()
    if REPO_DEFAULT_SEED_PATH.is_file():
        return REPO_DEFAULT_SEED_PATH.resolve()
    return REPO_DEFAULT_SEED_PATH
