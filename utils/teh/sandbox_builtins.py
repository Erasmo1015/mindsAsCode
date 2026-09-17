"""Restricted builtins for TEH choose() programs.

Shared by teh.py (source/target/person/eval/parallel workers) and
utils/mem/rescore_initial_pool.py so every T-PICS execution path has the
same sandbox. Do not add I/O, eval/exec, or filesystem helpers.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

# Names exposed inside choose(). __import__ is pre-existing (math fallback);
# it is not a newly added capability.
TEH_SAFE_BUILTIN_CALLABLES: Dict[str, Any] = {
    "zip": zip,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "reversed": reversed,
    "sum": sum,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "float": float,
    "int": int,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "bool": bool,
    "set": set,
    "frozenset": frozenset,
    "sorted": sorted,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
    "iter": iter,
    "next": next,
    "ord": ord,
    "chr": chr,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "__import__": __import__,
}

TEH_SAFE_BUILTIN_NAMES = tuple(TEH_SAFE_BUILTIN_CALLABLES.keys())

# Must stay absent from the sandbox.
TEH_UNSAFE_BUILTIN_NAMES = (
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "help",
    "memoryview",
    "globals",
    "locals",
    "vars",
    "dir",
    "print",
)


def teh_program_namespace() -> Dict[str, Any]:
    return {
        "__builtins__": dict(TEH_SAFE_BUILTIN_CALLABLES),
        "__import__": __import__,
        "math": math,
    }


def compile_choose_with_error(
    code_str: str,
) -> Tuple[Optional[Callable[..., Any]], Optional[BaseException]]:
    """Compile choose(); return (fn, None) or (None, error)."""
    global_ns = teh_program_namespace()
    local_ns: Dict[str, Any] = {}
    try:
        exec(code_str, global_ns, local_ns)
    except Exception as exc:
        return None, exc
    choose_fn = local_ns.get("choose") or global_ns.get("choose")
    if callable(choose_fn):
        try:
            setattr(choose_fn, "__teh_source_code", str(code_str or ""))
        except Exception:
            pass
        return choose_fn, None
    return None, TypeError("missing callable choose(problem, history)")
