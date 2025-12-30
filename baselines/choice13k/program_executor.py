from typing import Callable, Optional


def compile_program(code_str: str) -> Optional[Callable]:
    """Safely compile program code and return choose callable if present."""
    global_ns = {"__builtins__": {}}
    local_ns = {}
    try:
        exec(code_str, global_ns, local_ns)
    except Exception:
        return None
    choose_fn = local_ns.get("choose") or global_ns.get("choose")
    if callable(choose_fn):
        return choose_fn
    return None

