from typing import Callable, Optional
import math


def compile_program(code_str: str) -> Optional[Callable]:
    """Safely compile program code and return choose callable if present."""
    # Provide minimal safe builtins needed for the program to run
    # Only include what's necessary for pure Python computation
    safe_builtins = {
        'zip': zip,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'sum': sum,
        'abs': abs,
        'min': min,
        'max': max,
        'float': float,
        'int': int,
        'str': str,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'bool': bool,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,
        '__import__': __import__,  # Needed for dynamic imports like __import__("math")
    }
    global_ns = {
        "__builtins__": safe_builtins,
        "math": math,  # Pre-import math module for convenience
    }
    local_ns = {}
    try:
        exec(code_str, global_ns, local_ns)
    except Exception:
        return None
    choose_fn = local_ns.get("choose") or global_ns.get("choose")
    if callable(choose_fn):
        return choose_fn
    return None

