"""Read-only aliases for missing dated records moved into bef_AAMAS_2027_sub."""
import os
from pathlib import Path
_original_open = Path.open
_original_stat = Path.stat
def relocated(path):
    text = os.fspath(path)
    marker = "/analysis_2026Sep/Codex/"
    if marker not in text or "/bef_AAMAS_2027_sub/" in text:
        return path
    prefix, suffix = text.split(marker, 1)
    if not suffix.split("/", 1)[0].startswith("Oct"):
        return path
    try:
        _original_stat(path)
        return path
    except FileNotFoundError:
        candidate = Path(prefix + marker + "bef_AAMAS_2027_sub/" + suffix)
        try:
            _original_stat(candidate)
            return candidate
        except FileNotFoundError:
            return path
def alias_open(self, mode="r", *args, **kwargs):
    if any(char in mode for char in "wax+"):
        return _original_open(self, mode, *args, **kwargs)
    return _original_open(relocated(self), mode, *args, **kwargs)
def alias_stat(self, *args, **kwargs):
    return _original_stat(relocated(self), *args, **kwargs)
Path.open = alias_open
Path.stat = alias_stat
