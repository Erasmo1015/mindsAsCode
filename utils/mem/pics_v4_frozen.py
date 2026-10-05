"""Fail-closed configuration/code locks for official uniform-v8 MEM commands."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = 'analysis/config/mem/pics_v4_uniform_v8_frozen.json'
INPUT_LOCK = 'analysis/config/mem/pics_v4_uniform_v8_inputs_frozen.json'
INPUT_LOCK_SHA256 = 'bf1a9fa84616335f68b86f3307581e0ee5bf846a2315a5fbff4d6f0468530206'
# Filled once when the approved specification is sealed; not a runtime override.
CONFIG_SHA256 = '28475dec54db08a4b8d116ba7d252d4dcc29f897195f454a9a193baaed21ac1b'


@lru_cache(maxsize=4)
def check_frozen(repo=REPO):
    repo = Path(repo)
    payload = (repo/CONFIG).read_bytes()
    if hashlib.sha256(payload).hexdigest() != CONFIG_SHA256:
        raise ValueError('Uniform-v8 frozen configuration changed; explicit new scientific freeze required')
    config = json.loads(payload)
    for rel, digest in config['code_hashes'].items():
        if hashlib.sha256((repo/rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen uniform-v8 implementation changed: {rel}')
    lock = (repo/INPUT_LOCK).read_bytes()
    if hashlib.sha256(lock).hexdigest() != INPUT_LOCK_SHA256:
        raise ValueError('Uniform-v8 frozen input lock changed')
    for rel, digest in json.loads(lock)['input_hashes'].items():
        if hashlib.sha256((repo/rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen uniform-v8 input changed: {rel}')
    return config


def guard_participant_frame(frame, *, fitting=False):
    if 'reference_policy' not in frame or not (frame.reference_policy == 'pics_v4_uniform_v8').any():
        return
    config = check_frozen()
    if not (frame.reference_policy == 'pics_v4_uniform_v8').all():
        raise ValueError('Mixed participant reference policies refused')
    if fitting and config['participant']['focal_family'] is None:
        raise ValueError('Participant fits are frozen closed until final uniform-v8 coverage declares the focal family; historical Sep-26 family refused')
