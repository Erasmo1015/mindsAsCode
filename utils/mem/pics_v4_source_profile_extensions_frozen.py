"""Additive locks for secondary comparators and preferred figures; v1 untouched."""
import hashlib
import json
from pathlib import Path
from utils.mem.pics_v4_frozen import REPO, check_frozen

EXTENSION = 'analysis/config/mem/pics_v4_source_profile_extensions_frozen.json'
EXTENSION_SHA256 = '5d7ded078da7052224fc9d9bb27fb383a47904e39b0001c78a6f22cbb8fa2f13'


def check_extensions(repo=REPO):
    check_frozen(repo)
    repo=Path(repo);raw=(repo/EXTENSION).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXTENSION_SHA256:
        raise ValueError('Source-profile extension configuration changed')
    config=json.loads(raw)
    for field in ('code_hashes','input_hashes'):
        for path,digest in config[field].items():
            if hashlib.sha256((repo/path).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Frozen source-profile extension {field} changed: {path}')
    return config
