"""Exact, job-scoped compatibility for the eight submitted transfer recoveries.

The original scientific run identity stays frozen. Only explicitly attested
recovery code may execute against it; this is not a wildcard hash exemption.
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'analysis/config/pics_v4_recovery_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def participant_directories(root):
    """Files and noncanonical/nonnumeric directory names are not participants."""
    for path in sorted(Path(root).glob('participant_*')):
        tail = path.name[len('participant_'):]
        if path.is_dir() and tail.isascii() and tail.isdecimal() and str(int(tail)) == tail:
            yield path


def approved_code_hashes(found, *, mode):
    if all(sha(REPO / p) == h for p, h in found.items()):
        return dict(found)
    if mode != 'transfer_based_only' or not CONFIG.is_file():
        raise RuntimeError('method code changed after CPU preflight')
    config = json.loads(CONFIG.read_text())
    if config['schema'] != 'pics_v4_scoped_recovery_v1' or found != config['original_method_code_sha256']:
        raise RuntimeError('recovery update does not match the frozen method certificate')
    for rel, old in found.items():
        new = config['recovery_changes'].get(rel)
        if new and new['before'] != old:
            raise RuntimeError('recovery preimage hash mismatch')
        expected = new['after'] if new else old
        if sha(REPO / rel) != expected:
            raise RuntimeError(f'unapproved recovery/method code: {rel}')
    for rel, expected in config['recovery_support_sha256'].items():
        if sha(REPO / rel) != expected:
            raise RuntimeError(f'recovery support SHA mismatch: {rel}')
    certificate = REPO / config['original_certificate']
    if sha(certificate) != config['original_certificate_sha256']:
        raise RuntimeError('original recovery preflight certificate changed')
    return dict(found)


def method_identity_hashes(current, *, mode):
    if mode == 'transfer_based_only' and CONFIG.is_file():
        original = json.loads(CONFIG.read_text())['original_method_code_sha256']
        # Require the complete old/new attestation, including untouched files.
        approved_code_hashes(original, mode=mode)
        return original
    return current


def assert_recovery_scope(output, *, target, mode):
    config = json.loads(CONFIG.read_text())
    path = Path(output)
    if mode != 'transfer_based_only' or path.parent.name != 'pics_v4_transfer_based_only' or path.parent.parent.name != target:
        raise RuntimeError('recovery target/mode/output mismatch')
    if path.name not in config['job_targets'] or target not in config['job_targets'][path.name]:
        raise RuntimeError('recovery update is restricted to the original eight job identities')
    approved_code_hashes(config['original_method_code_sha256'], mode=mode)
    return config


def activate(teh, output, *, target, mode):
    config = assert_recovery_scope(output, target=target, mode=mode)
    from resume_pics_v4 import install_saved_candidate_replay
    install_saved_candidate_replay(teh, output)
    print('[recovery] exact scoped update active; original scientific identity retained; '
          f'config_sha256={sha(CONFIG)} changes={json.dumps(config["recovery_changes"], sort_keys=True)}', flush=True)
