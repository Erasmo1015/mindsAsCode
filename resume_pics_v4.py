"""Replay durable PICS-v4 candidates through the unchanged scorer/elite updates.

Recovery never uses test scores to select parents. Committed stages are verified,
not rewritten; incomplete batches reuse every durable candidate at its original
absolute slot/decoding seed. Only missing responses reach the existing generator.
"""
from pathlib import Path
import contextvars
import hashlib
import json
import math
import os
import shutil
import threading


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def key(row):
    return (row.get('record_type'), row.get('phase'), row.get('iteration'), row.get('candidate_id'))


class ResumeVerified(RuntimeError):
    pass


class CandidateReplay:
    def __init__(self, teh, root):
        from utils.teh.pics_v4_recovery import participant_directories
        self.teh, self.root = teh, Path(root).resolve()
        self.stages, self.records, self.protected = {}, {}, {}
        self.lock = threading.RLock()
        self.context = contextvars.ContextVar('pics_v4_durable_generation', default=None)
        self.original = (teh.generate_program_variants, teh._generate_iteration_candidate_codes,
                         teh._parallel_generate_children, teh.append_mem_trace_record,
                         Path.write_text, Path.unlink, shutil.rmtree)
        roots = [p / 'global_phase' for p in (self.root / 'target_population').glob('*') if p.is_dir()]
        roots += list(participant_directories(self.root / 'selected'))
        for owner in roots:
            for stage in [owner / 'explore_phase', *owner.glob('iteration_*')]:
                if not stage.is_dir():
                    continue
                for code in (stage / 'candidates').glob('candidate_*.py'):
                    self.protected[code.resolve()] = code.read_bytes()
                metrics_path = stage / 'metrics.json'
                if not metrics_path.is_file():
                    continue
                metrics = json.loads(metrics_path.read_text())
                count = int(metrics.get('n_candidates', metrics.get('explore_candidates_requested', 0)))
                if count <= 0 or any(not (stage / 'candidates' / f'candidate_{i}.py').is_file() for i in range(count)):
                    raise RuntimeError(f'Incomplete committed candidates: {metrics_path}')
                if 'candidate_results' in metrics and len(metrics['candidate_results']) != count:
                    raise RuntimeError(f'Incomplete committed metrics: {metrics_path}')
                self.stages[stage.resolve()] = metrics
                for p in stage.rglob('*'):
                    if p.is_file() and (p.suffix == '.py' or p.name in {'metrics.json', 'pool_manifest.json', 'summary.json', 'summary.csv'}):
                        self.protected[p.resolve()] = p.read_bytes()
            initial = owner / 'initial_pool_from_global'
            for p in initial.rglob('*'):
                if p.is_file():
                    self.protected[p.resolve()] = p.read_bytes()
            trace = owner / 'mem_trace.jsonl'
            if trace.is_file():
                rows = {}
                for line in trace.read_text().splitlines():
                    row = json.loads(line)  # Corrupt/truncated committed records fail closed.
                    k = key(row)
                    if k in rows:
                        raise RuntimeError(f'Duplicate saved event: {trace} {k}')
                    rows[k] = row
                self.records[trace.resolve()] = rows
        for p in self.root.rglob('pics_v4_panel_banks/*.json'):
            self.protected[p.resolve()] = p.read_bytes()
        for p in self.root.rglob('pics_v4_panel_banks/ATTESTATION.json'):
            self.verify_bank_attestation(p)

    def verify_bank_attestation(self, path):
        """Prove the historical phase-prefix attestation, including its old index.

        Evolution historically extended FINGERPRINTS after the exploration
        attestation. Reconstruct that exact old index rather than ignoring a
        hash mismatch or rewriting the historical record.
        """
        hashes = json.loads(path.read_text())
        index = path.parent / 'FINGERPRINTS.json'
        for name, expected in hashes.items():
            file = path.parent / name
            if hashlib.sha256(file.read_bytes()).hexdigest() == expected:
                continue
            if name != 'FINGERPRINTS.json':
                raise RuntimeError(f'Historical bank SHA mismatch: {file}')
            payload = json.loads(index.read_text())
            from utils.teh.pics_v4_panels import BANK_FILENAMES, FINGERPRINT_KEYS
            covered = {FINGERPRINT_KEYS[k] for k, filename in BANK_FILENAMES.items() if filename in hashes}
            old_index = {k: v for k, v in payload.items() if not k.endswith('_fingerprint') or k in covered}
            old_bytes = (json.dumps(old_index, indent=2) + '\n').encode()
            if hashlib.sha256(old_bytes).hexdigest() != expected:
                raise RuntimeError(f'Historical phase-prefix fingerprint proof failed: {index}')

    def install(self):
        replay = self
        self.teh.generate_program_variants = self.variants
        self.teh._generate_iteration_candidate_codes = self.iteration
        self.teh._parallel_generate_children = self.parallel
        self.teh.append_mem_trace_record = self.trace
        def write_text(path, data, *args, **kwargs):
            return replay.write_text(path, data, *args, **kwargs)
        def unlink(path, *args, **kwargs):
            if path.resolve() in replay.protected:
                return None
            return replay.original[5](path, *args, **kwargs)
        def rmtree(path, *args, **kwargs):
            p = Path(path).resolve()
            if any(p == f.parent or p in f.parents for f in replay.protected):
                if p.name == 'initial_pool_from_global':
                    return None  # The unchanged writer below verifies every exported entry.
                raise RuntimeError(f'Recovery refuses deletion of committed work: {p}')
            return replay.original[6](path, *args, **kwargs)
        Path.write_text, Path.unlink, shutil.rmtree = write_text, unlink, rmtree
        self.teh._pics_v4_candidate_replay = self
        print(f'[resume] Installed durable replay: {len(self.stages)} committed stages; root={self.root}', flush=True)
        return self.stages

    def restore(self):
        (self.teh.generate_program_variants, self.teh._generate_iteration_candidate_codes,
         self.teh._parallel_generate_children, self.teh.append_mem_trace_record,
         Path.write_text, Path.unlink, shutil.rmtree) = self.original
        self.teh._pics_v4_candidate_replay = None

    def write_text(self, path, data, *args, **kwargs):
        p = path.resolve()
        with self.lock:
            old = self.protected.get(p)
            if old is not None:
                new = data.encode(kwargs.get('encoding') or 'utf-8')
                if p.name == 'ATTESTATION.json' and p.parent.name == 'pics_v4_panel_banks':
                    self.verify_bank_attestation(path)
                    actual = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in p.parent.glob('*.json') if f.name != 'ATTESTATION.json'}
                    if json.loads(new) != actual:
                        raise RuntimeError(f'Recovery bank attestation reconstruction mismatch: {p}')
                    return len(data)  # Keep the historical attestation bytes.
                if p.name == 'FINGERPRINTS.json' and p.parent.name == 'pics_v4_panel_banks':
                    prior, proposed = json.loads(old), json.loads(new)
                    from utils.teh.pics_v4_panels import FINGERPRINT_KEYS
                    if all(proposed.get(k) == v for k, v in prior.items()) and set(proposed) - set(prior) <= set(FINGERPRINT_KEYS.values()):
                        if proposed != prior:
                            result = self.original[4](path, data, *args, **kwargs)
                            self.protected[p] = path.read_bytes()
                            return result
                same = old == new
                if not same and p.suffix == '.json':
                    same = equivalent(json.loads(old), json.loads(new))
                if not same:
                    raise RuntimeError(f'Recovery reconstruction mismatch before overwrite: {p}')
                return len(data)
            atomic = self.root in p.parents and (p.name in {'metrics.json', 'mem_trace.jsonl'} or (p.parent.name == 'candidates' and p.name.startswith('candidate_')))
            if not atomic:
                return self.original[4](path, data, *args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f'.recovery-{os.getpid()}-{threading.get_ident()}.tmp')
            result = self.original[4](temporary, data, *args, **kwargs)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            if path.parent.name == 'candidates':
                self.protected[p] = path.read_bytes()
            return result

    def variants(self, *args, **kwargs):
        stats = kwargs.get('prompt_stats_path')
        if not stats:
            return self.original[0](*args, **kwargs)
        stage = Path(stats).parent.resolve()
        offset, count = int(kwargs.get('pics_lossless_candidate_offset', 0)), int(kwargs['n_variants'])
        metrics = self.stages.get(stage)
        if metrics:
            total = int(metrics.get('n_candidates', metrics.get('explore_candidates_requested', 0)))
            if offset + count > total:
                raise RuntimeError(f'Changed committed candidate budget: {stage}')
        token = self.context.set((stage, offset))
        try:
            # Preserve original batch size, role, seed base, parents and panel slots.
            return self.original[0](*args, **kwargs)
        finally:
            self.context.reset(token)

    def iteration(self, **kwargs):
        stats = kwargs['variant_kwargs'].get('prompt_stats_path')
        stage = Path(stats).parent.resolve() if stats else None
        metrics = self.stages.get(stage)
        if metrics:
            expected = ['fresh'] * int(kwargs['fresh_n_candidates'])
            expected += ['normal'] * (int(kwargs['n_candidates']) - len(expected))
            if int(metrics['n_candidates']) != int(kwargs['n_candidates']) or metrics['candidate_sources'] != expected:
                raise RuntimeError(f'Changed committed candidate schedule: {stage}')
        return self.original[1](**kwargs)

    def parallel(self, n, generate_one, *args, **kwargs):
        context = self.context.get()
        if context is None:
            return self.original[2](n, generate_one, *args, **kwargs)
        stage, offset = context
        def durable(index):
            path = stage / 'candidates' / f'candidate_{offset + index}.py'
            if path.is_file():
                return path.read_text()
            if os.environ.get('PICS_RESUME_VERIFY_ONLY') == '1':
                raise ResumeVerified(f'Next missing durable candidate: {path}')
            code = generate_one(index)
            path.write_text(code or '', encoding='utf-8')
            return code
        return self.original[2](n, durable, *args, **kwargs)

    def trace(self, path, row):
        if path is None:
            return self.original[3](path, row)
        from utils.mem.trace import json_safe_value
        row = json_safe_value(row)
        resolved = Path(path).resolve()
        with self.lock:
            rows = self.records.setdefault(resolved, {})
            old = rows.get(key(row))
            if old is not None:
                if not equivalent(old, row):
                    raise RuntimeError(f'Recovery reconstruction event mismatch: {path} {key(row)}')
                return
            # Keep the existing trace serializer, but commit the complete new
            # event with an atomic replacement instead of two append writes.
            out = Path(path)
            previous = out.read_text() if out.exists() else ''
            if previous and not previous.endswith('\n'):
                previous += '\n'
            line = json.dumps(row, ensure_ascii=False, separators=(',', ':'), sort_keys=False)
            out.write_text(previous + line + '\n', encoding='utf-8')
            rows[key(row)] = row


def install_saved_candidate_replay(teh, root):
    installed = getattr(teh, '_pics_v4_candidate_replay', None)
    if installed is not None:
        if installed.root == Path(root).resolve():
            return installed.stages
        installed.restore()
    return CandidateReplay(teh, root).install()


if __name__ == '__main__':
    import sys
    import teh
    if '--pics_v4' not in sys.argv:
        raise RuntimeError('Saved candidate replay requires --pics_v4')
    output = sys.argv[sys.argv.index('--output_dir') + 1]
    install_saved_candidate_replay(teh, output)
    teh.main()
