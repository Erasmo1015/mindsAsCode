"""Fresh participant attempts after a read-only completed population handoff.

Process-local adapter to the existing coordinator; no candidate replay or rescoring
of population checkpoints. Every invocation needs a new scheduler/output identity.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
PREDECESSORS = {
    '2plonsky2018when': 313161, '4wulff2018description': 313162,
    '7hilbig2014generalized': 313162, '12badham2017deficits': 313162,
    '14kool2016when': 313163, 'steyvers_2009_bandit': 313163,
    '10frey2017risk': 313164, '1peterson2021using': 313164,
    'mixed_gambles': 313165, '13schulz2020finding': 313165,
}


def predecessor(target):
    return REPO / 'generated_outputs/psych101_train/teh' / target / 'pics_v4_transfer_based_only' / f'job_{PREDECESSORS[target]}'


def completed_participants(root):
    from utils.teh.pics_v4_recovery import participant_directories
    from utils.teh.t_pics_gated_transfer import participant_run_is_complete
    return {int(p.name[12:]): p for p in participant_directories(root / 'selected')
            if participant_run_is_complete(p, expected_n_iterations=10, expected_explore_candidates=50)}


def protect_old_outputs(new_root):
    """Refuse writes/deletion through predecessor paths or completed-person links."""
    generated = (REPO / 'generated_outputs').resolve()
    allowed = Path(new_root).resolve()
    def as_path(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        return Path(os.fsdecode(value)).resolve()
    def protected(value):
        p = as_path(value)
        return p is not None and p.is_relative_to(generated) and not p.is_relative_to(allowed)
    def missing_new_root_parent(value):
        """parents=True must create the new job directory's absent ancestors."""
        p = as_path(value)
        return p is not None and p != allowed and allowed.is_relative_to(p)
    def audit(event, args):
        if event == 'open':
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and protected(args[0]):
                raise PermissionError(f'Continuation refuses old-output write: {args[0]}')
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate', 'os.rename', 'os.link', 'os.symlink'):
            paths = args[:2] if event in ('os.rename', 'os.link') else args[1:2] if event == 'os.symlink' else args[:1]
            if event == 'os.mkdir':
                paths = [p for p in paths if not missing_new_root_parent(p)]
            if any(protected(p) for p in paths):
                raise PermissionError(f'Continuation refuses old-output mutation: {event} {paths}')
    sys.addaudithook(audit)


def install(teh, target, new_root):
    from utils.teh import pics_v4_recovery as recovery, pics_v4_transfer as identity
    old = predecessor(target)
    new_root = Path(new_root).resolve()
    if new_root.parent.name != 'pics_v4_transfer_based_only' or new_root.parent.parent.name != target or new_root == old.resolve():
        raise RuntimeError('Continuation must use a separate target/transfer job root')
    if (new_root / 'CONTINUATION.json').exists() or (new_root / 'selected').exists():
        raise RuntimeError('Attempt already exists; use another new continuation directory')
    original_coordinator = teh._run_t_pics_gated_population_arms
    original_initialize = identity.initialize_participant
    skipped = {}
    plan = {}

    def activate(module, output, *, target, mode):
        if module is not teh or Path(output).resolve() != new_root or mode != 'transfer_based_only':
            raise RuntimeError('Continuation dispatch mismatch')
        expected = identity.current_identity()
        if expected is None or expected['target'] != target:
            raise RuntimeError('Missing frozen continuation identity')
        reuse = old.exists()
        if reuse:
            identity.assert_identity(identity.read(old / 'TRIAL_PROMPT_POLICY.json')['run_identity'])
            identity.verify_completed(old / 'target_population/transfer', target=target, kind='pics_v4_transfer_based_only')
            skipped.update(completed_participants(old))
            for pid, path in skipped.items():
                identity.assert_participant_resume(path, pid)
        cohort = expected['cohort']['participant_ids']
        if not set(skipped).issubset(set(cohort)):
            raise RuntimeError('Completed predecessor participant outside frozen cohort')
        plan.update(schema='pics_v4_fresh_continuation_v1', predecessor_root=str(old.resolve()),
                    predecessor_output_job=old.name, execution_job=os.environ.get('SLURM_JOB_ID'),
                    continuation_root=str(new_root), population_root=str(old.resolve()) if reuse else str(new_root),
                    population_reused=reuse, run_identity=expected,
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    canonical_participant_roots={str(pid): str(skipped.get(pid, new_root / 'selected' / f'participant_{pid}').resolve()) for pid in cohort},
                    skipped_participants=sorted(skipped), fresh_participants=[pid for pid in cohort if pid not in skipped],
                    policy='predecessor completed participants retained; unfinished participants fresh; no partial replay')
        print(f'[continuation] population_reused={reuse}; skip={sorted(skipped)}; fresh={plan["fresh_participants"]}', flush=True)

    def coordinator(**kwargs):
        if not plan:
            raise RuntimeError('Continuation identity was not validated')
        # Main has now written the ordinary root policy/prompt records. Do not
        # create a nonempty, marker-less root during its earlier identity check.
        selected = new_root / 'selected'
        selected.mkdir(parents=True, exist_ok=True)
        for pid, path in skipped.items():
            (selected / f'participant_{pid}').symlink_to(path.resolve(), target_is_directory=True)
        with (new_root / 'CONTINUATION.json').open('x') as stream:
            json.dump(plan, stream, indent=2)
        if not plan['population_reused']:
            return original_coordinator(**kwargs)
        pool = old / 'target_population/transfer/global_phase/global_elite_pool'
        (selected / 'retained_global_elite_pool').symlink_to(pool.resolve(), target_is_directory=True)
        (selected / 'SELECTED_ARM.txt').write_text('transfer\n')
        # Existing loader compiles programs and loads stored scores; no evaluation.
        return teh._load_gated_arm_pool(old / 'target_population/transfer')

    def initialize(path, pid):
        if pid in skipped:
            if Path(path).resolve() != skipped[pid].resolve():
                raise RuntimeError('Completed-participant continuation path mismatch')
            identity.assert_participant_resume(skipped[pid], pid)
            return  # Native completed-person skip follows; no mkdir/write in predecessor.
        return original_initialize(path, pid)

    recovery.activate = activate
    teh._run_t_pics_gated_population_arms = coordinator
    identity.initialize_participant = initialize
    protect_old_outputs(new_root)
    return plan


def main():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import teh
    target = sys.argv[sys.argv.index('--dataset') + 1]
    output = Path(sys.argv[sys.argv.index('--output_dir') + 1])
    install(teh, target, output)
    teh.main()


if __name__ == '__main__':
    main()
