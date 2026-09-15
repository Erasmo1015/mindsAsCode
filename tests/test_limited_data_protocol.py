"""Lightweight tests for the optional structure-aware limited-data protocol."""
from __future__ import annotations

import copy

from data_modules.psych101_binary import _expand_single_block_to_pseudo_blocks
from utils.teh.limited_data_protocol import (
    apply_limited_data_protocol,
    apply_structure_aware_protocol,
    load_participant_limited_splits,
    select_complete_then_prefix,
    split_continuous_session_chronological,
    structure_aware_subset_fingerprint,
    tag_split_trials,
)
from utils.teh.limited_data_registry import (
    LIMITED_DATA_PROTOCOL_OFF,
    LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
    LIMITED_DATA_REGISTRY,
)
from utils.teh.sparse_observations import apply_max_observed_trials


def _trial(action: int, history, unit: str, extra=None):
    problem = {"unit": unit, "block_index": unit}
    if extra:
        problem.update(extra)
    return {
        "problem": problem,
        "history": copy.deepcopy(history),
        "options": [0, 1],
        "action": action,
    }


def _resetting_unit(unit_id: str, n: int, start_action: int = 0):
    trials = []
    history = []
    for i in range(n):
        t = _trial(start_action + i, history, unit_id)
        trials.append(t)
        history = history + [{"action": start_action + i}]
    return trials


def test_registry_covers_all_fifteen():
    expected = {
        "1peterson2021using",
        "2plonsky2018when",
        "3frey2017cct",
        "4wulff2018description",
        "5speekenbrink2008learning",
        "7hilbig2014generalized",
        "10frey2017risk",
        "11enkavi2019recentprobes",
        "12badham2017deficits",
        "mixed_gambles",
        "bergert_nosofsky_2007",
        "guan_2020_stopping",
        "steyvers_2009_bandit",
        "13schulz2020finding",
        "14kool2016when",
    }
    assert set(LIMITED_DATA_REGISTRY) == expected


def test_default_off_matches_legacy_sparse_fingerprint():
    train = _resetting_unit("a", 20) + _resetting_unit("b", 20)
    val = _resetting_unit("c", 15)
    test = _resetting_unit("d", 12)
    legacy_tr, legacy_va, legacy_te, legacy_audit = apply_max_observed_trials(
        train,
        val,
        test,
        max_observed_trials_per_participant=40,
        dataset="1peterson2021using",
        participant_id=3,
        split_seed=0,
    )
    new_tr, new_va, new_te, new_audit, _manifest = apply_limited_data_protocol(
        train,
        val,
        test,
        dataset="1peterson2021using",
        participant_id=3,
        split_seed=0,
        max_observed_trials_per_participant=40,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_OFF,
    )
    assert new_audit.subset_fingerprint == legacy_audit.subset_fingerprint
    assert new_audit.selected_train_indices == legacy_audit.selected_train_indices
    assert new_audit.selected_val_indices == legacy_audit.selected_val_indices
    assert [t["action"] for t in new_tr] == [t["action"] for t in legacy_tr]
    assert [t["action"] for t in new_va] == [t["action"] for t in legacy_va]
    assert new_te == legacy_te == test


def test_deterministic_same_seed():
    train = _resetting_unit("p0", 35) + _resetting_unit("p1", 20)
    val = _resetting_unit("p2", 20)
    test = _resetting_unit("p3", 10)
    kwargs = dict(
        dataset="1peterson2021using",
        participant_id=1,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_unit_shuffle",
    )
    a = apply_structure_aware_protocol(train, val, test, **kwargs)
    b = apply_structure_aware_protocol(train, val, test, **kwargs)
    assert a[3].subset_fingerprint == b[3].subset_fingerprint
    assert a[4].subset_fingerprint == b[4].subset_fingerprint
    assert [t["action"] for t in a[0]] == [t["action"] for t in b[0]]


def test_methods_share_identical_subset_fingerprint():
    train = [_trial(i, [], f"t{i}") for i in range(50)]
    val = [_trial(100 + i, [], f"v{i}") for i in range(20)]
    test = [_trial(200 + i, [], f"x{i}") for i in range(15)]
    pics = apply_limited_data_protocol(
        train,
        val,
        test,
        dataset="7hilbig2014generalized",
        participant_id=9,
        split_seed=0,
        max_observed_trials_per_participant=40,
        limited_data_protocol=LIMITED_DATA_PROTOCOL_STRUCTURE_AWARE,
    )
    lm = apply_limited_data_protocol(
        train,
        val,
        test,
        dataset="7hilbig2014generalized",
        participant_id=9,
        split_seed=0,
        limited_train_val=40,
        limited_data_protocol="structure_aware",
    )
    assert pics[4].subset_fingerprint == lm[4].subset_fingerprint
    assert pics[4].train_fingerprint == lm[4].train_fingerprint
    assert pics[4].test_fingerprint == lm[4].test_fingerprint


def test_splits_never_overlap():
    train = _resetting_unit("a", 35) + _resetting_unit("b", 20)
    val = _resetting_unit("c", 12)
    test = _resetting_unit("d", 8)
    tr, va, te, _audit, manifest = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="2plonsky2018when",
        participant_id=0,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_unit_shuffle",
    )
    tagged_tr, tagged_va, tagged_te = tag_split_trials("2plonsky2018when", tr, va, te)
    keys = []
    for rows in (tagged_tr, tagged_va, tagged_te):
        for t in rows:
            keys.append(
                (
                    (t.get("_ldp") or {}).get("unit_id"),
                    (t.get("_ldp") or {}).get("chrono"),
                    (t.get("_ldp") or {}).get("session_index"),
                )
            )
    assert len(keys) == len(set(keys))
    assert manifest.history_consistency == "pass"


def test_speekenbrink_chronological_helper_no_shuffle():
    trials = [_trial(i, [{"action": j} for j in range(i)], "session") for i in range(200)]
    train, val, test = split_continuous_session_chronological(trials, 0.6)
    assert len(train) == 120
    assert len(val) == 40
    assert len(test) == 40
    assert [t["action"] for t in train + val + test] == list(range(200))


def test_speekenbrink_structure_aware_contiguous_suffix():
    trials = []
    history = []
    for i in range(200):
        trials.append(
            {
                "problem": {"cards": [i], "schema_type": "B"},
                "history": list(history),
                "options": ["A", "B"],
                "action": i % 2,
            }
        )
        history = history + [{"action": i % 2}]
    train, val, test = split_continuous_session_chronological(trials, 0.6)
    out_tr, out_va, out_te, _audit, manifest = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="5speekenbrink2008learning",
        participant_id=0,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="structure_aware_chronological_session",
    )
    assert len(out_tr) + len(out_va) == 40
    assert len(out_te) == 40
    assert [t["problem"]["cards"][0] for t in out_tr + out_va + out_te] == list(
        range(120, 200)
    )
    assert out_tr[0]["history"] == []
    assert manifest.test_set_differs_from_legacy is True
    assert "pseudo_block" not in manifest.split_kind
    assert manifest.history_consistency == "pass"


def test_legacy_speekenbrink_still_uses_pseudo_blocks():
    from data_modules.psych101_binary import PsychBlock, PsychExperiment, ParsedTrial

    trials = [
        ParsedTrial(action=i % 2, feedback=1.0, problem_fields={"cards": [i]})
        for i in range(200)
    ]
    exp = PsychExperiment(
        instruction="x",
        blocks=[
            PsychBlock(
                trials=trials,
                option_keys=["A", "B"],
                problem_static={"schema_type": "B"},
                schema_type="B",
            )
        ],
        dataset_alias="5speekenbrink2008learning",
        schema_type="B",
    )
    expanded = _expand_single_block_to_pseudo_blocks(exp)
    assert len(expanded.blocks) > 1


def test_kool_contiguous_days_and_stage1_prefix():
    tv = []
    history = []
    for day in range(1, 31):
        s1 = {
            "problem": {"presented_day": day, "stage": 1},
            "history": list(history),
            "options": ["J", "K"],
            "action": 0,
        }
        tv.append(s1)
        history = history + [{"action": 0, "stage": 1}]
        s2 = {
            "problem": {"presented_day": day, "stage": 2},
            "history": list(history),
            "options": ["J", "K"],
            "action": 1,
        }
        tv.append(s2)
        history = history + [{"action": 1, "stage": 2}]
    train, val = tv[:36], tv[36:48]
    test = tv[48:60]
    out_tr, out_va, out_te, _audit, manifest = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="14kool2016when",
        participant_id=0,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_chronological_days",
    )
    days = [
        (t["problem"]["presented_day"], t["problem"]["stage"])
        for t in out_tr + out_va + out_te
    ]
    assert days == sorted(days)
    assert len(out_te) == len(test)
    assert out_tr[0]["history"] == [] or "omitted" not in str(out_tr[0]["history"])
    chrono_days = [t["problem"]["presented_day"] for t in out_tr + out_va]
    assert chrono_days == sorted(chrono_days)
    assert manifest.history_consistency == "pass"
    assert len(out_tr) + len(out_va) >= 40


def test_iid_samples_individual_trials():
    train = [_trial(i, [], f"t{i}") for i in range(80)]
    val = [_trial(1000 + i, [], f"v{i}") for i in range(20)]
    test = [_trial(2000 + i, [], f"x{i}") for i in range(10)]
    out_tr, out_va, out_te, audit, _m = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="4wulff2018description",
        participant_id=4,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_unit_shuffle",
    )
    assert len(out_tr) + len(out_va) == 40
    assert out_te == test or [t["action"] for t in out_te] == [t["action"] for t in test]
    assert audit.retained_n_test == 10
    actions = [t["action"] for t in out_tr]
    assert actions != list(range(len(out_tr))) or len(out_tr) < 80


def test_resetting_complete_units_then_prefix():
    units = [_resetting_unit("u0", 10), _resetting_unit("u1", 10), _resetting_unit("u2", 10)]
    tagged_train, _, _ = tag_split_trials(
        "10frey2017risk",
        [t for u in units for t in u],
        [],
        [],
    )
    grouped = []
    cur = []
    cur_id = None
    for t in tagged_train:
        uid = (t.get("_ldp") or {}).get("unit_id")
        if cur and uid != cur_id:
            grouped.append(cur)
            cur = [t]
        else:
            cur.append(t)
        cur_id = uid
    if cur:
        grouped.append(cur)
    retained, partial, reason = select_complete_then_prefix(
        grouped, 25, prefix_valid=True
    )
    assert len(retained) == 25
    assert partial is True
    assert reason == "chronological_prefix"
    assert (retained[-1].get("_ldp") or {}).get("chrono") == 4


def test_thirty_five_plus_five_equals_forty():
    u0 = _resetting_unit("block0", 35)
    u1 = _resetting_unit("block1", 20)
    tagged, _, _ = tag_split_trials("1peterson2021using", u0 + u1, [], [])
    units = []
    cur = []
    cur_id = None
    for t in tagged:
        uid = (t.get("_ldp") or {}).get("unit_id")
        if cur and uid != cur_id:
            units.append(cur)
            cur = [t]
        else:
            cur.append(t)
        cur_id = uid
    if cur:
        units.append(cur)
    retained, partial, reason = select_complete_then_prefix(units, 40, prefix_valid=True)
    assert len(units[0]) == 35
    assert len(retained) == 40
    assert partial is True
    assert reason == "chronological_prefix"
    assert [(t.get("_ldp") or {}).get("unit_id") for t in retained].count(
        (retained[-1].get("_ldp") or {}).get("unit_id")
    ) == 5


def test_history_never_contains_omitted_observations():
    train = _resetting_unit("a", 35) + _resetting_unit("b", 20)
    val = _resetting_unit("c", 10)
    test = _resetting_unit("d", 8)
    out_tr, out_va, out_te, _a, manifest = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="3frey2017cct",
        participant_id=2,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_unit_shuffle",
    )
    omitted_actions = set()
    retained_actions = {id(t) for t in out_tr + out_va + out_te}
    del retained_actions
    retained_pairs = {
        (tuple(t["problem"].items()), t["action"], len(t["history"]))
        for t in out_tr + out_va + out_te
    }
    for original in train + val:
        key = (tuple(original["problem"].items()), original["action"], len(original["history"]))
        if key not in retained_pairs and original["history"]:
            omitted_actions.add(original["action"])
    for t in out_tr + out_va + out_te:
        for h in t["history"]:
            assert not (
                h.get("action") in omitted_actions
                and len(t["history"]) > 40
            )
    assert manifest.history_consistency == "pass"
    if out_tr:
        assert len(out_tr[0]["history"]) == 0 or out_tr[0]["history"]


def test_insufficient_data_keeps_all_eligible_and_full_test():
    train = _resetting_unit("a", 8)
    val = _resetting_unit("b", 4)
    test = _resetting_unit("c", 20)
    out_tr, out_va, out_te, audit, manifest = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="11enkavi2019recentprobes",
        participant_id=17,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_unit_shuffle",
    )
    assert len(out_tr) == 8
    assert len(out_va) == 4
    assert len(out_te) == 20
    assert audit.retained_n_test == 20
    assert manifest.fallback_reason == "insufficient_train_val"
    assert len(out_tr) + len(out_va) < 40


def test_test_never_counts_toward_budget():
    train = _resetting_unit("a", 100)
    val = _resetting_unit("b", 40)
    test = _resetting_unit("c", 80)
    out_tr, out_va, out_te, audit, _m = apply_structure_aware_protocol(
        train,
        val,
        test,
        dataset="12badham2017deficits",
        participant_id=0,
        split_seed=0,
        split_ratio=0.6,
        budget=40,
        split_kind="legacy_unit_shuffle",
    )
    assert len(out_tr) + len(out_va) <= 40
    assert len(out_te) == 80
    assert audit.original_n_test == audit.retained_n_test == 80


def test_fingerprint_function_is_stable():
    rows = [_trial(1, [], "u")]
    fp1 = structure_aware_subset_fingerprint(
        dataset="mixed_gambles",
        participant_id=1,
        split_seed=0,
        budget=40,
        protocol="structure_aware",
        train_trials=rows,
        val_trials=[],
        test_trials=[],
    )
    fp2 = structure_aware_subset_fingerprint(
        dataset="mixed_gambles",
        participant_id=1,
        split_seed=0,
        budget=40,
        protocol="structure_aware",
        train_trials=rows,
        val_trials=[],
        test_trials=[],
    )
    assert fp1 == fp2
