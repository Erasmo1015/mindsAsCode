"""Reporting-only: PICS v3 must not upload dynamic per-participant W&B scalars."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from utils.teh.t_pics_gated_wandb import (
    GatedWandbReporter,
    is_dynamic_participant_wandb_key,
    mean_from_complete_rows,
    strip_dynamic_participant_wandb_keys,
    suppress_dynamic_participant_wandb_scalars,
)

# Uploaded keys must not look like per-person Runs columns.
# Raw ``^p[^/]*/`` also matches fixed ``participant/`` / ``progress/``; exclude those.
_FIXED_PREFIXES = (
    "participant/",
    "progress/",
    "gate/",
    "final/",
    "status/",
    "global/",
    "g2/",
)
_PID_SLASH = re.compile(r"^p\d+/")
_PID_UNDERSCORE = re.compile(r"^p\d+_")
_P_THEN_SLASH = re.compile(r"^p[^/]*/")


def _assert_no_dynamic_keys(keys) -> None:
    for key in keys:
        s = str(key)
        if s == "dataset" or any(s.startswith(p) for p in _FIXED_PREFIXES):
            # Still forbid pid-shaped keys even under odd names
            assert not _PID_SLASH.match(s), s
            assert not _PID_UNDERSCORE.match(s), s
            continue
        assert not is_dynamic_participant_wandb_key(s), s
        assert not _PID_SLASH.match(s), s
        assert not _PID_UNDERSCORE.match(s), s
        assert not _P_THEN_SLASH.match(s), s


class _FakeSummary(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)


class _FakeRun:
    def __init__(self) -> None:
        self.summary = _FakeSummary()


class _FakeTable:
    def __init__(self, columns, data) -> None:
        self.columns = list(columns)
        self.data = list(data)


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.Table = _FakeTable

    def log(self, data, **kwargs):
        self.logged.append(dict(data))

    def define_metric(self, *args, **kwargs):
        return None

    def finish(self):
        return None


def test_dynamic_key_detector_and_strip():
    assert is_dynamic_participant_wandb_key("p0/test_loglik")
    assert is_dynamic_participant_wandb_key("p12/test_acc")
    assert is_dynamic_participant_wandb_key("p3_test_accuracy")
    assert not is_dynamic_participant_wandb_key("pfoo/bar")
    assert not is_dynamic_participant_wandb_key("participant/test_loglik")
    assert not is_dynamic_participant_wandb_key("final/mean_test_loglik")
    assert not is_dynamic_participant_wandb_key("progress/completed_participants")
    cleaned = strip_dynamic_participant_wandb_keys(
        {
            "p0/test_loglik": -1.0,
            "p0_train_loglik": -0.9,
            "participant/best_train_val_loglik": -0.8,
            "final/is_complete": True,
        }
    )
    assert set(cleaned) == {
        "participant/best_train_val_loglik",
        "final/is_complete",
    }


def test_suppress_flag_scoped_to_pics_v3(monkeypatch):
    monkeypatch.delenv("KIND", raising=False)
    monkeypatch.delenv("WANDB_PROJECT_NAME", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    assert suppress_dynamic_participant_wandb_scalars() is False
    monkeypatch.setenv("KIND", "pics_v3_g1")
    assert suppress_dynamic_participant_wandb_scalars() is True
    monkeypatch.setenv("KIND", "t_pics_g1_sa40_v2")
    monkeypatch.setenv("WANDB_PROJECT_NAME", "teh_pics_v3")
    assert suppress_dynamic_participant_wandb_scalars() is True
    monkeypatch.setenv("KIND", "something_else")
    monkeypatch.setenv("WANDB_PROJECT_NAME", "openevolve")
    assert suppress_dynamic_participant_wandb_scalars() is False


def test_gated_reporter_strips_dynamic_keys_keeps_fixed_contract(tmp_path: Path):
    selected = tmp_path / "selected"
    expected = [0, 1]
    rows_meta = []
    for ordinal, pid in enumerate(expected):
        person = selected / f"participant_{pid}"
        person.mkdir(parents=True)
        (person / "best_program.py").write_text(
            "def choose(problem, history):\n    return 0.5\n", encoding="utf-8"
        )
        tv = -1.0 - 0.5 * ordinal
        te = -1.5 - 0.5 * ordinal
        (person / "results.json").write_text(
            json.dumps(
                {
                    "overall_best_train": {
                        "program_id": f"p{pid}_best",
                        "selection_score": tv,
                        "train_loglik": tv,
                        "val_loglik": tv,
                    },
                    "overall_best_test": {"program_id": f"p{pid}_best", "test_loglik": te},
                }
            ),
            encoding="utf-8",
        )
        explore = person / "explore"
        explore.mkdir()
        (explore / "metrics.json").write_text(
            json.dumps({"explore_candidates_requested": 50}),
            encoding="utf-8",
        )
        for i in range(1, 11):
            (person / f"iteration_{i}").mkdir()
        rows_meta.append({"final_train_val_loglik": tv, "final_test_loglik": te})

    mean_te = sum(r["final_test_loglik"] for r in rows_meta) / len(rows_meta)
    mean_tv = sum(r["final_train_val_loglik"] for r in rows_meta) / len(rows_meta)
    (selected / "summary_loglik.csv").write_text(
        "num_of_participants,avg_train_loglik,avg_test_loglik,avg_val_loglik\n"
        f"2,{mean_tv},{mean_te},{mean_tv}\n",
        encoding="utf-8",
    )

    fake = _FakeWandb()
    reporter = GatedWandbReporter()
    reporter._wandb = fake
    reporter._run = _FakeRun()
    reporter._enabled = True
    reporter.attach_context(
        expected_participant_ids=expected,
        selected_dir=selected,
        n_iterations=10,
        explore_candidates=50,
        output_root=tmp_path,
    )

    reporter.log(
        {
            "p0/test_loglik": -1.5,
            "p0/train_loglik": -1.0,
            "p0/selection_score": -1.0,
            "p0_step": 3,
            "p0_test_acc": 0.7,
        }
    )
    reporter.log(
        {
            "progress/event": 1,
            "progress/completed_participants": 1,
            "p1/test_loglik": -9.9,
        }
    )
    reporter._safe_summary(
        {
            "dataset": "toy",
            "status/state": "running",
            "p0/test_loglik": -99.0,
        }
    )
    reporter._safe_summary(
        {
            "gate/selected_arm": "control",
            "gate/score_difference": 0.01,
        }
    )

    is_complete = reporter.publish_final()
    assert is_complete is True

    all_keys = set()
    for payload in fake.logged:
        all_keys.update(payload.keys())
    all_keys.update(reporter._run.summary.keys())
    _assert_no_dynamic_keys(all_keys)

    assert "participant/best_train_val_loglik" in all_keys or "participant/step" in all_keys
    assert "progress/completed_participants" in all_keys or "progress/event" in all_keys
    assert reporter._run.summary.get("gate/selected_arm") == "control"
    assert reporter._run.summary.get("final/is_complete") is True
    assert reporter._run.summary.get("final/mean_test_loglik") == pytest.approx(mean_te)
    assert reporter._run.summary.get("final/mean_train_val_loglik") == pytest.approx(mean_tv)

    table_payloads = [p for p in fake.logged if "final/participant_table" in p]
    assert len(table_payloads) == 1
    table = table_payloads[0]["final/participant_table"]
    assert isinstance(table, _FakeTable)
    assert "final_test_loglik" in table.columns
    assert len(table.data) == 2

    rows = [
        {
            "completion_status": "complete",
            "final_train_val_loglik": r["final_train_val_loglik"],
            "final_test_loglik": r["final_test_loglik"],
        }
        for r in rows_meta
    ]
    assert mean_from_complete_rows(rows, field="final_test_loglik") == pytest.approx(mean_te)


def test_publish_final_withholds_mean_until_complete(tmp_path: Path):
    selected = tmp_path / "selected"
    selected.mkdir()
    person = selected / "participant_0"
    person.mkdir()
    (person / "best_program.py").write_text(
        "def choose(problem, history):\n    return 0.5\n", encoding="utf-8"
    )
    (person / "results.json").write_text(
        json.dumps(
            {
                "overall_best_train": {"selection_score": -1.0, "train_loglik": -1.0},
                "overall_best_test": {"test_loglik": -1.2},
            }
        ),
        encoding="utf-8",
    )
    for i in range(1, 11):
        (person / f"iteration_{i}").mkdir()
    explore = person / "explore"
    explore.mkdir()
    (explore / "metrics.json").write_text(
        json.dumps({"explore_candidates_requested": 50}),
        encoding="utf-8",
    )

    fake = _FakeWandb()
    reporter = GatedWandbReporter()
    reporter._wandb = fake
    reporter._run = _FakeRun()
    reporter._enabled = True
    reporter.attach_context(
        expected_participant_ids=[0, 1],
        selected_dir=selected,
        n_iterations=10,
        explore_candidates=50,
        output_root=tmp_path,
    )
    assert reporter.publish_final() is False
    assert reporter._run.summary.get("final/is_complete") is False
    assert "final/mean_test_loglik" not in reporter._run.summary
    _assert_no_dynamic_keys(reporter._run.summary.keys())
    for payload in fake.logged:
        _assert_no_dynamic_keys(payload.keys())
