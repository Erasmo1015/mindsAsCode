"""6×6 T-PICS packed-grid guards. No job launch, no source-prediction ranking."""
from __future__ import annotations

import unittest
from pathlib import Path

from utils.teh.t_pics_6x6 import (
    DEFAULT_SCHEMA4_CONFIG,
    REPRESENTATIVES,
    all_arms,
    arms_for_target,
    forbidden_cli_tokens,
    grid_counts,
    required_freeze_flags,
    slurm_jobs,
    validate_schema4_sources,
)


ROOT = Path("cluster/2026Sep18_T_PICS_6x6")


class TestTPics6x6Grid(unittest.TestCase):
    def test_counts_are_30_transfers_and_6_controls(self) -> None:
        counts = grid_counts()
        self.assertEqual(counts["targets"], 6)
        self.assertEqual(counts["jobs"], 6)
        self.assertEqual(counts["arms"], 36)
        self.assertEqual(counts["controls"], 6)
        self.assertEqual(counts["transfers"], 30)
        self.assertEqual(counts["same_dataset"], 0)

    def test_no_same_dataset_transfer(self) -> None:
        for arm in all_arms():
            if arm.source is not None:
                self.assertNotEqual(arm.source, arm.target)
        for target in REPRESENTATIVES:
            sources = [a.source for a in arms_for_target(target) if a.source]
            self.assertEqual(len(sources), 5)
            self.assertNotIn(target, sources)

    def test_control_is_first_arm(self) -> None:
        for target in REPRESENTATIVES:
            arms = arms_for_target(target)
            self.assertEqual(arms[0].arm_id, "control")
            self.assertIsNone(arms[0].source)

    def test_gpu_split_is_two_h100_and_four_l40s(self) -> None:
        jobs = slurm_jobs()
        self.assertEqual(len(jobs), 6)
        gpus = [j["gpu"] for j in jobs]
        self.assertEqual(gpus.count("h100nvl"), 2)
        self.assertEqual(gpus.count("l40s_tp2"), 4)
        by_target = {j["target"]: j for j in jobs}
        self.assertEqual(by_target["1peterson2021using"]["gpu"], "h100nvl")
        self.assertEqual(by_target["bergert_nosofsky_2007"]["gpu"], "h100nvl")
        self.assertEqual(by_target["guan_2020_stopping"]["partition"], "researchlong")
        self.assertEqual(by_target["steyvers_2009_bandit"]["partition"], "researchlong")
        self.assertEqual(by_target["11enkavi2019recentprobes"]["partition"], "researchshort")
        self.assertEqual(by_target["12badham2017deficits"]["partition"], "researchshort")

    def test_schema4_rank1_files_exist_for_representatives(self) -> None:
        errors = validate_schema4_sources(config_path=DEFAULT_SCHEMA4_CONFIG)
        self.assertEqual(errors, [])

    def test_workers_keep_freeze_flags_and_omit_t_pics(self) -> None:
        common = (ROOT / "_6x6_common.sh").read_text(encoding="utf-8")
        for token in required_freeze_flags():
            self.assertIn(token, common)
        self.assertIn("--global_prompt_source_program", common)
        self.assertIn('if [[ -n "${source_best}" ]]', common)
        for name in ("job_6x6_h100.sh", "job_6x6_l40s.sh", "_6x6_common.sh"):
            text = (ROOT / name).read_text(encoding="utf-8")
            for banned in forbidden_cli_tokens():
                self.assertNotIn(banned + " ", text)
                self.assertNotIn(banned + "\n", text)
            self.assertNotIn("--t_pics_source_config", text)
        for worker in ("job_6x6_h100.sh", "job_6x6_l40s.sh"):
            text = (ROOT / worker).read_text(encoding="utf-8")
            self.assertIn('LIMITED_DATA_PROTOCOL="${LIMITED_DATA_PROTOCOL:-structure_aware}"', text)
            self.assertIn('LIMITED_TRAIN_VAL="${LIMITED_TRAIN_VAL:-40}"', text)
            self.assertIn('GLOBAL_ITERS="${GLOBAL_ITERS:-5}"', text)
            self.assertIn('N_ITERS="${N_ITERS:-5}"', text)

    def test_source_prompt_suffix_covers_all_representatives(self) -> None:
        from utils.teh.explore_source_prompt import build_rank1_explore_prompt_suffix
        from utils.teh.t_pics_6x6 import source_rank1
        from utils.teh_transfer.prompts import task_description_for_dataset

        for alias in REPRESENTATIVES:
            desc = task_description_for_dataset(alias)
            self.assertTrue(desc.strip(), msg=alias)
            suf = build_rank1_explore_prompt_suffix(
                source_dataset=alias,
                program_path=str(source_rank1(alias)),
                split_seed=0,
                psych_dataset_split="train",
                limited_data_protocol="structure_aware",
                limited_train_val=40,
            )
            self.assertGreater(len(suf), 100, msg=alias)
            self.assertIn("Cross-task transfer context", suf)
        text = (ROOT / "submit_6x6.sh").read_text(encoding="utf-8")
        self.assertIn('DRY_RUN="${DRY_RUN:-1}"', text)
        self.assertIn('CONFIRM_SUBMIT="${CONFIRM_SUBMIT:-0}"', text)
        self.assertIn("GLOBAL_ITERS:-5", text)
        self.assertIn("N_ITERS:-5", text)


if __name__ == "__main__":
    unittest.main()
