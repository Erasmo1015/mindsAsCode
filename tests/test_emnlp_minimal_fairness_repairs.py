from __future__ import annotations

import csv
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from baseline_methods.Psych101.Centaur import (
    build_centaur_prompt_prefix_indexed,
)
from data_modules.mixed_gambles import load_mixed_gambles_trials
from data_modules.psych101_binary import (
    experiment_to_trial_dicts,
    get_filtered_psych101_split,
    get_psych101_binary_experiment,
    parse_psych101_binary_row,
    split_psych_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_RATIO = 0.6
SPLIT_SEED = 0


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _prompt_hash(trials: list[dict], instruction: str) -> str:
    prompts = (
        build_centaur_prompt_prefix_indexed(trials, i, instruction=instruction)
        for i in range(len(trials))
    )
    return hashlib.sha256("\0".join(prompts).encode()).hexdigest()


def test_plonsky_feedback_is_bounded_to_the_same_decision_event() -> None:
    no_feedback = "\n".join(
        ["You press <<U>>.", "You press <<B>>.", "You press <<U>>.",
         "You press <<B>>.", "You press <<U>>."]
    )
    feedback = "\n".join(
        [
            "You press <<U>> and lose 17 points. "
            "You would have lost 8 points had you chosen option B.",
            "You press <<B>> and lose 8 points. "
            "You would have gained 47 points had you chosen option U.",
        ]
    )
    row = {
        "text": (
            "You will choose between two options. In the first five encounters "
            "you will not receive feedback. In the remaining encounters you will.\n\n"
            "Option U delivers -8 points with 100.0% chance.\n"
            "Option U delivers -17 points with unknown chance, "
            "47 points with unknown chance.\n"
            f"{no_feedback}\n{feedback}"
        )
    }

    exp = parse_psych101_binary_row(row, "2plonsky2018when")
    block = exp.blocks[0]

    assert block.option_keys == ["B", "U"]
    assert len(block.trials) == 7
    assert [trial.feedback for trial in block.trials[:5]] == [None] * 5
    assert [trial.feedback for trial in block.trials[5:]] == [-17.0, -8.0]
    assert [trial.action for trial in block.trials] == [1, 0, 1, 0, 1, 1, 0]
    assert block.problem_static["has_feedback"] is True

    trials = experiment_to_trial_dicts(exp)
    prompt = build_centaur_prompt_prefix_indexed(
        trials, 6, instruction=exp.instruction
    )
    assert "You receive -17.0 points" in prompt
    assert "You receive -8.0 points" not in prompt


def test_plonsky_full_source_reconciles_without_response_dependent_drops() -> None:
    rows = get_filtered_psych101_split("2plonsky2018when", split="train")
    raw_count = 0
    parsed_count = 0
    for row in rows:
        raw_count += len(re.findall(r"You press <<[A-Z]>>", row["text"], re.I))
        exp = parse_psych101_binary_row(dict(row), "2plonsky2018when")
        parsed_count += sum(len(block.trials) for block in exp.blocks)
        for block in exp.blocks:
            assert len(block.option_keys) == 2
            assert len(set(block.option_keys)) == 2
            assert [trial.feedback for trial in block.trials[:5]] == [None] * 5
            assert all(trial.feedback is not None for trial in block.trials[5:])

    assert raw_count == 162_000
    assert parsed_count == raw_count


@pytest.mark.parametrize(
    ("alias", "forbidden_fields", "history_fields"),
    [
        (
            "5speekenbrink2008learning",
            {"weather_outcome", "was_correct"},
            {"weather_outcome", "was_correct"},
        ),
        (
            "10frey2017risk",
            {"outcome_marker", "exploded"},
            {"outcome_marker", "exploded"},
        ),
        (
            "12badham2017deficits",
            {"response_key", "correct_category"},
            {"correct_category"},
        ),
    ],
)
def test_post_choice_fields_are_history_only(
    alias: str, forbidden_fields: set[str], history_fields: set[str]
) -> None:
    exp = get_psych101_binary_experiment(alias, 0, split="train")
    trials = experiment_to_trial_dicts(exp)

    assert trials
    assert all(forbidden_fields.isdisjoint(trial["problem"]) for trial in trials)

    trial_with_history = next(trial for trial in trials if trial["history"])
    assert history_fields.issubset(trial_with_history["history"][-1])


def test_repaired_centaur_current_prompts_remain_causal() -> None:
    forbidden_text = {
        "5speekenbrink2008learning": (
            "You are correct",
            "You are wrong",
            "the weather is",
        ),
        "10frey2017risk": (
            "explodes",
            "stop inflating the balloon and get",
        ),
        "12badham2017deficits": ("The correct category is",),
    }
    for alias, phrases in forbidden_text.items():
        exp = get_psych101_binary_experiment(alias, 0, split="train")
        trials = experiment_to_trial_dicts(exp)
        for i, trial in enumerate(trials):
            if trial["history"]:
                continue
            prompt = build_centaur_prompt_prefix_indexed(
                trials, i, instruction=exp.instruction
            )
            current_text = prompt.rsplit("\n\n", 2)[-2]
            for phrase in phrases:
                assert phrase not in current_text


@pytest.mark.parametrize(
    ("alias", "fields"),
    [
        ("5speekenbrink2008learning", {"weather_outcome", "was_correct"}),
        ("10frey2017risk", {"outcome_marker", "exploded"}),
        ("12badham2017deficits", {"response_key", "correct_category"}),
    ],
)
def test_centaur_prompts_are_unchanged_by_pics_leak_redaction(
    alias: str, fields: set[str]
) -> None:
    exp = get_psych101_binary_experiment(alias, 0, split="train")
    after_trials = experiment_to_trial_dicts(exp)
    before_trials = copy.deepcopy(after_trials)
    flat_parsed_trials = [
        parsed_trial
        for block in exp.blocks
        for parsed_trial in block.trials
    ]
    for trial, parsed_trial in zip(before_trials, flat_parsed_trials):
        trial["problem"].update(
            {
                field: parsed_trial.problem_fields[field]
                for field in fields
                if field in parsed_trial.problem_fields
            }
        )

    before_prompts = [
        build_centaur_prompt_prefix_indexed(
            before_trials, i, instruction=exp.instruction
        )
        for i in range(len(before_trials))
    ]
    after_prompts = [
        build_centaur_prompt_prefix_indexed(
            after_trials, i, instruction=exp.instruction
        )
        for i in range(len(after_trials))
    ]
    assert after_prompts == before_prompts


PROTECTED_PSYCH_GOLDENS = {
    "1peterson2021using": {
        "experiment": "8645ab366c881ffb93722a76ba7204e78c7c6262a016ea7d3151bc086e34b362",
        "counts": [60, 20, 20],
        "splits": [
            "5b27fcba72a6890f6394515123a2ee93b4d220b6a16dadc9736cdee80b37ba0f",
            "cb5e87351ffde231353d33cf84e05b5717ead714a65841f88eea229264b015e3",
            "f010f3b9b6dc61aba9f1e6b8a8838ed0b59b94be7fe1fed07ca8ad9b60066572",
        ],
        "prompts": [
            "73c1bac051dfd8f76e5644c3a7e37a3a934f77143223f28a5e4781c8e3f1d978",
            "cd9011d569d66e31627faa8d555f53c507edb454b692291882b0798ad395d179",
            "ba936ec8aa5be06d674d9e8dd6aa3c5b19dae368934c2d9d6224ced741cb076b",
        ],
    },
    "3frey2017cct": {
        "experiment": "497428b7d3b8251c6982b15e4070be340e5689a51f2a263dcf02e4006f2f58b0",
        "counts": [243, 97, 103],
        "splits": [
            "d5414454f5071c28563c4ec440e5ad1f86257a53aafa48abf83ffc44e0a4d749",
            "f27d82584dd6b9a570b3de6241a56316430673dc5ede751cde52daeae076d9e3",
            "ae1d6d3ed60f2170cf8adeee5c7a9d729d4acae9f8f173a41b8d56c9b3e3e87c",
        ],
        "prompts": [
            "f5f7d4b180361eaf20f7a77a35a92a091ce2441e89bc37c54fa7de67aed5ed18",
            "c84d0d17c5cf622a945ae2d7b5896562f83c5733712f3d1bce3ec0b2183f1590",
            "be59e473c454f6e941ae17c3f080931b358e5ae1b5b26da70aeda6e09d4df76e",
        ],
    },
    "4wulff2018description": {
        "experiment": "d842748d0cfba5b6c488aeb62e0193f3c65dd4957420acb41e17c328f1f0e1a3",
        "counts": [1, 1, 1],
        "splits": [
            "66783da45b36c760d993970cf781407db8c2fd7277eba78043c23ee58c58df3b",
            "d7fc7eeb52b4a5c2d53ed63de32f79e877dc65619f9c85beff0ea7805b9286e5",
            "6a1c67bae2c51d2b5e9188a090aaf88e9abacd3fbd9c4c62b0375fbfd896fbdb",
        ],
        "prompts": [
            "1e47acb763d7be7147d1b5b8b9a6111a09bdd766e8276f8266136ff525946e2e",
            "07fd2ac5de2c6d164534539736035bcb355f0f8229accfcc38341a6763dce600",
            "dad8a5578cf9fe6caccbf2b0927df4232836de6c5248a0f8f9e09fcb6dfa0e98",
        ],
    },
    "7hilbig2014generalized": {
        "experiment": "f9d62e19725b3c0febf095e76000be917d49be3b9ccd7fc24d7a122d4125e11e",
        "counts": [48, 32, 16],
        "splits": [
            "68cf8cd7287399593026c8da2951621e8ec3cf7b0ec7b8fd33efcf730cf1aecd",
            "4b115158ebf11dca9f76f7f489466b5f38683eceab03a58ae62d535bf57a441f",
            "b7c13fad59eb816ed4a4f667ac36137a2411a1a0e5578a876ede8747455699ca",
        ],
        "prompts": [
            "55594505240a295a8eed695fd8638075d0b49259cadda6beabe0a10e0cb4e734",
            "02001327019670ee40ec4a5be130d4bd1d52b48bcb6ca2a62eeb041cf7dc8738",
            "86e5bce7623721230da2c938139fe4f8e2356a34b3962e1a243ceb57d906d1f0",
        ],
    },
    "11enkavi2019recentprobes": {
        "experiment": "a35933f83cd316230b22283f57766f312954b06d98e666a6346ac3da748e5380",
        "counts": [45, 15, 15],
        "splits": [
            "ec3512156001b3531aeee8951cda5783a93f86bd1e0a9f88a1eee9d38d576377",
            "e77a201c5fac4f0cbc58ea2f9d3f1a2d9454529ef917761cc259f69228280ab3",
            "db47cfcf4a311a0a456208a454147128793477e9dcfc4ee5253ea6f8f19da7fa",
        ],
        "prompts": [
            "e0ed2cc123c47cb6fc6bbef1fa36db70d822c4931befea7ef33ec321b75cf8df",
            "4cc813e61dfa885fb5781863f4f0f373ccda2a993da91cfa0007aeb3f2320536",
            "7abc8630c8d13a704a6568bd8eb1e527f87e55f8efa85f838bc546e5f7ab5b29",
        ],
    },
}


@pytest.mark.parametrize("alias", sorted(PROTECTED_PSYCH_GOLDENS))
def test_protected_psych101_outputs_are_unchanged(alias: str) -> None:
    expected = PROTECTED_PSYCH_GOLDENS[alias]
    exp = get_psych101_binary_experiment(alias, 0, split="train")
    splits = split_psych_experiment(
        exp, split_ratio=SPLIT_RATIO, split_seed=SPLIT_SEED
    )[:3]

    assert _canonical_hash(exp._asdict()) == expected["experiment"]
    assert [len(split) for split in splits] == expected["counts"]
    assert [_canonical_hash(split) for split in splits] == expected["splits"]
    assert [
        _prompt_hash(split, exp.instruction) for split in splits
    ] == expected["prompts"]


def test_mixed_gambles_output_is_unchanged() -> None:
    csv_path = REPO_ROOT / "datasets/mixed_gambles/data_all_2021-01-08.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        participant_id = int(next(csv.DictReader(handle))["subject"])
    splits = load_mixed_gambles_trials(
        participant_id,
        csv_path=str(csv_path),
        split_ratio=SPLIT_RATIO,
        split_seed=SPLIT_SEED,
    )[:3]

    assert participant_id == 101
    assert [len(split) for split in splits] == [129, 43, 43]
    assert [_canonical_hash(split) for split in splits] == [
        "d9165a0f258ff8822150dcc69b59abba65eeeabf23e81342e137a523a41ba885",
        "6b935cc2c1b34611635e32e8bf8374fe20f5ec0628f5338226c768a574023b11",
        "71216d8074a1a61c1b954b09a49ebb1a5c1a098c227eaf39c367436ba0d03343",
    ]
