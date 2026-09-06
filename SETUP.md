# mindsAsCode environment setup

This document replaces the old monolithic conda export. Use it when moving the
repo to a new GPU server.

Historical snapshot (do not use for installs):
`env_exports/environment_original_kjha.yml` (from KJha02/mindsAsCode; many
unrelated / transitive packages; a few pip entries that failed are commented).

## New-server workflow

```bash
git clone <repo-url> mindsAsCode
cd mindsAsCode
bash setup_env.sh
conda activate evo310
bash scripts/check_environment.sh
```

Optional flags:

```bash
bash setup_env.sh --with-human      # NiceGUI / FastAPI / tortoise-orm
bash setup_env.sh --with-centaur    # unsloth for Centaur baseline
bash setup_env.sh --skip-flash-attn # only if flash-attn build must be deferred
bash setup_env.sh --force-recreate  # delete and recreate evo310
```

## What each file does

| File | Role |
|------|------|
| `environment.yml` | Minimal conda env: Python 3.12 + `cuda-toolkit=12.4` + pip |
| `pyproject.toml` | Direct Python deps, split into core / extras |
| `setup_env.sh` | Creates env, installs **pinned GPU stack**, then `pip install -e .[extras]` |
| `scripts/check_environment.sh` | Verifies pins + imports + `python -m pip check` |
| `env_exports/` | Frozen snapshots only (reference, not the install path) |

## Known-working GPU stack (must preserve)

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| CUDA toolkit (conda) | 12.4 |
| torch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| torchaudio | 2.5.1+cu124 |
| vllm | 0.7.3 |
| xformers | 0.0.28.post3 |
| transformers | 4.48.3 |
| ray | 2.40.0 |
| numpy | 1.26.4 |
| fsspec | 2024.9.0 |
| flash-attn | 2.7.0.post2 |

These are installed **explicitly** by `setup_env.sh` (PyTorch cu124 index first).
Do not install them via a blind `pip install -r` dump from the old environment.

`setup_env.sh` skips reinstalling Torch/CUDA/vLLM packages when their versions
already match. At the end it **always** re-asserts `numpy==1.26.4` and
`fsspec==2024.9.0` (Torch can otherwise upgrade `fsspec` and break
`datasets==3.2.0`).

JAX CUDA plugins matching the PICS stack are also installed by `setup_env.sh`:

- `jax==0.5.0`, `jaxlib==0.5.0`
- `jax-cuda12-plugin==0.5.0`, `jax-cuda12-pjrt==0.5.0`
- `flax==0.10.3` (via pyproject core)

## Dependency audit (from repository imports)

Scanned ~188 Python files under the repo (excluding generated outputs / nested
third-party clones). Only **direct** third-party imports are listed below.
Transitive packages from the old export are omitted on purpose.

### Core PICS / runtime (`pyproject.toml` `[project].dependencies`)

| Package | Why included (direct import evidence) |
|---------|----------------------------------------|
| `numpy` | Ubiquitous arrays in TE/TEH/gridworld/data modules (~70 files) |
| `tqdm` | Progress bars in runners and baselines |
| `PyYAML` | `utils/teh_transfer/config.py`, OpenEvolve config dumps, analysis compare scripts |
| `msgpack` | Human trajectory decode + `plot_and_eval` / `train_baselines` |
| `tiktoken` | Token budgeting in `utils/rbu.py` (and OpenEvolve helper paths) |
| `rich` | Console formatting in ROTE / LLM baselines and `prompts/generate_gt_single.py` |
| `openai` | LLM client for TE / TEH / Template_evo* and local-vLLM OpenAI-compatible mode |
| `wandb` | Experiment logging in TE / TEH / Template_evo* / `plot_and_eval` |
| `datasets` | Hugging Face `load_dataset` / `load_from_disk` in `data_modules/*`, `te_aggregate.py`, `te_dr.py`, `Template_evo_non_strict.py` |
| `fsspec==2024.9.0` | Peer pin for `datasets==3.2.0` (prevents Torch/HF stacks from pulling incompatible `fsspec`) |
| `jax` / `jaxlib` | Gridworld env + PICS evaluation (`environment*.py`, `gen_data.py`, TE runners, …) |
| `flax` | Serialization / modules used with the JAX env stack |
| `optax` | Optimizers in `plot_and_eval.py` and `train_baselines.py` |
| `matplotlib` / `seaborn` / `pandas` | Core eval plotting and CSV summaries (`plot_and_eval.py`, `eval_partnr.py`, …) |
| `imageio` | GIF / frame IO in `environment.py`, `gen_data.py`, `plot_and_eval.py` |
| `opencv-python` | `cv2` usage in `environment*.py`, `gen_data.py`, video helpers |

### GPU / CUDA-sensitive (installed by `setup_env.sh`, documented under `[project.optional-dependencies].gpu`)

| Package | Why |
|---------|-----|
| `torch` / `torchvision` / `torchaudio` | Direct `torch` imports in ROTE / Centaur / partnr paths; cu124 build required |
| `transformers` | Direct imports in `baselines/*ROTE*`, `plot_and_eval.py`, `eval_partnr.py` |
| `vllm` | In-process `vllm.LLM` in ROTE / NaiveLLM; also used to serve local OpenAI-compatible endpoints |
| `xformers` / `flash-attn` / `ray` | Not imported directly by this repo, but **required pins** of the known-working `vllm==0.7.3` stack |
| `numpy==1.26.4` | Pinned with that stack (also a core runtime dep) |
| `fsspec==2024.9.0` | Final pin re-asserted by `setup_env.sh` so Torch cannot break `datasets` |
| `jax-cuda12-plugin` / `jax-cuda12-pjrt` | CUDA backends for `jax==0.5.0` on GPU nodes |

### Baseline (`[baseline]` / special installs)

| Package | Why |
|---------|-----|
| `scipy` | `baseline_methods/MLE.py`, `prospect_theory.py`, `teh_psych/MLE.py`, `baselines/analysis.py` |
| `unsloth` | Optional Centaur path (`baseline_methods/Centaur.py`) — `setup_env.sh --with-centaur` |
| `openevolve` | Imported by OpenEvolve runners; install by cloning into `reference_repos/openevolve` (see below), not via the default pip set |
| AutoToM | Nested clone `baselines/AutoToM` (see README); not a pip dependency of this repo |

### Analysis / dev

| Package | Why |
|---------|-----|
| `scipy` | Also used by `analysis/code/mixed_gambles/train_MLE.py` |
| `matplotlib` / `pandas` / `seaborn` / `PyYAML` | Analysis scripts under `analysis/code/**` (already in core) |
| `pytest` | `tests/` and `utils/teh_psych/test_*.py` |

### Human experiments (`[human]` + NiceWebRL)

| Package | Why |
|---------|-----|
| `nicegui` / `fastapi` / `aiofiles` / `tortoise-orm` | Web apps: `play_human_web_app.py`, `prediction_*_web_app.py`, … |
| `nicewebrl` | Direct imports in human experiment entrypoints — install from [NiceWebRL](https://github.com/KempnerInstitute/nicewebrl) separately |

### Present in the old environment export but **not** direct deps of this repo

Examples (non-exhaustive; the export has ~400 pip pins): `ai2thor`, `dash*`, `deepspeed`, `openrlhf` (commented), `verl` (commented), `gemma` (commented), `kauldron` (commented), `lamorel` (commented), `xmanager`, `clu`, `array-record`, `scikit-learn`, `statsmodels`, `black`, `jupyterlab`, `cupy-cuda12x`, `bitsandbytes`, `accelerate`, `peft`, `trl`, `anthropic`, and most `nvidia-*-cu12` wheels pulled transitively by torch/vllm.

`scikit-learn` and `statsmodels` appear at the bottom of the old export under a “pkg for PICS” comment, but **no file in this repository imports them**.

## Extra baseline setup (unchanged research behavior)

### AutoToM

```bash
git clone https://github.com/KJha02/AutoToM.git baselines/AutoToM
```

### OpenEvolve

Runners expect a local checkout (see `baseline_methods/deprecated_run_openevolve.py` /
`baseline_methods/Psych101/run_openevolve.py`):

```bash
mkdir -p reference_repos
git clone <openevolve-repo-url> reference_repos/openevolve
```

### Centaur

Prefer a dedicated large-model workflow; or:

```bash
bash setup_env.sh --with-centaur
```

See `baseline_methods/AGENT_Centaur.md`.

### Partnr

Clone the Partnr fork separately and follow that repository’s setup (see README).

## Smoke checks after install

```bash
conda activate evo310
bash scripts/check_environment.sh

# Local vLLM server (example)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# TE-style local mode then routes OpenAI client calls to that server
# (see AGENT.md for full flags)
```

## Design rules for future maintainers

1. Add a dependency to `pyproject.toml` only if something in **this** repo imports it (or it is a documented pinned peer of the GPU stack).
2. Never re-copy `env_exports/environment_original_kjha.yml` into the install path.
3. Change GPU pins only together (torch / vision / audio / vllm / xformers / flash-attn / transformers / ray / numpy / fsspec) and re-run `scripts/check_environment.sh` on a GPU node.
4. Keep final-env pins `numpy==1.26.4` and `fsspec==2024.9.0`; `scripts/check_environment.sh` must pass `python -m pip check`.
5. Do not change algorithm or experiment code when adjusting packaging.
