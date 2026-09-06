#!/usr/bin/env bash
# Create / refresh the evo310 conda env and install the known-working GPU stack
# plus mindsAsCode Python dependencies from pyproject.toml.
#
# Usage (from repo root):
#   bash setup_env.sh
#   bash setup_env.sh --with-human
#   bash setup_env.sh --with-centaur
#   bash setup_env.sh --skip-flash-attn
#   bash setup_env.sh --extras "baseline,analysis,dev"
#
# Then:
#   conda activate evo310
#   bash scripts/check_environment.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="evo310"
WITH_HUMAN=0
WITH_CENTAUR=0
SKIP_FLASH_ATTN=0
EXTRAS="baseline,analysis,dev"
FORCE_RECREATE=0

# Final-env pins that must win over transitive upgrades (e.g. torch bumping fsspec).
PIN_NUMPY="1.26.4"
PIN_FSSPEC="2024.9.0"

usage() {
  cat <<'EOF'
setup_env.sh — bootstrap mindsAsCode conda env (evo310)

Options:
  --with-human        Also install human web-experiment extras (nicegui/fastapi/...)
  --with-centaur      Also install unsloth (Centaur baseline)
  --skip-flash-attn   Skip flash-attn build/install (not recommended)
  --extras LIST       Comma-separated pyproject extras (default: baseline,analysis,dev)
  --force-recreate    Remove existing evo310 env and recreate from environment.yml
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-human) WITH_HUMAN=1; shift ;;
    --with-centaur) WITH_CENTAUR=1; shift ;;
    --skip-flash-attn) SKIP_FLASH_ATTN=1; shift ;;
    --extras) EXTRAS="${2:-}"; shift 2 ;;
    --force-recreate) FORCE_RECREATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# Return 0 if importable package version starts with expected prefix.
pkg_version_ok() {
  local mod="$1"
  local expect="$2"
  python - "$mod" "$expect" <<'PY'
import importlib, sys
mod, expect = sys.argv[1], sys.argv[2]
try:
    m = importlib.import_module(mod)
except Exception:
    sys.exit(1)
ver = getattr(m, "__version__", None)
if ver is None or not str(ver).startswith(expect):
    sys.exit(1)
sys.exit(0)
PY
}

# pip show version equals exact string (distribution name, not import name).
dist_version_eq() {
  local dist="$1"
  local expect="$2"
  python - "$dist" "$expect" <<'PY'
import importlib.metadata as md
import sys
dist, expect = sys.argv[1], sys.argv[2]
try:
    ver = md.version(dist)
except md.PackageNotFoundError:
    sys.exit(1)
sys.exit(0 if ver == expect else 1)
PY
}

torch_stack_ok() {
  python - <<'PY'
import importlib
import sys

need = {
    "torch": ("2.5.1", "+cu124", "12.4"),
    "torchvision": ("0.20.1", None, None),
    "torchaudio": ("2.5.1", None, None),
}
for mod, (prefix, tag, cuda) in need.items():
    try:
        m = importlib.import_module(mod)
    except Exception:
        sys.exit(1)
    ver = getattr(m, "__version__", "")
    if not str(ver).startswith(prefix):
        sys.exit(1)
    if tag and tag not in str(ver):
        sys.exit(1)
    if cuda is not None:
        if getattr(m, "version", None) is None or m.version.cuda != cuda:
            sys.exit(1)
sys.exit(0)
PY
}

ensure_final_pins() {
  echo "==> Ensuring final pins: numpy==${PIN_NUMPY}, fsspec==${PIN_FSSPEC}"
  python -m pip install --force-reinstall --no-deps \
    "numpy==${PIN_NUMPY}" \
    "fsspec==${PIN_FSSPEC}"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda/Mambaforge and retry." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "$FORCE_RECREATE" -eq 1 ]] && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Removing existing conda env: $ENV_NAME"
  conda env remove -y -n "$ENV_NAME"
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Conda env '$ENV_NAME' already exists; updating from environment.yml"
  conda env update -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune
else
  echo "==> Creating conda env from environment.yml"
  conda env create -n "$ENV_NAME" -f "$ROOT/environment.yml"
fi

conda activate "$ENV_NAME"

echo "==> Python: $(python -V) @ $(which python)"
echo "==> Ensuring pip/setuptools/wheel"
python -m pip install -U pip setuptools wheel

# ---------------------------------------------------------------------------
# Known-working GPU stack (must preserve these pins)
# Python 3.12 + CUDA toolkit 12.4 (conda) + torch 2.5.1+cu124 + vllm 0.7.3 ...
# Skip reinstall when the stack is already correct.
# ---------------------------------------------------------------------------
if torch_stack_ok; then
  echo "==> PyTorch cu124 stack already correct; skipping reinstall"
else
  echo "==> Installing PyTorch 2.5.1 (cu124)"
  python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
fi

echo "==> Pinning numpy==${PIN_NUMPY} (required by the working GPU stack)"
python -m pip install --force-reinstall --no-deps "numpy==${PIN_NUMPY}"

if pkg_version_ok transformers 4.48.3 && pkg_version_ok ray 2.40.0; then
  echo "==> transformers/ray already at pinned versions; skipping"
else
  echo "==> Installing transformers / ray (vLLM-compatible pins)"
  python -m pip install "transformers==4.48.3" "ray==2.40.0"
fi

if pkg_version_ok xformers 0.0.28.post3; then
  echo "==> xformers already at 0.0.28.post3; skipping"
else
  echo "==> Installing xformers==0.0.28.post3"
  python -m pip install "xformers==0.0.28.post3"
fi

if pkg_version_ok vllm 0.7.3; then
  echo "==> vllm already at 0.7.3; skipping"
else
  echo "==> Installing vllm==0.7.3"
  python -m pip install "vllm==0.7.3"
fi

if [[ "$SKIP_FLASH_ATTN" -eq 1 ]]; then
  echo "==> Skipping flash-attn (--skip-flash-attn)"
elif pkg_version_ok flash_attn 2.7.0.post2; then
  echo "==> flash-attn already at 2.7.0.post2; skipping"
else
  echo "==> Installing flash-attn==2.7.0.post2"
  # Needs the env's nvcc from cuda-toolkit=12.4 and an already-installed torch.
  export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
  export PATH="$CONDA_PREFIX/bin:$PATH"
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
    "flash-attn==2.7.0.post2" --no-build-isolation
fi

# Build extras list for editable install
EXTRA_LIST="$EXTRAS"
if [[ "$WITH_HUMAN" -eq 1 ]]; then
  EXTRA_LIST="${EXTRA_LIST},human"
fi
if [[ "$WITH_CENTAUR" -eq 1 ]]; then
  EXTRA_LIST="${EXTRA_LIST},centaur"
fi
# Drop leading/trailing commas and duplicates is fine for pip
EXTRA_LIST="$(echo "$EXTRA_LIST" | sed 's/^,//;s/,$//;s/,,/,/g')"

echo "==> Editable install: pip install -e .[${EXTRA_LIST}]"
python -m pip install -e ".[${EXTRA_LIST}]"

# JAX CUDA plugins AFTER editable install so pip does not replace them with CPU-only jaxlib.
if pkg_version_ok jax 0.5.0 && pkg_version_ok jaxlib 0.5.0 \
  && dist_version_eq jax-cuda12-plugin 0.5.0 \
  && dist_version_eq jax-cuda12-pjrt 0.5.0; then
  echo "==> JAX 0.5.0 + CUDA 12 plugins already present; skipping"
else
  echo "==> Installing JAX 0.5.0 CUDA 12 plugins (matches flax/jaxlib pins in pyproject)"
  python -m pip install \
    "jax==0.5.0" "jaxlib==0.5.0" \
    "jax-cuda12-plugin==0.5.0" "jax-cuda12-pjrt==0.5.0"
fi

# Re-assert critical non-torch pins only when wrong (avoid force-reinstalling torch/vLLM).
echo "==> Re-asserting critical pins if needed (without touching a correct Torch stack)"
if ! pkg_version_ok transformers 4.48.3; then
  python -m pip install --force-reinstall --no-deps "transformers==4.48.3"
fi
if ! pkg_version_ok ray 2.40.0; then
  python -m pip install --force-reinstall --no-deps "ray==2.40.0"
fi
if ! pkg_version_ok vllm 0.7.3; then
  python -m pip install --force-reinstall --no-deps "vllm==0.7.3"
fi
if ! pkg_version_ok xformers 0.0.28.post3; then
  python -m pip install --force-reinstall --no-deps "xformers==0.0.28.post3"
fi
if ! torch_stack_ok; then
  echo "==> Torch stack drifted; reinstalling cu124 wheels"
  python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
fi

# Always win last: numpy + fsspec (torch/datasets-compatible).
ensure_final_pins

echo
echo "======================================================================"
echo " Environment bootstrap finished."
echo " Next:"
echo "   conda activate ${ENV_NAME}"
echo "   bash scripts/check_environment.sh"
echo
echo " Optional baselines (not installed by default):"
echo "   OpenEvolve: clone into reference_repos/openevolve (see SETUP.md)"
echo "   AutoToM:    git clone https://github.com/KJha02/AutoToM.git baselines/AutoToM"
echo "   NiceWebRL:  install from KempnerInstitute/nicewebrl if using --with-human"
echo "======================================================================"
