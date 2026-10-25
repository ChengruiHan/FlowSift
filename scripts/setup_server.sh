#!/usr/bin/env bash
# Prefer a working preinstalled CUDA stack. Run from the project root.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
mkdir -p results third_party
if "$PYTHON_BIN" -c 'import torch, torchvision; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  "$PYTHON_BIN" -m venv --system-site-packages .venv
  "$PYTHON_BIN" - <<'PY' > results/reuse-runtime-constraints.txt
import torch, torchvision
print(f'torch=={torch.__version__}')
print(f'torchvision=={torchvision.__version__}')
if torch.__version__.startswith('2.8.'):
    print('torchcodec==0.7.0')
PY
  CONSTRAINTS=results/reuse-runtime-constraints.txt
else
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX_URL"
  CONSTRAINTS=configs/server-constraints.txt
fi
.venv/bin/python -m pip install --upgrade pip
if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo >/dev/null 2>&1
fi
if [[ ! -d third_party/lerobot/.git ]]; then
  git clone --branch v0.6.0 --depth 1 https://github.com/huggingface/lerobot.git third_party/lerobot
fi
if [[ "$(git -C third_party/lerobot describe --tags --exact-match)" != "v0.6.0" ]]; then
  echo 'third_party/lerobot must be checked out at v0.6.0' >&2
  exit 1
fi
.venv/bin/python -m pip install -c "$CONSTRAINTS" -e 'third_party/lerobot[libero,smolvla,evaluation]' -e '.[test,server]'
.venv/bin/python scripts/configure_libero.py
.venv/bin/python scripts/inspect_environment.py --output results/server_environment.json
.venv/bin/python -m pip freeze > results/server-requirements.lock.txt
git -C third_party/lerobot rev-parse HEAD > results/lerobot.commit
.venv/bin/python -m pytest -q
printf '\nInstallation complete. Activate .venv and follow docs/server_training.md.\n'
