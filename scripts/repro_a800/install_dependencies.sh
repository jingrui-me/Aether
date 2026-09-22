#!/usr/bin/env bash
set -euo pipefail
WC_ROOT="${WC_ROOT:-/root/autodl-tmp/worldcache-repro}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TMPDIR="$WC_ROOT/tmp"
export PIP_NO_CACHE_DIR=1
mkdir -p "$TMPDIR" "$WC_ROOT/wheelhouse"
PYTHON="$WC_ROOT/envs/aether/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
"$PYTHON" -m pip install --index-url https://mirrors.aliyun.com/pypi/simple \
  --find-links "$WC_ROOT/wheelhouse" -r "$REPO_ROOT/requirements.txt" -c "$REPO_ROOT/scripts/repro_a800/constraints.txt"
"$PYTHON" -m pip check
"$PYTHON" -c 'import torch, diffusers, transformers; print("torch", torch.__version__, "cuda", torch.version.cuda); print("diffusers", diffusers.__version__, "transformers", transformers.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
