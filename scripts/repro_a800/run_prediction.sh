#!/usr/bin/env bash
set -euo pipefail
WC_ROOT="${WC_ROOT:-/root/autodl-tmp/worldcache-repro}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="$WC_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$WC_ROOT/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$WC_ROOT/outputs" "$TMPDIR"
cd "$REPO_ROOT"
exec "$WC_ROOT/envs/aether/bin/python" scripts/demo.py \
  --task prediction \
  --image assets/example_obs/car.png \
  --raymap_action assets/example_raymaps/raymap_forward_right.npy \
  --cogvideox_pretrained_model_name_or_path "$WC_ROOT/models/CogVideoX-5b-I2V" \
  --aether_pretrained_model_name_or_path "$WC_ROOT/models/AetherV1" \
  --seed 42 --height 480 --width 720 --num_frames 41 \
  --num_inference_steps 50 \
  --output_dir "$WC_ROOT/outputs/aether_worldcache_car_seed42"
