# Small paired Aether / WorldCache experiment

Use the same `aether` environment as README.md. Add `lpips==0.1.4` (no changes to torch/diffusers). LPIPS downloads the official AlexNet weights on first use. Set `TORCH_HOME=$WC_ROOT/cache/torch`.

Inputs are `$WC_ROOT/repos/Aether/assets/example_obs/car.png`, `$WC_ROOT/inputs/cafe.jpg` and `$WC_ROOT/inputs/waterfall.jpg`. The extra images are from `Howieeeee/WorldScore`, revision `42c4e267e1cc0529af1b0284ce018e04d43906e3`, `static/train` rows 0 and 500. Keep the downloaded image hashes in `inputs/sources.json`. All three use Aether's same forward-right raymap; this is not the full WorldScore evaluation protocol.

Run each method sequentially on one GPU, with a fresh output root:

```bash
export WC_ROOT=/root/autodl-tmp/worldcache-repro
export HF_HOME="$WC_ROOT/cache/huggingface" TORCH_HOME="$WC_ROOT/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export TMPDIR="$WC_ROOT/tmp"
export WORLDCACHE_MODE=worldcache WORLDCACHE_PERCENTILE_STABLE=0.30 WORLDCACHE_PERCENTILE_CHAOTIC=0.60 WORLDCACHE_N_MAX=2 WORLDCACHE_ERROR_THRESHOLD=0.2
out="$WC_ROOT/outputs/matrix_NEW"
mkdir -p "$out"
for mode in original worldcache; do
  "$WC_ROOT/envs/aether/bin/python" scripts/repro_a800/run_matrix.py --mode "$mode" --output "$out" || exit $?
done
echo 0 > "$out/exit_code"
"$WC_ROOT/envs/aether/bin/python" scripts/repro_a800/summarize_matrix.py --root "$out"
```

The original source must be at `$WC_ROOT/repos/Aether` and the adapted source at `$WC_ROOT/repos/Aether-worldcache`. Scripts import the corresponding demo; the original code is not patched. Each process loads one pipeline, runs a full car/42 warmup excluded from statistics, then the six input/seed cases. This means 14 runs total, 12 measured outputs. No measured case is dropped. All default demo post-reconstruction and exports are preserved. The model remains loaded between cases; CUDA synchronization surrounds the same denoising context for both methods. `result.json` records both the 50-step prediction and 4-step reconstruction separately. Per-sample elapsed time excludes model loading but includes preprocessing and export. Logging remains enabled in both methods.

Compare existing videos without rerunning generation:

```bash
"$WC_ROOT/envs/aether/bin/python" scripts/repro_a800/compare_cache_quality.py \
  --ref "$WC_ROOT/outputs/aether_original_pair_seed42" \
  --test "$WC_ROOT/outputs/aether_worldcache_car_seed42" \
  --output "$WC_ROOT/logs/first_pair_quality.json"
```

Explicit MP4 paths are also accepted. Directory inputs must contain exactly one `*_rgb.mp4`, to prevent accidentally comparing the wrong video. Frames/FPS/dimensions must match. Metrics average per-frame PSNR, RGB SSIM (Gaussian sigma=1.5, population covariance) and official LPIPS AlexNet v0.1 across all frames including frame 0. There is no resizing or registration. These measure agreement with baseline compressed video, not ground-truth accuracy, temporal quality, or a guarantee of lossless acceleration.

Aggregation reports each pair, sample mean/std, mean of pairwise speedups and ratio of total durations separately. Six different inputs/seeds are exploratory data, not repeated timing trials. Original runs precede cached runs, so temporal drift remains a limitation.