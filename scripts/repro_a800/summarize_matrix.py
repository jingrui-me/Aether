"""Aggregate six complete pairs; warmup directories are deliberately excluded."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import torch
import lpips
from compare_cache_quality import compare, video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    if (root / 'exit_code').read_text().strip() != '0':
        raise RuntimeError('Generation did not finish successfully')
    metric = lpips.LPIPS(net='alex', version='0.1').cuda().eval()
    rows = []
    for name in ('car', 'cafe', 'waterfall'):
        for seed in (42,1234):
            case = f'{name}_seed{seed}'
            dirs = [root / mode / case for mode in ('original', 'worldcache')]
            runs = [json.loads((d / 'result.json').read_text()) for d in dirs]
            if any(r['warmup'] or r['name'] != name or r['seed'] != seed for r in runs):
                raise ValueError('Invalid case pairing')
            quality = compare(video_path(dirs[0]), video_path(dirs[1]), metric, 'cuda')
            if (quality['frames'], quality['height'], quality['width'], quality['fps']) != (41,480,720,12):
                raise ValueError('Unexpected output dimensions or FPS')
            (root / f'{case}_quality.json').write_text(json.dumps(quality, indent=2))
            original, cache = runs
            orig_s, cache_s = [r['stages'][0]['seconds'] for r in runs]
            row = dict(input=name, seed=seed, original_denoising_seconds=orig_s, worldcache_denoising_seconds=cache_s,
                denoising_speedup=orig_s/cache_s, original_sample_seconds=original['elapsed_seconds'],
                worldcache_sample_seconds=cache['elapsed_seconds'], sample_speedup=original['elapsed_seconds']/cache['elapsed_seconds'],
                original_full=original['prediction_full'], worldcache_full=cache['prediction_full'], worldcache_cache=cache['prediction_cache'],
                **quality['mean'])
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (root / 'paired_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [k for k in rows[0] if k not in ('input','seed')]
    summary = dict(pairs=len(rows), generated_outputs=2*len(rows), warmup_runs_excluded=2,
        mean={k:statistics.mean(r[k] for r in rows) for k in numeric},
        sample_std={k:statistics.stdev(r[k] for r in rows) for k in numeric},
        denoising_speedup_ratio_of_totals=sum(r['original_denoising_seconds'] for r in rows)/sum(r['worldcache_denoising_seconds'] for r in rows),
        sample_speedup_ratio_of_totals=sum(r['original_sample_seconds'] for r in rows)/sum(r['worldcache_sample_seconds'] for r in rows),
        caveats=['Six different input/seed pairs, not repeated timing trials or the full WorldScore benchmark.',
                 'Each method loads one model then runs one full excluded warmup; six measured samples are all retained.',
                 'Methods run sequentially (all original, then all WorldCache); no contention, but order/time drift is not controlled.',
                 'Quality is decoded-MP4 RGB agreement with baseline, all 41 frames; no ground-truth or temporal metric.',
                 'Denoising duration uses synchronized boundaries around the 50-step progress-bar context; no per-step sync.',
                 'Sample duration excludes model loading, includes preprocessing, prediction, reconstruction and file export.'])
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()