"""Run six paired cases with one resident pipeline and a separate full warmup."""
import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['original', 'worldcache'], required=True)
    parser.add_argument('--root', default='/root/autodl-tmp/worldcache-repro')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    repo = root / 'repos' / ('Aether' if args.mode == 'original' else 'Aether-worldcache')
    out = Path(args.output) / args.mode
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    spec = importlib.util.spec_from_file_location('repro_demo', repo / 'scripts/demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    images = [('car', repo / 'assets/example_obs/car.png'), ('cafe', root / 'inputs/cafe.jpg'), ('waterfall', root / 'inputs/waterfall.jpg')]
    raymap = repo / 'assets/example_raymaps/raymap_forward_right.npy'
    common = ['--task', 'prediction', '--raymap_action', str(raymap),
        '--cogvideox_pretrained_model_name_or_path', str(root / 'models/CogVideoX-5b-I2V'),
        '--aether_pretrained_model_name_or_path', str(root / 'models/AetherV1'),
        '--height', '480', '--width', '720', '--num_frames', '41', '--num_inference_steps', '50']
    for _, path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = dict(mode=args.mode, repo=str(repo), common_args=common,
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [x[1] for x in images] + [raymap]},
        torch=torch.__version__, protocol='One persistent model per method. Full car/42 warmup excluded before six measured cases. CUDA synchronize at denoising boundaries. No deletion of measured cases. Same forward-right raymap for all images. Main duration includes encoding, prediction, reconstruction, exports, but excludes model loading.',
        cache_environment={k:v for k,v in os.environ.items() if k.startswith('WORLDCACHE_')})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    sys.argv = ['demo.py'] + common + ['--image', str(images[0][1]), '--seed', '42', '--output_dir', str(out / 'warmup')]
    demo.seed_all(42)
    start = time.perf_counter()
    pipeline = demo.build_pipeline(demo.parse_args())
    torch.cuda.synchronize()
    manifest['model_load_seconds'] = time.perf_counter() - start
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    demo.build_pipeline = lambda _: pipeline
    base_progress = pipeline.progress_bar
    stages = []

    @contextlib.contextmanager
    def timed_progress(*pa, **kw):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with base_progress(*pa, **kw) as bar:
            yield bar
        torch.cuda.synchronize()
        stages.append(dict(steps=kw.get('total'), completed=int(bar.n), seconds=time.perf_counter() - started))

    pipeline.progress_bar = timed_progress
    cases = [('warmup', images[0][1], 42, True)] + [(name, path, seed, False) for name,path in images for seed in (42,1234)]
    for name, path, seed, warmup in cases:
        run_name = 'warmup' if warmup else f'{name}_seed{seed}'
        directory = out / run_name
        directory.mkdir()
        sys.argv = ['demo.py'] + common + ['--image', str(path), '--seed', str(seed), '--output_dir', str(directory)]
        stages.clear()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(f'START {args.mode} {run_name}', flush=True)
        started = time.perf_counter()
        with (directory / 'run.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            demo.main()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if [(s['steps'],s['completed']) for s in stages] != [(50,50),(4,4)]:
            raise RuntimeError(f'Incomplete denoising: {stages}')
        log_text = (directory / 'run.log').read_text()
        types = re.findall(r'Calculation Type: (full|worldcache)', log_text)
        if args.mode == 'worldcache' and len(types) != 54:
            raise RuntimeError(f'Expected 54 calculation type records, got {len(types)}')
        prediction = types[:50] if types else ['full'] * 50
        result = dict(mode=args.mode, name=name, seed=seed, warmup=warmup, stages=list(stages),
            elapsed_seconds=elapsed, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            prediction_full=prediction.count('full'), prediction_cache=prediction.count('worldcache'),
            count_basis='WorldCache log' if types else 'Pristine full forward and verified 50 completed steps',
            videos=[str(p) for p in directory.glob('*_rgb.mp4')])
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        print('DONE ' + json.dumps(result), flush=True)
    print(f'COMPLETE {args.mode}', flush=True)


if __name__ == '__main__':
    main()