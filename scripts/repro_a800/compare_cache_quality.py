"""Compare matching decoded RGB videos, with explicit reproducible metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def video_path(value):
    path = Path(value)
    if path.is_dir():
        paths = list(path.glob('*_rgb.mp4'))
        if len(paths) != 1:
            raise ValueError(f'Expected one RGB video in {path}; found {len(paths)}. Pass explicit video path.')
        path = paths[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def compare(ref, test, metric, device):
    caps = [cv2.VideoCapture(str(p)) for p in (ref, test)]
    if not all(c.isOpened() for c in caps):
        raise ValueError('Cannot open both videos')
    fps = [c.get(cv2.CAP_PROP_FPS) for c in caps]
    if abs(fps[0] - fps[1]) > 1e-5:
        raise ValueError(f'FPS mismatch: {fps}')
    rows = []
    try:
        with torch.inference_mode():
            while True:
                ok_a, a = caps[0].read()
                ok_b, b = caps[1].read()
                if ok_a != ok_b:
                    raise ValueError('Frame count mismatch')
                if not ok_a:
                    break
                if a.shape != b.shape:
                    raise ValueError('Frame shape mismatch')
                height, width = a.shape[:2]
                a, b = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in (a, b)]
                def tensor(x):
                    return torch.from_numpy(x.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1
                rows.append(dict(frame=len(rows), psnr=float(peak_signal_noise_ratio(a, b, data_range=255)),
                    ssim=float(structural_similarity(a, b, data_range=255, channel_axis=2,
                        gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
                    lpips=float(metric(tensor(a), tensor(b)).item())))
    finally:
        for cap in caps:
            cap.release()
    if not rows:
        raise ValueError('No decoded frames')
    return dict(reference=str(ref), test=str(test), sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ref, test)},
        frames=len(rows), fps=fps[0], height=int(height), width=int(width),
        mean={key: float(np.mean([r[key] for r in rows])) for key in ('psnr', 'ssim', 'lpips')}, per_frame=rows,
        protocol='Decoded MP4 RGB, native resolution, all frames including frame 0, arithmetic mean of per-frame metrics; PSNR data_range=255; SSIM Gaussian sigma=1.5 population covariance; LPIPS alex v0.1, pretrained calibrated weights, RGB [-1,1]. Measures agreement with baseline, not ground-truth accuracy.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    metric = lpips.LPIPS(net='alex', version='0.1').to(args.device).eval()
    result = compare(video_path(args.ref), video_path(args.test), metric, args.device)
    result['versions'] = dict(torch=torch.__version__, opencv=cv2.__version__)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['mean']))


if __name__ == '__main__':
    main()