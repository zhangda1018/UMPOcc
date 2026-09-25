#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from tools.test_ood_fast import maybe_import_plugins, soft_occupancy_reliable_calibrate


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark SGN OOD inference.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--score-mode', choices=('echoood', 'ppsc_v3'), required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def build_runtime(args):
    cfg = Config.fromfile(args.config)
    maybe_import_plugins(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.pts_bbox_head.return_umpof_features = False
    cfg.model.pts_bbox_head.ood_flag = True
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    return dataset, data_loader, model


def score_result(result, mode):
    score = result['ood_pred']
    if mode == 'echoood':
        return score
    return soft_occupancy_reliable_calibrate(
        score,
        result['output_voxels'],
        kernel_size=7,
        alpha=0.75,
        gamma=2.0,
        boost=0.25,
        suppress=0.0,
        min_support=0.0,
        support_scale=0.2,
        support_power=1.0)


def main():
    args = parse_args()
    if args.warmup < 0 or args.samples < 1:
        raise ValueError('--warmup must be non-negative and --samples positive.')
    dataset, data_loader, model = build_runtime(args)
    required = args.warmup + args.samples
    if len(dataset) < required:
        raise ValueError(f'Dataset has {len(dataset)} samples, but {required} are required.')

    timings = []
    torch.cuda.reset_peak_memory_stats()
    for index, data in enumerate(data_loader):
        if index >= required:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            score = score_result(result, args.score_mode)
            score.sum().item()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if index + 1 == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        if index >= args.warmup:
            timings.append(elapsed)

    timings = np.asarray(timings, dtype=np.float64)
    module = model.module
    total_params = sum(parameter.numel() for parameter in module.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in module.parameters()
        if parameter.requires_grad)
    result = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'score_mode': args.score_mode,
        'warmup_samples': args.warmup,
        'measured_samples': len(timings),
        'mean_latency_ms': float(1000.0 * timings.mean()),
        'median_latency_ms': float(1000.0 * np.median(timings)),
        'p95_latency_ms': float(1000.0 * np.percentile(timings, 95)),
        'fps_from_mean': float(1.0 / timings.mean()),
        'peak_allocated_gib': float(torch.cuda.max_memory_allocated() / 2**30),
        'peak_reserved_gib': float(torch.cuda.max_memory_reserved() / 2**30),
        'parameters_m': float(total_params / 1e6),
        'trainable_parameters_m': float(trainable_params / 1e6),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
