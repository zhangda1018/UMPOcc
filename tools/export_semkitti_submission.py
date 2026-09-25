#!/usr/bin/env python
import argparse
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
from tools.test_ood_fast import maybe_import_plugins


LEARNING_TO_KITTI = np.asarray(
    [0, 10, 11, 15, 18, 20, 30, 31, 32, 40,
     44, 48, 49, 50, 51, 70, 71, 72, 80, 81],
    dtype=np.uint16)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export SemanticKITTI SSC test predictions.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--sequences', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--progress-interval', type=int, default=25)
    return parser.parse_args()


def build_runtime(args):
    cfg = Config.fromfile(args.config)
    maybe_import_plugins(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.pts_bbox_head.save_flag = False
    cfg.model.pts_bbox_head.ood_flag = False
    cfg.model.pts_bbox_head.return_umpof_features = False
    cfg.data.test.split = 'test'
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    requested = set(args.sequences)
    available = {str(scan['sequence']) for scan in dataset.scans}
    missing = requested - available
    if missing:
        raise ValueError(f'Unknown test sequences: {sorted(missing)}')
    dataset.scans = [
        scan for scan in dataset.scans
        if str(scan['sequence']) in requested]
    dataset.scans.sort(
        key=lambda scan: (str(scan['sequence']), Path(scan['voxel_path']).stem))
    dataset.set_group_flag()

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


def main():
    args = parse_args()
    dataset, data_loader, model = build_runtime(args)
    limit = len(dataset)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError('--max-samples must be positive.')
        limit = min(limit, args.max_samples)

    root = Path(args.out_dir)
    start = time.time()
    written = 0
    for index, data in enumerate(data_loader):
        if index >= limit:
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        prediction = torch.argmax(result['output_voxels'], dim=1)[0]
        prediction = prediction.detach().cpu().numpy().astype(np.int64)
        mapped = LEARNING_TO_KITTI[prediction]

        scan = dataset.scans[index]
        sequence = str(scan['sequence'])
        frame_id = Path(scan['voxel_path']).stem
        destination = root / 'sequences' / sequence / 'predictions' / f'{frame_id}.label'
        destination.parent.mkdir(parents=True, exist_ok=True)
        mapped.tofile(destination)
        written += 1
        if written % args.progress_interval == 0 or written == limit:
            print(
                f'Exported {written}/{limit} predictions in '
                f'{time.time() - start:.1f}s', flush=True)

    print(f'Finished sequences={sorted(args.sequences)}, files={written}, root={root}')


if __name__ == '__main__':
    main()
