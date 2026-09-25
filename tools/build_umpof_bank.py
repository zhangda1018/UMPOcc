import argparse
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from sklearn.cluster import MiniBatchKMeans

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Build UMPOF prototype bank.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--modes', type=int, default=4)
    parser.add_argument('--per-sample-class', type=int, default=2048)
    parser.add_argument('--max-features-per-class', type=int, default=40000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_import_plugins(cfg):
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = cfg.get('plugin_dir', None)
        if plugin_dir is None:
            return
        module_dir = os.path.dirname(plugin_dir).split('/')
        module_path = module_dir[0]
        for item in module_dir[1:]:
            module_path = module_path + '.' + item
        importlib.import_module(module_path)


def prepare_cfg(args):
    cfg = Config.fromfile(args.config)
    maybe_import_plugins(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.pts_bbox_head.return_umpof_features = True
    cfg.model.pts_bbox_head.ood_flag = False
    cfg.data.workers_per_gpu = args.workers

    data_cfg = cfg.data[args.split]
    data_cfg.test_mode = True
    data_cfg.pop('samples_per_gpu', None)
    return cfg, data_cfg


@torch.no_grad()
def collect_features(model, data_loader, class_ids, args):
    class_features = defaultdict(list)
    class_counts = defaultdict(int)
    processed = 0

    model.eval()
    for i, data in enumerate(data_loader):
        if args.max_samples is not None and processed >= args.max_samples:
            break

        result = model(return_loss=False, rescale=True, **data)
        feats = result['vox_feats_full'][0].detach()
        target = result['target_voxels'][0].detach().long()

        _, h, w, z = feats.shape
        feats_flat = feats.permute(1, 2, 3, 0).reshape(h * w * z, -1)
        target_flat = target.reshape(-1)

        for class_id in class_ids:
            indices = torch.nonzero(target_flat == class_id, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            if indices.numel() > args.per_sample_class:
                perm = torch.randperm(indices.numel(), device=indices.device)[:args.per_sample_class]
                indices = indices[perm]

            sampled = feats_flat[indices].float().cpu()
            current = class_counts[class_id]
            if current < args.max_features_per_class:
                keep = min(sampled.shape[0], args.max_features_per_class - current)
                class_features[class_id].append(sampled[:keep])
                class_counts[class_id] += keep

        processed += 1
        if processed % 10 == 0:
            print('Collected features from {} samples.'.format(processed), flush=True)

    return class_features, class_counts, processed


def normalize_np(x, eps=1e-6):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def fit_prototypes_for_class(features, modes, seed):
    x = torch.cat(features, dim=0).numpy().astype(np.float32)
    x = normalize_np(x)
    num_samples, feat_dim = x.shape

    if num_samples == 0:
        raise ValueError('Cannot fit prototypes with zero samples.')

    if num_samples < modes:
        center = normalize_np(x.mean(axis=0, keepdims=True))
        centers = np.repeat(center, modes, axis=0)
        variances = np.ones((modes,), dtype=np.float32)
        priors = np.ones((modes,), dtype=np.float32) / modes
        counts = np.zeros((modes,), dtype=np.int64)
        counts[:num_samples] = 1
        return centers, variances, priors, counts

    kmeans = MiniBatchKMeans(
        n_clusters=modes,
        random_state=seed,
        batch_size=min(8192, max(1024, num_samples)),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01)
    labels = kmeans.fit_predict(x)
    centers = normalize_np(kmeans.cluster_centers_.astype(np.float32))

    cosine = np.matmul(x, centers.T)
    dist = 1.0 - cosine
    nearest = np.argmin(dist, axis=1)

    variances = np.zeros((modes,), dtype=np.float32)
    counts = np.zeros((modes,), dtype=np.int64)
    for mode_idx in range(modes):
        mask = nearest == mode_idx
        counts[mode_idx] = int(mask.sum())
        if counts[mode_idx] > 0:
            variances[mode_idx] = float(dist[mask, mode_idx].mean())
        else:
            variances[mode_idx] = float(dist.min(axis=1).mean())

    priors = counts.astype(np.float32)
    if priors.sum() > 0:
        priors = priors / priors.sum()
    else:
        priors = np.ones((modes,), dtype=np.float32) / modes

    variances = np.maximum(variances, 0.02).astype(np.float32)
    return centers, variances, priors.astype(np.float32), counts


def build_bank(class_features, class_counts, class_ids, class_names, cfg, args, processed):
    centers_all = []
    variances_all = []
    priors_all = []
    counts_all = []
    kept_class_ids = []

    for class_id in class_ids:
        if class_counts[class_id] == 0:
            print('Skip class {} ({}) because no features were collected.'.format(
                class_id, class_names[class_id]), flush=True)
            continue

        centers, variances, priors, counts = fit_prototypes_for_class(
            class_features[class_id], args.modes, args.seed + class_id)
        centers_all.append(torch.from_numpy(centers))
        variances_all.append(torch.from_numpy(variances))
        priors_all.append(torch.from_numpy(priors))
        counts_all.append(torch.from_numpy(counts))
        kept_class_ids.append(class_id)
        print('Class {:02d} {:>15s}: {:6d} features'.format(
            class_id, class_names[class_id], class_counts[class_id]), flush=True)

    if not centers_all:
        raise RuntimeError('No class features were collected.')

    bank = dict(
        centers=torch.stack(centers_all, dim=0),
        variances=torch.stack(variances_all, dim=0),
        priors=torch.stack(priors_all, dim=0),
        counts=torch.stack(counts_all, dim=0),
        class_ids=torch.tensor(kept_class_ids, dtype=torch.long),
        metadata=dict(
            source_config=args.config,
            checkpoint=args.checkpoint,
            split=args.split,
            processed_samples=processed,
            modes=args.modes,
            per_sample_class=args.per_sample_class,
            max_features_per_class=args.max_features_per_class,
            class_names=class_names,
            dataset_type=cfg.data[args.split].type,
            depthmodel=cfg.data[args.split].get('depthmodel', None),
            temporal=list(cfg.data[args.split].get('temporal', [])),
            labels_tag=cfg.data[args.split].get('labels_tag', None),
        ))
    return bank


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg, data_cfg = prepare_cfg(args)

    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    class_names = list(model.module.pts_bbox_head.class_names)
    class_ids = list(range(1, len(class_names)))
    class_features, class_counts, processed = collect_features(
        model, data_loader, class_ids, args)

    bank = build_bank(class_features, class_counts, class_ids, class_names, cfg, args, processed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(bank, args.out)
    print('Saved UMPOF bank to {}'.format(args.out), flush=True)


if __name__ == '__main__':
    main()
