import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.proood.prooodmodule.umpof_bank import UMPOFScorer
from projects.mmdet3d_plugin.proood.utils.ssc_metric import SSCMetrics


def parse_args():
    parser = argparse.ArgumentParser(description='Fast single-GPU OOD evaluation.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument(
        '--score-mode',
        default='echoood',
        choices=[
            'echoood',
            'umpof', 'max', 'mean',
            'umpof_predcls', 'max_predcls', 'mean_predcls',
            'gated_max_conf', 'gated_max_margin',
            'tadc_reliable_max', 'tadc_tail_max',
            'tadc_v2_reliable_max', 'tadc_v2_tail_max',
            'tadc_v2_reliable_predcls', 'tadc_v2_tail_predcls',
            'tadc_quantile_max', 'tadc_quantile_tail',
            'tadc_capacity_select', 'tadc_capacity_mean',
            'tadc_capacity_max',
            'tadc_spatial_avg', 'tadc_spatial_maxpool',
            'echoood_spatial_avg',
            'tadc_spatial_echo_fusion',
            'echoood_spatial_avg_valid',
            'echoood_spatial_support',
            'tadc_spatial_support_fusion',
            'tadc_spatial_echo_fusion_valid',
            'echoood_spatial_peak',
            'tadc_spatial_peak_fusion',
            'echoood_ppsc_v2',
            'ppsc_v2_fusion',
            'echoood_ppsc_v3',
            'ppsc_v3_fusion',
            'echoood_ppsc_v4',
            'echoood_ppsc_v5',
        ],
        help='OOD score source. max/mean fuse EchoOOD and UMPOF scores.')
    parser.add_argument('--umpof-bank', default=None)
    parser.add_argument('--umpof-tail-bank', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument(
        '--include-sequences',
        nargs='+',
        default=None,
        help='Evaluate only these dataset sequence IDs.')
    parser.add_argument(
        '--metric-max-points',
        type=int,
        default=None,
        help='Optional per-run subsampling for quick screening; omit for exact metrics.')
    parser.add_argument(
        '--metric-histogram-bins',
        type=int,
        default=None,
        help='Use bounded-memory deterministic histogram metrics with this many bins.')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--progress-interval', type=int, default=10)
    parser.add_argument('--chunk-size', type=int, default=262144)
    parser.add_argument('--umpof-tau', type=float, default=0.07)
    parser.add_argument('--gate-lambda', type=float, default=1.0)
    parser.add_argument('--gate-power', type=float, default=1.0)
    parser.add_argument('--tadc-tail-boost', type=float, default=0.25)
    parser.add_argument('--tadc-min-gate', type=float, default=0.0)
    parser.add_argument('--tadc-max-gate', type=float, default=1.0)
    parser.add_argument('--tadc-quantile-low', type=float, default=0.05)
    parser.add_argument('--tadc-quantile-high', type=float, default=0.95)
    parser.add_argument('--tadc-quantile-min-voxels', type=int, default=64)
    parser.add_argument('--tadc-spatial-kernel', type=int, default=3)
    parser.add_argument('--tadc-spatial-alpha', type=float, default=0.5)
    parser.add_argument('--tadc-peak-gamma', type=float, default=1.0)
    parser.add_argument('--ppsc-v2-boost', type=float, default=0.75)
    parser.add_argument('--ppsc-v2-suppress', type=float, default=0.75)
    parser.add_argument(
        '--ppsc-v2-occupancy',
        choices=['all', 'predicted'],
        default='predicted',
        help='Neighborhood support used by PPSC v2.')
    parser.add_argument('--ppsc-v3-min-support', type=float, default=0.10)
    parser.add_argument('--ppsc-v3-support-scale', type=float, default=0.40)
    parser.add_argument('--ppsc-v3-support-power', type=float, default=1.0)
    parser.add_argument('--ppsc-v4-min-support', type=float, default=0.10)
    parser.add_argument('--ppsc-v4-support-scale', type=float, default=0.40)
    parser.add_argument('--ppsc-v4-support-power', type=float, default=1.0)
    parser.add_argument('--ppsc-v5-min-support', type=float, default=0.10)
    parser.add_argument('--ppsc-v5-support-scale', type=float, default=0.40)
    parser.add_argument('--ppsc-v5-support-power', type=float, default=1.0)
    parser.add_argument(
        '--ood-dilation-radius',
        type=int,
        default=6,
        help='OOD label dilation radius in voxels (4/5/6 = 0.8/1.0/1.2 m).')
    parser.add_argument(
        '--ood-dilation-radii',
        nargs='+',
        type=int,
        default=None,
        help='Compute several dilation radii from one model forward pass.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


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
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    maybe_import_plugins(cfg)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    needs_umpof = args.score_mode in (
        'umpof', 'max', 'mean',
        'umpof_predcls', 'max_predcls', 'mean_predcls',
        'gated_max_conf', 'gated_max_margin',
        'tadc_reliable_max', 'tadc_tail_max',
        'tadc_v2_reliable_max', 'tadc_v2_tail_max',
        'tadc_v2_reliable_predcls', 'tadc_v2_tail_predcls',
        'tadc_quantile_max', 'tadc_quantile_tail',
        'tadc_capacity_select', 'tadc_capacity_mean',
        'tadc_capacity_max',
        'tadc_spatial_avg', 'tadc_spatial_maxpool',
        'tadc_spatial_echo_fusion',
        'tadc_spatial_support_fusion',
        'tadc_spatial_echo_fusion_valid',
        'tadc_spatial_peak_fusion',
        'ppsc_v2_fusion',
        'ppsc_v3_fusion')
    needs_echoood = args.score_mode in (
        'echoood', 'max', 'mean', 'max_predcls', 'mean_predcls',
        'gated_max_conf', 'gated_max_margin',
        'tadc_reliable_max', 'tadc_tail_max',
        'tadc_v2_reliable_max', 'tadc_v2_tail_max',
        'tadc_v2_reliable_predcls', 'tadc_v2_tail_predcls',
        'tadc_quantile_max', 'tadc_quantile_tail',
        'tadc_capacity_select', 'tadc_capacity_mean',
        'tadc_capacity_max',
        'tadc_spatial_avg', 'tadc_spatial_maxpool',
        'echoood_spatial_avg', 'tadc_spatial_echo_fusion',
        'echoood_spatial_avg_valid', 'echoood_spatial_support',
        'tadc_spatial_support_fusion',
        'tadc_spatial_echo_fusion_valid',
        'echoood_spatial_peak', 'tadc_spatial_peak_fusion',
        'echoood_ppsc_v2', 'ppsc_v2_fusion',
        'echoood_ppsc_v3', 'ppsc_v3_fusion', 'echoood_ppsc_v4',
        'echoood_ppsc_v5')
    cfg.model.pts_bbox_head.return_umpof_features = needs_umpof
    cfg.model.pts_bbox_head.ood_flag = needs_echoood

    if args.workers is not None:
        cfg.data.workers_per_gpu = args.workers

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        raise TypeError('Only dict cfg.data.test is supported by test_ood_fast.py.')

    return cfg, samples_per_gpu


def calculate_auroc(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def calculate_aupr_fast(scores, labels, added_label_count):
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order].astype(np.uint8)

    tp = np.cumsum(sorted_labels == 1, dtype=np.float64)
    fp = np.cumsum(sorted_labels == 0, dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)

    num_positives = float(np.sum(labels == 1) - int(added_label_count))
    if num_positives <= 0:
        return float('nan')
    recall = tp / num_positives
    return auc(recall, precision)


def calculate_standard_aupr(scores, labels):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


class HistogramOODMetrics:
    def __init__(self, num_bins):
        if num_bins < 256:
            raise ValueError('--metric-histogram-bins must be at least 256.')
        self.num_bins = num_bins
        self.original_pos = np.zeros(num_bins, dtype=np.int64)
        self.original_neg = np.zeros(num_bins, dtype=np.int64)
        self.dilated_pos = np.zeros(num_bins, dtype=np.int64)
        self.dilated_neg = np.zeros(num_bins, dtype=np.int64)
        self.added_label_count = 0

    def _update_pair(self, score, labels, pos_hist, neg_hist):
        valid = (labels == 0) | (labels == 1)
        valid_scores = score[valid].detach().float().cpu().numpy()
        valid_labels = labels[valid].detach().cpu().numpy()
        valid_scores = np.ascontiguousarray(
            np.clip(valid_scores, 0.0, 1.0), dtype=np.float32)
        score_bits = valid_scores.view(np.uint32).astype(np.uint64)
        max_score_bits = np.uint64(np.float32(1.0).view(np.uint32))
        indices = (
            score_bits * np.uint64(self.num_bins - 1) // max_score_bits
        ).astype(np.int32)
        pos_hist += np.bincount(
            indices[valid_labels == 1], minlength=self.num_bins)
        neg_hist += np.bincount(
            indices[valid_labels == 0], minlength=self.num_bins)

    def update(self, score, labels_original, labels_dilated, added_count):
        self._update_pair(
            score, labels_original, self.original_pos, self.original_neg)
        self._update_pair(
            score, labels_dilated, self.dilated_pos, self.dilated_neg)
        self.added_label_count += int(added_count)

    @staticmethod
    def _descending_counts(pos_hist, neg_hist):
        tp = np.cumsum(pos_hist[::-1], dtype=np.float64)
        fp = np.cumsum(neg_hist[::-1], dtype=np.float64)
        populated = (pos_hist[::-1] + neg_hist[::-1]) > 0
        return tp[populated], fp[populated]

    @classmethod
    def _auroc(cls, pos_hist, neg_hist):
        tp, fp = cls._descending_counts(pos_hist, neg_hist)
        total_pos = float(pos_hist.sum())
        total_neg = float(neg_hist.sum())
        if total_pos <= 0 or total_neg <= 0:
            return float('nan')
        tpr = np.concatenate(([0.0], tp / total_pos))
        fpr = np.concatenate(([0.0], fp / total_neg))
        return float(np.trapz(tpr, fpr))

    @classmethod
    def _standard_aupr(cls, pos_hist, neg_hist):
        tp, fp = cls._descending_counts(pos_hist, neg_hist)
        total_pos = float(pos_hist.sum())
        if total_pos <= 0:
            return float('nan')
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / total_pos
        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))
        return float(np.trapz(precision, recall))

    @classmethod
    def _legacy_aupr(cls, pos_hist, neg_hist, added_label_count):
        tp, fp = cls._descending_counts(pos_hist, neg_hist)
        num_positives = float(pos_hist.sum() - int(added_label_count))
        if num_positives <= 0 or tp.size < 2:
            return float('nan')
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / num_positives
        return float(np.trapz(precision, recall))

    def compute(self):
        return {
            'auroc': self._auroc(self.original_pos, self.original_neg),
            'aupr_original': self._standard_aupr(
                self.original_pos, self.original_neg),
            'aupr_dilated_standard': self._standard_aupr(
                self.dilated_pos, self.dilated_neg),
            'aupr_area': self._legacy_aupr(
                self.dilated_pos, self.dilated_neg, self.added_label_count),
        }


def sample_metric_points(scores, labels, max_points, rng):
    if max_points is None or scores.shape[0] <= max_points:
        return scores, labels

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    pos_keep = min(pos_idx.shape[0], max(1, max_points // 2))
    neg_keep = min(neg_idx.shape[0], max_points - pos_keep)

    sampled = []
    if pos_keep > 0:
        sampled.append(rng.choice(pos_idx, size=pos_keep, replace=False))
    if neg_keep > 0:
        sampled.append(rng.choice(neg_idx, size=neg_keep, replace=False))
    if not sampled:
        return scores[:0], labels[:0]

    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return scores[indices], labels[indices]


def build_ood_labels(target, dilation_radius=6):
    target = target.clone()
    target_original = target.clone()

    label_original = torch.full_like(target_original, 255, dtype=torch.uint8)
    label_original[target_original < 20] = 0
    label_original[(target_original >= 20) & (target_original < 255)] = 1

    label_20_mask = target == 20
    kernel_size = 2 * dilation_radius + 1
    dilation_kernel = torch.ones(
        (1, 1, kernel_size, kernel_size, kernel_size),
        device=target.device,
        dtype=torch.float32)
    expanded_label_20 = F.conv3d(
        label_20_mask.float().unsqueeze(1),
        dilation_kernel,
        padding=dilation_radius).squeeze(1) > 0
    valid_region = (target == 0) | (target == 255)
    expanded_label_20 = expanded_label_20 & valid_region

    original_count = torch.sum(label_20_mask)
    new_count = torch.sum(expanded_label_20)
    added_count = int((new_count - original_count).item())

    target[expanded_label_20] = 20
    label_dilated = torch.full_like(target, 255, dtype=torch.uint8)
    label_dilated[target < 20] = 0
    label_dilated[(target >= 20) & (target < 255)] = 1
    return label_dilated, label_original, added_count


def extract_valid_arrays(score, labels, max_points, rng):
    valid = (labels == 0) | (labels == 1)
    score_np = score[valid].detach().float().cpu().numpy().astype(np.float32)
    label_np = labels[valid].detach().cpu().numpy().astype(np.uint8)
    return sample_metric_points(score_np, label_np, max_points, rng)


def robust_normalize(values, low_quantile, high_quantile):
    low = torch.quantile(values.float(), low_quantile)
    high = torch.quantile(values.float(), high_quantile)
    if high <= low:
        return torch.zeros_like(values)
    return torch.clamp((values - low) / (high - low), min=0.0, max=1.0)


def classwise_quantile_calibrate(score, pred_labels, low_quantile,
                                 high_quantile, min_voxels):
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError('TADC quantiles must satisfy 0 <= low < high <= 1.')
    if score.shape != pred_labels.shape:
        raise ValueError('Score and predicted labels must have the same shape.')

    calibrated = torch.zeros_like(score)
    for batch_idx in range(score.shape[0]):
        occupied = pred_labels[batch_idx] > 0
        if not occupied.any():
            continue

        occupied_values = score[batch_idx][occupied]
        calibrated[batch_idx][occupied] = robust_normalize(
            occupied_values, low_quantile, high_quantile)

        class_ids = torch.unique(pred_labels[batch_idx][occupied])
        for class_id in class_ids:
            class_mask = pred_labels[batch_idx] == class_id
            if int(class_mask.sum().item()) < min_voxels:
                continue
            class_values = score[batch_idx][class_mask]
            calibrated[batch_idx][class_mask] = robust_normalize(
                class_values, low_quantile, high_quantile)

    return calibrated


def spatial_calibrate(score, kernel_size, alpha, mode,
                      count_include_pad=True):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('TADC spatial kernel must be a positive odd integer.')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('TADC spatial alpha must be in [0, 1].')

    score_5d = score.unsqueeze(1)
    padding = kernel_size // 2
    if mode == 'avg':
        spatial = F.avg_pool3d(
            score_5d, kernel_size=kernel_size, stride=1,
            padding=padding,
            count_include_pad=count_include_pad).squeeze(1)
    elif mode == 'maxpool':
        spatial = F.max_pool3d(
            score_5d, kernel_size=kernel_size, stride=1,
            padding=padding).squeeze(1)
    else:
        raise ValueError('Unsupported TADC spatial mode: {}'.format(mode))

    return torch.clamp((1.0 - alpha) * score + alpha * spatial, 0.0, 1.0)


def spatial_support_calibrate(score, kernel_size, alpha):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('TADC spatial kernel must be a positive odd integer.')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('TADC spatial alpha must be in [0, 1].')

    score_5d = score.unsqueeze(1)
    padding = kernel_size // 2
    local_avg = F.avg_pool3d(
        score_5d, kernel_size=kernel_size, stride=1,
        padding=padding, count_include_pad=False).squeeze(1)
    local_max = F.max_pool3d(
        score_5d, kernel_size=kernel_size, stride=1,
        padding=padding).squeeze(1)
    support = local_avg / torch.clamp(local_max, min=1e-6)
    support = torch.clamp(support, min=0.0, max=1.0)
    calibrated = score * ((1.0 - alpha) + alpha * support)
    return torch.clamp(calibrated, min=0.0, max=1.0)


def peak_preserving_spatial_calibrate(score, kernel_size, alpha, gamma):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('TADC spatial kernel must be a positive odd integer.')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('TADC spatial alpha must be in [0, 1].')
    if gamma <= 0.0:
        raise ValueError('TADC peak gamma must be positive.')

    score_5d = score.unsqueeze(1)
    padding = kernel_size // 2
    local_avg = F.avg_pool3d(
        score_5d, kernel_size=kernel_size, stride=1,
        padding=padding, count_include_pad=False).squeeze(1)
    local_max = F.max_pool3d(
        score_5d, kernel_size=kernel_size, stride=1,
        padding=padding).squeeze(1)

    spatial = (1.0 - alpha) * score + alpha * local_avg
    peakness = score / torch.clamp(local_max, min=1e-6)
    peakness = torch.pow(torch.clamp(peakness, 0.0, 1.0), gamma)
    calibrated = peakness * score + (1.0 - peakness) * spatial
    return torch.clamp(calibrated, min=0.0, max=1.0)


def occupancy_aware_peak_calibrate(score, logits, kernel_size, boost,
                                   suppress, gamma, occupancy_mode):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('PPSC v2 spatial kernel must be a positive odd integer.')
    if not 0.0 <= boost <= 1.0:
        raise ValueError('PPSC v2 boost must be in [0, 1].')
    if not 0.0 <= suppress <= 1.0:
        raise ValueError('PPSC v2 suppress must be in [0, 1].')
    if gamma <= 0.0:
        raise ValueError('PPSC v2 peak gamma must be positive.')

    score_5d = score.unsqueeze(1)
    if occupancy_mode == 'predicted':
        occupied = torch.argmax(logits, dim=1, keepdim=True) > 0
    elif occupancy_mode == 'all':
        occupied = torch.ones_like(score_5d, dtype=torch.bool)
    else:
        raise ValueError('Unsupported PPSC v2 occupancy mode: {}'.format(
            occupancy_mode))

    padding = kernel_size // 2
    occupied_float = occupied.to(score.dtype)
    local_sum = F.avg_pool3d(
        score_5d * occupied_float, kernel_size=kernel_size, stride=1,
        padding=padding, count_include_pad=True)
    local_count = F.avg_pool3d(
        occupied_float, kernel_size=kernel_size, stride=1,
        padding=padding, count_include_pad=True)
    has_support = local_count > 0
    local_avg = torch.where(
        has_support,
        local_sum / torch.clamp(local_count, min=1e-6),
        score_5d)

    masked_score = torch.where(
        occupied, score_5d, torch.full_like(score_5d, -1.0))
    local_max = F.max_pool3d(
        masked_score, kernel_size=kernel_size, stride=1,
        padding=padding)
    local_max = torch.where(has_support, local_max, score_5d)

    delta_up = torch.clamp(local_avg - score_5d, min=0.0)
    delta_down = torch.clamp(score_5d - local_avg, min=0.0)
    peakness = score_5d / torch.clamp(local_max, min=1e-6)
    peakness = torch.pow(torch.clamp(peakness, 0.0, 1.0), gamma)
    non_peak = 1.0 - peakness
    calibrated = score_5d + non_peak * (
        boost * delta_up - suppress * delta_down)
    return torch.clamp(calibrated.squeeze(1), min=0.0, max=1.0)


def soft_occupancy_reliable_calibrate(score, logits, kernel_size, alpha,
                                      gamma, boost, suppress, min_support,
                                      support_scale, support_power):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('PPSC v3 spatial kernel must be a positive odd integer.')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('PPSC v3 alpha must be in [0, 1].')
    if gamma <= 0.0 or support_power <= 0.0:
        raise ValueError('PPSC v3 powers must be positive.')
    if not 0.0 <= min_support <= 1.0:
        raise ValueError('PPSC v3 minimum support must be in [0, 1].')
    if support_scale <= 0.0:
        raise ValueError('PPSC v3 support scale must be positive.')

    score_5d = score.unsqueeze(1)
    padding = kernel_size // 2
    occupied_probability = 1.0 - torch.softmax(logits, dim=1)[:, :1]

    # The v1 score is the conservative fallback. Soft support only controls
    # how far the occupancy-aware correction is allowed to move it.
    local_avg_v1 = F.avg_pool3d(
        score_5d, kernel_size=kernel_size, stride=1, padding=padding,
        count_include_pad=False)
    local_max = F.max_pool3d(
        score_5d, kernel_size=kernel_size, stride=1, padding=padding)
    peakness = score_5d / torch.clamp(local_max, min=1e-6)
    peakness = torch.pow(torch.clamp(peakness, 0.0, 1.0), gamma)
    v1 = peakness * score_5d + (1.0 - peakness) * (
        (1.0 - alpha) * score_5d + alpha * local_avg_v1)

    support = F.avg_pool3d(
        occupied_probability, kernel_size=kernel_size, stride=1,
        padding=padding, count_include_pad=False)
    weighted_avg = F.avg_pool3d(
        score_5d * occupied_probability, kernel_size=kernel_size, stride=1,
        padding=padding, count_include_pad=False)
    weighted_avg = weighted_avg / torch.clamp(support, min=1e-6)
    weighted_avg = torch.where(support > 1e-6, weighted_avg, local_avg_v1)

    delta_up = torch.clamp(weighted_avg - score_5d, min=0.0)
    delta_down = torch.clamp(score_5d - weighted_avg, min=0.0)
    v2 = score_5d + (1.0 - peakness) * (
        boost * delta_up - suppress * delta_down)

    reliability = torch.clamp(
        (support - min_support) / support_scale, min=0.0, max=1.0)
    reliability = torch.pow(reliability, support_power).squeeze(1)
    calibrated = (1.0 - reliability) * v1 + reliability * v2
    return torch.clamp(calibrated.squeeze(1), min=0.0, max=1.0)


def hard_occupancy_reliable_calibrate(score, logits, kernel_size, alpha,
                                      gamma, boost, suppress, min_support,
                                      support_scale, support_power):
    v1 = peak_preserving_spatial_calibrate(
        score, kernel_size, alpha, gamma)
    v2 = occupancy_aware_peak_calibrate(
        score, logits, kernel_size, boost, suppress, gamma, 'predicted')
    occupied_probability = 1.0 - torch.softmax(logits, dim=1)[:, :1]
    padding = kernel_size // 2
    support = F.avg_pool3d(
        occupied_probability, kernel_size=kernel_size, stride=1,
        padding=padding, count_include_pad=False)
    reliability = torch.clamp(
        (support - min_support) / support_scale, min=0.0, max=1.0)
    reliability = torch.pow(reliability, support_power).squeeze(1)
    calibrated = v1 + reliability * (v2 - v1)
    return torch.clamp(calibrated, min=0.0, max=1.0)


def echo_fallback_occupancy_calibrate(score, logits, kernel_size, gamma,
                                      boost, suppress, min_support,
                                      support_scale, support_power):
    if not 0.0 <= min_support <= 1.0:
        raise ValueError('PPSC v5 minimum support must be in [0, 1].')
    if support_scale <= 0.0 or support_power <= 0.0:
        raise ValueError('PPSC v5 support scale and power must be positive.')

    calibrated = occupancy_aware_peak_calibrate(
        score, logits, kernel_size, boost, suppress, gamma, 'predicted')
    occupied_probability = 1.0 - torch.softmax(logits, dim=1)[:, :1]
    padding = kernel_size // 2
    support = F.avg_pool3d(
        occupied_probability, kernel_size=kernel_size, stride=1,
        padding=padding, count_include_pad=False)
    reliability = torch.clamp(
        (support - min_support) / support_scale, min=0.0, max=1.0)
    reliability = torch.pow(reliability, support_power).squeeze(1)

    # Preserve EchoOOD ordering where occupancy support is unreliable.
    output = score + reliability * (calibrated - score)
    return torch.clamp(output, min=0.0, max=1.0)


def compute_score(result, scorer, args, tail_scorer=None):
    echo_score = result.get('ood_pred', None)
    umpof_score = None

    if args.score_mode in (
            'umpof', 'max', 'mean',
            'umpof_predcls', 'max_predcls', 'mean_predcls',
            'gated_max_conf', 'gated_max_margin',
            'tadc_reliable_max', 'tadc_tail_max',
            'tadc_v2_reliable_max', 'tadc_v2_tail_max',
            'tadc_v2_reliable_predcls', 'tadc_v2_tail_predcls',
            'tadc_quantile_max', 'tadc_quantile_tail',
            'tadc_capacity_select', 'tadc_capacity_mean',
            'tadc_capacity_max',
            'tadc_spatial_avg', 'tadc_spatial_maxpool',
            'tadc_spatial_echo_fusion',
            'tadc_spatial_support_fusion',
            'tadc_spatial_echo_fusion_valid',
            'tadc_spatial_peak_fusion',
            'ppsc_v2_fusion',
            'ppsc_v3_fusion'):
        class_conditional = args.score_mode.endswith('_predcls')
        umpof_score = scorer.score(
            result['vox_feats_full'],
            logits=result['output_voxels'],
            chunk_size=args.chunk_size,
            class_conditional=class_conditional)

    if args.score_mode in (
            'tadc_capacity_select', 'tadc_capacity_mean',
            'tadc_capacity_max'):
        if tail_scorer is None:
            raise RuntimeError('TADC capacity modes require --umpof-tail-bank.')
        tail_umpof_score = tail_scorer.score(
            result['vox_feats_full'],
            logits=result['output_voxels'],
            chunk_size=args.chunk_size,
            class_conditional=False)
        pred_labels = torch.argmax(result['output_voxels'], dim=1)
        tail_mask = scorer.tail_mask(pred_labels).to(umpof_score.device)

        if args.score_mode == 'tadc_capacity_select':
            adaptive_score = torch.where(
                tail_mask, tail_umpof_score, umpof_score)
        elif args.score_mode == 'tadc_capacity_mean':
            tail_score = 0.5 * (umpof_score + tail_umpof_score)
            adaptive_score = torch.where(tail_mask, tail_score, umpof_score)
        else:
            tail_score = torch.max(umpof_score, tail_umpof_score)
            adaptive_score = torch.where(tail_mask, tail_score, umpof_score)

        if echo_score is None:
            raise RuntimeError('TADC capacity fusion requires EchoOOD score.')
        return torch.max(echo_score, adaptive_score)

    if args.score_mode == 'echoood':
        if echo_score is None:
            raise RuntimeError('EchoOOD score requested, but result has no ood_pred.')
        return echo_score
    if args.score_mode == 'echoood_spatial_avg':
        if echo_score is None:
            raise RuntimeError('Spatial EchoOOD score requested, but result has no ood_pred.')
        return spatial_calibrate(
            echo_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            'avg')
    if args.score_mode == 'echoood_spatial_avg_valid':
        if echo_score is None:
            raise RuntimeError('Spatial EchoOOD score requested, but result has no ood_pred.')
        return spatial_calibrate(
            echo_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            'avg',
            count_include_pad=False)
    if args.score_mode == 'echoood_spatial_support':
        if echo_score is None:
            raise RuntimeError('Spatial support score requested, but result has no ood_pred.')
        return spatial_support_calibrate(
            echo_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha)
    if args.score_mode == 'echoood_spatial_peak':
        if echo_score is None:
            raise RuntimeError('Peak-preserving score requested, but result has no ood_pred.')
        return peak_preserving_spatial_calibrate(
            echo_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            args.tadc_peak_gamma)
    if args.score_mode == 'echoood_ppsc_v2':
        if echo_score is None:
            raise RuntimeError('PPSC v2 score requested, but result has no ood_pred.')
        return occupancy_aware_peak_calibrate(
            echo_score,
            result['output_voxels'],
            args.tadc_spatial_kernel,
            args.ppsc_v2_boost,
            args.ppsc_v2_suppress,
            args.tadc_peak_gamma,
            args.ppsc_v2_occupancy)
    if args.score_mode in ('umpof', 'umpof_predcls'):
        return umpof_score
    if args.score_mode in ('max', 'max_predcls'):
        if echo_score is None:
            raise RuntimeError('Max fusion requested, but result has no EchoOOD score.')
        return torch.max(echo_score, umpof_score)
    if args.score_mode in ('tadc_spatial_avg', 'tadc_spatial_maxpool'):
        if echo_score is None:
            raise RuntimeError('TADC spatial fusion requires EchoOOD score.')
        base_score = torch.max(echo_score, umpof_score)
        spatial_mode = 'avg' if args.score_mode == 'tadc_spatial_avg' else 'maxpool'
        return spatial_calibrate(
            base_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            spatial_mode)
    if args.score_mode == 'tadc_spatial_echo_fusion':
        if echo_score is None:
            raise RuntimeError('TADC spatial fusion requires EchoOOD score.')
        spatial_echo = spatial_calibrate(
            echo_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            'avg')
        delta = torch.clamp(umpof_score - spatial_echo, min=0.0)
        fused = spatial_echo + args.gate_lambda * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    if args.score_mode == 'tadc_spatial_echo_fusion_valid':
        if echo_score is None:
            raise RuntimeError('TADC spatial fusion requires EchoOOD score.')
        spatial_echo = spatial_calibrate(
            echo_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            'avg',
            count_include_pad=False)
        delta = torch.clamp(umpof_score - spatial_echo, min=0.0)
        fused = spatial_echo + args.gate_lambda * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    if args.score_mode == 'tadc_spatial_support_fusion':
        if echo_score is None:
            raise RuntimeError('TADC spatial support fusion requires EchoOOD score.')
        supported_echo = spatial_support_calibrate(
            echo_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha)
        delta = torch.clamp(umpof_score - supported_echo, min=0.0)
        fused = supported_echo + args.gate_lambda * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    if args.score_mode == 'tadc_spatial_peak_fusion':
        if echo_score is None:
            raise RuntimeError('TADC peak-preserving fusion requires EchoOOD score.')
        peak_echo = peak_preserving_spatial_calibrate(
            echo_score,
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            args.tadc_peak_gamma)
        delta = torch.clamp(umpof_score - peak_echo, min=0.0)
        fused = peak_echo + args.gate_lambda * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    if args.score_mode == 'ppsc_v2_fusion':
        if echo_score is None:
            raise RuntimeError('PPSC v2 fusion requires EchoOOD score.')
        calibrated_echo = occupancy_aware_peak_calibrate(
            echo_score,
            result['output_voxels'],
            args.tadc_spatial_kernel,
            args.ppsc_v2_boost,
            args.ppsc_v2_suppress,
            args.tadc_peak_gamma,
            args.ppsc_v2_occupancy)
        delta = torch.clamp(umpof_score - calibrated_echo, min=0.0)
        fused = calibrated_echo + args.gate_lambda * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    if args.score_mode == 'echoood_ppsc_v3':
        if echo_score is None:
            raise RuntimeError('PPSC v3 score requested, but result has no ood_pred.')
        return soft_occupancy_reliable_calibrate(
            echo_score,
            result['output_voxels'],
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            args.tadc_peak_gamma,
            args.ppsc_v2_boost,
            args.ppsc_v2_suppress,
            args.ppsc_v3_min_support,
            args.ppsc_v3_support_scale,
            args.ppsc_v3_support_power)
    if args.score_mode == 'echoood_ppsc_v4':
        if echo_score is None:
            raise RuntimeError('PPSC v4 score requested, but result has no ood_pred.')
        return hard_occupancy_reliable_calibrate(
            echo_score,
            result['output_voxels'],
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            args.tadc_peak_gamma,
            args.ppsc_v2_boost,
            args.ppsc_v2_suppress,
            args.ppsc_v4_min_support,
            args.ppsc_v4_support_scale,
            args.ppsc_v4_support_power)
    if args.score_mode == 'echoood_ppsc_v5':
        if echo_score is None:
            raise RuntimeError('PPSC v5 score requested, but result has no ood_pred.')
        return echo_fallback_occupancy_calibrate(
            echo_score,
            result['output_voxels'],
            args.tadc_spatial_kernel,
            args.tadc_peak_gamma,
            args.ppsc_v2_boost,
            args.ppsc_v2_suppress,
            args.ppsc_v5_min_support,
            args.ppsc_v5_support_scale,
            args.ppsc_v5_support_power)
    if args.score_mode == 'ppsc_v3_fusion':
        if echo_score is None:
            raise RuntimeError('PPSC v3 fusion requires EchoOOD score.')
        calibrated_echo = soft_occupancy_reliable_calibrate(
            echo_score,
            result['output_voxels'],
            args.tadc_spatial_kernel,
            args.tadc_spatial_alpha,
            args.tadc_peak_gamma,
            args.ppsc_v2_boost,
            args.ppsc_v2_suppress,
            args.ppsc_v3_min_support,
            args.ppsc_v3_support_scale,
            args.ppsc_v3_support_power)
        delta = torch.clamp(umpof_score - calibrated_echo, min=0.0)
        fused = calibrated_echo + args.gate_lambda * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    if args.score_mode in ('mean', 'mean_predcls'):
        if echo_score is None:
            raise RuntimeError('Mean fusion requested, but result has no EchoOOD score.')
        return 0.5 * (echo_score + umpof_score)
    if args.score_mode in ('gated_max_conf', 'gated_max_margin'):
        if echo_score is None:
            raise RuntimeError('Gated fusion requested, but result has no EchoOOD score.')
        gate = compute_uncertainty_gate(result['output_voxels'], args.score_mode)
        gate = torch.clamp(gate, min=0.0, max=1.0)
        if args.gate_power != 1.0:
            gate = torch.pow(gate, args.gate_power)
        delta = torch.clamp(umpof_score - echo_score, min=0.0)
        fused = echo_score + args.gate_lambda * gate * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    if args.score_mode in ('tadc_quantile_max', 'tadc_quantile_tail'):
        if echo_score is None:
            raise RuntimeError('TADC quantile fusion requires EchoOOD score.')
        pred_labels = torch.argmax(result['output_voxels'], dim=1)
        calibrated = classwise_quantile_calibrate(
            umpof_score,
            pred_labels,
            args.tadc_quantile_low,
            args.tadc_quantile_high,
            args.tadc_quantile_min_voxels)
        if args.score_mode == 'tadc_quantile_max':
            return torch.max(echo_score, calibrated)

        gate = scorer.tadc_gate(
            pred_labels,
            mode='tail',
            tail_boost=args.tadc_tail_boost,
            min_gate=args.tadc_min_gate,
            max_gate=args.tadc_max_gate).to(echo_score.device)
        if args.gate_power != 1.0:
            gate = torch.pow(gate, args.gate_power)
        delta = torch.clamp(calibrated - echo_score, min=0.0)
        fused = echo_score + args.gate_lambda * gate * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    if args.score_mode in (
            'tadc_reliable_max', 'tadc_tail_max',
            'tadc_v2_reliable_max', 'tadc_v2_tail_max',
            'tadc_v2_reliable_predcls', 'tadc_v2_tail_predcls'):
        if echo_score is None:
            raise RuntimeError('TADC fusion requested, but result has no EchoOOD score.')
        pred_labels = torch.argmax(result['output_voxels'], dim=1)
        tadc_mode = 'tail' if 'tail' in args.score_mode else 'reliable'
        gate = scorer.tadc_gate(
            pred_labels,
            mode=tadc_mode,
            tail_boost=args.tadc_tail_boost,
            min_gate=args.tadc_min_gate,
            max_gate=args.tadc_max_gate).to(echo_score.device)
        if args.gate_power != 1.0:
            gate = torch.pow(gate, args.gate_power)
        delta = torch.clamp(umpof_score - echo_score, min=0.0)
        fused = echo_score + args.gate_lambda * gate * delta
        return torch.clamp(fused, min=0.0, max=1.0)
    raise ValueError('Unsupported score mode: {}'.format(args.score_mode))


def compute_uncertainty_gate(logits, score_mode):
    probs = F.softmax(logits, dim=1)
    if score_mode == 'gated_max_conf':
        top1 = torch.max(probs, dim=1)[0]
        return 1.0 - top1

    if score_mode == 'gated_max_margin':
        top2 = torch.topk(probs, k=2, dim=1)[0]
        margin = top2[:, 0] - top2[:, 1]
        return 1.0 - margin

    raise ValueError('Unsupported gated score mode: {}'.format(score_mode))


def evaluate_ssc(dataset, ssc_metric):
    return dataset.evaluate({'ssc_scores': ssc_metric.compute()}, metric=['bbox'])


def filter_dataset_sequences(dataset, include_sequences):
    if include_sequences is None:
        return None
    if not hasattr(dataset, 'scans'):
        raise TypeError('--include-sequences requires a dataset with a scans attribute.')

    requested = set(include_sequences)
    available = {str(scan.get('sequence')) for scan in dataset.scans}
    missing = requested - available
    if missing:
        raise ValueError(
            'Requested sequences are unavailable: {}. Available: {}'.format(
                sorted(missing), sorted(available)))

    dataset.scans = [
        scan for scan in dataset.scans
        if str(scan.get('sequence')) in requested
    ]
    if not dataset.scans:
        raise ValueError('--include-sequences selected zero samples.')
    if hasattr(dataset, 'set_group_flag'):
        dataset.set_group_flag()
    return sorted(requested)


def main():
    args = parse_args()
    dilation_radii = args.ood_dilation_radii or [args.ood_dilation_radius]
    dilation_radii = list(dict.fromkeys(dilation_radii))
    if any(radius < 0 for radius in dilation_radii):
        raise ValueError('--ood-dilation-radius must be non-negative.')
    if args.ood_dilation_radius not in dilation_radii:
        dilation_radii.insert(0, args.ood_dilation_radius)
    if args.metric_max_points is not None and args.metric_histogram_bins is not None:
        raise ValueError(
            '--metric-max-points and --metric-histogram-bins are mutually exclusive.')
    if len(dilation_radii) > 1 and args.metric_histogram_bins is None:
        raise ValueError(
            'Multiple dilation radii require --metric-histogram-bins.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_random_seed(args.seed, deterministic=args.deterministic)

    needs_umpof = args.score_mode in (
        'umpof', 'max', 'mean',
        'umpof_predcls', 'max_predcls', 'mean_predcls',
        'gated_max_conf', 'gated_max_margin',
        'tadc_reliable_max', 'tadc_tail_max',
        'tadc_v2_reliable_max', 'tadc_v2_tail_max',
        'tadc_v2_reliable_predcls', 'tadc_v2_tail_predcls',
        'tadc_quantile_max', 'tadc_quantile_tail',
        'tadc_capacity_select', 'tadc_capacity_mean',
        'tadc_capacity_max',
        'tadc_spatial_avg', 'tadc_spatial_maxpool',
        'tadc_spatial_echo_fusion',
        'tadc_spatial_support_fusion',
        'tadc_spatial_echo_fusion_valid',
        'tadc_spatial_peak_fusion',
        'ppsc_v2_fusion',
        'ppsc_v3_fusion')
    if needs_umpof and args.umpof_bank is None:
        raise ValueError('--umpof-bank is required for score mode {}'.format(args.score_mode))
    needs_tail_bank = args.score_mode in (
        'tadc_capacity_select', 'tadc_capacity_mean', 'tadc_capacity_max')
    if needs_tail_bank and args.umpof_tail_bank is None:
        raise ValueError('--umpof-tail-bank is required for score mode {}'.format(
            args.score_mode))

    cfg, samples_per_gpu = prepare_cfg(args)
    dataset = build_dataset(cfg.data.test)
    dataset_size_before_filter = len(dataset)
    evaluated_sequences = filter_dataset_sequences(
        dataset, args.include_sequences)
    eval_len = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    metric_max_per_frame = None
    histogram_metrics = None
    histogram_metrics_by_radius = None
    if args.metric_max_points is not None:
        metric_max_per_frame = max(1, args.metric_max_points // max(eval_len, 1))
        warnings.warn(
            '--metric-max-points enables approximate metrics for quick screening.')
    if args.metric_histogram_bins is not None:
        histogram_metrics_by_radius = {
            radius: HistogramOODMetrics(args.metric_histogram_bins)
            for radius in dilation_radii}
        histogram_metrics = histogram_metrics_by_radius[args.ood_dilation_radius]
        warnings.warn(
            '--metric-histogram-bins enables deterministic bounded-memory metrics.')

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    scorer = None
    tail_scorer = None
    if needs_umpof:
        scorer = UMPOFScorer.from_file(
            args.umpof_bank,
            device='cuda',
            tau=args.umpof_tau)
    if needs_tail_bank:
        tail_scorer = UMPOFScorer.from_file(
            args.umpof_tail_bank,
            device='cuda',
            tau=args.umpof_tau)

    ssc_metric = SSCMetrics(len(dataset.class_names)).cuda()
    auroc_scores = []
    auroc_labels = []
    aupr_scores = []
    aupr_labels = []
    added_label_count = 0
    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    processed = 0

    print('Evaluation protocol: sequences={}, dilation_radius={} voxels, samples={}'.format(
        evaluated_sequences if evaluated_sequences is not None else 'all',
        dilation_radii,
        eval_len), flush=True)

    for i, data in enumerate(data_loader):
        if args.max_samples is not None and i >= args.max_samples:
            break

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        output_voxels = torch.argmax(result['output_voxels'], dim=1)
        target_voxels = result['target_voxels'].clone().long()
        ssc_metric.update(y_pred=output_voxels.clone(), y_true=target_voxels.clone())

        score = compute_score(result, scorer, args, tail_scorer=tail_scorer)
        if histogram_metrics_by_radius is not None:
            for radius, radius_metrics in histogram_metrics_by_radius.items():
                labels_dilated, labels_original, added_count = build_ood_labels(
                    target_voxels, dilation_radius=radius)
                radius_metrics.update(
                    score, labels_original, labels_dilated, added_count)
                if radius == args.ood_dilation_radius:
                    added_label_count += added_count
        else:
            labels_dilated, labels_original, added_count = build_ood_labels(
                target_voxels, dilation_radius=args.ood_dilation_radius)
            added_label_count += added_count
        if histogram_metrics_by_radius is None:
            score_auroc, label_auroc = extract_valid_arrays(
                score, labels_original, metric_max_per_frame, rng)
            score_aupr, label_aupr = extract_valid_arrays(
                score, labels_dilated, metric_max_per_frame, rng)
            auroc_scores.append(score_auroc)
            auroc_labels.append(label_auroc)
            aupr_scores.append(score_aupr)
            aupr_labels.append(label_aupr)

        processed = i + 1
        if processed % args.progress_interval == 0 or processed == eval_len:
            elapsed = time.time() - start_time
            print('Processed {}/{} samples in {:.1f}s'.format(
                processed, eval_len, elapsed), flush=True)

    ssc_results = evaluate_ssc(dataset, ssc_metric)

    ood_metrics_by_radius = None
    if histogram_metrics_by_radius is not None:
        ood_metrics_by_radius = {
            str(radius): metrics.compute()
            for radius, metrics in histogram_metrics_by_radius.items()}
        ood_metrics = ood_metrics_by_radius[str(args.ood_dilation_radius)]
    else:
        auroc_scores = np.concatenate(auroc_scores)
        auroc_labels = np.concatenate(auroc_labels)
        aupr_scores = np.concatenate(aupr_scores)
        aupr_labels = np.concatenate(aupr_labels)

        ood_metrics = {
            'auroc': float(calculate_auroc(auroc_scores, auroc_labels)),
            'aupr_original': float(calculate_standard_aupr(
                auroc_scores, auroc_labels)),
            'aupr_dilated_standard': float(calculate_standard_aupr(
                aupr_scores, aupr_labels)),
            'aupr_area': float(calculate_aupr_fast(
                aupr_scores, aupr_labels, added_label_count)),
        }
    print('>>> ssc_metrics: ', ssc_results, flush=True)
    print('>>> ood_metrics: ', ood_metrics, flush=True)

    if args.out is not None:
        output = dict(
            config=args.config,
            checkpoint=args.checkpoint,
            score_mode=args.score_mode,
            umpof_bank=args.umpof_bank,
            umpof_tail_bank=args.umpof_tail_bank,
            gate_lambda=args.gate_lambda,
            gate_power=args.gate_power,
            tadc_tail_boost=args.tadc_tail_boost,
            tadc_min_gate=args.tadc_min_gate,
            tadc_max_gate=args.tadc_max_gate,
            tadc_quantile_low=args.tadc_quantile_low,
            tadc_quantile_high=args.tadc_quantile_high,
            tadc_quantile_min_voxels=args.tadc_quantile_min_voxels,
            tadc_spatial_kernel=args.tadc_spatial_kernel,
            tadc_spatial_alpha=args.tadc_spatial_alpha,
            tadc_peak_gamma=args.tadc_peak_gamma,
            ppsc_v2_boost=args.ppsc_v2_boost,
            ppsc_v2_suppress=args.ppsc_v2_suppress,
            ppsc_v2_occupancy=args.ppsc_v2_occupancy,
            ppsc_v3_min_support=args.ppsc_v3_min_support,
            ppsc_v3_support_scale=args.ppsc_v3_support_scale,
            ppsc_v3_support_power=args.ppsc_v3_support_power,
            ppsc_v4_min_support=args.ppsc_v4_min_support,
            ppsc_v4_support_scale=args.ppsc_v4_support_scale,
            ppsc_v4_support_power=args.ppsc_v4_support_power,
            ppsc_v5_min_support=args.ppsc_v5_min_support,
            ppsc_v5_support_scale=args.ppsc_v5_support_scale,
            ppsc_v5_support_power=args.ppsc_v5_support_power,
            ood_dilation_radius=args.ood_dilation_radius,
            ood_dilation_radii=dilation_radii,
            include_sequences=evaluated_sequences,
            dataset_size_before_filter=dataset_size_before_filter,
            evaluated_samples=processed,
            max_samples=args.max_samples,
            metric_max_points=args.metric_max_points,
            metric_histogram_bins=args.metric_histogram_bins,
            metric_histogram_mapping=(
                'float32_ordered_bits'
                if args.metric_histogram_bins is not None else None),
            ssc_metrics=ssc_results,
            ood_metrics=ood_metrics,
            ood_metrics_by_radius=ood_metrics_by_radius)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
