#!/usr/bin/env python3
"""Generate official SPIdepth metric depth for SSCBench-KITTI-360 frames."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--spidepth-root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--sequences', nargs='+', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--all-images', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def config_line_to_args(line):
    for arg in line.split():
        if arg.strip():
            yield arg


def collect_items(args):
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    items = []
    for sequence in args.sequences:
        sequence_root = data_root / 'data_2d_raw' / sequence
        if args.all_images:
            frame_ids = sorted(
                p.stem for p in
                (sequence_root / 'image_00' / 'data_rect').glob('*.png'))
        else:
            frame_ids = sorted(
                p.stem for p in (sequence_root / 'voxels').glob('*.bin'))
        for frame_id in frame_ids:
            image_path = (
                sequence_root / 'image_00' / 'data_rect' / f'{frame_id}.png')
            output_path = output_root / sequence / f'{frame_id}_disp.npy'
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if args.overwrite or not output_path.is_file():
                items.append((image_path, output_path))
    if args.limit is not None:
        items = items[:args.limit]
    return items


class ImageDataset(Dataset):
    def __init__(self, items, feed_size):
        self.items = items
        self.feed_size = feed_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, output_path = self.items[index]
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            original_size = (image.height, image.width)
            image = image.resize(self.feed_size, Image.Resampling.LANCZOS)
            tensor = pil_to_tensor(image).float().div_(255.0)
        return tensor, original_size, str(output_path)


def collate_batch(batch):
    images, sizes, outputs = zip(*batch)
    return torch.stack(images), sizes, outputs


def atomic_save(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, array.astype(np.float32, copy=False))
    os.replace(str(temporary), str(path))


def build_model(args):
    sys.path.insert(0, args.spidepth_root)
    from SQLdepth import MonodepthOptions, SQLdepth

    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = config_line_to_args
    config = str(Path(args.spidepth_root) / 'conf' / 'cvnXt.txt')
    opt = options.parser.parse_args(['@' + config])
    opt.load_pt_folder = args.weights
    opt.no_cuda = False
    return SQLdepth(opt).cuda().eval(), opt


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for SPIdepth preprocessing')

    model, opt = build_model(args)
    items = collect_items(args)
    print(f'Queued {len(items)} frames from sequences {args.sequences}',
          flush=True)
    if not items:
        return

    dataset = ImageDataset(items, (opt.width, opt.height))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_batch)

    completed = 0
    started = time.time()
    with torch.inference_mode():
        for images, original_sizes, output_paths in loader:
            images = images.cuda(non_blocking=True)
            augmented = torch.cat((images, torch.flip(images, dims=(-1,))),
                                  dim=0)
            predictions = model(augmented)
            normal, flipped = predictions.chunk(2, dim=0)
            predictions = 0.5 * (normal + torch.flip(flipped, dims=(-1,)))

            for prediction, original_size, output_path in zip(
                    predictions, original_sizes, output_paths):
                prediction = F.interpolate(
                    prediction[None], size=tuple(original_size),
                    mode='bilinear', align_corners=False)[0, 0]
                atomic_save(output_path, prediction.cpu().numpy())

            completed += len(output_paths)
            if completed % args.log_interval < len(output_paths) or completed == len(items):
                elapsed = time.time() - started
                print(
                    f'Processed {completed}/{len(items)} frames '
                    f'({completed / max(elapsed, 1e-6):.2f} frame/s)',
                    flush=True)


if __name__ == '__main__':
    main()
