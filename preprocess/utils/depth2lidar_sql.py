#!/usr/bin/env python3
"""Project SPIdepth metric-depth arrays to KITTI pseudo-LiDAR files."""

import argparse
import os
from pathlib import Path

import numpy as np

import kitti_util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib-dir', required=True)
    parser.add_argument('--depth-dir', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--max-high', type=float, default=80.0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def project_depth_to_velodyne(calibration, depth, max_high):
    rows, cols = depth.shape
    columns, rows_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    image_points = np.stack((columns, rows_grid, depth), axis=-1).reshape(-1, 3)
    cloud = calibration.project_image_to_velo(image_points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]


def atomic_write(path, cloud):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    cloud.tofile(str(temporary))
    os.replace(str(temporary), str(path))


def main():
    args = parse_args()
    depth_dir = Path(args.depth_dir)
    save_dir = Path(args.save_dir)
    calibration = kitti_util.Calibration(str(Path(args.calib_dir) / 'calib.txt'))
    depth_paths = sorted(depth_dir.glob('*_disp.npy'))
    print(f'Queued {len(depth_paths)} depth maps from {depth_dir}', flush=True)

    for index, depth_path in enumerate(depth_paths, start=1):
        frame_id = depth_path.name[:-len('_disp.npy')]
        output_path = save_dir / f'{frame_id}.bin'
        if output_path.is_file() and not args.overwrite:
            continue
        depth = np.load(str(depth_path))
        cloud = project_depth_to_velodyne(calibration, depth, args.max_high)
        cloud = np.concatenate(
            (cloud, np.ones((cloud.shape[0], 1), dtype=cloud.dtype)), axis=1)
        atomic_write(output_path, cloud.astype(np.float32, copy=False))
        if index % 50 == 0 or index == len(depth_paths):
            print(f'Processed {index}/{len(depth_paths)} depth maps', flush=True)


if __name__ == '__main__':
    main()
