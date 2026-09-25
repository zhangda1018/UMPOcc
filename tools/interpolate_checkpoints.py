#!/usr/bin/env python
import argparse
import copy
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Interpolate common floating-point parameters between checkpoints.')
    parser.add_argument('base')
    parser.add_argument('finetuned')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--alphas', type=float, nargs='+', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    base = torch.load(args.base, map_location='cpu')
    finetuned = torch.load(args.finetuned, map_location='cpu')
    base_state = base.get('state_dict', base)
    tuned_state = finetuned.get('state_dict', finetuned)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_float = [
        key for key, value in tuned_state.items()
        if key in base_state and value.shape == base_state[key].shape
        and torch.is_floating_point(value)
    ]
    print(f'Interpolating {len(common_float)} floating-point tensors; '
          f'keeping {len(tuned_state) - len(common_float)} tuned tensors unchanged.')

    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')
        output = copy.copy(finetuned)
        state = {}
        for key, tuned_value in tuned_state.items():
            if key in common_float:
                base_value = base_state[key].to(dtype=tuned_value.dtype)
                state[key] = base_value.mul(1.0 - alpha).add(tuned_value, alpha=alpha)
            else:
                state[key] = tuned_value
        output['state_dict'] = state
        output.pop('optimizer', None)
        output.setdefault('meta', {})['checkpoint_interpolation'] = {
            'base': str(Path(args.base).resolve()),
            'finetuned': str(Path(args.finetuned).resolve()),
            'alpha': alpha,
        }
        tag = str(alpha).replace('.', 'p')
        destination = out_dir / f'interp_alpha_{tag}.pth'
        torch.save(output, destination)
        print(f'Saved alpha={alpha:g} to {destination}')


if __name__ == '__main__':
    main()
