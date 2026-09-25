# UMPOcc

Source release of the UMPOcc implementation developed on top of ProOOD, using
the SGN backbone for 3D semantic occupancy prediction and out-of-distribution
(OOD) detection.

The code includes **UMPOF** multi-prototype modeling and **PPSC** spatial score
calibration. This package contains the SGN implementation from the modified
ProOOD project. It does not include a VoxDet implementation.

## Getting started

1. Follow [Installation](docs/install.md) to set up the Python 3.8 / PyTorch 1.9
   environment.
2. Prepare the datasets and depth inputs using [Data preparation](docs/dataset.md).
3. Follow [Training and evaluation](docs/run.md) for adaptation, occupancy
   evaluation, PPSC evaluation, and SemanticKITTI test submission export.

Run commands from this repository's root. Dataset paths default to `data/`;
externally obtained checkpoints go under `ckpts/`. No datasets, model weights,
prototype banks, predictions, or experiment results are bundled.

## Main configurations

All configurations below are under `projects/configs/sgn/`.

| Configuration | Use | Prototype modes |
| --- | --- | ---: |
| `umpocc-semkitti-msnet.py` | SemanticKITTI, MobileStereoNet input | 4 |
| `umpocc-kitti360-msnet.py` | SSCBench-KITTI-360, MobileStereoNet input | 4 |
| `umpocc-semkitti-sql.py` | SemanticKITTI, SQL input | 2 |
| `umpocc-kitti360-sql.py` | SSCBench-KITTI-360, SQL input | 4 |
| `umpocc-ood-vaakitti.py` | VAA-KITTI evaluation | 2 |
| `umpocc-ood-vaakitti360.py` | VAA-KITTI-360 evaluation | 4 |
| `umpocc-ood-stu.py` | VAA-STU evaluation | 2 |

The four training configurations adapt an existing ProOOD checkpoint; they
are not training-from-scratch recipes. The corresponding `proood-*-portable.py`
configurations provide the inherited baseline settings and portable data paths.

## Code map

| Path | Purpose |
| --- | --- |
| `projects/mmdet3d_plugin/proood/prooodmodule/distributional_prototype.py` | Multi-prototype modeling |
| `projects/mmdet3d_plugin/proood/prooodmodule/tailvoxelselector.py` | Tail voxel selection |
| `projects/mmdet3d_plugin/proood/dense_heads/proood.py` | Model head and training losses |
| `tools/test_ood_fast.py` | OOD scoring, PPSC, and tolerance metrics |
| `tools/eval_ood.sh` | PPSC evaluation with the current parameter preset |
| `tools/build_umpof_bank.py` | Optional offline prototype bank construction |
| `tools/interpolate_checkpoints.py` | Checkpoint interpolation |
| `preprocess/` | Label, stereo depth, and pseudo-LiDAR preparation |

Internal `ProOOD` / `ProOODHead` registry names and module paths are retained
for compatibility with existing checkpoints. In the evaluation CLI, the PPSC
implementation is named `echoood_ppsc_v3`; older score variants remain available
for comparison. The standard PPSC path does not require an offline bank.

## License and acknowledgments

The upstream Apache-2.0 [LICENSE](LICENSE) is retained. See [NOTICE](NOTICE) for
the origin of inherited code and changes in this package. Bundled MobileStereoNet
sources retain their [license](preprocess/mobilestereonet/LICENSE).
We acknowledge ProOOD, SGN, OpenMMLab, MobileStereoNet, SPIdepth, and the benchmark
authors. Dataset and externally downloaded checkpoint licenses are separate.
