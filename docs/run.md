# Training and evaluation

All examples run from the repository root with the environment activated:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## Checkpoints

No weights or banks are included. The adaptation configurations expect the
following separately obtained ProOOD weights:

| Configuration | Initial checkpoint under `ckpts/` |
| --- | --- |
| `umpocc-semkitti-msnet.py` | `proood_sgn_semkitti_occ.pth` |
| `umpocc-kitti360-msnet.py` | `proood_sgn_kitti360_occ.pth` |
| `umpocc-semkitti-sql.py` | `proood_sgn_sql_ood.pth` |
| `umpocc-kitti360-sql.py` | `proood_sgn_sql_kitti360_ood.pth` |

Inherited training configs also reference the ImageNet backbone checkpoint
`ckpts/resnet50-19c8e357.pth`. Obtain it separately, or when loading a complete
model checkpoint disable backbone initialization with
`--cfg-options model.pretrained=None`. Evaluation entry points disable this
initialization automatically.

## Adaptation

Example using one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh \
  projects/configs/sgn/umpocc-semkitti-sql.py 1 \
  --cfg-options model.pretrained=None
```

Each UMPOcc training configuration retains the existing adaptation settings:
learning rate `2e-5`, 200 warm-up iterations, eight configured epochs, and
distributional prototype loss weight `0.1`. Checkpoints are saved every epoch.
Use the checkpoint corresponding to the intended experiment; these settings
do not imply that the final epoch is the selected model. Changing GPU count
changes the effective batch size.

To supply a checkpoint outside `ckpts/`:

```bash
bash tools/dist_train.sh projects/configs/sgn/umpocc-semkitti-sql.py 1 \
  --cfg-options model.pretrained=None load_from=/path/to/base.pth
```

For inherited baseline training use the matching `proood-*-portable.py`
configuration. For example, `proood-sql-semkitti-portable.py` retains the
original SemanticKITTI SQL training recipe. A baseline run needs the separately
obtained backbone initialization weights.

## Checkpoint interpolation

The utility retains prototype buffers and other tensors found only in the
adapted model. Common floating-point tensors are interpolated as
`(1 - alpha) * base + alpha * adapted`.

```bash
python tools/interpolate_checkpoints.py \
  ckpts/proood_sgn_sql_ood.pth \
  work_dirs/umpocc-semkitti-sql/epoch_1.pth \
  --out-dir work_dirs/umpocc-semkitti-sql/interpolated --alphas 0.1
```

This produces `interp_alpha_0p1.pth`. Choose the raw or interpolated checkpoint
explicitly for each experiment; interpolation is not performed automatically.

## Semantic occupancy

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh \
  projects/configs/sgn/umpocc-semkitti-msnet.py \
  /path/to/umpocc_semkitti_msnet.pth 1
```

Use `umpocc-kitti360-msnet.py` and the corresponding checkpoint for
SSCBench-KITTI-360. Match the prototype mode count and input depth branch
between the checkpoint and its evaluation configuration.

SemanticKITTI local evaluation uses validation labels. To export predictions
for the official test server instead:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_semkitti_submission.py \
  projects/configs/sgn/umpocc-semkitti-msnet.py \
  /path/to/umpocc_semkitti_msnet.pth \
  --out-dir submissions/semantic_kitti \
  --sequences 11 12 13 14 15 16 17 18 19 20 21
```

## OOD evaluation with PPSC

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/eval_ood.sh \
  projects/configs/sgn/umpocc-ood-vaakitti.py \
  /path/to/umpocc_semkitti_sql.pth --out outputs/vaa_kitti.json

CUDA_VISIBLE_DEVICES=0 bash tools/eval_ood.sh \
  projects/configs/sgn/umpocc-ood-vaakitti360.py \
  /path/to/umpocc_kitti360_sql.pth --out outputs/vaa_kitti360.json

CUDA_VISIBLE_DEVICES=0 bash tools/eval_ood.sh \
  projects/configs/sgn/umpocc-ood-stu.py \
  /path/to/umpocc_semkitti_sql.pth \
  --include-sequences 144 --out outputs/vaa_stu_144.json
```

The STU example selects sequence 144. Omitting `--include-sequences` evaluates
both 125 and 144. If 125 is used for parameter selection, the full set includes
those development samples and should be identified as such.

The preset calls `test_ood_fast.py` with `--score-mode echoood_ppsc_v3`, kernel
size 7, alpha 0.75, gamma 2, boost 0.25, suppression 0, minimum support 0,
support scale 0.2, and support power 1. The inherited `tadc-*` and `ppsc-v*`
argument names are retained. Append CLI arguments to override the preset.

For an uncalibrated comparison on the same checkpoint, append
`--score-mode echoood`. For the inherited single-prototype ProOOD baseline,
also use the corresponding `proood-ood-*-portable.py` config and baseline
checkpoint. Fast OOD evaluation supports one GPU per process.

## Metric conventions

The preset reports dilation radii 4, 5, and 6 voxels, corresponding to
0.8, 1.0, and 1.2 m on the 0.2 m grid. It uses the existing deterministic
65,536-bin histogram implementation for bounded memory; this is an
approximation of sorting all scores. Keep the same metric settings for
comparisons. `--metric-max-points` is optional subsampling for quick checks
and should not be used for full benchmark reporting.

Output distinguishes the inherited tolerance metric (`aupr_area`), ordinary
PR area on the original labels (`aupr_original`), and PR area on dilated labels
(`aupr_dilated_standard`). The inherited tolerance metric applies an
added-label correction and is not necessarily bounded by one. Do not treat
these three fields as interchangeable or silently clip the legacy metric.
AUROC uses the original OOD labels. See the metric implementations in
`tools/test_ood_fast.py` for exact conventions.

## Optional tools

`tools/build_umpof_bank.py` builds a bank from labeled source features for
optional density/fusion score variants. The `echoood_ppsc_v3` path above does
not need it. Use `--split train` to build a bank from the source training set;
the command accepts a config, checkpoint, `--out`, and `--modes`.

`tools/benchmark_ood_inference.py` measures inference latency and GPU memory
with `--score-mode echoood` or `--score-mode ppsc_v3`, and requires `--out`.
Use `--help` for its warm-up and sample-count settings.

The legacy model-head prediction writers default to `outputs/semantic/` and
`outputs/ood/`. Override these with `UMPOCC_PRED_DIR` and
`UMPOCC_OOD_PRED_DIR`. The dedicated submission exporter uses its own
`--out-dir`.
