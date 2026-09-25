# Data preparation

Run commands from the UMPOcc root. Obtain SemanticKITTI, SSCBench-KITTI-360,
and the VAA datasets separately. The VAA downloads are documented by OccOoD,
and KITTI-360 preparation is documented by
[SSCBench](https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360).

## Dataset roots

The public configurations use the following relative paths. Create symlinks
to prepared datasets, or update the nested `data.train`, `data.val`, and
`data.test` paths in the configuration. Changing only the top-level `data_root`
through `--cfg-options` does not update already constructed nested dictionaries.

```text
data/
  semantic_kitti/
    dataset/
      sequences/<sequence>/{image_2,image_3,voxels,calib.txt,poses.txt}
      labels/<sequence>/*_1_{1,2,8}.npy
      sequences_msnet3d_lidar/sequences/<sequence>/*.bin
      sequences_sql_lidar/sequences/<sequence>/*.bin
  kitti360/
    data_2d_raw/<sequence>/{image_00/data_rect,image_01/data_rect,voxels,poses.txt}
    preprocess/labels/<sequence>/*_1_{1,2,8}.npy
    msnet3d_pseudo_lidar/<sequence>/*.bin
    sql_pseudo_lidar/<sequence>/*.bin
  vaa_kitti/
    dataset/
      sequences/<sequence>/{image,voxels,calib.txt,poses.txt}
      labels/<sequence>/*_1_1.npy
      sequences_sql_lidar/sequences/<sequence>/*.bin
  vaa_kitti360/
    data_2d_raw/<sequence>/{image,voxels,poses.txt}
    labels/<sequence>/*_1_1.npy
    sql_pseudo_lidar/<sequence>/*.bin
  vaa_stu/
    <sequence>/{image/port_a_cam_0,voxels,calib.txt,poses.txt}
    labels/<sequence>/*_1_1.npy
    sequences_sql_lidar/<sequence>/*.bin
```

Example linking one prepared dataset:

```bash
mkdir -p data
ln -s /path/to/semantic_kitti data/semantic_kitti
```

The supplied loaders determine frame selection, label conventions, and temporal
inputs. Training configurations with temporal inputs need pseudo-LiDAR for the
referenced neighboring frames as well as the labeled frames.
The VAA loaders enumerate `voxels/*.label` and read the corresponding prepared
`labels/*_1_1.npy` targets. Their image directory names differ from the source
training datasets, as shown above.

## SemanticKITTI labels

Generate the semantic labels and their downsampled versions:

```bash
python preprocess/label/label_preprocess.py \
  --kitti_root data/semantic_kitti \
  --kitti_preprocess_root data/semantic_kitti/dataset
```

For SSCBench-KITTI-360, use the benchmark's prepared labels and aligned poses.
OOD labels must use the benchmark conventions handled by the VAA loaders;
do not preprocess them using the SemanticKITTI known-class remapping.

## MobileStereoNet depth

The bundled `preprocess/mobilestereonet/` contains the inherited inference
sources and image filename lists. Obtain its checkpoint separately from
[MobileStereoNet](https://github.com/cogsys-tuebingen/mobilestereonet).
For example, for SemanticKITTI sequence 00:

```bash
python preprocess/mobilestereonet/prediction.py \
  --datapath data/semantic_kitti/dataset/sequences/00 \
  --testlist preprocess/mobilestereonet/filenames/00.txt \
  --num_seq 00 --loadckpt ckpts/MSNet3D_SF_DS_KITTI2015.ckpt \
  --dataset kitti --model MSNet3D \
  --savepath data/semantic_kitti/dataset/sequences_msnet3d_depth \
  --baseline 388.1823

python preprocess/utils/depth2lidar.py \
  --calib_dir data/semantic_kitti/dataset/sequences/00 \
  --depth_dir data/semantic_kitti/dataset/sequences_msnet3d_depth/sequences/00 \
  --save_dir data/semantic_kitti/dataset/sequences_msnet3d_lidar/sequences/00
```

The inherited `--baseline` argument means baseline multiplied by focal length:
388.1823 for sequences 00–02 and 13–21, 389.6304 for 03, and 381.8293
for 04–12. For KITTI-360 use `--dataset kitti360`, its sequence filename list,
331.5325566, and `preprocess/utils/depth2lidar_kitti360.py`.
The conversion helper accepts `--depth_dir` and `--save_dir`.

## SQL / SPIdepth inputs

The historical configuration name `sql` identifies the monocular depth input
path. The included preprocessing entry points use an external SPIdepth checkout
and SQL weights. For example:

```bash
python tools/preprocess_spidepth_semkitti.py \
  --data-root data/semantic_kitti/dataset \
  --output-root data/semantic_kitti/dataset/sequences_sql_depth \
  --spidepth-root /path/to/SPIdepth --weights /path/to/sql_weights \
  --sequences 00 --all-images

python preprocess/utils/depth2lidar_sql.py \
  --calib-dir data/semantic_kitti/dataset/sequences/00 \
  --depth-dir data/semantic_kitti/dataset/sequences_sql_depth/sequences/00 \
  --save-dir data/semantic_kitti/dataset/sequences_sql_lidar/sequences/00
```

The SPIdepth outputs named `*_disp.npy` contain metric depth. Use the SQL
conversion helper to handle these filenames. For KITTI-360 and STU, use
`tools/preprocess_spidepth_kitti360.py` and `tools/preprocess_spidepth_stu.py`;
their `--help` describes the arguments. These entry points generate depth;
the loaders consume the converted pseudo-LiDAR directories shown above.

## Evaluation splits

SemanticKITTI occupancy configurations default to validation sequence 08 for
local metric computation. Official test predictions are exported separately
with `tools/export_semkitti_submission.py`; test ground truth is not bundled.
SSCBench-KITTI-360 configurations retain their inherited validation/test split.
STU uses sequences 125 and 144 by default; `--include-sequences 144` selects
sequence 144 only. State the actual split whenever reporting results.
