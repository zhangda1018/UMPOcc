# UMPOcc release: portable paths and public configuration names.
"""Short UMPOF-M4 adaptation from the released KITTI-360 MSN checkpoint."""

_base_ = ['./proood-kitti360-portable.py']

work_dir = 'work_dirs/umpocc-kitti360-msnet'

# Table 2 occupancy uses the MobileStereoNet input path, so this branch must
# start from the matching released occupancy checkpoint.
load_from = 'ckpts/proood_sgn_kitti360_occ.pth'

model = dict(
    pts_bbox_head=dict(
        prototype_modes=4,
        distributional_proto_loss_weight=0.1))

optimizer = dict(lr=2e-5)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=0.1)
total_epochs = 8
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
