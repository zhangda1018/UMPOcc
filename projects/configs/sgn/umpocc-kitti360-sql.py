# UMPOcc release: portable paths and public configuration names.
"""Low-risk UMPOF fine-tuning from the released KITTI-360 SQL checkpoint."""

_base_ = ['./proood-sql-kitti360-portable.py']

work_dir = 'work_dirs/umpocc-kitti360-sql'

# Keep the released occupancy solution as the starting point. The added
# distributional prototype state is initialized and warmed up during training.
load_from = 'ckpts/proood_sgn_sql_kitti360_ood.pth'

model = dict(
    pts_bbox_head=dict(
        prototype_modes=4,
        distributional_proto_loss_weight=0.1))

# A short, conservative adaptation is used as the first diagnostic run.
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
