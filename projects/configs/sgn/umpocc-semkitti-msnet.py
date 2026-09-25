# UMPOcc release: portable paths and public configuration names.
"""Short UMPOF-M4 adaptation from the released SemanticKITTI MSN checkpoint."""

_base_ = ['./proood-semkitti-portable.py']

work_dir = 'work_dirs/umpocc-semkitti-msnet'

load_from = 'ckpts/proood_sgn_semkitti_occ.pth'

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
