# UMPOcc release: portable paths and public configuration names.
_base_ = ['./proood-ood-stu-portable.py']

work_dir = 'work_dirs/umpocc-ood-stu'

model = dict(
    pts_bbox_head=dict(
        prototype_modes=2,
        distributional_proto_loss_weight=0.1))
