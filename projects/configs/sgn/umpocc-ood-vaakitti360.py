# UMPOcc release: portable paths and public configuration names.
_base_ = ['./proood-ood-vaakitti360-portable.py']

work_dir = 'work_dirs/umpocc-ood-vaakitti360'

model = dict(
    pts_bbox_head=dict(
        prototype_modes=4,
        distributional_proto_loss_weight=0.1))
