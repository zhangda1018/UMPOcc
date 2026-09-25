# UMPOcc release: portable paths and public configuration names.
_base_ = ['./proood-ood-stu.py']

custom_imports = dict(
   imports=['projects.mmdet3d_plugin.datasets'],
   allow_failed_imports=False)

work_dir = 'work_dirs/proood-ood-stu-portable'
data_root = 'data/semantic_kitti/'
data_test_root = 'data/vaa_stu/'

model = dict(
   pts_bbox_head=dict(
       ood_flag=True))

data = dict(
   train=dict(
       data_root=data_root,
       preprocess_root=data_root + 'dataset'),
   val=dict(
       data_root=data_test_root,
       preprocess_root=data_test_root),
   test=dict(
       data_root=data_test_root,
       preprocess_root=data_test_root))
