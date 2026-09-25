# UMPOcc release: portable paths and public configuration names.
_base_ = ['./proood-ood-vaakitti360.py']

custom_imports = dict(
   imports=['projects.mmdet3d_plugin.datasets'],
   allow_failed_imports=False)

work_dir = 'work_dirs/proood-ood-vaakitti360-portable'
data_root = 'data/vaa_kitti360/'
data_test_root = 'data/vaa_kitti360/'

data = dict(
   train=dict(
       data_root=data_root,
       preprocess_root=data_root),
   val=dict(
       data_root=data_test_root,
       preprocess_root=data_test_root),
   test=dict(
       data_root=data_test_root,
       preprocess_root=data_test_root))
