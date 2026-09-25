# UMPOcc release: portable paths and public configuration names.
_base_ = ['./proood-kitti360.py']

custom_imports = dict(
   imports=['projects.mmdet3d_plugin.datasets'],
   allow_failed_imports=False)

work_dir = 'work_dirs/proood-kitti360-portable'
data_root = 'data/kitti360/'

data = dict(
   train=dict(
       data_root=data_root,
       preprocess_root=data_root + 'preprocess'),
   val=dict(
       data_root=data_root,
       preprocess_root=data_root + 'preprocess'),
   test=dict(
       data_root=data_root,
       preprocess_root=data_root + 'preprocess'))
