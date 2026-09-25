# UMPOcc release: portable paths and public configuration names.
_base_ = ['./proood-semkitti.py']

work_dir = 'work_dirs/proood-semkitti-portable'
data_root = 'data/semantic_kitti/'

data = dict(
   train=dict(
       data_root=data_root,
       preprocess_root=data_root + 'dataset'),
   val=dict(
       data_root=data_root,
       preprocess_root=data_root + 'dataset'),
   test=dict(
       data_root=data_root,
       preprocess_root=data_root + 'dataset'))
