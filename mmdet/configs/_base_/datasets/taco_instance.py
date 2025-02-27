# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms.loading import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler

from mmdet.datasets.taco import TACODataset
from mmdet.datasets.samplers.batch_sampler import AspectRatioBatchSampler
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, Resize, PhotoMetricDistortion
from mmdet.evaluation.metrics.coco_metric import CocoMetric

# dataset settings
dataset_type = 'TACODataset'
data_root = 'data/taco/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True, with_seg=True),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5, direction=["horizontal", "vertical"]),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=CocoDataset,
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CocoDataset,
        data_root=data_root,
        ann_file='valid/valid.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'valid/valid.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type=DefaultSampler, shuffle=False),
#     dataset=dict(
#         type=CocoDataset,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type=CocoMetric,
#     metric=['bbox', 'segm'],
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_instance/test')

# Load pretrained model to obtain higher results
# https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets