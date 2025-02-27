# dataset settings
dataset_type = 'TACODataset'
data_root = '/root/PIPE/'

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
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type="PhotoMetricDistortion"),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    #dict(type='Mosaic', prob=0.3),
    dict(type='RandomFlip', prob=0.4, direction=['horizontal', 'vertical']),
    #dict(type='Normalize', mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), 
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    #dict(type='Normalize', mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), 
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

trainTacoDataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)

#balancedTrainTacoDataset=dict(
#        type='ClassBalancedDataset',
#        oversample_thr=1.0,
#        dataset=trainTacoDataset)

valTacoDataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/valid.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=trainTacoDataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=valTacoDataset)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/valid.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator