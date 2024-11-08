# dataset settings
dataset_type = 'GC10Dataset'

# classes = ("1_chongkong","2_hanfeng","3_yueyawan","4_shuiban","5_youban","6_siban","7_yiwu","8_yahen","9_zhehen","10_yaozhe")

data_root = '../mmdetection2.0/data/New_GC-DET/'




img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         # scale=(640, 640),
#         # flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# test_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         file_client_args=file_client_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(640, 640), keep_ratio=True),
#     dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(640,640), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            # VOCDataset will add different `dataset_type` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='main/trainval.txt',
                    data_prefix=dict(sub_data_root='images/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline)
            ])))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='main/test.txt',
        data_prefix=dict(sub_data_root='images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=2,
#     train=dict(
#         type='RepeatDataset',
#         times=3,
#         dataset=dict(
#             type=dataset_type,
#             ann_file=data_root + 'main/trainval.txt',
#             img_prefix=data_root + 'images/',
#             pipeline=train_pipeline)),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'main/test.txt',
#         img_prefix=data_root + 'images/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'main/test.txt',
#         img_prefix=data_root + 'images/',
#         pipeline=test_pipeline))
# evaluation = dict(interval=4, metric='mAP',save_best='auto')

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
