# model settings
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', prefix='backbone.',checkpoint=checkpoint)),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=1,
    #     add_extra_convs='on_input',
    #     num_outs=5),
    neck=dict(
            type='VOV',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            deploy=False,
            start_level=1),
    # bbox_head=dict(
    #     type='RetinaHead',
    #     num_classes=80,
    #     in_channels=256,
    #     stacked_convs=4,
    #     feat_channels=256,
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         octave_base_scale=4,
    #         scales_per_octave=3,
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[8, 16, 32, 64, 128]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0]),
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # # model training and testing settings
    # train_cfg=dict(
    #     assigner=dict(
    #         type='MaxIoUAssigner',
    #         pos_iou_thr=0.5,
    #         neg_iou_thr=0.4,
    #         min_pos_iou=0,
    #         ignore_iof_thr=-1),
    #     sampler=dict(
    #         type='PseudoSampler'),  # Focal loss should use PseudoSampler
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),
    # test_cfg=dict(
    #     nms_pre=1000,
    #     min_bbox_size=0,
    #     score_thr=0.05,
    #     nms=dict(type='nms', iou_threshold=0.5),
    #     max_per_img=100))
bbox_head=dict(
        type='ATSSHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
