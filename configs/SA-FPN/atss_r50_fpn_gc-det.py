# from tools.work_dirs.retinanet_r50_fpn_1x_coco.retinanet_r50_fpn_1x_coco import load_from
from pickletools import optimize

_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../_base_/datasets/gc_det.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=10))


max_epochs = 15
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning rate

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3,7],
        gamma=0.1)
]

optim_wrapper = dict(

    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.1),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
#

#
optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))




