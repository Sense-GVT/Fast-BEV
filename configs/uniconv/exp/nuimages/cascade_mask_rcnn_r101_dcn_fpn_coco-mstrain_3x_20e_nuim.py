_base_ = 'cascade_mask_rcnn_r101_fpn_coco-mstrain_3x_20e_nuim.py'

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        with_cp=True),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
)

load_from = 'pretrained_models/cascade_mask_rcnn_r101_dcn_fpn_mstrain_3x_coco_bbox_mAP_0.4610_segm_mAP_0.4000.pth'  # noqa
