_base_ = 'cascade_mask_rcnn_regnetx-3.2GF_fpn_coco-mstrain_3x_20e_nuim.py'
model = dict(
    backbone=dict(
        type='RegNet',
        arch='regnetx_800mf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_800mf'),
        style='pytorch',
        with_cp=True),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 288, 672],
        out_channels=256,
        num_outs=5))

load_from = 'http://download.openmmlab.sensetime.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco_20210715_211616-dcbd13f4.pth'  # noqa
