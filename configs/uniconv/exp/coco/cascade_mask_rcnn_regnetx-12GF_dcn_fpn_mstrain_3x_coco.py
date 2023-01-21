_base_ = 'cascade_mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py'
model = dict(
    backbone=dict(
        type='RegNet',
        arch='regnetx_12gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_12gf'),
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        with_cp=True),
    neck=dict(
        type='FPN',
        in_channels=[224, 448, 896, 2240],
        out_channels=256,
        num_outs=5))
