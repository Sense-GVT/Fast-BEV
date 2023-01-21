_base_ = 'cascade_mask_rcnn_regnetx-3.2GF_fpn_coco-mstrain_3x_20e_nuim.py'
model = dict(
    backbone=dict(
        type='RegNet',
        arch='regnetx_8.0gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_8.0gf'),
        style='pytorch',
        with_cp=True),
    neck=dict(
        type='FPN',
        in_channels=[80, 240, 720, 1920],
        out_channels=256,
        num_outs=5))

load_from = ''
