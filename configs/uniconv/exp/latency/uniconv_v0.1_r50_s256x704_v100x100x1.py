_base_ = 'uniconv_v0.1_r18_s256x704_v200x200x6.py'
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'
    ),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=256, out_channels=64),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64*1*4,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    n_voxels=(100, 100, 1),
    voxel_size=[1.0, 1.0, 6.0],
)
