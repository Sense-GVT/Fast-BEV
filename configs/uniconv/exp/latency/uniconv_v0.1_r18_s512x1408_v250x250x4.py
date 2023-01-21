_base_ = 'uniconv_v0.1_r18_s512x1408_v200x200x6.py'
model = dict(
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64*4*4,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    n_voxels=(250, 250, 4),
    voxel_size=[0.4, 0.4, 1.5],
)
