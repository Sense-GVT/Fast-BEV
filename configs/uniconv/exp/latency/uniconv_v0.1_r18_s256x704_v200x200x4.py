_base_ = 'uniconv_v0.1_r18_s256x704_v200x200x6.py'
model = dict(
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64*4*4,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    n_voxels=(200, 200, 4),
    voxel_size=[0.5, 0.5, 1.5],
)
