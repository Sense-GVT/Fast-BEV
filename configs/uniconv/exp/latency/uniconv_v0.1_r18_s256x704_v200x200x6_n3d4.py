_base_ = 'uniconv_v0.1_r18_s256x704_v200x200x6.py'
model = dict(
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64*6*4,
        out_channels=256,
        num_layers=4,
        stride=2,
        is_transpose=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
)
