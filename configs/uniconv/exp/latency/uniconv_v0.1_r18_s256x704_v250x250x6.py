_base_ = 'uniconv_v0.1_r18_s256x704_v200x200x6.py'
model = dict(
    n_voxels=(250, 250, 6),
    voxel_size=[0.4, 0.4, 1.0],
)
