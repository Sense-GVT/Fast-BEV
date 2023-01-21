_base_ = 'uniconv_v0.1_r18_s256x704_v200x200x6.py'
model = dict(
    n_voxels=(100, 100, 6),
    voxel_size=[1.0, 1.0, 1.0],
)
