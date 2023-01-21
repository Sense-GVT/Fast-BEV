_base_ = 'uniconv_v0.1_r18_s256x704_v200x200x6.py'
model = dict(
    n_voxels=(125, 125, 6),
    voxel_size=[0.8, 0.8, 1.0],
)
