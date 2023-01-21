# -*- coding: utf-8 -*-
_base_ = 'uniconv_cla_t0.2.2_s540x960_v425x250x6_e12_interval10.0.py'
class_names = [
    'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'
]

dataset_type = 'InternalDataset'
data_root = 'data/cla/annotations/'
is_debug = True
repeat = 20
tmp = './figs/sequential'

# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-100, -50, -5, 70, 50, 3]
input_size = (540, 960)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_config = {
    'src_size': (1080, 1920),
    'input_size': input_size,
    # # train-aug
    'resize': (0.0, 0.0),
    'crop': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    # train-aug
    # 'resize': (-0.25, 0.25),
    # 'crop': (-0.5, 0.5),
    # 'rot': (-45, 45),
    # 'flip': True,
    # test-aug
    'test_input_size': input_size,
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

# file_client_args = dict(backend='disk')
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        data_root: 'adc:s3://sh1984_datasets/'}))

train_pipeline = [
    dict(type='MultiViewPipeline', n_images=6, sequential=True, n_times=2, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D',
         with_bbox=False,
         with_label=False,
         with_bev_seg=False,
         with_attr_label=False),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    # dict(
    #     type='InternalRandomFlip3D',
    #     flip_2d=False,
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5,
    #     update_img2lidar=True),
    # dict(
    #     type='InternalGlobalRotScaleTrans',
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.75, 1.25],
    #     translation_std=[0.0, 0.0, 0.0],
    #     update_img2lidar=True),
    # dict(type='InternalRandomAugImageMultiViewImage', data_config=data_config, is_debug=is_debug, tmp=tmp),
    dict(type='InternalRandomAugImageMultiViewImageDebug', data_config=data_config, is_debug=is_debug, tmp=tmp, repeat=repeat),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]
test_pipeline = [
    dict(type='MultiViewPipeline', n_images=6, sequential=True, n_times=2, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    # dict(type='InternalRandomAugImageMultiViewImage', data_config=data_config, is_debug=is_debug, tmp=tmp),
    dict(type='InternalRandomAugImageMultiViewImageDebug', data_config=data_config, is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        dataset=dict(
            data_root=data_root,
            ann_file=data_root + 'annotation_0616/sweeps/val_detr3d_with_seq_dataset.json',
            pipeline=train_pipeline,
            sequential=True,
            n_times=2,
            speed_mode='abs_velo',
            prev_only=True,
            train_adj_ids=[0],
            test_adj_ids=[0],
            verbose=True,
        )
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'annotation_0616/sweeps/val_detr3d_with_seq_dataset.json',
        pipeline=test_pipeline,
        sequential=True,
        n_times=2,
        speed_mode='abs_velo',
        prev_only=True,
        train_adj_ids=[0],
        test_adj_ids=[0],
        verbose=True,
    ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'annotation_0616/sweeps/val_detr3d_with_seq_dataset.json',
        pipeline=test_pipeline,
        sequential=True,
        n_times=2,
        speed_mode='abs_velo',
        prev_only=True,
        train_adj_ids=[0],
        test_adj_ids=[0],
        verbose=True,
    )
)
