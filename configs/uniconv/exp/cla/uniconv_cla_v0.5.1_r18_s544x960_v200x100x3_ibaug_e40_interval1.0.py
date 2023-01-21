# -*- coding: utf-8 -*-
_base_ = 'uniconv_cla_v0.5.0_r18_s544x960_v200x100x3_e40_interval1.0.py'
class_names = [
    'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'
]

dataset_type = 'InternalDataset'
data_root = 'data/cla/annotations/'

# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-20, -25, -5, 80, 25, 3]
load_interval_train = 1
load_interval_test = 1
input_size = (544, 960)
is_debug = False

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_config = {
    'src_size': (1080, 1920),
    'input_size': input_size,
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
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
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         data_root: 'adc:s3://sh1984_datasets/'}))
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         data_root+'cla-datasets/': 'adc:s3://sh1984_datasets/cla-datasets/',
#         data_root+'detr3d/': 'adc:s3://sh1984_datasets/detr3d/',
#         data_root+'Pilot/': 'adc-1424:s3://sh1424_datasets/Pilot/'}))
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        data_root: 'zf-1424:s3://NVDATA/cla/images/'}))

train_pipeline = [
    dict(type='MultiViewPipeline', n_images=6, transforms=[
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
    dict(
        type='InternalRandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='InternalGlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.0, 0.0, 0.0],
        update_img2lidar=True),
    dict(type='InternalRandomAugImageMultiViewImage', data_config=data_config, is_debug=is_debug),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]
test_pipeline = [
    dict(type='MultiViewPipeline', n_images=6, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='InternalRandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'annotation_0000/pilot_add_061_071_081_1of2.json',
            pipeline=train_pipeline,
            classes=class_names,
            load_interval=load_interval_train,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotation_0616/val_detr3d_dataset_v0_6_1.json',
        pipeline=test_pipeline,
        classes=class_names,
        load_interval=load_interval_test,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotation_0616/val_detr3d_dataset_v0_6_1.json',
        pipeline=test_pipeline,
        classes=class_names,
        load_interval=load_interval_test,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
