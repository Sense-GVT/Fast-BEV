# -*- coding: utf-8 -*-
n_voxels = (200, 200, 4)
voxel_size = [0.5, 0.5, 1.5]
# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
post_center_range = [-60, -60, -10, 60, 60, 10]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

model = dict(
    type='M2BevNetSeq',
    style="v3",
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch'
    ),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=256, out_channels=64),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64*4,
        out_channels=192,
        num_layers=2,
        stride=2,
        is_transpose=True,
        fuse=dict(in_channels=64*4*2, out_channels=64*4),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    seg_head=None,
    bbox_head=dict(
        type='CenterHead',
        in_channels=192,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']), 
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=256,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=post_center_range,
            max_num=500,
            score_threshold=0.1,
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    n_voxels=n_voxels,
    voxel_size=voxel_size,
    # model training and testing settings
    train_cfg=dict(
        point_cloud_range=point_cloud_range,
        grid_size=n_voxels,
        voxel_size=voxel_size,
        out_size_factor=2,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
    test_cfg=dict(
        pc_range=point_cloud_range[:2],
        post_center_limit_range=post_center_range,
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        out_size_factor=2,
        voxel_size=voxel_size[:2],
        pre_max_size=1000,
        post_max_size=83,
        # scale nms
        nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
        nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
        nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]]
    )
)

dataset_type = 'NuScenesMultiView_Map_Dataset2'
data_root = './data/nuscenes/'
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
    'src_size': (900, 1600),
    'input_size': (256, 704),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': (256, 704),
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
        './sweeps': 'public-1424:s3://openmmlab/datasets/detection3d/nuscenes/sweeps',
        data_root: 'public-1424:s3://openmmlab/datasets/detection3d/nuscenes/',
        './samples': 'public-1424:s3://openmmlab/datasets/detection3d/nuscenes/samples'}))

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=6, n_times=2, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D',
         with_bbox=True,
         with_label=True,
         with_bev_seg=True),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='RandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05],
        update_img2lidar=True),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels',
                                 'gt_bboxes_3d', 'gt_labels_3d',
                                 'gt_bev_seg'])]
test_pipeline = [
    dict(type='MultiViewPipeline', sequential=True, n_images=6, n_times=2, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    # dict(type='TestTimeAugImageMultiViewImage', data_config=data_config, is_train=False),
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
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            with_box2d=True,
            box_type_3d='LiDAR',
            ann_file='data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl',
            load_interval=1,
            sequential=True,
            n_times=2,
            train_adj_ids=[3],
            speed_mode='abs_velo',
            max_interval=10,
            min_interval=0,
            fix_direction=True,
            prev_only=True,
            test_adj='prev',
            test_adj_ids=[3],
            test_time_id=None,
        )
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_box2d=True,
        box_type_3d='LiDAR',
        ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl',
        load_interval=1,
        sequential=True,
        n_times=2,
        train_adj_ids=[3],
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[3],
        test_time_id=None,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_box2d=True,
        box_type_3d='LiDAR',
        ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl',
        load_interval=1,
        sequential=True,
        n_times=2,
        train_adj_ids=[3],
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[3],
        test_time_id=None,
    )
)

optimizer = dict(
    type='AdamW2',
    lr=0.0002,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False
)

total_epochs = 24 
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
evaluation = dict(interval=100)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'

load_from = 'pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'
resume_from = None
workflow = [('train', 1)]

# fp16 settings, the loss scale is specifically tuned to avoid Nan
fp16 = dict(loss_scale='dynamic')
