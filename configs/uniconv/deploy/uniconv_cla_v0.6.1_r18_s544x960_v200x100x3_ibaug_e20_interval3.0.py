class_names = ['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN']
dataset_type = 'InternalDataset'
data_root = 'data/cla/annotations/'
point_cloud_range = [-20, -25, -5, 80, 25, 3]
load_interval_train = 1
load_interval_test = 1
input_size = (544, 960)
is_debug = False
neck_2d_channel = 64
neck_3d_channel = 64
n_voxels = (200, 100, 3)
n_z = 3
voxel_size = [0.5, 0.5, 1.5]
anchor_range = [-20, -25, -5, 80, 25, 3]
with_cp = False
find_unused_parameters = True
lr = 0.0008
total_epochs = 40
load_from = 'pretrained_models/cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth'
model = dict(
    type='M2BevNet',
    style='v3',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch',
        with_cp=False),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=256, out_channels=64),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=192,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        norm_cfg=dict(type='BN', requires_grad=True)),
    seg_head=None,
    bbox_head=dict(
        type='FreeAnchor3DHead',
        is_transpose=True,
        num_classes=4,
        in_channels=256,
        feat_channels=256,
        num_convs=0,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-20, -25, -5, 80, 25, 3]],
            sizes=[[0.866, 2.5981, 1.0], [0.5774, 1.7321, 1.0],
                   [1.0, 1.0, 1.0], [0.4, 0.4, 1]],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    n_voxels=(200, 100, 3),
    voxel_size=[0.5, 0.5, 1.5],
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.2,
        score_thr=0.05,
        min_bbox_size=0,
        max_num=500))
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_config = dict(
    src_size=(1080, 1920),
    input_size=(544, 960),
    resize=(0.0, 0.0),
    crop=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    test_input_size=(544, 960),
    test_resize=0.0,
    test_rotate=0.0,
    test_flip=False,
    pad=(0, 0, 0, 0),
    pad_divisor=32,
    pad_color=(0, 0, 0))
file_client_args = dict(
    backend='petrel',
    path_mapping=dict(
        {'data/cla/annotations/': 'zf-1424:s3://NVDATA/cla/images/'}))
train_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(
                    backend='petrel',
                    path_mapping=dict({
                        'data/cla/annotations/':
                        'zf-1424:s3://NVDATA/cla/images/'
                    })))
        ]),
    dict(
        type='LoadAnnotations3D',
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
        type='InternalRandomAugImageMultiViewImage',
        data_config=dict(
            src_size=(1080, 1920),
            input_size=(544, 960),
            resize=(0.0, 0.0),
            crop=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            test_input_size=(544, 960),
            test_resize=0.0,
            test_rotate=0.0,
            test_flip=False,
            pad=(0, 0, 0, 0),
            pad_divisor=32,
            pad_color=(0, 0, 0)),
        is_debug=False),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[-20, -25, -5, 80, 25, 3]),
    dict(type='KittiSetOrigin', point_cloud_range=[-20, -25, -5, 80, 25, 3]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'
        ]),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(
                    backend='petrel',
                    path_mapping=dict({
                        'data/cla/annotations/':
                        'zf-1424:s3://NVDATA/cla/images/'
                    })))
        ]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='InternalRandomAugImageMultiViewImage',
        data_config=dict(
            src_size=(1080, 1920),
            input_size=(544, 960),
            resize=(0.0, 0.0),
            crop=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            test_input_size=(544, 960),
            test_resize=0.0,
            test_rotate=0.0,
            test_flip=False,
            pad=(0, 0, 0, 0),
            pad_divisor=32,
            pad_color=(0, 0, 0)),
        is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=[-20, -25, -5, 80, 25, 3]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='InternalDataset',
            data_root='data/cla/annotations/',
            ann_file=
            'data/cla/annotations/annotation_0000/pilot_add_061_071_081_1of2.json',
            pipeline=[
                dict(
                    type='MultiViewPipeline',
                    n_images=6,
                    transforms=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(
                                backend='petrel',
                                path_mapping=dict({
                                    'data/cla/annotations/':
                                    'zf-1424:s3://NVDATA/cla/images/'
                                })))
                    ]),
                dict(
                    type='LoadAnnotations3D',
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
                    type='InternalRandomAugImageMultiViewImage',
                    data_config=dict(
                        src_size=(1080, 1920),
                        input_size=(544, 960),
                        resize=(0.0, 0.0),
                        crop=(0.0, 0.0),
                        rot=(0.0, 0.0),
                        flip=False,
                        test_input_size=(544, 960),
                        test_resize=0.0,
                        test_rotate=0.0,
                        test_flip=False,
                        pad=(0, 0, 0, 0),
                        pad_divisor=32,
                        pad_color=(0, 0, 0)),
                    is_debug=False),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-20, -25, -5, 80, 25, 3]),
                dict(
                    type='KittiSetOrigin',
                    point_cloud_range=[-20, -25, -5, 80, 25, 3]),
                dict(
                    type='NormalizeMultiviewImage',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE',
                        'PEDESTRIAN'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            classes=[
                'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'
            ],
            load_interval=1,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=True),
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='InternalDataset',
        data_root='data/cla/annotations/',
        ann_file=
        'data/cla/annotations/annotation_0616/val_detr3d_dataset_v0_6_1.json',
        pipeline=[
            dict(
                type='MultiViewPipeline',
                n_images=6,
                transforms=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(
                            backend='petrel',
                            path_mapping=dict({
                                'data/cla/annotations/':
                                'zf-1424:s3://NVDATA/cla/images/'
                            })))
                ]),
            dict(
                type='LoadPointsFromFile',
                dummy=True,
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(
                type='InternalRandomAugImageMultiViewImage',
                data_config=dict(
                    src_size=(1080, 1920),
                    input_size=(544, 960),
                    resize=(0.0, 0.0),
                    crop=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    test_input_size=(544, 960),
                    test_resize=0.0,
                    test_rotate=0.0,
                    test_flip=False,
                    pad=(0, 0, 0, 0),
                    pad_divisor=32,
                    pad_color=(0, 0, 0)),
                is_train=False),
            dict(
                type='KittiSetOrigin',
                point_cloud_range=[-20, -25, -5, 80, 25, 3]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE',
                    'PEDESTRIAN'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ],
        classes=['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'],
        load_interval=1,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='InternalDataset',
        data_root='data/cla/annotations/',
        ann_file=
        'data/cla/annotations/annotation_0616/val_detr3d_dataset_v0_6_1.json',
        pipeline=[
            dict(
                type='MultiViewPipeline',
                n_images=6,
                transforms=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(
                            backend='petrel',
                            path_mapping=dict({
                                'data/cla/annotations/':
                                'zf-1424:s3://NVDATA/cla/images/'
                            })))
                ]),
            dict(
                type='LoadPointsFromFile',
                dummy=True,
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(
                type='InternalRandomAugImageMultiViewImage',
                data_config=dict(
                    src_size=(1080, 1920),
                    input_size=(544, 960),
                    resize=(0.0, 0.0),
                    crop=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    test_input_size=(544, 960),
                    test_resize=0.0,
                    test_rotate=0.0,
                    test_flip=False,
                    pad=(0, 0, 0, 0),
                    pad_divisor=32,
                    pad_color=(0, 0, 0)),
                is_train=False),
            dict(
                type='KittiSetOrigin',
                point_cloud_range=[-20, -25, -5, 80, 25, 3]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE',
                    'PEDESTRIAN'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ],
        classes=['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'],
        load_interval=1,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR'))
optimizer = dict(
    type='AdamW2',
    lr=0.0008,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(interval=100)
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
work_dir = 'work_dirs/uniconv/exp/cla/uniconv_cla_v0.6.1_r18_s544x960_v200x100x3_ibaug_e20_interval3.0'
gpu_ids = range(0, 1)
