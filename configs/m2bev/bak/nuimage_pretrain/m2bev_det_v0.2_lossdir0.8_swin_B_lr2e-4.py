# -*- coding: utf-8 -*-
# debug专用参数

# debug专用参数
load_from = './work_dirs/cascade_mask_rcnn_swin_B_fpn_coco-20e_20e_nuim/latest.pth'
fp16 = dict(loss_scale='dynamic')


model = dict(
    type='ImVoxelNet_MAP_4in1',
    backbone=dict(
        type='SwinTransformer',
        frozen_stages=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=64,
        relu_before_extra_convs=True,
        num_outs=4),
    neck_fuse=dict(in_channels=64*4, out_channels=64),
    neck_3d=dict(
        type='NuScenesImVoxelNeck',
        in_channels=64,
        out_channels=256),
    seg_head=None,
    bbox_head=dict(
        type='FreeAnchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            # scales=[1, 2, 4],
            sizes=[
                [0.8660, 2.5981, 1.],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.],  # 1/sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    
    n_voxels=(400, 400, 12),
    voxel_size=[0.25, 0.25, 0.5],
    # model training and testing settings
    train_cfg = dict(
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
    test_cfg = dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=2000,
        nms_thr=0.2,
        score_thr=0.005,
        min_bbox_size=0,
        max_num=500))



# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
dataset_type = 'NuScenesMultiView_Map_Dataset'
data_root = 'data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadAnnotations3D', with_bev_seg=True),
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bev_seg'])]
test_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
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
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
                     'absolute_pos_embed': dict(decay_mult=0.),
                     'relative_position_bias_table': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.)
                    }))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    step=[9,11]
)

total_epochs = 12

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='3d-det')), 
    ])

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]

