# -*- coding: utf-8 -*-
'''
Training Script:
    bash tools/dist_train.sh configs/fastbev/exp/paper/fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.py 4
Pay attention to : 
(1) mmdet3d->datasets->nuscenes_monocular_dataset_map_2->NuScenesMultiView_Map_Dataset2->get_data_info->get_data_info_nsmvds
(2) nsds.sequential is False, maybe set it True.

Tesing script :
    bash tools/dist_test.sh custom_cfg/fastbev_m2_export_stg123_s576x1024_v100x50x2x32_nfc32_3dc224.py  2 /mnt/share_disk/zhangyu/fastbev/work_dir/fastbev_m2_export_stg123_s576x1024_v100x50x2x32_nfc32_3dc224/epoch_12.pth --out /mnt/share_disk/zhangyu/fastbev/work_dir/fastbev_m2_export_stg123_s576x1024_v100x50x2x32_nfc32_3dc224/eval/epoch_12.pkl --eval nds

Problems:
(1) RuntimeError: DataLoader worker (pid 1137903) is killed by signal: Killed. 
'''
debug_prep  = False
debug_run=False
if debug_run:
    norm_cfg=dict(type='BN')
    samples_per_gpu = 1
    workers_per_gpu = 2
else:
    norm_cfg=dict(type='SyncBN', requires_grad=True)
    samples_per_gpu = 12
    workers_per_gpu = 8

img_size=(480, 736) # (height,width)
stage_out_idx=(1, 2, 3)
resnet_out_channel = [512, 1024, 2048]
num_out_stages = len(stage_out_idx)
neck_out_channel=64
anchor_load_check = False # Make sure created anchors are the same with loaded anchors from disk for onnx exporting.
n_times = 1
sequential = False
bev_z = 3   # bev height
bev_y = 100 # bev length/depth
bev_x = 200 # bev width 
bev_c = neck_out_channel # channels of per bev pixel 
bev_ds = 4
bev_w_ = bev_x // bev_ds
bev_h_ = bev_y // bev_ds
with_box2d = False
suffix = '_verify'
neck3d_channel = 256
embed_dim = neck3d_channel

# If point cloud range is changed, the models should also change their point cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]

model = dict(
    type='FastBEVTR',
    style="v1",
    feat2didx = -2, # feature index for transformer bev generation.
    tr_bev_h = bev_h_,
    tr_bev_w = bev_w_,
    shared_3d2d_proj = False,  # Sharing reference between xray and transformer.
    with_mlvl2d_feat=True, # return multi-level 2D image features when extracting image feature.
    img_size = img_size,
    backproject = 'inplace', # 'inplace_step', 'inplace', 'vanilla_step', 'vanilla' , 'custom_op', 'custom_export'
    bbox_head_2d = None if not with_box2d  else dict(),
    suffix=suffix , # for loading extrinsic from disk when exporting onnx 
    box_head_decode=dict(
        type='GetBboxes',
        anchor3d_head = None,
        anchor3d_path=f'debug/anchors/fastbev_anchor3d_head_py_{str(img_size[0])}x{str(img_size[1])}_{str(bev_x)}x{str(bev_y)}.json',
        coder = dict(
                type='DeltaXYZWLHRBBoxCoderExport',
                code_size=9)
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=stage_out_idx,
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'
    ),
    neck=dict(
        type='FPNFloat32',
        norm_cfg=norm_cfg,
        in_channels=resnet_out_channel,
        out_channels=neck_out_channel,
        num_outs=num_out_stages,
        upsample_cfg = dict(
            mode='bilinear', # nearest -> bilinear
            scale_factor=2,
            align_corners=False,
        )),
    neck_fuse=dict(in_channels=[neck_out_channel*num_out_stages], out_channels=[neck_out_channel]),
    msda_bev_encoder=dict(
        type='MSDABEVEncoder',
        num_query = 1250,
        bev_h = bev_h_,
        bev_w = bev_w_,
        embed_dims = embed_dim,
        pc_range = point_cloud_range,
        feat_inchs = neck_out_channel,
        num_cams = 6,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=embed_dim//2,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        transformer = dict(
 	    type='FASTBEVTRANSFORMER',
            num_feature_levels=1,
            num_cams=6,
            two_stage_num_proposals=300,
            embed_dims=embed_dim,
            use_cams_embeds=True,
            encoder=dict(
                type='FastBEVFormerEncoder',
                pc_range=point_cloud_range,
                num_points_in_pillar=bev_z,  # default 4. 
                num_layers=1,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=embed_dim,
                            num_levels=1,
                            num_points=4, #4 for accuracy, epxect smaller value for speed.
                        ),
                        dict(
                            type='FastbevSptialCrossAttention',
                            pc_range=point_cloud_range,
                            embed_dims=embed_dim,
                            batch_first = True,
                            attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=embed_dim,
                                num_levels=1,
                                num_heads=4,
                                num_points=2*bev_z, #2 for accuracy, epxect smaller value for speed.
                                im2col_step=samples_per_gpu
                            ), # TODO: Checking this setting.
                        ),],
                    feedforward_channels=embed_dim,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm'),
                    ffn_num_fcs=2,
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dim,
                        feedforward_channels=embed_dim,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                ),
            ),
        ),
    ),
    # neck_3d=dict(
    #     type='M2BevNeck',
    #     in_channels=neck_out_channel*bev_z,
    #     out_channels=neck3d_channel,
    #     num_layers=6,
    #     stride=1, # do not sample down anymore, as the bev size already be half (100x100) in view-transformer stage.
    #     is_transpose=True,
    #     fuse=dict(in_channels=neck_out_channel*bev_z*n_times, out_channels=neck_out_channel*bev_z),
    #     norm_cfg=norm_cfg),
    neck_3d=dict(
        type='M2BevFPNNeck',
        in_channels=neck_out_channel*bev_z,
        out_channels=neck3d_channel,
        num_layers=6,
        strides = (1, 1, 2, 1, 2, 1), # (200x100) -> (100x50) -> (50x25)
        out_indices = (1, 3, 5),  #  (100x50) -> (50x25)  -> (100x50)
        resize_fcs = (1, 2, 2),  # (100x50) + ((50x25) -> (100x50)) 
        is_transpose=True, # 
        fuse=dict(in_channels=neck_out_channel*bev_z*n_times, out_channels=neck_out_channel*bev_z),
        norm_cfg=norm_cfg,
        with_mlvl_bev=True),
    bev_fusion=dict(
        type='BEVBranchFusion',
        ups_fcs2 = (2,2),
        out_shape = (bev_y,bev_x),
        in_channel1 = neck3d_channel,
        in_channel2 = embed_dim,
        out_channel = neck3d_channel,
        bev_short_cut=True,
        num_last_fuse_blks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU',inplace=True),
    ),
    seg_head=None,
    bbox_head=dict(
        type='FreeAnchor3DHead',
        is_transpose=False,
        num_classes=10,
        in_channels=neck3d_channel,
        feat_channels=neck3d_channel,
        num_convs=0,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_load_check=anchor_load_check,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGeneratorExport',
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
            reshape_out=True,
            anchor_load_check=anchor_load_check),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoderExport', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    multi_scale_id=[0],
    n_voxels=[(bev_x, bev_y, bev_z)],
    voxel_size=[[100/bev_x, 100/bev_y, 6/bev_z]],
    # model training and testing settings
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
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
        use_scale_nms=True,
        use_tta=False,
        # Normal-NMS
        nms_across_levels=False,
        use_rotate_nms=True,
        nms_thr=0.2,
        # Scale-NMS
        nms_type_list=[
            'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'circle'],
        nms_thr_list=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2],
        nms_radius_thr_list=[4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1],
	nms_rescale_factor=[1.0, 0.7, 0.55, 0.4, 0.7, 1.0, 1.0, 4.5, 9.0, 1.0],
    )
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

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
    'input_size': img_size,
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': img_size,
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}


file_client_args = dict(backend='disk')
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         data_root: 'public-1984:s3://openmmlab/datasets/detection3d/nuscenes/'}))

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=sequential, n_images=6, n_times=n_times, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D', # nop
         with_bbox=with_box2d,
         with_label=with_box2d,
 with_bev_seg=with_box2d),
    dict(
        type='LoadPointsFromFile', # nop
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='RandomFlip3D', # nop
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='GlobalRotScaleTrans', # nop
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05],
        update_img2lidar=True),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', #'gt_bboxes', 'gt_labels','gt_bev_seg'
                                 'gt_bboxes_3d', 'gt_labels_3d',
                                 ])]
test_pipeline = [
    dict(type='MultiViewPipeline', sequential=sequential, n_images=6, n_times=n_times, transforms=[
        dict(
            type='LoadImageFromFile',
            file_client_args=file_client_args)]),
    dict(
        type='LoadPointsFromFile',  # nop
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False), # yes, resize and crop
    # dict(type='TestTimeAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range), # yes, to onnx and post process
    dict(type='NormalizeMultiviewImage', **img_norm_cfg), # yes
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])]

data_info_path = '/mnt/share_disk/zhangyu/bevs/fastbev/data/nuscenes/'
data_root = '/mnt/share_disk/dataset/nuscenes/nuscenes/'
# dataset_type = 'NuScenesMultiView_Map_Dataset2' # this 
dataset_type = 'NuScenesMultiView_Det_Dataset'
# data_root = './data/nuscenes/'
# data_info_path = './data/nuscenes/'
data = dict(
    samples_per_gpu=samples_per_gpu, # 12 ,8,12 
    workers_per_gpu=workers_per_gpu, # 3  ,4,6
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            with_box2d=with_box2d,
            box_type_3d='LiDAR',
            # ann_file='data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl',
            ann_file= data_info_path + 'nuscenes_infos_train.pkl',
            load_interval=1,
            sequential=sequential,
            n_times=n_times,
            train_adj_ids=[1, 3, 5],
            speed_mode='abs_velo',
            max_interval=10,
            min_interval=0,
            fix_direction=True,
            prev_only=True,
            test_adj='prev',
            test_adj_ids=[1, 3, 5],
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
        with_box2d=with_box2d,
        box_type_3d='LiDAR',
        # ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl',
        ann_file=data_info_path + 'nuscenes_infos_val.pkl',
        load_interval=1,
        sequential=sequential,
        n_times=n_times,
        train_adj_ids=[1, 3, 5],
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_box2d=with_box2d,
        box_type_3d='LiDAR',
        # ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl',
        ann_file=data_info_path + 'nuscenes_infos_val.pkl',
        load_interval=1,
        sequential=sequential,
        n_times=n_times,
        train_adj_ids=[1, 3, 5],
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None,
    )
)

optimizer = dict(
    type='AdamW2',
    lr=0.0004,
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

total_epochs = 40
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
evaluation = dict(interval=5)
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
resume_from = "/mnt/share_disk/zhangyu/fastbev/work_dir/fastbev_r50_stg123_s480x736_v200x100x3x64_nfc64_3dc256_nl6/latest.pth" # 
# load_from = "/mnt/share_disk/zhangyu/fastbev/work_dir/fastbev_r50_stg123_s480x736_v200x100x3x64_nfc64_3dc256_nl6/with_short_cut/epoch_1.pth" 
load_from = None
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
# fp16 settings, the loss scale is specifically tuned to avoid Nan
fp16 = dict(loss_scale='dynamic')
# runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
runner = dict(type='DebugEpochBasedRunner', max_epochs=total_epochs)
no_validate=True
build_dataset = False

##########################################################################################################################
# For exporting torch to onnx 
##########################################################################################################################
opset_version = 13
forward_func='forward_export_meta' # 'forward_export_bboxes', 'forward_export', 'forward_export_whole', 'forward_export_meta'
work_dir='/mnt/share_disk/zhangyu/fastbev/work_dir/'
json_file_name = f'debug/fastbev_forward_export/fastbev_1batch_6cam_1seq_{str(img_size[0])}x{str(img_size[1])}{suffix}.json' # for exporting onnx 

##########################################################################################################################
# For debuging custom 3D-NMS codes 
##########################################################################################################################
dump_nms_data = False
dump_nms_path = './debug/NMS/3DNMS/'
##########################################################################################################################
# For debuging "CUDA error: device-side assert triggered" 
##########################################################################################################################
custom_hooks=[

]

