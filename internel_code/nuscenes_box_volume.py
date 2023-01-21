from mmdet3d.datasets import build_dataset
from tqdm import tqdm
import argparse
import torch
import time
import os 


parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=1)


if __name__ == '__main__':
    args = parser.parse_args()
    rank = args.rank
    world_size = args.world_size

    point_cloud_range = [-50, -50, -5, 50, 50, 3]
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    dataset_type = 'NuScenesMultiView_Map_Dataset2'
    data_root = './data/nuscenes/'

    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    data_config = {
        'src_size': (900, 1600),
        'input_size': (320, 880),
        # train-aug
        'resize': (-0.06, 0.11),
        'crop': (-0.05, 0.05),
        'rot': (-5.4, 5.4),
        'flip': True,
        # test-aug
        'test_input_size': (320, 880),
        'test_resize': 0.0,
        'test_rotate': 0.0,
        'test_flip': False,
        # top, right, bottom, left
        'pad': (0, 0, 0, 0),
        'pad_divisor': 32,
        'pad_color': (0, 0, 0),
    }

    file_client_args = dict(
        backend='petrel',
        path_mapping=dict({
            data_root: 'public-1424:s3://openmmlab/datasets/detection3d/nuscenes/'}))

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

    input_modality = dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=True)

    data = dict(
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
            ann_file='./data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl',
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
    )

    dataset = build_dataset(data)
    stat = {'0-1': 0, '1-2': 0, '2-4': 0, '4-6': 0, '6-8': 0, '8-10': 0, '10-12': 0, '12-14': 0, '14-16': 0, '16-18': 0, '18-20':0, '>20': 0, 'total': 0}
    num_data = len(dataset)
    num_patch = num_data // world_size
    start = rank * num_patch
    end = (rank + 1) * num_patch
    if rank == world_size - 1:
        end = num_data

    with tqdm(total=(end - start)) as pbar:
        for i in range(start, end):
            data_i = dataset.__getitem__(i)
            gt_bbox = data_i['gt_bboxes_3d'].data
            volumes = gt_bbox.volume.cpu().numpy().tolist()

            for volume in volumes:
                stat['total'] += 1
                if 0 < volume <= 1:
                    stat['0-1'] += 1
                elif 1 < volume <= 2:
                    stat['1-2'] += 1
                elif 2 < volume <= 4:
                    stat['2-4'] += 1
                elif 4 < volume <= 6:
                    stat['4-6'] += 1
                elif 6 < volume <= 8:
                    stat['6-8'] += 1
                elif 8 < volume <= 10:
                    stat['8-10'] += 1
                elif 10 < volume <= 12:
                    stat['10-12'] += 1
                elif 12 < volume <= 14:
                    stat['12-14'] += 1
                elif 14 < volume <= 16:
                    stat['14-16'] += 1
                elif 16 < volume <= 18:
                    stat['16-18'] += 1
                elif 18 < volume <= 20:
                    stat['18-20'] += 1
                elif volume > 20:
                    stat['>20'] += 1
            pbar.update(1)

    torch.save(stat, f'/mnt/lustre/chenzeren/stat_rank{rank}.pth')
    if rank == 0 and world_size > 1:
        while True:
            time.sleep(1)
            flag = True
            for rank_i in range(world_size):
                if not os.path.exists(f'/mnt/lustre/chenzeren/stat_rank{rank}.pth'):
                    flag = False
            if flag:
                break

        stat = {'0-1': 0, '1-2': 0, '2-4': 0, '4-6': 0, '6-8': 0, '8-10': 0, '10-12': 0, '12-14': 0, '14-16': 0, '16-18': 0, '18-20':0, '>20': 0, 'total': 0}
        
        for rank_i in range(world_size):
            stat_i = torch.load(f'/mnt/lustre/chenzeren/stat_rank{rank_i}.pth', 'cpu')
            for k, v in stat_i.items():
                stat[k] += v
            os.remove(f'/mnt/lustre/chenzeren/stat_rank{rank_i}.pth')

        torch.save(stat, f'/mnt/lustre/chenzeren/stat_nuscenes.pth')
    
    print('all done')
