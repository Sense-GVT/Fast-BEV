import numpy as np
import os
import mmcv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('--target', default='phx', help='target machine to generate anntoations')
    args = parser.parse_args()

    return args.target

machine_target = parse_args()
data_path = 'data/internal'
final_path = os.path.join(data_path, 'annotations')
os.makedirs(final_path, exist_ok=True)
if machine_target == 'phx':
    folder_list = os.listdir(data_path)
    try:
        folder_list.remove('train.json')
    except:
        pass
    try:
        folder_list.remove('val.json')
    except:
        pass
    try:
        folder_list.remove('annotations')
    except:
        pass
else:   # sh38
    seq_path = '/mnt/lustre/mig.cla/users/caijain/workspace/prepare_detr3d/manual_check_detr3d_dataset.txt'
    with open(seq_path, 'r') as f:
	    lines = f.readlines()
    folder_list = [mm.split('/')[-1].strip() for mm in lines]
    
num_seq = len(folder_list)
train_val_ratio = 0.8
train_num = int(train_val_ratio * num_seq)
val_num = num_seq - train_num

cam_names = [
    'center_camera_fov120',
    'left_front_camera',
    'left_rear_camera',
    'rear_camera',
    'right_rear_camera',
    'right_front_camera'
]

def merge_annotation(seq_list, mode='train'):
    cnt = 0
    merge_anno = dict()
    merge_anno['infos'] = []
    for seq in seq_list:
        if seq == '2021_12_21_10_57_54_AutoCollect':
            continue
        if machine_target != 'phx':
            seq = os.path.join(seq, 'detr3d_dataset')
        dataset_path = os.path.join(data_path, seq)
        anno_path = os.path.join(dataset_path, 'detr3d_train.json')
        anno = mmcv.load(anno_path)
        print(anno.keys())
        for ii in range(len(anno['infos'])):
            for cam_name in cam_names:
                anno['infos'][ii]['cams'][cam_name]['data_path'] = \
                    os.path.join(seq, anno['infos'][ii]['cams'][cam_name]['data_path'])
        merge_anno['infos'] += anno['infos']
        cnt += len(anno['infos'])
    print("%s set image num is %d" % (mode, cnt))
    mmcv.dump(merge_anno, '%s/%s.json' % (final_path, mode))

merge_annotation(folder_list[:val_num], mode='val')
merge_annotation(folder_list[val_num:], mode='train')