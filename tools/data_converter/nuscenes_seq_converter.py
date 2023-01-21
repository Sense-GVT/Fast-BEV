# Copyright (c) Phigent Robotics. All rights reserved.

import pickle
from nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
import ipdb


def add_adj_info():
    interval = 3
    max_adj = 60
    sample_num = None
    for set in ['test', 'val', 'train', ]:
        if set in ['val', 'train']:
            continue
        dataset = pickle.load(open('./data/nuscenes/nuscenes_infos_%s.pkl' % set, 'rb'))
        if set in ['train', 'val']:
            nuscenes_version = 'v1.0-trainval'
        else:
            nuscenes_version = 'v1.0-test'
        dataroot = './data/nuscenes/'
        nuscenes = NuScenes(nuscenes_version, dataroot)
        map_token_to_id = dict()
        for id in range(len(dataset['infos'])):
            map_token_to_id[dataset['infos'][id]['token']] = id
            if sample_num is not None and id > sample_num:
                break
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            if sample_num is not None and id > sample_num:
                break
            info = dataset['infos'][id]
            sample = nuscenes.get('sample', info['token'])
            for adj in ['next', 'prev']:
                sweeps = []
                adj_list = dict()
                for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                    adj_list[cam] = []

                    sample_data = nuscenes.get('sample_data', sample['data'][cam])
                    adj_list[cam] = []
                    count = 0
                    while count < max_adj:
                        if sample_data[adj] == '':
                            break
                        sd_adj = nuscenes.get('sample_data', sample_data[adj])
                        sample_data = sd_adj
                        adj_list[cam].append(dict(data_path='./data/nuscenes/' + sd_adj['filename'],
                                                  timestamp=sd_adj['timestamp'],
                                                  ego_pose_token=sd_adj['ego_pose_token']))
                        count += 1
                for count in range(interval - 1, min(max_adj, len(adj_list['CAM_FRONT'])), interval):
                    timestamp_front = adj_list['CAM_FRONT'][count]['timestamp']
                    # get ego pose
                    pose_record = nuscenes.get('ego_pose', adj_list['CAM_FRONT'][count]['ego_pose_token'])

                    # get cam infos
                    cam_infos = dict(CAM_FRONT=dict(data_path=adj_list['CAM_FRONT'][count]['data_path']))
                    for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                                'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                        timestamp_curr_list = np.array([t['timestamp'] for t in adj_list[cam]], dtype=np.long)
                        diff = np.abs(timestamp_curr_list - timestamp_front)
                        selected_idx = np.argmin(diff)
                        cam_infos[cam] = dict(data_path=adj_list[cam][int(selected_idx)]['data_path'])
                        # print('%02d-%s'%(selected_idx, cam))
                    sweeps.append(dict(timestamp=timestamp_front, cams=cam_infos,
                                       ego2global_translation=pose_record['translation'],
                                       ego2global_rotation=pose_record['rotation']))
                dataset['infos'][id][adj] = sweeps if len(sweeps) > 0 else None

            # get ego speed and transfrom the targets velocity from global frame into ego-relative mode
            previous_id = id
            if not sample['prev'] == '':
                sample_tmp = nuscenes.get('sample', sample['prev'])
                previous_id = map_token_to_id[sample_tmp['token']]
            next_id = id
            if not sample['next'] == '':
                sample_tmp = nuscenes.get('sample', sample['next'])
                next_id = map_token_to_id[sample_tmp['token']]
            time_pre = 1e-6 * dataset['infos'][previous_id]['timestamp']
            time_next = 1e-6 * dataset['infos'][next_id]['timestamp']
            time_diff = time_next - time_pre
            posi_pre = np.array(dataset['infos'][previous_id]['ego2global_translation'], dtype=np.float32)
            posi_next = np.array(dataset['infos'][next_id]['ego2global_translation'], dtype=np.float32)
            velocity_global = (posi_next - posi_pre) / time_diff

            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            e2g_r = info['ego2global_rotation']
            e2g_t = info['ego2global_translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            velocity_global = np.array([*velocity_global[:2], 0.0])
            velocity_lidar = velocity_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T
            velocity_lidar = velocity_lidar[:2]

            dataset['infos'][id]['velo'] = velocity_lidar
            if set in ['train', 'val']:
                dataset['infos'][id]['gt_velocity'] = dataset['infos'][id]['gt_velocity'] - velocity_lidar.reshape(1, 2)

        filename = './data/nuscenes/nuscenes_infos_%s_4d_interval%d_max%d.pkl' % (set, interval, max_adj)
        if sample_num is not None:
            filename = filename.replace('.pkl', f'_sample{sample_num}.pkl')
        with open(filename, 'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    add_adj_info()
