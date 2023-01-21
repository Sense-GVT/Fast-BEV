import numpy as np
from mmdet.datasets import DATASETS
import os
import torch
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
import mmcv
import cv2
import os.path as osp
from tqdm import tqdm
from pyquaternion import Quaternion
import copy
import random

import ipdb # noqa


@DATASETS.register_module()
class InternalDataset(Custom3DDataset):
    r"""Internal Dataset.
    """
    CLASSES = ('VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN')
    cams = [
        'center_camera_fov120', 'left_front_camera', 'left_rear_camera',
        'rear_camera', 'right_rear_camera', "right_front_camera"
    ]
    lidar2img_rts = []

    def __init__(self,
                 data_root,
                 ann_file=None,
                 pipeline=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 test_mode=False,
                 box_type_3d='LiDAR',
                 shuffle=False,
                 seed=0,
                 sequential=False,
                 n_times=2,
                 speed_mode='abs_velo',
                 prev_only=False,
                 next_only=False,
                 train_adj_ids=None,
                 test_adj_ids=None,
                 max_interval=3,
                 min_interval=0,
                 verbose=False):

        self.data_root = data_root
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            print(f"Building a shuffle dataset with seed {self.seed}")
        self.sequential = sequential
        self.n_times = n_times
        self.speed_mode = speed_mode
        self.prev_only = prev_only
        self.next_only = next_only
        self.train_adj_ids = train_adj_ids
        self.test_adj_ids = test_adj_ids
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.verbose = verbose

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            test_mode=test_mode,
            box_type_3d=box_type_3d)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(data_infos)
        return data_infos

    def pre_get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            timestamp=info['timestamp'],
            sample_idx=index,
        )
        center2lidar = np.matrix(info['center2lidar'])
        image_paths = []
        lidar2img_rts = []
        lidar2img_augs = []
        for cam_type, cam_info in info['cams'].items():
            img_path = os.path.join(self.data_root, cam_info['data_path'])
            image_paths.append(img_path)

            # obtain lidar to image transformation matrix
            intrinsic = np.array(cam_info['cam_intrinsic']).reshape(3, 3)  # 3x3

            extrinsic = np.array(cam_info['extrinsic']).reshape(4, 4)  # 4x4
            lidar2cam_rt = extrinsic @ center2lidar
            lidar2cam_rt = lidar2cam_rt.T

            # keep aug rts
            lidar2img_aug = {
                'intrin': np.array(cam_info['cam_intrinsic']).reshape(3, 3),
                'lidar2cam_rt': lidar2cam_rt,
                'post_rot': np.eye(3),
                'post_tran': np.zeros(3),

            }
            if self.sequential:
                lidar2img_aug.update({
                    'rot': np.array(cam_info['sensor2lidar_rotation']),
                    'tran': np.array(cam_info['sensor2lidar_translation']),
                })

            lidar2img_augs.append(lidar2img_aug)

            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            lidar2img_rt = np.array((viewpad @ lidar2cam_rt.T))
            lidar2img_rts.append(lidar2img_rt)

        if self.sequential:
            adjacent_type_list = []
            adjacent_id_list = []
            for time_id in range(1, self.n_times):
                if info['prev'] is None or info['next'] is None:
                    adjacent = 'prev' if info['next'] is None else 'next'
                else:
                    if self.prev_only or self.next_only:
                        adjacent = 'prev' if self.prev_only else 'next'
                    # stage: test
                    elif self.test_mode:
                        if self.test_adj_ids is not None:
                            assert len(self.test_adj_ids) == self.n_times - 1
                            select_id = self.test_adj_ids[time_id-1]
                            assert self.min_interval <= select_id <= self.max_interval
                            adjacent = {True: 'prev', False: 'next'}[select_id > 0]
                        else:
                            adjacent = self.test_adj
                    # stage: train
                    elif self.train_adj_ids is not None:
                        assert len(self.train_adj_ids) == self.n_times - 1
                        select_id = self.train_adj_ids[time_id-1]
                        assert self.min_interval <= select_id <= self.max_interval
                        adjacent = {True: 'prev', False: 'next'}[select_id > 0]
                    else:
                        adjacent = np.random.choice(['prev', 'next'])

                if type(info[adjacent]) is list:
                    # stage: test
                    if self.test_mode:
                        if len(info[adjacent]) <= self.min_interval:
                            select_id = len(info[adjacent]) - 1
                        elif self.test_adj_ids is not None:
                            assert len(self.test_adj_ids) == self.n_times - 1
                            select_id = self.test_adj_ids[time_id-1]
                            assert self.min_interval <= select_id <= self.max_interval
                            select_id = min(abs(select_id), len(info[adjacent])-1)
                        else:
                            assert self.min_interval >= 0 and self.max_interval >= 0, "single direction only here"
                            select_id_step = (self.max_interval+self.min_interval) // self.n_times
                            select_id = min(self.min_interval + select_id_step * time_id, len(info[adjacent])-1)
                    # stage: train
                    else:
                        if len(info[adjacent]) <= self.min_interval:
                            select_id = len(info[adjacent]) - 1
                        elif self.train_adj_ids is not None:
                            assert len(self.train_adj_ids) == self.n_times - 1
                            select_id = self.train_adj_ids[time_id-1]
                            assert self.min_interval <= select_id <= self.max_interval
                            select_id = min(abs(select_id), len(info[adjacent])-1)
                        else:
                            assert self.min_interval >= 0 and self.max_interval >= 0, "single direction only here"
                            select_id = np.random.choice([adj_id for adj_id in range(
                                min(self.min_interval, len(info[adjacent])),
                                min(self.max_interval, len(info[adjacent])))])
                    info_adj = info[adjacent][select_id]
                    if self.verbose:
                        print(' get_data_info: ', 'time_id: ', time_id, adjacent, select_id)
                else:
                    info_adj = info[adjacent]

                adjacent_type_list.append(adjacent)
                adjacent_id_list.append(select_id)

                egocurr2global = np.eye(4, dtype=np.float32)
                egocurr2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
                egocurr2global[:3, 3] = info['ego2global_translation']

                egoadj2global = np.eye(4, dtype=np.float32)
                egoadj2global[:3, :3] = Quaternion(info_adj['center_camera_fov120']['ego2global_rotation']).rotation_matrix
                egoadj2global[:3, 3] = info_adj['center_camera_fov120']['ego2global_translation']

                egoadj2egocurr = np.linalg.inv(egocurr2global) @ egoadj2global

                for cam_id, (cam_type, cam_info) in enumerate(info_adj.items()):

                    img_path = os.path.join(self.data_root, cam_info['data_path'])
                    image_paths.append(img_path)

                    lidar2img_aug = lidar2img_augs[cam_id].copy()

                    mat = np.eye(4, dtype=np.float32)
                    mat[:3, :3] = lidar2img_aug['rot']
                    mat[:3, 3] = lidar2img_aug['tran']
                    mat = egoadj2egocurr @ mat
                    lidar2img_aug['rot'] = mat[:3, :3]
                    lidar2img_aug['tran'] = mat[:3, 3]

                    lidar2cam_r = np.linalg.inv(lidar2img_aug['rot'])
                    lidar2cam_t = lidar2img_aug['tran'] @ lidar2cam_r.T

                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t

                    lidar2img_aug['lidar2cam_rt'] = lidar2cam_rt
                    lidar2img_augs.append(lidar2img_aug)

                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = lidar2img_aug['intrin']

                    lidar2img_rt = np.array((viewpad @ lidar2cam_rt.T))
                    lidar2img_rts.append(lidar2img_rt)

            if self.verbose:
                time_list = [0.0]
                for i in range(self.n_times-1):
                    time = 1e-6 * (info['timestamp'] - info[adjacent_type_list[i]][adjacent_id_list[i]]['center_camera_fov120']['timestamp'])
                    time_list.append(time)
                print(' get_data_info: ', 'time: ', time_list)

            info['adjacent_type'] = adjacent_type_list
            info['adjacent_id'] = adjacent_id_list

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                lidar2img_aug=lidar2img_augs,
            )
        )
        if self.sequential:
            input_dict.update(dict(info=info))

        return input_dict

    def get_data_info(self, index):
        data_info = self.pre_get_data_info(index)
        n_cameras = len(data_info['img_filename'])
        new_info = dict(
            sample_idx=data_info['sample_idx'],
            img_prefix=[None] * n_cameras,
            img_info=[dict(filename=x) for x in data_info['img_filename']],
            lidar2img=dict(
                extrinsic=[x.astype(np.float32) for x in data_info['lidar2img']],
                intrinsic=np.eye(4, dtype=np.float32),
                lidar2img_aug=data_info['lidar2img_aug'],
            )
        )
        if not self.test_mode:
            annos = self.get_ann_info(index)
            if self.sequential:
                bbox = annos['gt_bboxes_3d'].tensor
                annos['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                    bbox, box_dim=bbox.shape[-1], origin=(0.5, 0.5, 0.0))
            new_info['ann_info'] = annos
        return new_info

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = np.array(info['gt_boxes'])
        gt_names_3d = info['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info.get('gt_velocity', np.zeros((gt_bboxes_3d.shape[0], 2)))
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d
        )
        return anns_results

    def evaluate(self, results, vis_mode=False, *args, **kwargs):
        eval_seg = 'bev_seg' in results[0]
        eval_det = 'boxes_3d' in results[0]
        assert eval_seg is True or eval_det is True

        new_bevseg_results = None
        new_det_results = None
        if eval_seg:
            new_bevseg_results = []
            new_bevseg_gts_road, new_bevseg_gts_lane = [], []
        if eval_det:
            new_det_results = []

        for i in range(len(results)):
            if eval_det:
                box_type = type(results[i]['boxes_3d'])
                boxes_3d = results[i]['boxes_3d'].tensor
                boxes_3d = box_type(boxes_3d, box_dim=9, origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
                new_det_results.append(dict(
                    boxes_3d=boxes_3d,
                    scores_3d=results[i]['scores_3d'],
                    labels_3d=results[i]['labels_3d']))

            if eval_seg:
                assert results[i]['bev_seg'].shape[0] == 1
                seg_pred = results[i]['bev_seg'][0]
                seg_pred_road, seg_pred_lane = seg_pred[0], seg_pred[1]
                seg_pred_road = (seg_pred_road.sigmoid() > 0.5).int().data.cpu().numpy()
                seg_pred_lane = (seg_pred_lane.sigmoid() > 0.5).int().data.cpu().numpy()

                new_bevseg_results.append(dict(seg_pred_road=seg_pred_road,
                                               seg_pred_lane=seg_pred_lane))

                # bev seg gt path
                seg_gt_path = 'data/nuscenes/maps_bev_seg_gt_2class/'
                if not mmcv.is_filepath(seg_gt_path):
                    # online generate map, too slow
                    if i == 0:
                        print('### first time need generate bev seg map online ###')
                        print('### bev seg map is saved at:{} ###'.format(seg_gt_path))
                    sample_token = self.get_data_info(i)['sample_idx']
                    seg_gt = self._get_map_by_sample_token(sample_token)
                    seg_gt_road, seg_gt_lane = seg_gt[..., 0], seg_gt[..., 1]
                    mmcv.imwrite(seg_gt_road, seg_gt_path+'road/{}.png'.format(i), auto_mkdir=True)
                    mmcv.imwrite(seg_gt_lane, seg_gt_path+'lane/{}.png'.format(i), auto_mkdir=True)

                # load gt from local machine
                seg_gt_road = mmcv.imread(seg_gt_path+'road/{}.png'.format(i), flag='grayscale').astype('float64')
                seg_gt_lane = mmcv.imread(seg_gt_path+'lane/{}.png'.format(i), flag='grayscale').astype('float64')
                new_bevseg_gts_road.append(seg_gt_road)
                new_bevseg_gts_lane.append(seg_gt_lane)

        if vis_mode:
            print('### vis nus test data ###')
            self.show(new_det_results, 'figs', bev_seg_results=new_bevseg_results, thr=0.3, fps=3)
            print('### finish vis ###')
            exit()

        result_dict = dict()
        if eval_det:
            # eval detection
            result_dict = super().evaluate(new_det_results, *args, **kwargs)
        if eval_seg:
            # eval segmentation
            bev_res_dict = self.evaluate_bev(new_bevseg_results,
                                             new_bevseg_gts_road,
                                             new_bevseg_gts_lane)
            for k in bev_res_dict.keys():
                result_dict[k] = bev_res_dict[k]
        return result_dict

    def bev_to_corners(self, bev):
        n = bev.shape[0]
        bev[:, -1] = -bev[:, -1]
        corners = torch.stack((
            0.5 * bev[:, 2] * torch.cos(bev[:, -1]) - 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            0.5 * bev[:, 2] * torch.sin(bev[:, -1]) + 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            0.5 * bev[:, 2] * torch.cos(bev[:, -1]) + 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            0.5 * bev[:, 2] * torch.sin(bev[:, -1]) - 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            -0.5 * bev[:, 2] * torch.cos(bev[:, -1]) + 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            -0.5 * bev[:, 2] * torch.sin(bev[:, -1]) - 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
            -0.5 * bev[:, 2] * torch.cos(bev[:, -1]) - 0.5 * bev[:, 3] * torch.sin(bev[:, -1]) + bev[:, 0],
            -0.5 * bev[:, 2] * torch.sin(bev[:, -1]) + 0.5 * bev[:, 3] * torch.cos(bev[:, -1]) + bev[:, 1],
        ))
        corners = corners.reshape(4, 2, n).permute(2, 0, 1)
        return corners

    def draw_bev_result(self, img, pred_bev_corners, gt_bev_corners):
        bev_size = 1600
        scale = 10

        if img is None:
            img = np.zeros((bev_size, bev_size, 3))

        # draw circle
        for i in range(bev_size//(20*scale)):
            cv2.circle(img, (bev_size//2,bev_size//2), (i+1)*10*scale, (125,217,233), 2)
            if i == 4:
                cv2.circle(img, (bev_size//2,bev_size//2), (i+1)*10*scale, (255,255,255), 2)

        if gt_bev_corners is not None:
            gt_bev_buffer = copy.deepcopy(gt_bev_corners)
            gt_bev_corners[:, :, 0] = -gt_bev_buffer[:, :, 1] * scale + bev_size//2
            gt_bev_corners[:, :, 1] = -gt_bev_buffer[:, :, 0] * scale + bev_size//2

            gt_color = (61, 102, 255)
            for corners in gt_bev_corners:
                cv2.line(img, (int(corners[0, 0]), int(corners[0, 1])), (int(corners[1, 0]), int(corners[1, 1])), gt_color, 4)
                cv2.line(img, (int(corners[1, 0]), int(corners[1, 1])), (int(corners[2, 0]), int(corners[2, 1])), gt_color, 4)
                cv2.line(img, (int(corners[2, 0]), int(corners[2, 1])), (int(corners[3, 0]), int(corners[3, 1])), gt_color, 4)
                cv2.line(img, (int(corners[3, 0]), int(corners[3, 1])), (int(corners[0, 0]), int(corners[0, 1])), gt_color, 4)

        if pred_bev_corners is not None:
            pred_bev_buffer = copy.deepcopy(pred_bev_corners)
            pred_bev_corners[:, :, 0] = -pred_bev_buffer[:, :, 1] * scale + bev_size//2
            pred_bev_corners[:, :, 1] = -pred_bev_buffer[:, :, 0] * scale + bev_size//2
            pred_color = (241, 101, 72)
            for corners in pred_bev_corners:
                cv2.line(img, (int(corners[0, 0]), int(corners[0, 1])), (int(corners[1, 0]), int(corners[1, 1])), pred_color, 3)
                cv2.line(img, (int(corners[1, 0]), int(corners[1, 1])), (int(corners[2, 0]), int(corners[2, 1])), pred_color, 3)
                cv2.line(img, (int(corners[2, 0]), int(corners[2, 1])), (int(corners[3, 0]), int(corners[3, 1])), pred_color, 3)
                cv2.line(img, (int(corners[3, 0]), int(corners[3, 1])), (int(corners[0, 0]), int(corners[0, 1])), pred_color, 3)

        return img

    def draw_bev_lidar(self, img, points):
        bev_size = 1600
        scale = 10
        if img is None:
            img = np.zeros((bev_size, bev_size, 3))
        # draw circle
        for i in range(bev_size//(20*scale)):
            cv2.circle(img, (bev_size//2, bev_size//2), (i+1)*10*scale, (125, 217, 233), 2)

        idx = (points[:, 0] < bev_size // (2*scale)) &\
              (points[:, 0] > -bev_size // (2*scale)) &\
              (points[:, 1] < bev_size // (2*scale)) &\
              (points[:, 1] > -bev_size // (2*scale))
        points = points[idx]

        points[:, 0] = points[:, 0] * scale + bev_size//2
        points[:, 1] = -points[:, 1] * scale + bev_size//2
        for i in range(0, len(points), 10):
            point = points[i]
            cv2.circle(img, (int(point[0]), int(point[1])), 1, (255,255,255), 1)

        return img

    def show_bev(self, results, out_dir, pipeline=None):
        pipeline = self._get_pipeline(pipeline)
        for i, result in tqdm(list(enumerate(results))):
            data_info = self.data_infos[i]
            file_name = data_info

            pred_bboxes = result['boxes_3d']
            scores = result['scores_3d']

            idx = scores > 0.2
            pred_bboxes = pred_bboxes[idx]

            pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)

            img = self.draw_bev_result(None, pred_bev_corners, None)
            save_path = osp.join(out_dir, file_name + '.png')
            if img is not None:
                mmcv.imwrite(img, save_path)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in tqdm(list(enumerate(results))):
            data_info = self.data_infos[i]
            result_path = osp.join(out_dir, data_info[:-4])
            pred_bboxes = result['boxes_3d']
            scores = result['scores_3d']
            idx = scores > 0.2
            pred_bboxes = pred_bboxes[idx]

            mmcv.mkdir_or_exist(result_path)
            for num, cam in enumerate(self.cams):
                img_path = osp.join(self.data_root, "calib_images", cam, data_info)
                img = mmcv.imread(img_path)
                file_name = osp.split(img_path)[-1].split('.')[0]

                # need to transpose channel to first dim
                # anno_info = self.get_ann_info(i)
                # gt_bboxes = anno_info['gt_bboxes_3d']
                pred_bbox_color = (241, 101, 72)
                img = draw_lidar_bbox3d_on_img(
                    copy.deepcopy(pred_bboxes), img, self.lidar2img_rts[num], None, color=pred_bbox_color,thickness=3)
                mmcv.imwrite(img, osp.join(result_path, f'{cam}_pred.png'))

    def show_panorama(self, results, out_dir, pipeline=None, show_gt=True, show_pred=True, sample_rate=1, video_writer=None):
        pipeline = self._get_pipeline(pipeline)
        file_client = mmcv.FileClient(backend='petrel')
        for i, result in tqdm(list(enumerate(results))):
            if i % sample_rate != 0:
                continue
            data_info = self.data_infos[i]
            anno_info = self.get_ann_info(i)
            file_name = str(data_info['timestamp'])
            center2lidar = np.matrix(data_info['center2lidar'])
            gt_bboxes = anno_info['gt_bboxes_3d']
            pred_bboxes = result['boxes_3d']

            gt_bboxes.tensor[:, -1] = -gt_bboxes.tensor[:, -1]
            pred_bboxes.tensor[:, 6] = -pred_bboxes.tensor[:, 6]

            scores = result['scores_3d']
            idx = scores > 0.4
            pred_bboxes = pred_bboxes[idx]

            gt_bev_corners = self.bev_to_corners(gt_bboxes.bev)
            pred_bev_corners = self.bev_to_corners(pred_bboxes.bev)

            bev_img = self.draw_bev_result(None, pred_bev_corners, gt_bev_corners)
            cam_imgs = []

            for num, cam in enumerate(self.cams):
                img_path = data_info['cams'][cam]['data_path']

                img_bytes = file_client.get('zf-1424:s3://NVDATA/cla/images/' + img_path)
                img = mmcv.imfrombytes(img_bytes)
                cam_info = data_info['cams'][cam]

                intrinsic = np.matrix(cam_info['cam_intrinsic'])
                extrinsic = np.matrix(cam_info['extrinsic'])
                extrinsic = extrinsic @ center2lidar

                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = np.array((viewpad @ extrinsic))

                gt_bbox_color = (61, 102, 255)  # red
                pred_bbox_color = (241, 101, 72)  # blue
                # fp_bbox_color=(30, 105, 210) # origin
                # tp_bbox_color=(50, 205, 50) # green

                if len(gt_bboxes) != 0 and show_gt:
                    img = draw_lidar_bbox3d_on_img(
                        copy.deepcopy(gt_bboxes), img, lidar2img_rt, None, color=gt_bbox_color, thickness=3)
                if len(pred_bboxes) != 0 and show_pred:
                    img = draw_lidar_bbox3d_on_img(
                        copy.deepcopy(pred_bboxes), img, lidar2img_rt, None, color=pred_bbox_color, thickness=3)
                # if len(fp_boxes) != 0 and show_pred:
                #     img = draw_lidar_bbox3d_on_img(
                #         copy.deepcopy(fp_boxes), img, lidar2img_rt, None, color=fp_bbox_color, thickness=3)
                # if len(tp_boxes) != 0 and show_pred:
                #     img = draw_lidar_bbox3d_on_img(
                #         copy.deepcopy(tp_boxes), img, lidar2img_rt, None, color=tp_bbox_color, thickness=3)

                cam_imgs.append(img)

            img_size = (1600, 2400, 3)
            pano = np.zeros(img_size, np.uint8)
            bev_img = cv2.resize(bev_img, (800, 800))
            pano[400:1200, 800:1600] = bev_img

            cam1 = cam_imgs[0]
            cam1 = cv2.resize(cam1, (800, 400))
            pano[:400, 800:1600] = cam1

            cam2 = cam_imgs[1]
            cam2 = cv2.resize(cam2, (800, 400))
            pano[400:800, :800] = cam2

            cam3 = cam_imgs[2]
            cam3 = cv2.resize(cam3, (800, 400))
            pano[800:1200, :800] = cam3

            cam4 = cam_imgs[3]
            cam4 = cv2.resize(cam4, (800, 400))
            pano[-400:, 800:1600] = cam4

            cam5 = cam_imgs[4]
            cam5 = cv2.resize(cam5, (800, 400))
            pano[800:1200, -800:] = cam5

            cam6 = cam_imgs[5]
            cam6 = cv2.resize(cam6, (800, 400))
            pano[400:800, -800:] = cam6
            if img is not None:
                # if video_writer is not None:
                #     image = cv2.resize(pano, (1200, 800), interpolation=cv2.INTER_NEAREST)
                #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #     video_writer.writeFrame(image)
                save_path = osp.join(out_dir, file_name + '.jpg')
                mmcv.imwrite(pano, save_path)
                # else:
                #     save_path = osp.join(out_dir, file_name + '.jpg')
                #     mmcv.imwrite(pano, save_path)
        return video_writer
