import os
import sys
import mmcv
import shutil
import argparse
import numpy as np
from mmcv import Config, DictAction
import math
from collections import deque

proj_dir = os.getcwd()
lod_dir = os.path.join(
    proj_dir, 'internal_code/custom_eval/3d_lidar_detection_evaluation')
sys.path.append(proj_dir)
sys.path.append(lod_dir)

from nuscenes_eval_core import NuScenesEval

CLASS_MAPPING = ['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN']
CLASS_MAPPING_GT = {
    'VEHICLE_CAR': 'VEHICLE_CAR',
    'VEHICLE_TRUCK': 'VEHICLE_TRUCK',
    'BIKE_BICYCLE': 'PEDESTRIAN',
    'PEDESTRIAN': 'PEDESTRIAN'
}


class PRDEvaluatorTarget():

    def __init__(self, cfg, pred_file, save_path=None):
        """
        gt_file: val.json
        pred_file: work_dirs/internal_detr3d_res101_scale05_smallset/epoch_12/results.pkl
        """

        self.cfg = cfg

        gt_file = cfg.data.test.ann_file
        gt_load_interval = cfg.data.test.load_interval
        self.gt = mmcv.load(gt_file)
        self.gt['infos'] = list(
            sorted(self.gt['infos'], key=lambda e: e['timestamp']))
        self.gt['infos'] = self.gt['infos'][::gt_load_interval]
        self.pred = mmcv.load(pred_file)

        # results save path: cacheVEHICLE_CAR_pr_curve.png / cacheVEHICLE_TRUCK_pr_curve.png / gt / pred /
        if save_path is None:
            save_path = os.path.split(pred_file)[0]
        self.save_path = save_path

    def evaluate_nuscenes_data(self, point_cloud_range, distance_threshold):
        print("point_cloud_range: ", point_cloud_range)
        print("distance_threshold: ", distance_threshold)
        evaluator = NuScenesEval(self.predictions,
                                 self.gts,
                                 'class x y z l w h r score',
                                 self.save_path,
                                 distance_threshold=distance_threshold,
                                 point_cloud_range=point_cloud_range)
        return evaluator

    def evaluate_internal_data(self, prec_threshold=(0.7, 0.7, 0.2, 0.5), prec_threshold_key_zone=(0.8, 0.8, 0.2, 0.5)):
        gt_save_path = os.path.join(self.save_path, 'gt')
        pred_save_path = os.path.join(self.save_path, 'pred')
        self.convert_results(gt_save_path, pred_save_path)
        # prec_threshold = (0.749, 0.789, 0.29, 0.239)
        # prec_threshold_key_zone = (0.893, 0.886, 0.350, 0.257)
        print('==============================================================')
        print('==============================================================')
        print('==============================================================')
        print('==================== standard evaluation =====================')
        print('==============================================================')
        print('==============================================================')
        print('==============================================================')
        # run evaluation
        print("self.cfg.point_cloud_range: ", self.cfg.point_cloud_range)
        all_classes = ['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN']
        all_distances = [0.1, 0.1, 0.2, 0.2]
        real_eval_results = None
        for i, target_prec in enumerate(prec_threshold):
            score_deque = deque(maxlen=3)
            score_threshold = 0.4
            base_lr = 0.02
            while True:
                if len(score_deque)>2:
                    if score_deque[0]==score_deque[-1]:
                        base_lr = base_lr/2
                evaluator = NuScenesEval(self.predictions,
                                        self.gts,
                                        'class x y z l w h r score',
                                        self.save_path,
                                        distance_threshold=all_distances[i],
                                        score_threshold=score_threshold,
                                        classes=[all_classes[i]],
                                        point_cloud_range=self.cfg.point_cloud_range)
                eval_results = evaluator.get_metric_results()
                cur_prec = eval_results[all_classes[i]]['precision_range'][-1]
                if abs(prec_threshold[i] - cur_prec) < 0.003:
                    print('score_thresh: ', i, score_threshold)
                    if real_eval_results is None:
                        real_eval_results = evaluator.get_metric_results()
                    else:
                        real_eval_results.update(evaluator.get_metric_results())
                    break
                else:
                    if prec_threshold[i] - cur_prec > 0:
                        score_threshold += base_lr
                    else:
                        score_threshold = score_threshold-base_lr
                score_deque.append(score_threshold)

        mmcv.dump(real_eval_results, os.path.join(self.save_path,
                                             'eval_results.json'))
        # run internal evaluation
        print('==============================================================')
        print('==============================================================')
        print('==============================================================')
        print('======= key zone evaluation (distance_threshold=0.2) =========')
        print('==============================================================')
        print('==============================================================')
        print('==============================================================')
        key_zone = [-50, -20, -5.0, 50, 20, 3.0]
        real_eval_results = None
        for i, target_prec in enumerate(prec_threshold_key_zone):
            score_deque = deque(maxlen=3)
            score_threshold = 0.4
            base_lr = 0.02
            while True:
                if len(score_deque) > 2:
                    if score_deque[0] == score_deque[-1]:
                        base_lr = base_lr/2
                evaluator = NuScenesEval(self.predictions,
                                        self.gts,
                                        'class x y z l w h r score',
                                        self.save_path,
                                        distance_threshold=0.2,
                                        score_threshold=score_threshold,
                                        classes=[all_classes[i]],
                                        point_cloud_range=key_zone)
                eval_results = evaluator.get_metric_results()
                cur_prec = eval_results[all_classes[i]]['precision_range'][-1]
                if abs(prec_threshold_key_zone[i] - cur_prec) < 0.003:
                    print('score_thresh: ', i, score_threshold)
                    if real_eval_results is None:
                        real_eval_results = evaluator.get_metric_results()
                    else:
                        real_eval_results.update(evaluator.get_metric_results())
                    break
                else:
                    if prec_threshold_key_zone[i] - cur_prec > 0:
                        score_threshold += base_lr
                    else:
                        score_threshold = score_threshold-base_lr
                score_deque.append(score_threshold)
        mmcv.dump(real_eval_results, os.path.join(self.save_path,
                                             'eval_results_key_zone.json'))
        # # run internal evaluation
        # print('==============================================================')
        # print('==============================================================')
        # print('==============================================================')
        # print('======= tiny zone evaluation (distance_threshold=0.2) ========')
        # print('==============================================================')
        # print('==============================================================')
        # print('==============================================================')
        # tiny_zone = [-10, -15, -5.0, 10, 15, 3.0]
        # evaluator = NuScenesEval(self.predictions,
        #                          self.gts,
        #                          'class x y z l w h r score',
        #                          self.save_path,
        #                          distance_threshold=0.2,
        #                          point_cloud_range=tiny_zone)
        # eval_results = evaluator.get_metric_results()
        # mmcv.dump(eval_results, os.path.join(self.save_path,
        #                                      'eval_results_tiny_zone.json'))

        # # run internal evaluation
        # print('==============================================================')
        # print('==============================================================')
        # print('==============================================================')
        # print('======= mid zone evaluation (distance_threshold=0.2) =========')
        # print('==============================================================')
        # print('==============================================================')
        # print('==============================================================')
        # mid_zone = [-20, -30, -5.0, 20, 30, 3.0]
        # evaluator = NuScenesEval(self.predictions,
        #                          self.gts,
        #                          'class x y z l w h r score',
        #                          self.save_path,
        #                          distance_threshold=0.2,
        #                          point_cloud_range=mid_zone)
        # eval_results = evaluator.get_metric_results()
        # mmcv.dump(eval_results, os.path.join(self.save_path,
        #                                      'eval_results_mid_zone.json'))

        # # run internal evaluation
        # print('==============================================================')
        # print('==============================================================')
        # print('==============================================================')
        # print('========= A zone evaluation (distance_threshold=0.2) =========')
        # print('==============================================================')
        # print('==============================================================')
        # print('==============================================================')
        # zone = [-0, -1.5, -5.0, 100, 1.5, 3.0]  # [back, left_side, down, front, right_side, up]
        # evaluator = NuScenesEval(self.predictions,
        #                          self.gts,
        #                          'class x y z l w h r score',
        #                          self.save_path,
        #                          distance_threshold=0.1,
        #                          classes=['VEHICLE_CAR', 'VEHICLE_TRUCK'],
        #                          point_cloud_range=zone)
        # eval_results = evaluator.get_metric_results()
        # evaluator = NuScenesEval(self.predictions,
        #                          self.gts,
        #                          'class x y z l w h r score',
        #                          self.save_path,
        #                          distance_threshold=0.2,
        #                          classes=['BIKE_BICYCLE', 'PEDESTRIAN'],
        #                          point_cloud_range=zone)
        # eval_results.update(evaluator.get_metric_results())
        # mmcv.dump(eval_results, os.path.join(self.save_path,
        #                                      'eval_results_A_zone.json'))

        return

    def run_nuscene_evaluation(self):
        gt_save_path = os.path.join(self.save_path, 'gt')
        pred_save_path = os.path.join(self.save_path, 'pred')

        # run evaluation
        NuScenesEval(self.predictions,
                     self.gts,
                     'class x y z l w h r score',
                     self.save_path,
                     max_range=50)
        return

    def convert_results(self, gt_save_path, pred_save_path):
        if os.path.exists(gt_save_path):
            shutil.rmtree(gt_save_path)
        if os.path.exists(pred_save_path):
            shutil.rmtree(pred_save_path)
        os.makedirs(gt_save_path, exist_ok=True)
        os.makedirs(pred_save_path, exist_ok=True)

        self.predictions = []
        self.gts = []

        for anno, pred in zip(self.gt['infos'], self.pred):
            gt_bboxes = anno['gt_boxes']
            gt_names = anno['gt_names']

            pred_bboxes = pred['boxes_3d'].tensor.numpy()
            pred_scores = pred['scores_3d'].numpy()
            pred_labels = pred['labels_3d'].numpy()

            valid_idx = pred_scores >= 0.0
            pred_bboxes = pred_bboxes[valid_idx]
            pred_scores = pred_scores[valid_idx]
            pred_labels = pred_labels[valid_idx]

            # prepare gt
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            for gt_name, gt_bbox in zip(gt_names, gt_bboxes):
                # classes.append(str(CLASS_MAPPING_GT[gt_name]))
                classes.append(str(gt_name))
                x.append(gt_bbox[0])
                y.append(gt_bbox[1])
                z.append(gt_bbox[2])
                l.append(gt_bbox[3])
                w.append(gt_bbox[4])
                h.append(gt_bbox[5])
                r.append(gt_bbox[6])

            final_array = np.hstack(
                (np.array(classes).reshape(-1, 1), np.array(x).reshape(-1, 1),
                 np.array(y).reshape(-1, 1), np.array(z).reshape(-1, 1),
                 np.array(l).reshape(-1, 1), np.array(w).reshape(-1, 1),
                 np.array(h).reshape(-1, 1), np.array(r).reshape(-1, 1)))
            self.gts.append(final_array)

            # prepare prediction
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            for pred_label, pred_bbox, pred_score in zip(
                    pred_labels, pred_bboxes, pred_scores):
                classes.append(str(CLASS_MAPPING[pred_label]))
                x.append(pred_bbox[0])
                y.append(pred_bbox[1])
                z.append(pred_bbox[2])
                l.append(pred_bbox[3])
                w.append(pred_bbox[4])
                h.append(pred_bbox[5])
                r.append(pred_bbox[6])
                score.append(pred_score)

            final_array = np.hstack(
                (np.array(classes).reshape(-1, 1), np.array(x).reshape(-1, 1),
                 np.array(y).reshape(-1, 1), np.array(z).reshape(-1, 1),
                 np.array(l).reshape(-1, 1), np.array(w).reshape(-1, 1),
                 np.array(h).reshape(-1, 1), np.array(r).reshape(-1, 1)))
            final_array = np.hstack(
                (final_array, np.array(score).reshape(-1, 1)))
            self.predictions.append(final_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--config', help='config file in pickle format')
    parser.add_argument('--pred', help='output result file in pickle format')
    parser.add_argument('--eval_only',
                        action='store_true',
                        help='only do evaluation')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    pred = args.pred
    eval_only = args.eval_only
    internal_evaluator = PRDEvaluatorTarget(cfg, pred)
    if eval_only:
        internal_evaluator.run_nuscene_evaluation()
    else:
        internal_evaluator.evaluate_internal_data()
