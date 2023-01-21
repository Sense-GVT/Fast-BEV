import os
import sys
import mmcv
import shutil
import argparse
import numpy as np
from mmcv import Config, DictAction

proj_dir = os.getcwd()
lod_dir = os.path.join(
    proj_dir, 'internal_code/custom_eval/3d_lidar_detection_evaluation')
sys.path.append(proj_dir)
sys.path.append(lod_dir)

from nuscenes_eval_core import NuScenesEval

CLASS_MAPPING = ['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN']


class InternalEvaluator():

    def __init__(self, cfg, pred_file, save_path=None):
        """
        gt_file: val.json
        pred_file: work_dirs/internal_detr3d_res101_scale05_smallset/epoch_12/results.pkl
        """

        self.cfg = cfg

        gt_file = cfg.data.test.ann_file
        self.gt = mmcv.load(gt_file)
        self.gt['infos'] = list(
            sorted(self.gt['infos'], key=lambda e: e['timestamp']))
        self.pred = mmcv.load(pred_file)

        # results save path: cacheVEHICLE_CAR_pr_curve.png / cacheVEHICLE_TRUCK_pr_curve.png / gt / pred /
        if save_path is None:
            save_path = os.path.split(pred_file)[0]
        self.save_path = save_path

    def evaluate_internal_data(self):
        gt_save_path = os.path.join(self.save_path, 'gt')
        pred_save_path = os.path.join(self.save_path, 'pred')
        self.convert_results(gt_save_path, pred_save_path)

        # run evaluation
        evaluator = NuScenesEval(self.predictions,
                                 self.gts,
                                 'class x y z l w h r score',
                                 self.save_path,
                                 point_cloud_range=self.cfg.point_cloud_range)
        eval_results = evaluator.get_metric_results()
        mmcv.dump(eval_results, os.path.join(self.save_path,
                                             'eval_results.json'))
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

            pred_bboxes = pred['pts_bbox']['boxes_3d'].tensor.numpy()
            pred_scores = pred['pts_bbox']['scores_3d'].numpy()
            pred_labels = pred['pts_bbox']['labels_3d'].numpy()

            valid_idx = pred_scores >= 0.3
            pred_bboxes = pred_bboxes[valid_idx]
            pred_scores = pred_scores[valid_idx]
            pred_labels = pred_labels[valid_idx]

            # prepare gt
            classes = []
            score = []
            x, y, z, r = [], [], [], []
            l, w, h = [], [], []
            for gt_name, gt_bbox in zip(gt_names, gt_bboxes):
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
    internal_evaluator = InternalEvaluator(cfg, pred)
    if eval_only:
        internal_evaluator.run_nuscene_evaluation()
    else:
        internal_evaluator.evaluate_internal_data()
