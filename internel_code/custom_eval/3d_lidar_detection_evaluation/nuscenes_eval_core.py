import glob
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from label_parser import LabelParser
from kuhn_munkres import KMMatcher


def match_pairs_km(pred_label, gt_label, distance_threshold=0.1):
    true_preds = np.empty((0, 8))
    corresponding_gt = np.empty((0, 7))
    result_score = np.empty((0, 4))
    # calculate similarity between predictions and gts
    score_metric = np.zeros((len(pred_label), len(gt_label)))
    for gt_idx in range(len(gt_label)):
        distance = np.linalg.norm(
            gt_label[gt_idx, :2] - pred_label[:, :2], axis=-1)
        score_metric[:, gt_idx] = 1. / distance

    # km
    km_matcher = KMMatcher()
    score_metric = score_metric.transpose()
    km_matcher.setInformationMatrix(score_metric)
    km_matcher.processKM()
    matched_result = km_matcher.getMatchedResult()

    # process matched results
    for pred_idx in range(matched_result.size):
        gt_idx = matched_result[pred_idx]
        if gt_idx >= 0:
            gt_position = gt_label[gt_idx, :2]
            score = score_metric[gt_idx][pred_idx]
            score_criterion = 1. / (distance_threshold * np.linalg.norm(gt_position))
            if score_metric[gt_idx][pred_idx] > 1. / (distance_threshold * np.linalg.norm(gt_position)):
                true_preds = np.vstack(
                    (true_preds, pred_label[pred_idx, :].reshape(-1, 1).T))
                corresponding_gt = np.vstack(
                    (corresponding_gt, gt_label[gt_idx]))

                # Store score for mAP
                result_score = np.vstack(
                    (result_score, np.array([[1, pred_label[pred_idx, 7], score, score_criterion]])))
            else:
                # FP
                result_score = np.vstack(
                    (result_score, np.array([[0, pred_label[pred_idx, 7], score, score_criterion]])))
        else:
            # FP
            result_score = np.vstack(
                (result_score, np.array([[0, pred_label[pred_idx, 7], -1, -1]])))

    return true_preds, corresponding_gt, result_score


class NuScenesEval:
    def __init__(self, pred_label_path, gt_label_path, label_format, save_loc, 
                 distance_threshold=0.1, 
                 classes=['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN'],
                 score_threshold=0.0, point_cloud_range=None, run=True, area_name="full_zone"):

        # Initialize
        self.save_loc = save_loc
        self.distance_threshold_sq = distance_threshold  # distance_threshold**2
        self.distance_threshold_ate = 0.2  # used for ate calculation
        self.score_threshold = score_threshold
        self.point_cloud_range = point_cloud_range
        self.classes = classes
        self.total_N_pos = 0
        self.results_dict = {}
        self.area_name = area_name
        os.makedirs(os.path.join(self.save_loc, self.area_name), exist_ok=True)
        for single_class in classes:
            class_dict = {}
            class_dict['class'] = single_class
            class_dict['T_p'] = np.empty((0, 8))
            class_dict['gt'] = np.empty((0, 7))
            class_dict['total_N_pos'] = 0
            class_dict['result'] = np.empty((0, 5))
            class_dict['precision'] = []
            class_dict['recall'] = []
            class_dict['T_p_ate'] = np.empty((0, 8))
            class_dict['gt_ate'] = np.empty((0, 7))
            class_dict['result_ate'] = np.empty((0, 5))
            self.results_dict[single_class] = class_dict

        # metric results
        self.metric_results = {}
        for single_class in classes:
            self.metric_results[single_class] = {}

        # Run
        if run:
            self.time = time.time()
            self.evaluate(pred_label_path, gt_label_path, label_format)

    def get_metric_results(self):
        return self.metric_results

    def evaluate(self, all_predictions, all_gts, label_format):
        num_examples = len(all_predictions)
        print("Starting evaluation for {} file predictions".format(num_examples))
        print("--------------------------------------------")

        # Check missing files
        print("Confirmation prediction ground truth file pairs.")

        # Evaluate matches
        print("Evaluation examples")
        for predictions, ground_truth in zip(all_predictions, all_gts):
            if self.point_cloud_range is not None:
                predictions, ground_truth = self.filter_by_range(predictions, ground_truth, point_range=self.point_cloud_range)
            self.eval_pair(predictions, ground_truth)
        print("\nDone!")
        print("----------------------------------")

        # Calculate
        for single_class in self.classes:
            class_dict = self.results_dict[single_class]
            print("Calculating metrics for {} class".format(single_class))
            print("----------------------------------")
            print("Number of ground truth labels: ", class_dict['total_N_pos'])
            print("Number of detections:  ", class_dict['result'].shape[0])
            print("Number of true positives:  ", np.sum(class_dict['result'][:, 0] == 1))
            print("Number of false positives:  ", np.sum(class_dict['result'][:, 0] == 0))
            self.metric_results[single_class]["gt_num"] = class_dict[
                'total_N_pos']
            self.metric_results[single_class]["pred_num"] = class_dict[
                'result'].shape[0]
            self.metric_results[single_class]["tp_num"] = np.sum(
                class_dict['result'][:, 0] == 1)
            self.metric_results[single_class]["fp_num"] = np.sum(
                class_dict['result'][:, 0] == 0)
            self.metric_results[single_class]["match_pair"] = class_dict['result']
            if class_dict['total_N_pos'] == 0:
                print("No detections for this class!")
                print(" ")
                continue
            # Recall Precision
            self.compute_recall_precision(single_class)
            print('Recall: %.3f ' % self.metric_results[single_class]["recall_range"][-1])
            print('Precision: %.3f ' % self.metric_results[single_class]["precision_range"][-1])
            # AP
            self.compute_ap_curve(class_dict)
            mean_ap = self.compute_mean_ap(class_dict['precision'], class_dict['recall'])
            print('Mean AP: %.3f ' % mean_ap)
            self.metric_results[single_class]["ap"] = mean_ap
            f1 = self.compute_f1_score(class_dict['precision'], class_dict['recall'])
            print('F1 Score: %.3f ' % f1)
            self.metric_results[single_class]["f1_score"] = f1
            print(' ')
            # Positive Thresholds
            # ATE 2D
            ate2d, ate2d_pct = self.compute_ate2d(
                class_dict['T_p_ate'], class_dict['gt_ate'])
            print('Average 2D Translation Error [m]:  {:.4f}, percentage {:.2f}%'.format(
                ate2d, ate2d_pct*100))
            self.metric_results[single_class]["ate_2d"] = ate2d
            self.metric_results[single_class]["ate_2d_pct"] = ate2d_pct*100
            # ATE 3D
            ate3d, ate3d_pct = self.compute_ate3d(
                class_dict['T_p_ate'], class_dict['gt_ate'])
            print('Average 3D Translation Error [m]:  {:.4f}, {:.2f}%'.format(
                ate3d, ate3d_pct*100))
            self.metric_results[single_class]["ate_3d"] = ate3d
            self.metric_results[single_class]["ate_3d_pct"] = ate3d_pct*100
            # ASE
            ase = self.compute_ase(class_dict['T_p_ate'], class_dict['gt_ate'])
            print('Average Scale Error:  %.4f ' % ase)
            self.metric_results[single_class]["ase"] = ase
            # AOE
            aoe = self.compute_aoe(class_dict['T_p_ate'], class_dict['gt_ate'])
            print('Average Orientation Error [rad]:  {:.4f}, [degree] {:.4f}'.format(
                aoe, aoe * 180 / np.pi))
            self.metric_results[single_class]["aoe"] = aoe * 180 / np.pi
            print(" ")
        self.time = float(time.time() - self.time)
        print("Total evaluation time: %.5f " % self.time)

    def compute_recall_precision(self, single_class):
        match_pair = self.metric_results[single_class]['match_pair']
        tp = match_pair[match_pair[:, 0] == 1]
        fp = match_pair[match_pair[:, 0] == 0]
        gt = match_pair[(match_pair[:, 0] == 1) + (match_pair[:, 0] == -1)]
        pred = match_pair[match_pair[:, 0] >= 0]
        self.metric_results[single_class]["fp_pred"] = np.array([sum((fp[:, 2] > i*10) * (fp[:, 2] <= (i+1)*10)) for i in range(20)])            
        self.metric_results[single_class]["fp_gt"] = np.array([sum((fp[:, 3] > i*10) * (fp[:, 3] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["tp_pred"] = np.array([sum((tp[:, 2] > i*10) * (tp[:, 2] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["tp_gt"] = np.array([sum((tp[:, 3] > i*10) * (tp[:, 3] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["pred"] = np.array([sum((pred[:, 2] > i*10) * (pred[:, 2] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["gt"] = np.array([sum((gt[:, 3] > i*10) * (gt[:, 3] <= (i+1)*10)) for i in range(20)])
        self.metric_results[single_class]["recall_ring"] = self.metric_results[single_class]["tp_gt"] / (self.metric_results[single_class]["gt"] + 1e-5)
        self.metric_results[single_class]["recall_range"] = np.array([sum(tp[:, 3] < i*10) / (sum(gt[:, 3] < i*10) + 1e-5)for i in range(20)])
        self.metric_results[single_class]["precision_ring"] = self.metric_results[single_class]["tp_pred"] / (self.metric_results[single_class]["pred"] + 1e-5)
        self.metric_results[single_class]["precision_range"] = np.array([sum(tp[:, 2] < i*10) / (sum(pred[:, 2] < i*10) + 1e-5) for i in range(20)])
            
        plt.figure()
        l1, = plt.plot([10*i for i in range(self.metric_results[single_class]["recall_ring"].shape[0])], 
                self.metric_results[single_class]["recall_ring"])

        l3, = plt.plot([10*i for i in range(self.metric_results[single_class]["precision_ring"].shape[0])],
                self.metric_results[single_class]["precision_ring"])

        plt.legend(handles=[l1, l3],
                labels=['recall_range', 'precision_range'],
                loc='best')
        plt.title(single_class + "_pr_ring")
        plt.xlabel("distance")
        plt.ylabel("percentage")
        plt.savefig(os.path.join(self.save_loc, single_class + "_pr_ring.png"))
        plt.close()
        
        plt.figure()
        l2, = plt.plot([10*i for i in range(self.metric_results[single_class]["recall_range"].shape[0])],
                self.metric_results[single_class]["recall_range"])
        l4, = plt.plot([10*i for i in range(self.metric_results[single_class]["precision_range"].shape[0])],
                self.metric_results[single_class]["precision_range"])
        plt.legend(handles=[l2, l4],
                labels=['recall_range', 'precision_range'],
                loc='best')
        plt.title(single_class + "_pr_range")
        plt.xlabel("distance")
        plt.ylabel("percentage")
        plt.savefig(os.path.join(self.save_loc, self.area_name, single_class + "_pr_range.png"))
        plt.close()

    def compute_ap_curve(self, class_dict):
        t_pos = 0
        class_dict['precision'] = np.ones(class_dict['result'].shape[0]+2)
        class_dict['recall'] = np.zeros(class_dict['result'].shape[0]+2)
        sorted_detections = class_dict['result'][(-class_dict['result'][:, 1]).argsort(), :]
        for i, (result_bool, result_score, _, _, _) in enumerate(sorted_detections):
            if result_bool == 1:
                t_pos += 1
            class_dict['precision'][i+1] = t_pos / (i + 1)
            class_dict['recall'][i+1] = t_pos / class_dict['total_N_pos']
        class_dict['precision'][i+2] = 0
        class_dict['recall'][i+2] = class_dict['recall'][i+1]

        # Plot
        plt.figure()
        plt.plot(class_dict['recall'], class_dict['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve for {} Class'.format(class_dict['class']))
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.savefig(os.path.join(self.save_loc, self.area_name, class_dict['class'] + "_pr_curve.png"))
        plt.close()

    def compute_f1_score(self, precision, recall):
        p, r = precision[(precision+recall) > 0], recall[(precision+recall) > 0]
        f1_scores = 2 * p * r / (p + r)
        return np.max(f1_scores)

    def compute_mean_ap(self, precision, recall, precision_threshold=0.0, recall_threshold=0.0):
        mean_ap = 0
        threshold_mask = np.logical_and(precision > precision_threshold,
                                        recall > recall_threshold)
        # calculate mean AP
        precision = precision[threshold_mask]
        recall = recall[threshold_mask]
        recall_diff = np.diff(recall)
        precision_diff = np.diff(precision)
        # Square area under curve based on i+1 precision, then linear difference in precision
        mean_ap = np.sum(precision[1:]*recall_diff + recall_diff*precision_diff/2)
        # We need to divide by (1-recall_threshold) to make the max possible mAP = 1. In practice threshold by the first
        # considered recall value (threshold = 0.1 -> first considered value may be = 0.1123)
        mean_ap = mean_ap/(1-recall[0])
        return mean_ap

    def compute_ate2d(self, predictions, ground_truth):
        # euclidean distance 3d
        mean_ate2d = np.mean(np.linalg.norm(
            predictions[:, :2] - ground_truth[:, :2], axis=-1))
        mean_ate2d_pct = np.mean(np.linalg.norm(
            predictions[:, :2] - ground_truth[:, :2], axis=-1) / np.linalg.norm(ground_truth[:, :2], axis=-1))
        return mean_ate2d, mean_ate2d_pct

    def compute_ate3d(self, predictions, ground_truth):
        # euclidean distance 2d
        mean_ate3d = np.mean(np.linalg.norm(
            predictions[:, :3] - ground_truth[:, :3], axis=-1))
        mean_ate3d_pct = np.mean(np.linalg.norm(
            predictions[:, :3] - ground_truth[:, :3], axis=-1) / np.linalg.norm(ground_truth[:, :3], axis=-1))
        return mean_ate3d, mean_ate3d_pct

    def compute_ase(self, predictions, ground_truth):
        # simplified iou where boxes are centered and aligned with eachother
        pred_vol = predictions[:, 3]*predictions[:, 4]*predictions[:, 5]
        gt_vol = ground_truth[:, 3]*ground_truth[:, 4]*ground_truth[:, 5]
        iou3d = np.mean(1 - np.minimum(pred_vol, gt_vol)/np.maximum(pred_vol, gt_vol))
        return iou3d

    def compute_aoe(self, predictions, ground_truth):
        err = ground_truth[:,6] - predictions[:,6]
        aoe = np.mean(np.abs((err + np.pi) % (2*np.pi) - np.pi))
        return aoe

    def eval_pair(self, pred_label, gt_label):
        # Check
        assert pred_label.shape[1] == 9
        assert gt_label.shape[1] == 8

        # Threshold score
        if pred_label.shape[0] > 0:
            pred_label = pred_label[pred_label[:, 8].astype(np.float) > self.score_threshold, :]
        for single_class in self.classes:
            # get all pred labels, order by score
            valid_idx = pred_label[:, 0].astype(str) == single_class
            class_pred_label = pred_label[valid_idx, 1:]
            score = class_pred_label[:, 7].astype(np.float)
            class_pred_label = class_pred_label[(-score).argsort(), :].astype(np.float)  # sort decreasing

            # add gt label length to total_N_pos
            class_gt_label = gt_label[gt_label[:, 0].astype(str) == single_class, 1:].astype(np.float)
            self.results_dict[single_class]['total_N_pos'] += class_gt_label.shape[0]

            # match pairs for ap
            pred_array, gt_array, result_score_pair = self.match_pairs_km(class_pred_label, class_gt_label, self.distance_threshold_sq)

            # add to existing results
            self.results_dict[single_class]['T_p'] = np.vstack((self.results_dict[single_class]['T_p'], pred_array))
            self.results_dict[single_class]['gt'] = np.vstack((self.results_dict[single_class]['gt'], gt_array))
            self.results_dict[single_class]['result'] = np.vstack((self.results_dict[single_class]['result'],
                                                                   result_score_pair))

            # match pairs for ATE
            pred_array, gt_array, result_score_pair = self.match_pairs_km(class_pred_label, class_gt_label, self.distance_threshold_ate)

            # add to existing results
            self.results_dict[single_class]['T_p_ate'] = np.vstack((self.results_dict[single_class]['T_p_ate'], pred_array))
            self.results_dict[single_class]['gt_ate'] = np.vstack((self.results_dict[single_class]['gt_ate'], gt_array))
            self.results_dict[single_class]['result_ate'] = np.vstack((self.results_dict[single_class]['result_ate'],
                                                                   result_score_pair))

    def match_pairs(self, pred_label, gt_label):
        true_preds = np.empty((0, 8))
        corresponding_gt = np.empty((0, 7))
        result_score = np.empty((0, 2))
        # Initialize matching loop
        match_incomplete = True
        while match_incomplete and gt_label.shape[0] > 0:
            match_incomplete = False
            for gt_idx, single_gt_label in enumerate(gt_label):
                # Check is any prediction is in range
                distance_sq_array = (single_gt_label[0] - pred_label[:, 0])**2 + (single_gt_label[1] - pred_label[:, 1])**2
                # If there is a prediction in range, pick closest
                if np.any(distance_sq_array < self.distance_threshold_sq):
                    min_idx = np.argmin(distance_sq_array)
                    # Store true prediction
                    true_preds = np.vstack((true_preds, pred_label[min_idx, :].reshape(-1, 1).T))
                    corresponding_gt = np.vstack((corresponding_gt, gt_label[gt_idx]))

                    # Store score for mAP
                    result_score = np.vstack((result_score, np.array([[1, pred_label[min_idx, 7]]])))

                    # Remove prediction and gt then reset loop
                    pred_label = np.delete(pred_label, obj=min_idx, axis=0)
                    gt_label = np.delete(gt_label, obj=gt_idx, axis=0)
                    match_incomplete = True
                    break

        # If there were any false detections, add them.
        if pred_label.shape[0] > 0:
            false_positives = np.zeros((pred_label.shape[0], 2))
            false_positives[:, 1] = pred_label[:, 7]
            result_score = np.vstack((result_score, false_positives))
        return true_preds, corresponding_gt, result_score

    def match_pairs_km(self, pred_label, gt_label, distance_threshold=0.1):
        true_preds = np.empty((0, 8))
        corresponding_gt = np.empty((0, 7))
        result_score = np.empty((0, 5))
        # calculate similarity between predictions and gts
        score_metric = np.zeros((len(pred_label), len(gt_label)))
        gts_visited = np.ones(len(gt_label)) * -1
        for gt_idx in range(len(gt_label)):
            distance = np.linalg.norm(
                gt_label[gt_idx, :2] - pred_label[:, :2], axis=-1)
            score_metric[:, gt_idx] = 1. / distance

        # km
        km_matcher = KMMatcher()
        score_metric = score_metric.transpose()
        km_matcher.setInformationMatrix(score_metric)
        km_matcher.processKM()
        matched_result = km_matcher.getMatchedResult()

        # process matched results
        for pred_idx in range(matched_result.size):
            gt_idx = matched_result[pred_idx]
            pred_distance = np.linalg.norm(pred_label[pred_idx, :2])
            if gt_idx >= 0:
                gts_visited[gt_idx] = 1
                gt_distance = np.linalg.norm(gt_label[gt_idx, :2])
                if distance_threshold == "small_object":
                    if gt_distance < 50:
                        threshold = -0.0006389*gt_distance**2 + 0.22733382*gt_distance + 0.27651913
                        threshold = 1/threshold
                    else:
                        threshold = 1. / (10 + 0.2 * (gt_distance - 50))
                else:
                    threshold = 1. / (distance_threshold * gt_distance)
                
                if score_metric[gt_idx][pred_idx] > threshold:
                    true_preds = np.vstack(
                        (true_preds, pred_label[pred_idx, :].reshape(-1, 1).T))
                    corresponding_gt = np.vstack(
                        (corresponding_gt, gt_label[gt_idx]))

                    # Store score for mAP
                    result_score = np.vstack(
                        (result_score, np.array([[1, pred_label[pred_idx, 7], pred_distance, gt_distance, 1/score_metric[gt_idx][pred_idx]]])))
                else:
                    # FP
                    result_score = np.vstack(
                        (result_score, np.array([[0, pred_label[pred_idx, 7], pred_distance, gt_distance, 1/score_metric[gt_idx][pred_idx]]])))
            else:
                # FP
                result_score = np.vstack(
                    (result_score, np.array([[0, pred_label[pred_idx, 7], pred_distance, -1, -1]])))

        for gt_idx in range(len(gts_visited)):
            if gts_visited[gt_idx] == 1:
                continue
            gt_distance = np.linalg.norm(gt_label[gt_idx, :2])
            result_score = np.vstack(
                    (result_score, np.array([[-1, -1, -1, gt_distance, -1]])))

        return true_preds, corresponding_gt, result_score

    def filter_by_range(self,
                        pred_label,
                        gt_label,
                        point_range=[-50, -50, -5.0, 50, 50, 3.0]):
        valid_pred_index = [
            i for i in range(len(pred_label))
            if self.is_inside_point_cloud_range(
                pred_label[i, 1:4].astype(np.float32), point_range)
        ]
        valid_gt_index = [
            i for i in range(len(gt_label)) if self.is_inside_point_cloud_range(
                gt_label[i, 1:4].astype(np.float32), point_range)
        ]
        return pred_label[valid_pred_index, :], gt_label[valid_gt_index, :]

    def is_inside_point_cloud_range(self, point, point_range):
        if (point[0] <= point_range[3] and point[0] >= point_range[0]) and (
                point[1] <= point_range[4] and
                point[1] >= point_range[1]) and (point[2] <= point_range[5] and
                                                 point[2] >= point_range[2]):
            return True
        else:
            return False
