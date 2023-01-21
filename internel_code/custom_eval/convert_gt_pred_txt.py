import mmcv
import argparse
import os
import shutil

CLASS_MAPPING = ['VEHICLE_CAR', 'VEHICLE_TRUCK', 'BIKE_BICYCLE', 'PEDESTRIAN']
def parse_args():
    parser = argparse.ArgumentParser(description='Convert GT .txt')
    parser.add_argument('--gt', help='gt_annotation json path', type=str, default=None)
    parser.add_argument('--pred', help='prediction pkl path', type=str, default=None)

    args = parser.parse_args()
    return args.gt, args.pred

def convert_results(json_path, pkl_path):
    json_file = mmcv.load(json_path)
    pkl_file = mmcv.load(pkl_path)
    gt_path = 'internal_code/custom_eval/data/gt/'
    pred_path = 'internal_code/custom_eval/data/pred/'

    try:
        shutil.rmtree(gt_path)
        shutil.rmtree(pred_path)
    except:
        pass

    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)
    for anno, pred in zip(json_file['infos'], pkl_file):
        timestamp = anno['timestamp']

        gt_bboxes = anno['gt_boxes']
        gt_names = anno['gt_names']

        pred_bboxes = pred['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = pred['pts_bbox']['scores_3d'].numpy()
        pred_labels = pred['pts_bbox']['labels_3d'].numpy()

        valid_idx = pred_scores >= 0.3
        pred_bboxes = pred_bboxes[valid_idx]
        pred_scores = pred_scores[valid_idx]
        pred_labels = pred_labels[valid_idx]
        
        json_str = ""
        pkl_str = ""
        for gt_name, gt_bbox in zip(gt_names, gt_bboxes):
            json_str += str(gt_name)
            for ii, _ in enumerate(gt_bbox):
                json_str += " "
                json_str += str(gt_bbox[ii])
            json_str += " 1.0\n"
        for pred_label, pred_bbox, pred_score in zip(pred_labels, pred_bboxes, pred_scores):
            pkl_str += str(CLASS_MAPPING[pred_label])
            for ii, _ in enumerate(pred_bbox[:7]):
                pkl_str += " "
                pkl_str += str(pred_bbox[ii])
            pkl_str += " %.3f\n" % pred_score

        with open(os.path.join(gt_path, '%s.txt' % timestamp), 'w') as f:
            f.write(json_str)
        with open(os.path.join(pred_path, '%s.txt' % timestamp), 'w') as f:
            f.write(pkl_str)
        

if __name__ == '__main__':
    gt_path, pred_path = parse_args()
    convert_results(gt_path, pred_path)