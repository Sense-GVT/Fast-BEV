import argparse
from nuscenes_eval_core import NuScenesEval


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_labels', type=str,
                        help='Prediction labels data path', default='internal_code/custom_eval/data/pred')
    parser.add_argument('--gt_labels', type=str,
                        help='Ground Truth labels data path', default='internal_code/custom_eval/data/gt')
    parser.add_argument('--save_loc', type=str, help='Save location', default='internal_code/custom_eval/data/cache')
    parser.add_argument('--format', type=str, default='class x y z l w h r score')
    parser.add_argument('--max_range', type=float, default=50, help='max evaluation range')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    NuScenesEval(args.pred_labels, args.gt_labels, args.format, args.save_loc, max_range=args.max_range)


if __name__ == '__main__':
    main()