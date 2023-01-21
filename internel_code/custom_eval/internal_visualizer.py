import os
import sys
import mmcv
import shutil
from mmcv import Config
from mmdet3d.datasets import build_dataset
import cv2
from cv2 import VideoWriter_fourcc
import os
import numpy as np
import skvideo.io
import argparse

proj_dir = os.getcwd()
# lod_dir = os.path.join(proj_dir,
#                        'internal_code/custom_eval/3d_lidar_detection_evaluation')
sys.path.append(proj_dir)
# sys.path.append(lod_dir)


class InternalVisualizer(object):
    def __init__(self, cfg_file, pred_file, save_path=None, cfg_options=None):
        """
        gt_file: val.json
        pred_file: work_dirs/internal_detr3d_res101_scale05_smallset/epoch_12/results.pkl
        """
        if not pred_file.endswith(('.pkl', '.pickle')):
            raise ValueError('The results file must be a pkl file.')

        self.cfg_file = cfg_file
        self.pred_file = pred_file

        self.cfg_options = cfg_options

        # results save path: save_figs / save_videos
        if save_path is None:
            save_path = os.path.split(pred_file)[0]
        self.save_path = save_path
        self.save_figs_path = os.path.join(self.save_path, 'save_figs')
        self.save_videos_path = os.path.join(self.save_path, 'save_videos')
        if os.path.exists(self.save_figs_path):
            shutil.rmtree(self.save_figs_path)
        if os.path.exists(self.save_videos_path):
            shutil.rmtree(self.save_videos_path)
        os.makedirs(self.save_figs_path, exist_ok=True)
        os.makedirs(self.save_videos_path, exist_ok=True)

    def visualize(self, video_only=False, sample_rate=1):
        if video_only:
            self.create_videos(self.save_figs_path, self.save_videos_path)
            return

        cfg = Config.fromfile(self.cfg_file)
        if self.cfg_options is not None:
            cfg.merge_from_dict(self.cfg_options)
        cfg.data.test.test_mode = True
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])

        # build the dataset
        dataset = build_dataset(cfg.data.test)
        results = mmcv.load(self.pred_file)
        eval_pipeline = cfg.get('eval_pipeline', {})

        video_writer = self.init_videos(self.save_videos_path)
        if eval_pipeline:
            video_writer = dataset.show_panorama(results,
                                                 self.save_figs_path,
                                                 pipeline=eval_pipeline,
                                                 sample_rate=int(sample_rate),
                                                 video_writer=video_writer)
        else:
            # use default pipeline
            video_writer = dataset.show_panorama(results,
                                                 self.save_figs_path,
                                                 sample_rate=int(sample_rate),
                                                 video_writer=video_writer)

        video_writer.close()

    def create_videos(self, save_figs_path, save_videos_path):
        save_video_name = os.path.join(save_videos_path, 'internal_results.mp4')

        img_list = os.listdir(save_figs_path)
        img_list = sorted(img_list)

        video_fps = 10
        video_writer = skvideo.io.FFmpegWriter(
            save_video_name,
            inputdict={
                '-r': str(video_fps),
                '-s': '{}x{}'.format(int(1200), int(800))
            },
            outputdict={
                '-r': str(video_fps),
                '-vcodec': 'libx264'
            })
        for name in img_list:
            image = cv2.imread(os.path.join(save_figs_path, name))
            image = cv2.resize(image, (1200, 800),
                               interpolation=cv2.INTER_NEAREST)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_writer.writeFrame(image)
        video_writer.close()
        return

    def init_videos(self, save_videos_path):
        save_video_name = os.path.join(save_videos_path, 'internal_results.mp4')

        video_fps = 10
        video_writer = skvideo.io.FFmpegWriter(
            save_video_name,
            inputdict={
                '-r': str(video_fps),
                '-s': '{}x{}'.format(int(1200), int(800))
            },
            outputdict={
                '-r': str(video_fps),
                '-vcodec': 'libx264'
            })
        return video_writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--cfg', help='output result file in pickle format')
    parser.add_argument('--pred', help='output result file in pickle format')
    parser.add_argument('--sample_rate', help='output sample_rate')
    parser.add_argument('--video_only',
                        action='store_true',
                        help='only create video')
    args = parser.parse_args()

    cfg = args.cfg
    pred = args.pred
    video_only = args.video_only
    sample_rate = args.sample_rate
    internal_visualizer = InternalVisualizer(args.cfg, args.pred)
    internal_visualizer.visualize(video_only, sample_rate)
