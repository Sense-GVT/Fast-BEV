import cv2
from cv2 import VideoWriter_fourcc
import os
import numpy as np
from tqdm import tqdm

import skvideo.io
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='create videos based on images')
    parser.add_argument('--img_path', '-in',
                        help='output result file in pickle format')
    parser.add_argument('--video_path', '-out',
                        help='output result file in pickle format')
    args = parser.parse_args()

    img_dir = args.img_path  # "/mnt/lustre/mig.cla/users/fanglj/projects/detr3d/work_dirs/internal_detr3d_res101_scale05_range75/results_epoch_20/save_figs"
    savepath = args.video_path  # "/mnt/lustre/mig.cla/users/fanglj/projects/detr3d/work_dirs/internal_detr3d_res101_scale05_range75/results_epoch_20/save_videos/internal_results.mp4"

    if not os.path.exists(os.path.split(savepath)[0]):
        os.makedirs(os.path.split(savepath)[0], exist_ok=True)

    video_fps = 10
    video_writer = skvideo.io.FFmpegWriter(
        savepath,
        inputdict={
            '-r':
                str(video_fps),
            '-s':
                '{}x{}'.format(int(1200),
                               int(800))
        },
        outputdict={
            '-r': str(video_fps),
            '-vcodec': 'libx264'
        })

    img_list = os.listdir(img_dir)
    img_list = sorted(img_list)
    for name in tqdm(img_list):
        image = cv2.imread(os.path.join(img_dir, name))
        image = cv2.resize(image, (1200, 800), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_writer.writeFrame(image)
    video_writer.close()
