#!/usr/bin/env bash
T=`date +%m%d%H%M`

# nuscenes
 # python tools/analysis_tools/analyze_boxes.py \
 #    ./data/nuscenes/nuscenes_infos_train.pkl \
 #    ./data/nuscenes/nuscenes_infos_train_
 # python tools/analysis_tools/analyze_boxes.py \
 #    ./data/nuscenes/nuscenes_infos_val.pkl \
 #    ./data/nuscenes/nuscenes_infos_val_

# # cla
# python tools/analysis_tools/analyze_boxes.py \
#     ./data/cla/annotations/annotation_0616/all_detr3d_dataset_v0_6_1.json \
#     ./data/cla/annotations/annotation_0616/all_detr3d_dataset_v0_6_1_
# python tools/analysis_tools/analyze_boxes.py \
#     ./data/cla/annotations/annotation_0616/sample_detr3d_dataset_v0_6_1.json \
#     ./data/cla/annotations/annotation_0616/sample_detr3d_dataset_v0_6_1_
# python tools/analysis_tools/analyze_boxes.py \
#     ./data/cla/annotations/annotation_0616/val_detr3d_dataset_v0_6_1.json \
#     ./data/cla/annotations/annotation_0616/val_detr3d_dataset_v0_6_1_
