# Fast-BEV

[Fast-BEV: A Fast and Strong Bird’s-Eye View Perception Baseline](https://arxiv.org/abs/2301.12511)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/fast-bev++.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark_setting.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark.png)

## Usage

### Installation

* CUDA>=9.2
* GCC>=5.4
* Python>=3.6
* Pytorch>=1.8.1
* Torchvision>=0.9.1
* MMCV-full==1.4.0
* MMDetection==2.14.0
* MMSegmentation==0.14.1

### Dataset preparation

```
  .
  ├── data
  │   └── nuscenes
  │       ├── maps
  │       ├── maps_bev_seg_gt_2class
  │       ├── nuscenes_infos_test_4d_interval3_max60.pkl
  │       ├── nuscenes_infos_train_4d_interval3_max60.pkl
  │       ├── nuscenes_infos_val_4d_interval3_max60.pkl
  │       ├── v1.0-test
  │       └── v1.0-trainval
```

### Pretraining

```
  .
  ├── pretrained_models
  │   └── cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth
```

### Evaluation

|  Model  | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS | Download |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| M0 | 0.2770 | 0.7647 | 0.2905 | 0.5334 | 0.4699 | 0.2131 | 0.4114 | [model](https://drive.google.com/file/d/1_L3y6LMV9BAFJw0XaZRTNo-kDlhgTS17/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1SFs2XWvO1kJvybwgrafAbuYy_7-DtUUP/view?usp=share_link) | 

### Deployment

## View Transformation Latency on device
[2D-to-3D on CUDA & CPU](https://github.com/Sense-GVT/Fast-BEV/tree/dev/script/view_tranform_cuda)

## Citation
```
@article{li2023fast,
  title={Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline},
  author={Li, Yangguang and Huang, Bin and Chen, Zeren and Cui, Yufeng and Liang, Feng and Shen, Mingzhu and Liu, Fenggang and Xie, Enze and Sheng, Lu and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2301.12511},
  year={2023}
}
```
