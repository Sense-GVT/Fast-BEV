# nuscenes数据集实验

## joint exps
|id|exp|bs|mAP|NDS|mIoU|IoU@road|IoU@lane|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|00|m2bev_det_v0.2_lossdir0.8_200x200bev_6+0_4conv_seg_box2d(paper)|24|0.340|0.401|0.523|-|-|
|01|m2bev_det_v0.2_lossdir0.8_200x200bev_6+0_4conv_seg_box2d(repro)|8|0.352|0.407|0.532|0.722|0.341|
|02|m2bev_det_v0.2_lossdir0.8_200x200bev_6+0_4conv_seg_box2d_wodcn|8|0.3237|0.3795|0.505|0.690|0.319|
|03|m2bev_det_v0.2_lossdir0.8_200x200bev_6+0_4conv_seg_box2d_r101_20ep|8|0.3598|0.4258|0.547|0.738|0.356|

## detection exps
|id|exp|size|bs|mAP|NDS|FPS|GPU|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|00|m2bev_det_v0.2_lossdir0.8_200x200bev_6conv|1x6x3x928x1600|8|0.355|0.409|4.7|8082MiB|

## 3dhead exps
|id|exp|size|bs|mAP|NDS|FPS|GPU|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|00|m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_tiny_pure|1x6x3x928x1600|8|0.3135|0.3631|-|-|
|01|m2bev_det_v0.2_lossdir0.8_200x200bev_6conv_wodcn_tiny_pure_20e+imgaug|1x6x3x928x1600|8|0.3042|0.3615|-|-|


## augment exps
|id|exp|bs|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|
|00|m2bev_baseline|16|0.3209|0.4004|
|06|m2bev_baseline+imgaug|8|0.4376|0.5186|
|07|m2bev_baseline+imgaug_v1.0|16|0.4842|0.5548|
|08|m2bev_baseline+imgaug_v1.1|16|0.4774|0.5500|
|09|m2bev_baseline+imgaug_v1.2|16|0.4905|0.5577|
