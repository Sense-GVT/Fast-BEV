# UniconvNV@cla

## cla exps@p100e40
|id|exp|bs|epoch|CAR|TRUCK|BICYCLE|PEDESTRIAN|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|internal_detr3d_res50_scale05_side50_ceph_0616_wodeform_40e_2lr@e40|32x2|40|0.760|0.701|0.530|0.157|
|0|uniconv_cla_v0.2.2_s540x960_v425x250x6_e40_interval1.0@e27|64|40|0.778|0.645|0.464|0.140|
|0|uniconv_cla_v0.2.2_s540x960_v425x250x6_e40_interval1.0@e36|64|40|0.743|0.621|0.445|0.177|
|0|uniconv_cla_v0.2.2_s540x960_v425x250x6_e40_interval1.0@e40|64|40|0.732|0.635|0.416|0.19|
|0|uniconv_cla_v0.2.2_z1_s540x960_v425x250x6_ibaug_e40_interval1.0@e28|64|40|0.713|0.718|0.230|0.076|
|0|uniconv_cla_v0.2.2_z1_s540x960_v425x250x6_ibaug_e40_interval1.0@e37|64|40|0.728|0.715|0.251|0.078|
|0|uniconv_cla_v0.2.2_z1_s540x960_v425x250x6_ibaug_e40_interval1.0@e40|64|40|0.727|0.709|0.256|0.077|
|0|uniconv_cla_v0.2.2_z1.1_s540x960_v425x250x6_ibaug_e40_interval1.0_lastft5@e40|64|40|0.783|0.727|0.398|0.087|

## cla exps@p10e12
|id|exp|bs|epoch|CAR|TRUCK|BICYCLE|PEDESTRIAN|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|uniconv_cla_t0.2.2_s540x960_v425x250x6_e12_interval10.0|32|12|0.704|0.567|0.439|0.058|
|1|uniconv_cla_t0.2.2_z1_s540x960_v425x250x6_iaug_e12_interval10.0|32|12|0.678|0.626|0.225|0.058|
|1.1|uniconv_cla_t0.2.2_z1.1_s540x960_v425x250x6_iaug-resize_e12_interval10.0|32|12|0.691|0.626|0.243|0.058|
|1.2|uniconv_cla_t0.2.2_z1.2_s540x960_v425x250x6_iaug-crop_e12_interval10.0|32|12|0.691|0.626|0.267|0.059|
|1.3|uniconv_cla_t0.2.2_z1.3_s540x960_v425x250x6_iaug-rot_e12_interval10.0|32|12|0.688|0.627|0.285|0.055|
|1.4|uniconv_cla_t0.2.2_z1.4_s540x960_v425x250x6_iaug-flip_e12_interval10.0|32|12|0.717|0.609|0.349|0.062|
|2|uniconv_cla_t0.2.2_z2_s540x960_v425x250x6_baug_e12_interval10.0|32|12|0.663|0.538|0.192|0.055|
|2.1|uniconv_cla_t0.2.2_z2.1_s540x960_v425x250x6_baug-flip_e12_interval10.0|32|12|0.709|0.564|0.424|0.075|
|2.2.1|uniconv_cla_t0.2.2_z2.2.1_s540x960_v425x250x6_baug-rot_e12_interval10.0|32|12|0.700|0.578|0.403|0.058|
|2.2.2|uniconv_cla_t0.2.2_z2.2.2_s540x960_v425x250x6_baug-scale_e12_interval10.0|32|12|0.642|0.539|0.188|0.046|
|3|uniconv_cla_t0.2.2_z3_s540x960_v425x250x6_ibaug_e12_interval10.0|32|12|0.648|0.537|0.171|0.057|

### sort@CAR
 - good: iaug-flip, baug-flip, noaug
 - ok: baug-rot, iaug-resize, iaug-crop, iaug-rot
 - bad: baug-scale

|id|exp|bs|epoch|CAR|TRUCK|BICYCLE|PEDESTRIAN|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1.4|uniconv_cla_t0.2.2_z1.4_s540x960_v425x250x6_iaug-flip_e12_interval10.0|32|12|0.717|0.609|0.349|0.062|
|2.1|uniconv_cla_t0.2.2_z2.1_s540x960_v425x250x6_baug-flip_e12_interval10.0|32|12|0.709|0.564|0.424|0.075|
|0|uniconv_cla_t0.2.2_s540x960_v425x250x6_e12_interval10.0|32|12|0.704|0.567|0.439|0.058|
|2.2.1|uniconv_cla_t0.2.2_z2.2.1_s540x960_v425x250x6_baug-rot_e12_interval10.0|32|12|0.700|0.578|0.403|0.058|
|1.1|uniconv_cla_t0.2.2_z1.1_s540x960_v425x250x6_iaug-resize_e12_interval10.0|32|12|0.691|0.626|0.243|0.058|
|1.2|uniconv_cla_t0.2.2_z1.2_s540x960_v425x250x6_iaug-crop_e12_interval10.0|32|12|0.691|0.626|0.267|0.059|
|1.3|uniconv_cla_t0.2.2_z1.3_s540x960_v425x250x6_iaug-rot_e12_interval10.0|32|12|0.688|0.627|0.285|0.055|
|2.2.2|uniconv_cla_t0.2.2_z2.2.2_s540x960_v425x250x6_baug-scale_e12_interval10.0|32|12|0.642|0.539|0.188|0.046|

### sort@TRUCK
 - good: iaug-rot, iaug-resize, iaug-crop, iaug-flip, baug-rot, noaug
 - ok: baug-flip
 - bad: baug-scale

|id|exp|bs|epoch|CAR|TRUCK|BICYCLE|PEDESTRIAN|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1.3|uniconv_cla_t0.2.2_z1.3_s540x960_v425x250x6_iaug-rot_e12_interval10.0|32|12|0.688|0.627|0.285|0.055|
|1.1|uniconv_cla_t0.2.2_z1.1_s540x960_v425x250x6_iaug-resize_e12_interval10.0|32|12|0.691|0.626|0.243|0.058|
|1.2|uniconv_cla_t0.2.2_z1.2_s540x960_v425x250x6_iaug-crop_e12_interval10.0|32|12|0.691|0.626|0.267|0.059|
|1.4|uniconv_cla_t0.2.2_z1.4_s540x960_v425x250x6_iaug-flip_e12_interval10.0|32|12|0.717|0.609|0.349|0.062|
|2.2.1|uniconv_cla_t0.2.2_z2.2.1_s540x960_v425x250x6_baug-rot_e12_interval10.0|32|12|0.700|0.578|0.403|0.058|
|0|uniconv_cla_t0.2.2_s540x960_v425x250x6_e12_interval10.0|32|12|0.704|0.567|0.439|0.058|
|2.1|uniconv_cla_t0.2.2_z2.1_s540x960_v425x250x6_baug-flip_e12_interval10.0|32|12|0.709|0.564|0.424|0.075|
|2.2.2|uniconv_cla_t0.2.2_z2.2.2_s540x960_v425x250x6_baug-scale_e12_interval10.0|32|12|0.642|0.539|0.188|0.046|

### sort@BICYCLE
 - good: noaug
 - ok: baug-flip, baug-rot
 - bad: iaug-flip, iaug-rot, iaug-crop, iaug-resize, baug-scale

|id|exp|bs|epoch|CAR|TRUCK|BICYCLE|PEDESTRIAN|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|uniconv_cla_t0.2.2_s540x960_v425x250x6_e12_interval10.0|32|12|0.704|0.567|0.439|0.058|
|2.1|uniconv_cla_t0.2.2_z2.1_s540x960_v425x250x6_baug-flip_e12_interval10.0|32|12|0.709|0.564|0.424|0.075|
|2.2.1|uniconv_cla_t0.2.2_z2.2.1_s540x960_v425x250x6_baug-rot_e12_interval10.0|32|12|0.700|0.578|0.403|0.058|
|1.4|uniconv_cla_t0.2.2_z1.4_s540x960_v425x250x6_iaug-flip_e12_interval10.0|32|12|0.717|0.609|0.349|0.062|
|1.3|uniconv_cla_t0.2.2_z1.3_s540x960_v425x250x6_iaug-rot_e12_interval10.0|32|12|0.688|0.627|0.285|0.055|
|1.2|uniconv_cla_t0.2.2_z1.2_s540x960_v425x250x6_iaug-crop_e12_interval10.0|32|12|0.691|0.626|0.267|0.059|
|1.1|uniconv_cla_t0.2.2_z1.1_s540x960_v425x250x6_iaug-resize_e12_interval10.0|32|12|0.691|0.626|0.243|0.058|
|2.2.2|uniconv_cla_t0.2.2_z2.2.2_s540x960_v425x250x6_baug-scale_e12_interval10.0|32|12|0.642|0.539|0.188|0.046|

### sort@PEDESTRIAN
 - good: baug-flip, iaug-flip, iaug-crop, noaug
 - ok: baug-rot, iaug-resize, iaug-rot
 - bad: baug-scale

|id|exp|bs|epoch|CAR|TRUCK|BICYCLE|PEDESTRIAN|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|2.1|uniconv_cla_t0.2.2_z2.1_s540x960_v425x250x6_baug-flip_e12_interval10.0|32|12|0.709|0.564|0.424|0.075|
|1.4|uniconv_cla_t0.2.2_z1.4_s540x960_v425x250x6_iaug-flip_e12_interval10.0|32|12|0.717|0.609|0.349|0.062|
|1.2|uniconv_cla_t0.2.2_z1.2_s540x960_v425x250x6_iaug-crop_e12_interval10.0|32|12|0.691|0.626|0.267|0.059|
|0|uniconv_cla_t0.2.2_s540x960_v425x250x6_e12_interval10.0|32|12|0.704|0.567|0.439|0.058|
|2.2.1|uniconv_cla_t0.2.2_z2.2.1_s540x960_v425x250x6_baug-rot_e12_interval10.0|32|12|0.700|0.578|0.403|0.058|
|1.1|uniconv_cla_t0.2.2_z1.1_s540x960_v425x250x6_iaug-resize_e12_interval10.0|32|12|0.691|0.626|0.243|0.058|
|1.3|uniconv_cla_t0.2.2_z1.3_s540x960_v425x250x6_iaug-rot_e12_interval10.0|32|12|0.688|0.627|0.285|0.055|
|2.2.2|uniconv_cla_t0.2.2_z2.2.2_s540x960_v425x250x6_baug-scale_e12_interval10.0|32|12|0.642|0.539|0.188|0.046|
