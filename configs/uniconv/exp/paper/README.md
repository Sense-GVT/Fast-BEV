# nuscenes数据集实验

## m exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.2_m0_r18_s256x704_v200x200x4_c192_d2|32|20|0.2841|0.4272|
|00|uniconv_v0.2_m1_r34_s256x704_v200x200x4_c224_d4|32|20|0.3256|0.4695|
|00|uniconv_v0.2_m2_r34_s320x880_v250x250x6_c224_d4|32|20|0.3323|0.4716|
|00|uniconv_v0.2_m3_r50_s256x704_v200x200x6_c256_d6|32|20|0.3345|0.4731|
|00|uniconv_v0.2_m4_r50_s320x880_v250x250x6_c256_d6|32|20|0.3456|0.4824|
|00|uniconv_v0.2_m5_r50_s384x1056_v300x300x6_c256_d6|32|20|0.3485|0.4875|

## epoch exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_e12_s256x704_v200x200x6|32|12|0.2578|0.3299|
|00|uniconv_v0.1_e24_s256x704_v200x200x6|32|24|0.2565|0.3377|
|00|uniconv_v0.1_e36_s256x704_v200x200x6|32|36|0.2482|0.3199|
|00|uniconv_v0.1_e48_s256x704_v200x200x6|32|48|0.2472|0.3286|
|00|uniconv_v0.1_e12_s256x704_v200x200x6_seq2.0_ibaug_v2|32|12|0.2730|0.3810|
|00|uniconv_v0.1_e24_s256x704_v200x200x6_seq2.0_ibaug_v2|32|24|0.2941|0.4240|
|00|uniconv_v0.1_e36_s256x704_v200x200x6_seq2.0_ibaug_v2|32|36|0.3104|0.4403|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2|32|48|0.3206|0.4509|

## detach exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_e48_s256x704_v200x200x6|32|48|0.2472|0.3286|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2|32|48|0.3206|0.4509|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2_detach|32|48|0.2508|0.3899|

## cp exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2|32|48|0.3206|0.4509|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2_cp|32|48|0.3195|0.4451|

## aug exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_e48_s256x704_v200x200x6|32|48|0.2472|0.3286|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_imgaug|32|48|0.2853|0.3569|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_bevaug_v2|32|48|0.2650|0.3516|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_ibaug_v2|32|48|0.2926|0.3728|

## sequential exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_e48_s256x704_v200x200x6|32|48|0.2472|0.3286|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0|32|48|0.2692|0.4004|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_ibaug_v2|32|48|0.2926|0.3728|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2|32|48|0.3206|0.4509|

## image size exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_e48_s256x448_v200x200x6_seq2.0_ibaug_v2|32|48|0.2802|0.4191|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2|32|48|0.3206|0.4509|
|00|uniconv_v0.1_e48_s464x800_v200x200x6_seq2.0_ibaug_v2|32|48|0.3424|0.4656|
|00|uniconv_v0.1_e48_s544x960_v200x200x6_seq2.0_ibaug_v2|32|48|0.3448|0.4723|
|00|uniconv_v0.1_e48_s704x1208_v200x200x6_seq2.0_ibaug_v2|32|48|0.3583|0.4778|
|00|uniconv_v0.1_e48_s832x1440_v200x200x6_seq2.0_ibaug_v2|32|48|0.3680|0.4913|
|00|uniconv_v0.1_e48_s928x1600_v200x200x6_seq2.0_ibaug_v2_cp|32|48|0.3691|0.4876|

## voxel size exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_e48_s256x704_v200x200x6_seq2.0_ibaug_v2|32|48|0.3206|0.4509|
|00|uniconv_v0.1_e48_s256x704_v200x200x12_seq2.0_ibaug_v2|32|48|0.3240|0.4535|
|00|uniconv_v0.1_e48_s256x704_v400x400x6_seq2.0_ibaug_v2|32|48|0.3045|0.4415|
|00|uniconv_v0.1_e48_s256x704_v400x400x12_seq2.0_ibaug_v2|32|48|||

## 2d backbone exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_r18_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs|32|20|0.2925|0.4372|
|00|uniconv_v0.1_r18_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs_S|32|20|0.3023|0.4413|
|00|uniconv_v0.1_r18_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs_S_Tf|32|20|0.3079|0.4452|
|00|uniconv_v0.1_r18_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs_S_Tfs|32|20|0.3115|0.4444
|00|uniconv_v0.1_r50_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs|32|20|0.3345|0.4731|
|00|uniconv_v0.1_r50_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs_S|32|20|0.3431|0.4769|
|00|uniconv_v0.1_r50_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs_S_Tf|32|20|0.3457|0.4774|
|00|uniconv_v0.1_r101_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs|32|20|0.3452|0.4817|
|00|uniconv_v0.1_r101_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs_S|32|20|0.3523|0.4852|
|00|uniconv_v0.1_r101_e20_s256x704_v200x200x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs_S_Tf|32|20|0.3574|0.4907|

## bigmodel exps
|id|exp|bs|epoch|mAP|NDS|
|:-:|:-:|:-:|:-:|:-:|:-:|
|00|uniconv_v0.1_r50_dcn_e48_s928x1600_v400x400x12_seq2.0_ibaug_v2|32|48|0.3822|0.5034|
|00|uniconv_v0.1_r101_dcn_e48_s928x1600_v400x400x12_seq2.0_ibaug_v2|32|48|0.3829|0.5092|
|00|uniconv_v0.1_r101_e20_s928x1600_v400x400x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs|32|48|0.3660|0.4992|
|00|uniconv_v0.1_r101_dcn_e20_s928x1600_v400x400x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs@epoch13|32|48|0.4018|0.5308|
|00|uniconv_v0.1_r101_dcn_e20_s928x1600_v400x400x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs@epoch13_S|32|48|0.4066|0.5331|
|00|uniconv_v0.1_r101_dcn_e20_s928x1600_v400x400x6_seq3.1_ibaug_v2_interval1_f4s135_cbgs@epoch13_S_Tf|32|48|0.4126|0.5354|
