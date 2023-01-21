## 3D LiDAR BBox Detection Evaluation
This code allows you to evaluate your `3D bounding box predictions with ground truth labels using **(approximately)** the NuScenes standards of evaluation. This won't do NMS or score thresholding for you, just evaluate what you input.

**DISCLAMER**: This isn't an exact replication of the NuScenes benchmarking code. Their code is really difficult to use & understand so maybe they do something different I didn't notice, I just implemented it to the best of my understanding of the documentation & code. My mAP calculation is slightly simpler & I also don't calculate mAVE, mAAE and NDS. Feel free to use it to compare between the models you have internally, but I am not encouraging you to compare the results of my code with the official NuScenes benchmarks. This code is (hopefully) readable so please review it fully before using it for something more serious than internal comparison of models.

Requirements:
* numpy
* matplotlib

### Usage
You should two folders with .txt or .csv prediction and ground truth labels with corresponding file names.
```shell script
python nuscenes_eval.py --pred_labels path/to/pred_labels/ --gt_labels path/to/gt_labels/ --format "class x y z l w h r score" --classes "Car, Pedestrian, Cyclist"
```
The format string defines your input label format. It can be anything but those 8 fields "class x y z l w h r score" must be defined, other fields will be ignored. If you have extra fields you can end up with something like:
```shell script
--format "timestamp class color x y z field1 field2 field3 l w h r pitch roll score id"
```

### Metrics
The [NuScenes Detection Benchmark](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) is more descriptive than the KITTI one and integrates the precision-recall curve instead of an 11-point approximation, but also doesn't stick to Intersection-of-Union standard for determining true positives - instead it just considers 2D euclidean distance. It's a bit of a BEV evaluation though height error is considered in the ASE metrics.

The NuScenes evaluation code is open source but embedded deep in their API and not really usable on other custom data. So this allows you to obtain a similar evaluation on any sets of labels. Here is a description of the metrics:
Average Precision metric

#### mean Average Precision (mAP)
**NuScenes** state: "define a match by considering the 2D center distance on the ground plane rather than intersection over union[...]. Specifically, we match predictions with the ground truth objects that have the smallest center-distance up to a certain threshold. For a given match threshold we calculate average precision (AP) by integrating the recall vs precision curve for recalls and precisions > 0.1. We finally average over match thresholds of {0.5, 1, 2, 4} meters and compute the mean across classes."

**This code** approaches AP the same way, but for a single threshold (so far) and keeps class-based results separate. We also include F1 score.

#### True Positive metrics
NuScenes has true positives metrics comparing positive matches at threshold = 2.0 meters with the ground truth. The metrics implemented here are:

**Average Translation Error (ATE)**: Euclidean center distance in 2D in meters. We implement ATE2D and ATE3D in this code.

**Average Scale Error (ASE)**: Calculated as 1 - IOU after aligning centers and orientation. Currently just smallest ratio of volumes, but that's not exactly right.

**Average Orientation Error (AOE)**: Smallest yaw angle difference between prediction and ground-truth in radians.

####Change log
**0.1** (2020-10-07): alpha release with minimal testing and some approximations.

#### To do:
* Save precision-recall curves and results in some file.
* Fix ASE approximation to calculate 3D IOU exactly.
* Improve input flexibility such as BEV only, no rotation input, no score input (though all treated as same score will lead to inconsistent eval)
* Figure out if F1-score is actually correct (is it maximum harmonic mean of precision-recall pairs, or mean of all precision recall pairs?)
* Implement mAP as average of many thresholds and True Positive Metrics from one threshold.
