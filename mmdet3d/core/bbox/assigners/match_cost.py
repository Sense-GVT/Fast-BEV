# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet3d.core.bbox.iou_calculators import axis_aligned_bbox_overlaps_3d


@MATCH_COST.register_module()
class L1Cost:
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, dir_pred, gt_dirs):
        dir_cost = torch.cdist(dir_pred, gt_dirs, p=1)
        return dir_cost * self.weight

    
@MATCH_COST.register_module()
class CrossEntropyCost:
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, dir_pred, gt_dirs):
        dir_cost = torch.cdist(dir_pred, gt_dirs, p=1)
        return dir_cost * self.weight

    
@MATCH_COST.register_module()
class IoU3DCost:
    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        assert bboxes.shape[-1] == 6 and gt_bboxes.shape[-1] == 6
        overlaps = axis_aligned_bbox_overlaps_3d(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight