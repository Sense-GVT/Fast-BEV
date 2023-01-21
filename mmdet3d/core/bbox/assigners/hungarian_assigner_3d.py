# Copyright (c) OpenMMLab. All rights reserved.
# mmdet/core/bbox/assigners/hungarian_assigner.py
import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmcv.runner import get_dist_info
from mmdet3d.core.bbox.structures import xyzwhlr2xyzxyzr

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from IPython import embed
import random

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.0),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                 dir_cost=dict(type='DirL1Cost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        ##
        self.center_only = reg_cost.get('center_only', False)
        self.match3Dbox = reg_cost.get('match3Dbox', False)
        ##
        if 'center_only' in reg_cost:
            reg_cost.pop('center_only')
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.dir_cost = build_match_cost(dir_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               dir_pred,
               gt_bboxes,
               gt_labels,
               gt_dirs,
               img_meta,
               gt_bboxes_ignore=None,
               code_weight=None,
               eps=1e-7):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification cost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        if gt_bboxes.dtype == torch.float16:
            bbox_pred = bbox_pred.type(torch.float32)
            gt_bboxes = gt_bboxes.type(torch.float32)
        
        
        factor_xyz_whl_r_vxvy = gt_bboxes.new_tensor([100,100,8,20,20,20,1,1,1])
        '''debug 只考虑xy wh'''
        factor_xyz_whl_r_vxvy = factor_xyz_whl_r_vxvy[[0,1,3,4]]
        bbox_pred = bbox_pred[:, [0,1,3,4]]
        gt_bboxes = gt_bboxes[:, [0,1,3,4]]
        code_weight = code_weight[[0,1,3,4]]
        '''
        _bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        _gt_bboxes = bbox_cxcywh_to_xyxy(gt_bboxes)
        '''
        norm_bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred*factor_xyz_whl_r_vxvy) / factor_xyz_whl_r_vxvy[0]
        norm_gt_bboxes = bbox_cxcywh_to_xyxy(gt_bboxes*factor_xyz_whl_r_vxvy) / factor_xyz_whl_r_vxvy[0]
        
        real_bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred * factor_xyz_whl_r_vxvy)
        real_gt_bboxes = bbox_cxcywh_to_xyxy(gt_bboxes * factor_xyz_whl_r_vxvy)
        
        '''对于reg_cost 是对gt和pred归一化 并且转成xyxy去计算'''
        if code_weight is not None:
            if self.center_only:
                reg_cost = self.reg_cost((bbox_pred*code_weight)[:,:2],
                                         (gt_bboxes*code_weight)[:,:2])
            else:
                reg_cost = self.reg_cost(norm_bbox_pred*code_weight,
                                         norm_gt_bboxes*code_weight)
            iou_cost = self.iou_cost(real_bbox_pred*code_weight,
                                     real_gt_bboxes*code_weight)
            
        else:
            if self.center_only:
                reg_cost = self.reg_cost(bbox_pred[:,:2], gt_bboxes[:,:2])
            else:
                reg_cost = self.reg_cost(norm_bbox_pred, norm_gt_bboxes)
            iou_cost = self.iou_cost(real_bbox_pred, real_gt_bboxes)
        
        # dir cost
        dir_cost = self.dir_cost(dir_pred, gt_dirs)
        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost + dir_cost
        # embed(header='debug costs')    

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        assign_res = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        return assign_res
    
    
@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D_v1(BaseAssigner):
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.0),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                 dir_cost=dict(type='DirL1Cost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.dir_cost = build_match_cost(dir_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               dir_pred,
               gt_bboxes,
               gt_labels,
               gt_dirs,
               img_meta,
               gt_bboxes_ignore=None,
               assign_weight=None,
               eps=1e-7):
        assert gt_bboxes_ignore is None
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification cost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        if gt_bboxes.dtype == torch.float16:
            bbox_pred = bbox_pred.type(torch.float32)
            gt_bboxes = gt_bboxes.type(torch.float32)
        
        
        def hack_box_convert(box, factor):
            box = box * factor
            box = xyzwhlr2xyzxyzr(box)
            real_xyzxyz, rvxvy = box.split(6, dim=1)
            norm_xyzxyz = real_xyzxyz / factor[:3].repeat(2)
            real_box = torch.cat([real_xyzxyz, rvxvy], 1)
            norm_box = torch.cat([norm_xyzxyz, rvxvy], 1)
            return real_box, norm_box
        
        # 暂时写死这个 factor_xyz_whl_r_vxvy
        factor_xyz_whl_r_vxvy = gt_bboxes.new_tensor([100,100,8,20,20,20,1,1,1])
        real_bbox_preds, norm_bbox_preds = hack_box_convert(bbox_pred,
                                                            factor_xyz_whl_r_vxvy)
        real_bbox_gts, norm_bbox_gts = hack_box_convert(gt_bboxes,
                                                        factor_xyz_whl_r_vxvy)
        '''
        目前gt和pred都转成了xyzxyz_r_vxvy
        match只考虑xyzxyz'''
        factor_xyzxyz = factor_xyz_whl_r_vxvy[:3].repeat(2)
        '''对于reg_cost 是对gt和pred归一化 并且转成xyxy去计算'''
        assert assign_weight is not None
        reg_cost = self.reg_cost(norm_bbox_preds*assign_weight,
                                 norm_bbox_gts*assign_weight)
        # 先不考虑IoU cost
        real_bbox_gts_xyzxyz = real_bbox_gts[:,:6] * assign_weight[:6]
        real_bbox_preds_xyzxyz = real_bbox_preds[:,:6] * assign_weight[:6]
        iou_cost = self.iou_cost(real_bbox_preds_xyzxyz, real_bbox_gts_xyzxyz)
        # dir cost
        dir_cost = self.dir_cost(dir_pred, gt_dirs)
        
        # weighted sum of above all costs
        cost = cls_cost + reg_cost + iou_cost + dir_cost        
        
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        assign_res = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        return assign_res