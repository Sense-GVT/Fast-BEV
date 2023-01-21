# -*- coding: utf-8 -*-
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16

import copy
import ipdb  # noqa


def build_bbox_head(head_cfg, train_cfg, test_cfg):
    head_cfg.update(train_cfg=train_cfg)
    head_cfg.update(test_cfg=test_cfg)
    bbox_head = build_head(head_cfg)
    return bbox_head


@DETECTORS.register_module()
class MultiScaleM2BevNetSeq(BaseDetector):
    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        bbox_head,
        seg_head,
        n_voxels,
        voxel_size,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        seq_detach=False,
        multi_scale_id=None,
        multi_scale_3d_scaler=None,
        z_fuse=None,
        mlvl_anchor_sizes=None,
        mlvl_tgt_volume=None,
        with_cp=False,
        style="v4",
        debug=False,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if neck_fuse is None:
            self.neck_fuse = None
        else:
            if isinstance(neck_fuse['in_channels'], list):
                for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                    self.add_module(
                        f'neck_fuse_{i}', 
                        nn.Conv2d(in_channels, out_channels, 3, 1, 1))
            else:
                self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        self.z_fuse = z_fuse
        if self.z_fuse is not None:
            # z_fuse['type']
            for i, (in_channels, out_channels) in enumerate(zip(self.z_fuse['in_channels'], self.z_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_z_{i}',
                    nn.Conv2d(in_channels, out_channels, 1, 1))

        # style
        self.style = style
        # v4: multi-scale 2d + late/early fusion
        # v5: multi-scale 2d/3d (different x/y, same z) + wo fusion + multi-head
        # v6: multi-scale 2d/3d (different x/y, same z) + wo fusion + share-head
        # v7: multi-scale 2d/3d (different x/y/z) + x/y pooling/upsample + late/early fusion
        assert self.style in ["v4", "v5", "v6", "v7"], self.style
        self.multi_scale_id = multi_scale_id
        self.mlvl_tgt_volume = mlvl_tgt_volume
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        if bbox_head is not None:
            if self.style in ['v4', 'v6', 'v7']:
                self.bbox_head = build_bbox_head(bbox_head, train_cfg, test_cfg)
                self.bbox_head.voxel_size = voxel_size
            else:  # v5 multi-head
                self.bbox_head = None
                for lvl in range(len(self.multi_scale_id)):
                    bbox_head_cfg = copy.deepcopy(bbox_head)
                    bbox_head_cfg['anchor_generator']['sizes'] = mlvl_anchor_sizes[lvl]
                    self.add_module(f'bbox_head_lvl{lvl}', build_bbox_head(bbox_head_cfg, train_cfg, test_cfg))
                    getattr(self, f'bbox_head_lvl{lvl}').voxel_size = voxel_size[lvl]
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = build_head(bbox_head_2d)
        else:
            self.bbox_head_2d = None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # detach adj feature
        self.seq_detach = seq_detach
        # checkpoint
        self.with_cp = with_cp

        self.debug = debug

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def extract_feat(self, img, img_metas, mode):
        batch_size = img.shape[0]
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        x = self.backbone(
            img
        )  # [6, 256, 232, 400]; [6, 512, 116, 200]; [6, 1024, 58, 100]; [6, 2048, 29, 50]

        # use for vovnet
        if isinstance(x, dict):
            tmp = []
            for k in x.keys():
                tmp.append(x[k])
            x = tmp

        # fuse features
        def _inner_forward(x):
            out = self.neck(x)
            return out  # [6, 64, 232, 400]; [6, 64, 116, 200]; [6, 64, 58, 100]; [6, 64, 29, 50])

        if self.with_cp and x.requires_grad:
            mlvl_feats = cp.checkpoint(_inner_forward, x)
        else:
            mlvl_feats = _inner_forward(x)
        mlvl_feats = list(mlvl_feats)

        features_2d = None
        if self.bbox_head_2d:
            features_2d = mlvl_feats

        if self.debug:
            debug = []

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    # v4/v7 early fusion
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    # wo fusion
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*6:(seq_id+1)*6]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    if self.style == 'v4':
                        # v4 single-scale 3d
                        n_voxels, voxel_size = self.n_voxels, self.voxel_size
                    else:  
                        # v5/v6/v7 multi-scale 3d
                        n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    points = get_points(  # [3, vx, vy, vz]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                    ).to(feat_i.device)

                    if self.debug:
                        debug.append({
                            'lvl': lvl,
                            'stride': stride_i,
                            'height': height,
                            'width': width,
                            'feature_raw': feat_i,
                            'points': points,
                            'projection': projection,
                        })

                    # volume, debug_dict = backproject_v4(
                        # feat_i[:, :, :height, :width], points, projection, self.debug)  # [c, vx, vy, vz]
                    debug_dict = {}
                    volume, valid = backproject(
                        feat_i[:, :, :height, :width], points, projection)
                    volume = volume.sum(dim=0)
                    valid = valid.sum(dim=0)
                    volume = volume / valid
                    valid = valid > 0
                    volume[:, ~valid[0]] = 0.0

                    volumes.append(volume)
                    if self.debug:
                        debug[-1].update({'backp': debug_dict})
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])

            if self.z_fuse is not None:
                for i in range(len(volume_list)):  # seq
                    z_volumes = []
                    for j, z_slice in enumerate(self.z_fuse['z_slice']):
                        z_volume = volume_list[i][..., z_slice]
                        bs, c, x, y, z = z_volume.shape
                        # [bs, c, x, y, z'] -> [bs, c*z', x, y]
                        z_volume = z_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z*c).permute(0, 3, 1, 2)
                        # [bs, c', x, y]
                        z_volume = getattr(self, f'neck_fuse_z_{j}')(z_volume)
                        z_volumes.append(z_volume)
                    z_volumes = torch.cat(z_volumes, dim=1).unsqueeze(-1)  # fake z
                    volume_list[i] = z_volumes
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])
        
        if self.style == 'v4':
            # v4 single-scale 3d
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
        elif self.style == 'v7': 
            # v7 multi-scale 3d (different x/y/z)
            for i in range(len(mlvl_volumes)):
                mlvl_volume = mlvl_volumes[i]
                bs, c, x, y, z = mlvl_volume.shape
                # collapse h, [bs, seq*c, vx, vy, vz] -> [bs, seq*c*vz, vx, vy]
                mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z*c).permute(0, 3, 1, 2)
                
                # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
                if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                    # pooling to bottom level
                    mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
                elif self.multi_scale_3d_scaler == 'upsample' and i != 0:  
                    # upsampling to top level 
                    mlvl_volume = resize(
                        mlvl_volume,
                        mlvl_volumes[0].size()[2:4],
                        mode='bilinear',
                        align_corners=False)
                else:
                    # same x/y
                    pass

                # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
                mlvl_volume = mlvl_volume.unsqueeze(-1)
                mlvl_volumes[i] = mlvl_volume
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]
        else:
            # v5/v6 multi-scale 3d (different x/y, same z) wo fusion
            pass

        if self.debug:
            torch.save({'debug': debug}, '/mnt/lustre/chenzeren/debug2.pth')
            print('all done')
            exit(0)

        x = mlvl_volumes
        if self.style in ['v4', 'v7']:
            def _inner_forward(x):
                # v4 late fusion: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy]
                # v7 late fusion: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
                out = self.neck_3d(x)
                return out
            
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = _inner_forward(x)
        else:
            # v5/v6 wo fusion: list([bs, seq*c, vx, vy, vz]) -> list([bs, c', vx, vy])
            # not support checkpoint
            x = [self.neck_3d(x_lvl)[0] for x_lvl in x]

        return x, None, features_2d

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            if kwargs["export_2d"]:
                return self.onnx_export_2d(img, img_metas)
            elif kwargs["export_3d"]:
                return self.onnx_export_3d(img, img_metas)
            else:
                raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
        self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    ):
        feature_bev, valids, features_2d = self.extract_feat(img, img_metas, "train")
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None or \
            getattr(self, f'bbox_head_lvl0', None) is not None or \
            self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None or getattr(self, f'bbox_head_lvl0', None) is not None:
            if self.style in ['v4', 'v6', 'v7']:
                x = self.bbox_head(feature_bev)
                loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
                losses.update(loss_det)

            else:  # v5
                raise NotImplementedError
                # mlvl_gt_bboxes_3d, mlvl_gt_labels_3d = self.prepare_mlvl_tgt(gt_bboxes_3d, gt_labels_3d)
                # for lvl, bev_lvl in zip(self.multi_scale_id, feature_bev):
                #     x_lvl = getattr(self, f'bbox_head_lvl{lvl}')([bev_lvl])
                #     loss_det = getattr(self, f'bbox_head_lvl{lvl}').loss(
                #         *x_lvl, mlvl_gt_bboxes_3d[lvl], mlvl_gt_labels_3d[lvl], img_metas)
                #     losses.update({f'{k}_{lvl}': v for k, v in loss_det.items()})

        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            gt_bev = gt_bev_seg[0][None, ...].long()
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        if self.bbox_head_2d is not None:
            gt_bboxes = kwargs["gt_bboxes"][0]
            gt_labels = kwargs["gt_labels"][0]
            assert len(kwargs["gt_bboxes"]) == 1 and len(kwargs["gt_labels"]) == 1
            # hack a img_metas_2d
            img_metas_2d = []
            img_info = img_metas[0]["img_info"]
            for idx, info in enumerate(img_info):
                tmp_dict = dict(
                    filename=info["filename"],
                    ori_filename=info["filename"].split("/")[-1],
                    ori_shape=img_metas[0]["ori_shape"],
                    img_shape=img_metas[0]["img_shape"],
                    pad_shape=img_metas[0]["pad_shape"],
                    scale_factor=img_metas[0]["scale_factor"],
                    flip=False,
                    flip_direction=None,
                )
                img_metas_2d.append(tmp_dict)

            rank, world_size = get_dist_info()
            loss_2d = self.bbox_head_2d.forward_train(
                features_2d, img_metas_2d, gt_bboxes, gt_labels
            )
            losses.update(loss_2d)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        if not self.test_cfg.get('use_tta', False):
            return self.simple_test(img, img_metas)
        return self.aug_test(img, img_metas)

    def onnx_export_2d(self, img, img_metas):
        """
        input: 6, 3, 544, 960
        output: 6, 64, 136, 240
        """
        x = self.backbone(img)
        c1, c2, c3, c4 = self.neck(x)
        c2 = resize(
            c2, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c3 = resize(
            c3, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c4 = resize(
            c4, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.neck_fuse(x)

        if bool(os.getenv("DEPLOY", False)):
            x = x.permute(0, 2, 3, 1)
            return x

        return x

    def onnx_export_3d(self, x, _):
        # x: [6, 200, 100, 3, 256]
        # if bool(os.getenv("DEPLOY_DEBUG", False)):
        #     x = x.sum(dim=0, keepdim=True)
        #     return [x]
        if self.style == "v1":
            x = x.sum(dim=0, keepdim=True)  # [1, 200, 100, 3, 256]
            x = self.neck_3d(x)  # [[1, 256, 100, 50], ]
        elif self.style == "v2":
            x = self.neck_3d(x)  # [6, 256, 100, 50]
            x = [x[0].sum(dim=0, keepdim=True)]  # [1, 256, 100, 50]
        elif self.style == "v3":
            x = self.neck_3d(x)  # [1, 256, 100, 50]
        else:
            raise NotImplementedError

        if self.bbox_head is not None:
            cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
            cls_score = [item.sigmoid() for item in cls_score]

        if os.getenv("DEPLOY", False):
            if dir_cls_preds is None:
                x = [cls_score, bbox_pred]
            else:
                x = [cls_score, bbox_pred, dir_cls_preds]
            return x

        return x

    def simple_test(self, img, img_metas):
        bbox_results = []
        feature_bev, _, features_2d = self.extract_feat(img, img_metas, "test")
        if self.bbox_head is not None or getattr(self, f'bbox_head_lvl0', None) is not None:
            if self.style in ['v4', 'v6', 'v7']:
                x = self.bbox_head(feature_bev)
                bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
                bbox_results = [
                    bbox3d2result(det_bboxes, det_scores, det_labels)
                    for det_bboxes, det_scores, det_labels in bbox_list
                ]
            else:  # v5
                mlvl_cls_scores, mlvl_bbox_preds, mlvl_dir_cls_preds = [], [], []
                for lvl, bev_lvl in zip(self.multi_scale_id, feature_bev):
                    cls_score, bbox_pred, dir_cls_pred = getattr(self, f'bbox_head_lvl{lvl}')([bev_lvl])
                    mlvl_cls_scores.append(cls_score)
                    mlvl_bbox_preds.append(bbox_pred)
                    mlvl_dir_cls_preds.append(dir_cls_pred)

                # TODO
                bbox_results = [dict()]
                # bbox_list = self.bbox_head[0].get_bboxes(
                #     cls_scores, bbox_preds, dir_cls_preds, img_metas, valid=None)
                # bbox_results = [
                #     bbox3d2result(det_bboxes, det_scores, det_labels)
                #     for det_bboxes, det_scores, det_labels in bbox_list
                # ]
        else:
            bbox_results = [dict()]

        # BEV semantic seg
        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            bbox_results[0]['bev_seg'] = x_bev

        return bbox_results

    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):

            img_metas[0]['img_shape'] = img_shape_copy[24*tta_id:24*(tta_id+1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24*tta_id:24*(tta_id+1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24*tta_id:24*(tta_id+1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


def backproject_v4(features, points, projection, debug=False):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    if debug:
        save_dict = {}
        save_dict.update({
            'features_raw': features,
            'points_raw': points,
            'projection_raw': projection,
        })

    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    if debug:
        save_dict.update({
            'points': points,
            'points_proj': points_2d_3,
            'x': x,
            'y': y,
            'z': z,
            'valid': valid,
            'volume': volume
        })

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    if debug:
        return volume, save_dict
    return volume, None
