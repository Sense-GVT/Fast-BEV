# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16

import copy
import ipdb  # noqa


@DETECTORS.register_module()
class M2BevNetSeq(BaseDetector):
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
        with_cp=False,
        style="v1",
        feature_align=None,
        debug=False,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_fuse = nn.Conv2d(
            neck_fuse["in_channels"],
            neck_fuse["out_channels"],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.neck_3d = build_neck(neck_3d)

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.voxel_size = voxel_size
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
        # style
        self.style = style
        assert self.style in ["v1", "v2", "v3", "v4"], self.style

        self.feature_align = feature_align
        if self.feature_align is not None:
            if feature_align['type'] == 'pool':
                self.align_pool = nn.AdaptiveAvgPool2d(feature_align['output_size'])
            elif feature_align['type'] == 'top_pool':
                self.align_pool = nn.AdaptiveAvgPool2d(feature_align['output_size'])
            elif feature_align['type'] == 'top_crop':
                self.align_crop = feature_align['crop_val']
            else:
                raise RuntimeError()

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
            c1, c2, c3, c4 = cp.checkpoint(_inner_forward, x)
        else:
            c1, c2, c3, c4 = _inner_forward(x)

        features_2d = None
        if self.bbox_head_2d:
            features_2d = [c1, c2, c3, c4]

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

        def _inner_forward(x):
            out = self.neck_fuse(x)  # [6, 64, 232, 400]
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        if self.feature_align is not None:
            if self.feature_align['type'] == 'pool':
                x = self.align_pool(x)
                stride = img.shape[-1] / x.shape[-1]
            elif self.feature_align['type'] == 'top_pool':
                pool_range = self.feature_align['pool_range']
                x_split = torch.split(x, [pool_range, x.shape[2] - pool_range], dim=2)
                pooled_x_top = self.align_pool(x_split[0])
                pooled_x_top = resize(pooled_x_top, (pool_range, x.shape[3]), mode='nearest')
                x = torch.cat([pooled_x_top, x_split[1]], dim=2).contiguous()

                stride = img.shape[-1] // x.shape[-1]  # 4.0
                assert stride == 4
                stride = int(stride)
        else:
            stride = img.shape[-1] // x.shape[-1]  # 4.0
            assert stride == 4
            stride = int(stride)

        x = x.reshape([batch_size, -1] + list(x.shape[1:]))  # [1, 6, 64, 232, 400]

        if self.debug:
            save_dict = {}
        # reconstruct 3d voxels
        x_list = []
        valids_list = []
        x_split = torch.split(x, 6, dim=1)
        for seq_id in range(len(x_split)):
            volumes, valids = [], []
            for batch_id, (seq_feature, seq_img_meta) in enumerate(zip(x, img_metas)):
                feature = x_split[seq_id][batch_id]  # [6, 64, 232, 400]
                img_meta = copy.deepcopy(seq_img_meta)
                img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*6:(seq_id+1)*6]
                if isinstance(img_meta["img_shape"], list):
                    img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                    img_meta["img_shape"] = img_meta["img_shape"][0]
                projection = self._compute_projection(
                    img_meta, stride, noise=self.extrinsic_noise
                ).to(
                    x.device
                )  # [6, 3, 4]
                points = get_points(  # [3, 200, 200, 12]
                    n_voxels=torch.tensor(self.n_voxels),
                    voxel_size=torch.tensor(self.voxel_size),
                    origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                ).to(x.device)

                if self.feature_align is not None and self.feature_align['type'] in ['pool']:
                    height = self.feature_align['output_size'][0]
                    width = self.feature_align['output_size'][1]
                else:
                    height = img_meta["img_shape"][0] // stride
                    width = img_meta["img_shape"][1] // stride

                if self.debug:
                    save_dict.update({
                        'stride': stride,
                        'height': height,
                        'width': width,
                        'feature_raw': feature,
                        'points': points,
                        'projection': projection,
                    })
                    torch.save(save_dict, '/mnt/lustre/chenzeren/debug0.pth')

                if self.style == "v1":
                    volume, valid = backproject(
                        feature[:, :, :height, :width], points, projection
                    )
                    volume = volume.sum(dim=0)  # [6, 64, 200, 200, 12] -> [64, 200, 200, 12]
                    valid = valid.sum(dim=0)  # [6, 1, 200, 200, 12] -> [1, 200, 200, 12]
                    volume = volume / valid
                    valid = valid > 0
                    volume[:, ~valid[0]] = 0.0
                elif self.style == "v2":
                    volume = backproject_v2(
                        feature[:, :, :height, :width], points, projection
                    )  # [64, 200, 200, 12]
                else:
                    crop_val = 0
                    if self.feature_align is not None and self.feature_align['type'] == 'top_crop':
                        crop_val = self.align_crop
                    volume = backproject_v3(
                        feature[:, :, :height, :width], points, projection, crop_val, self.debug
                    )  # [64, 200, 200, 12]
                volumes.append(volume)

                if self.debug:
                    print('all done')
                    exit(0)

            x = torch.stack(volumes)  # [1, 64, 200, 200, 12]
            x_list.append(x)

        if self.seq_detach:
            for adj_id in range(len(x_list[1:])):
                x_list[adj_id] = x_list[adj_id].detach()
                valids_list[adj_id] = valids_list[adj_id].detach()
        x = torch.cat(x_list, dim=1)

        def _inner_forward(x):
            out = self.neck_3d(x)  # [[1, 256, 100, 100]]
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

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
        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            if 'CenterHead' in type(self.bbox_head).__name__:
                x = self.bbox_head(feature_bev)
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, x]
                loss_det = self.bbox_head.loss(*loss_inputs)
                losses.update(loss_det)
            else:
                x = self.bbox_head(feature_bev)
                loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
                losses.update(loss_det)

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
            if 'CenterHead' in type(self.bbox_head).__name__:
                output_dicts = self.bbox_head(x)
                
                # tasks [car, truck, bike, pedestrian]
                cls_score = []
                bbox_pred = []

                for task_id, task_pred in enumerate(output_dicts):
                    assert len(task_pred) == 1
                    task_pred = task_pred[0]

                    # heatmap
                    hm = task_pred['heatmap']
                    cls_score.append(hm)

                    ## bbox coder
                    reg = task_pred['reg']
                    height = task_pred['height']
                    dim = task_pred['dim']
                    rot = task_pred['rot']
                    vel = task_pred['vel']
                    boxes = torch.cat([reg, height, dim, rot, vel], dim=1)
                    bbox_pred.append(boxes)
                
                cls_score = torch.cat(cls_score, dim=1)
                bbox_pred = torch.cat(bbox_pred, dim=1)
                dir_cls_preds = None

            else:
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
        if self.bbox_head is not None:
            if 'CenterHead' in type(self.bbox_head).__name__:
                x = self.bbox_head(feature_bev)
                bbox_list = self.bbox_head.get_bboxes(
                    x, img_metas, rescale=True)
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels in bbox_list
                ]
            else:
                x = self.bbox_head(feature_bev)
                bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
                bbox_results = [
                    bbox3d2result(det_bboxes, det_scores, det_labels)
                    for det_bboxes, det_scores, det_labels in bbox_list
                ]
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


# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
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


def backproject_v2(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
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
    # print(f"valid: {valid.shape}, percept: {valid.sum() / (valid.shape[0] * valid.shape[1])}")

    # method1：特征填充，只填充有效特征，重复特征加和平均
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    count = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] += features[i, :, y[i, valid[i]], x[i, valid[i]]]
        count[:, valid[i]] += 1
    volume[count > 0] /= count[count > 0]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume


def backproject_v3(features, points, projection, crop_val=0, debug=False):
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
    valid = (x >= 0) & (y >= crop_val) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

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
        torch.save(save_dict, '/mnt/lustre/chenzeren/debug1.pth')

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume
