# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from mmseg.models.builder import HEADS, build_loss
from .bev_decoder_head import BEV_BaseDecodeHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.core import build_pixel_sampler
from mmseg.models.losses import accuracy


@HEADS.register_module()
class BEV_FCNHead(BEV_BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 use_centerness=False,
                 loss_ce=None,
                 loss_dice=None,
                 is_transpose=False,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.use_centerness = use_centerness
        self.is_transpose = is_transpose
        for i in range(3):
            print('seg head transpose: {}'.format(is_transpose))

        super(BEV_FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        self.loss_ce = build_loss(loss_ce)
        self.loss_dice = build_loss(loss_dice)

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))

        if self.is_transpose:
            # Anchor3DHead axis order is (y, x).
            output = output.transpose(-1, -2)
        output = self.cls_seg(output)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[1:3],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        seg_pred_road = seg_logit[:, 0, :, :]
        seg_pred_lane = seg_logit[:, 1, :, :]
        seg_label_road = seg_label[..., 0]
        seg_label_lane = seg_label[..., 1]

        loss_dice = 0.5 * (self.loss_dice(seg_pred_road.sigmoid(), seg_label_road) + \
                           self.loss_dice(seg_pred_lane.sigmoid(), seg_label_lane))

        if not self.use_centerness:
            loss_ce = 0.5 * (self.loss_ce(seg_pred_road, seg_label_road) + \
                             self.loss_ce(seg_pred_lane, seg_label_lane))
        else:
            self.loss_ce.reduction = 'none'
            tmp_loss = 0.5 * (self.loss_ce(seg_pred_road, seg_label_road) + \
                              self.loss_ce(seg_pred_lane, seg_label_lane))
            centerness = bev_centerness_weight(tmp_loss.shape[-1], tmp_loss.shape[-2]).to(tmp_loss.device)
            centerness = centerness[None, ...]
            loss_ce = (centerness*tmp_loss).mean()

        loss['loss_seg_dice'] = loss_dice
        loss['loss_seg_ce'] = loss_ce
        loss['iou_road'] = iou(seg_pred_road, seg_label_road)
        loss['iou_lane'] = iou(seg_pred_lane, seg_label_lane)

        return loss  


def bev_centerness_weight(nx, ny):
    assert nx == ny == 200
    xs, ys = torch.meshgrid(torch.arange(0, nx), torch.arange(0, nx))
    grid = torch.cat([xs[:, :, None], ys[:, :, None]], -1)
    grid = grid - nx//2
    grid = grid / (nx//2)
    centerness = (grid[..., 0]**2 + grid[..., 1]**2) / 2 
    centerness = centerness.sqrt() + 1
    return centerness


def iou(pred, tgt):
    pred = (pred.sigmoid() > 0.5).int()
    inter = ((pred == 1) & (tgt == 1)).float()
    union = ((pred == 1) | (tgt == 1)).float()
    iou = torch.sum(inter) / (torch.sum(union) + 1e-6)
    return iou
