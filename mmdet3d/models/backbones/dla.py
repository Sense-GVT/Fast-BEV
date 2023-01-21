from collections import OrderedDict

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.backbone import BACKBONE_REGISTRY, FPN, Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, LastLevelP6P7
from ..builder import BACKBONES
from mmcv.runner import auto_fp16


def get_num_groups_gn(out_channels):
    num_channels_per_group = max(4, out_channels // 32)
    assert out_channels % num_channels_per_group == 0
    num_groups = out_channels // num_channels_per_group
    return num_groups


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, norm="BN"):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=norm == "",
            dilation=dilation,
            norm=get_norm(norm, planes)
        )

        self.conv2 = Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=norm == "",
            dilation=dilation,
            norm=get_norm(norm, planes)
        )
        self.stride = stride
    
    @auto_fp16()
    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)

        out += residual
        out = F.relu_(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm="BN"):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = Conv2d(inplanes, bottle_planes, kernel_size=1, bias=norm == "", norm=get_norm(norm, bottle_planes))
        self.conv2 = Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=norm == "",
            dilation=dilation,
            norm=get_norm(norm, bottle_planes)
        )
        self.conv3 = Conv2d(bottle_planes, planes, kernel_size=1, bias=norm == "", norm=get_norm(norm, planes))
        self.stride = stride
    
    @auto_fp16()
    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        out += residual
        out = F.relu_(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm="BN"):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32

        self.conv1 = Conv2d(inplanes, bottle_planes, kernel_size=1, bias=norm == "", norm=get_norm(norm, bottle_planes))
        self.conv2 = Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=norm == "",
            dilation=dilation,
            groups=cardinality,
            norm=get_norm(norm, bottle_planes)
        )
        self.conv3 = Conv2d(bottle_planes, planes, kernel_size=1, bias=norm == "", norm=get_norm(norm, planes))
        self.stride = stride
    
    @auto_fp16()
    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        out += residual
        out = F.relu_(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual, norm="BN"):
        super(Root, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=norm == "",
            padding=(kernel_size - 1) // 2,
            norm=get_norm(norm, out_channels)
        )
        self.residual = residual
    
    @auto_fp16()
    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        if self.residual:
            x += children[0]
        x = F.relu_(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
        norm="BN"
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation, norm=norm)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation, norm=norm)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                norm=norm
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                norm=norm
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual, norm=norm)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        # (dennis.park) If 'self.tree1' is a Tree (not BasicBlock), then the output of project is not used.
        # if in_channels != out_channels:
        if in_channels != out_channels and not isinstance(self.tree1, Tree):
            self.project = Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=norm == "", norm=get_norm(norm, out_channels)
            )
    
    @auto_fp16()
    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        # (dennis.park) If 'self.tree1' is a 'Tree', then 'residual' is not used.
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@BACKBONES.register_module()
class DLA(Backbone):
    @configurable
    def __init__(
        self,
        levels,
        channels,
        block=BasicBlock,
        residual_root=False,
        # return_levels=False,
        out_features=None,
        norm="BN",
        pretrained=None
    ):
        """
        configurable is experimental.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "level0", "level1", "level2", ...
                If None, will return the output of the last layer.
        """
        super(DLA, self).__init__()
        self.channels = channels
        self.base_layer = Conv2d(
            3,
            channels[0],
            kernel_size=7,
            stride=1,
            padding=3,
            bias=norm == "",
            norm=get_norm(norm, channels[0]),
            activation=F.relu_
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0], norm=norm)
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2, norm=norm)
        self.level2 = Tree(
            levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root, norm=norm
        )
        self.level3 = Tree(
            levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root, norm=norm
        )
        self.level4 = Tree(
            levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root, norm=norm
        )
        self.level5 = Tree(
            levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root, norm=norm
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)

        if out_features is None:
            out_features = ['level5']
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

        out_feature_channels, out_feature_strides = {}, {}
        for lvl in range(6):
            name = f"level{lvl}"
            out_feature_channels[name] = channels[lvl]
            out_feature_strides[name] = 2**lvl

        self._out_feature_channels = {name: out_feature_channels[name] for name in self._out_features}
        self._out_feature_strides = {name: out_feature_strides[name] for name in self._out_features}
        
        if pretrained is not None:
            self.load_pretrain(pretrained)

    @property
    def size_divisibility(self):
        return 32

    @classmethod
    def from_config(cls, cfg, levels, channels, block):
        out_features = cfg.OUT_FEATURES
        norm = cfg.NORM
        return OrderedDict(levels=levels, channels=channels, block=block, out_features=out_features, norm=norm)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, norm="BN"):
        modules = []
        for i in range(convs):
            modules.append(
                Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=norm == "",
                    dilation=dilation,
                    norm=get_norm(norm, planes),
                    activation=F.relu_
                )
            )
            inplanes = planes
        return nn.Sequential(*modules)
    
    @auto_fp16()
    def forward(self, x):
        assert x.dim() == 4, f"DLA takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = []
        x = self.base_layer(x)
        for i in range(6):
            name = f"level{i}"
            x = self._modules[name](x)
            if name in self._out_features:
                outputs.append(x)
        return outputs
    
    def load_pretrain(self, weight_path):
        weight = torch.load(weight_path)['model']
        converted_weight = OrderedDict()
        for k in weight.keys():
            if 'bottom_up' in k:
                new_k = k.replace('backbone.bottom_up.', '')
                converted_weight[new_k] = weight[k]
        
        info = self.load_state_dict(converted_weight)
        print(info)
