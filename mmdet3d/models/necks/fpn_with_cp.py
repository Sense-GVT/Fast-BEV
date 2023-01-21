import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmdet.models import NECKS, FPN


@NECKS.register_module()
class FPNWithCP(FPN):

    def __init__(self, with_cp=False, **kwargs):
        super().__init__(**kwargs)
        self.with_cp = with_cp

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        inputs = list(inputs)
        if self.input_upsample_cfg is not None:
            for i in range(len(inputs)):
                inputs[i] = F.interpolate(inputs[i], **self.input_upsample_cfg)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            def _inner_forward(x):
                out = lateral_conv(x)
                return out

            if self.with_cp and inputs[i + self.start_level].requires_grad:
                lateral_out = cp.checkpoint(_inner_forward, inputs[i + self.start_level])
            else:
                lateral_out = _inner_forward(inputs[i + self.start_level])
            laterals.append(lateral_out)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = []
        for i in range(used_backbone_levels):
            def _inner_forward(x):
                out = self.fpn_convs[i](x)
                return out 

            if self.with_cp and laterals[i].requires_grad:
                fpn_out = cp.checkpoint(_inner_forward, laterals[i])
            else:
                fpn_out = _inner_forward(laterals[i])
            outs.append(fpn_out)
        
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
