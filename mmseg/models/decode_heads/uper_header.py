# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
#from mmcv.cnn import build_attention


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y



class PPMWithAttention(PPM):
    """Pyramid Pooling Module (PPM) with Attention.

    This class adds attention modules to the original PPM to enhance
    feature aggregation.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid Module.
        in_channels (int): Input channels for the PPM module.
        channels (int): Number of channels after pooling and convolution.
        attention_cfg (dict | None): Configuration for attention modules.
    """
    def __init__(self, pool_scales, in_channels, channels, **kwargs):
        #super(PPMWithAttention, self).__init__(pool_scales, in_channels, channels, **kwargs)
        super().__init__(pool_scales, in_channels, channels, **kwargs)
        # 初始化 stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=scale),
                ConvModule(in_channels, channels, kernel_size=1, stride=1, bias=False)
            ) for scale in pool_scales
        ])


        self.attention_modules = nn.ModuleList([
            SEBlock(channels)
            for _ in pool_scales
        ])

    def forward(self, x):
        psp_outs = []
        for stage, attention in zip(self.stages, self.attention_modules):
            # Apply attention after stage processing
            stage_out = stage(x)
            attention_out = attention(stage_out)

            # Resize the feature to the same size as the input feature map
            attention_out = resize(attention_out, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)

            psp_outs.append(attention_out)
        return psp_outs



@MODELS.register_module()
class UPerHeader(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        # PSP Module with Attention
        self.psp_modules = PPMWithAttention(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners

        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)



    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output




    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)


        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)

        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, **kwargs):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList([
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                conv_cfg=kwargs.get('conv_cfg'),
                norm_cfg=kwargs.get('norm_cfg'),
                act_cfg=kwargs.get('act_cfg')
            )
            for dilation in dilations
        ])
        self.project = ConvModule(
            len(dilations) * out_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=kwargs.get('conv_cfg'),
            norm_cfg=kwargs.get('norm_cfg'),
            act_cfg=kwargs.get('act_cfg')
        )

    def forward(self, x):
        aspp_outs = [block(x) for block in self.aspp_blocks]
        aspp_outs = torch.cat(aspp_outs, dim=1)
        return self.project(aspp_outs)