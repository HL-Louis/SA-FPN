
# Copyright (c) OpenMMLab. All rights reserved.
from re import S
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import warnings
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from ..utils import drop_path
from mmdet.registry import MODELS
from einops import rearrange, reduce, repeat


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, norm_cfg=dict(type='BN'),
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Intialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        # self.nonlinearity = nn.ReLU()
        self.nonlinearity = nn.SiLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_identity = None
            # self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_dense = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=groups, norm_cfg=norm_cfg)
            # self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            self.rbr_1x1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                      padding=padding_11, groups=groups, norm_cfg=norm_cfg)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class up_layer(BaseModule):
    def __init__(self, inchs, outchs, kernel_size=3, stride=1, norm_cfg=dict(type='BN'), deploy=False,
                 drop_path_rate=0.,
                 act_cfg=None):
        super(up_layer, self).__init__()
        self.inchannels = inchs
        self.outchannles = outchs
        self.midchannles = inchs // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_cfg = norm_cfg

        self.act_cfg = act_cfg

        self.conv1 = ConvModule(
            in_channels=self.inchannels,
            out_channels=self.midchannles,
            kernel_size=1,
            stride=self.stride,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = ConvModule(
            in_channels=self.inchannels,
            out_channels=self.midchannles,
            kernel_size=1,
            stride=self.stride,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.conv_out = ConvModule(
            in_channels=512,
            out_channels=self.outchannles,
            kernel_size=1,
            stride=self.stride,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.block = nn.Sequential(*(RepVGGBlock(self.midchannles, self.midchannles, deploy=deploy) for _ in range(4)))

        self.drop_path_rate = drop_path_rate
        self.tanh = torch.nn.Hardtanh()

    def forward(self, x1, x2):
        gather_size = x1.size()[2:]

        batch, _, height, width = x1.size()
        a = torch.zeros(batch, 256, height, width, device='cuda')
        #         x_o = x1

        x2 = F.interpolate(x2, size=gather_size, mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2], dim=1)

        # 合并宽高
        shortcut = x
        #

        out1 = self.conv1(x)
        out2 = self.conv2(x)

        out2 = self.block(out2)

        x_out = torch.cat([out1, out2], dim=1)

        if self.drop_path_rate > 0:
            x_out = drop_path(x_out, self.drop_path_rate, self.training)
            x_out += shortcut
        x_out = self.conv_out(x_out)

        # x_out = x1 + x2

        x1 = rearrange(x1, 'b c h w -> b c (h w)')

        x2 = rearrange(x2, 'b c h w -> b c (h w)')
        x1 = rearrange(x1, 'b c n -> b n c')

        x_matrix = torch.matmul(x1, x2)
        # x_gap = self.GAP_1(x_matrix)
        # x_matrix = x_matrix - x_gap

        # x_matrix = rearrange(x_matrix,' b c (h1 w) -> b c h1 w', h1=height)

        _x_gap = reduce(x_matrix, 'b c n -> b () n ', 'max')

        x_matrix = x_matrix - _x_gap

        # _x_gap = rearrange(_x_gap, 'b c (h1 w) -> b c h1 w', h1=height)

        #
        x_sig1 = self.tanh(x_matrix)
        # x_sig = self.sig(_x_gap)
        # x_sig = self.sig(x_matrix)
        # x_refine = rearrange(x_sig1, 'b c (h1 w) -> b c h1 w', h1=height)
        #
        #
        x_out_re = rearrange(x_out, 'b c h w -> b c (h w)')
        x_refine = torch.matmul(x_out_re, x_sig1)
        x_refine = rearrange(x_refine, 'b c (h1 w) -> b c h1 w', h1=height)
        x_refine = a * x_refine

        x_out = x_out + x_refine

        return x_out

@MODELS.register_module()
class VOV(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,

                 start_level=0,
                 refine_level=1,

                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 #  norm_cfg=None,
                 #  act_cfg=dict(type='SiLU'),
                 act_cfg=None,
                 deploy=False,
                 drop_path_rate=0.2,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(VOV, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.no_norm_on_lateral = no_norm_on_lateral
        #         self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.backbone_end_level = 5

        self.start_level = start_level
        self.refine_level = refine_level
        self.drop_path_rate = drop_path_rate
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, 4):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(5):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.fpn_convs.append(fpn_conv)

        self.p5_to_p6 = nn.Sequential(
            ConvModule(2048, 256, 1, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False),
            ConvModule(256, 256, 3, padding=1, dilation=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                       inplace=False),
            ConvModule(256, 256, 3, padding=2, dilation=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                       inplace=False),
            ConvModule(256, 256, 3, padding=5, dilation=5, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                       inplace=False))

        self.p6_to_p7 = nn.Sequential(
            ConvModule(256, 256, 1, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False),
            ConvModule(256, 256, 3, padding=1, dilation=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                       inplace=False),
            ConvModule(256, 256, 3, padding=2, dilation=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                       inplace=False),
            ConvModule(256, 256, 3, padding=5, dilation=5, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                       inplace=False))

        #         self.conv_out = ConvModule(1280, 256, kernel_size=3, stride=1, padding=1, conv_cfg=None, norm_cfg=norm_cfg,
        #                                    act_cfg=act_cfg)

        up_list = [up_layer(512, 256, deploy=deploy, drop_path_rate=self.drop_path_rate * i / 4) for i in range(4)]
        self.upRep = nn.Sequential(*up_list)

        # conv_list = [ConvModule(512, 2, kernel_size=3, stride=1, padding=1, conv_cfg=None, norm_cfg=norm_cfg,
        #                         act_cfg=act_cfg) for i in range(4)]
        # self.conv = nn.Sequential(*conv_list)
        # #
        # self.GAP = nn.AdaptiveAvgPool2d(1)
        #
        # self.conv_out = ConvModule(1280, 256, kernel_size=1, stride=1, conv_cfg=None, norm_cfg=norm_cfg,
        #                            act_cfg=dict(type='ReLU'))
        #
        # self.conv_out1 = ConvModule(1280, 1280, kernel_size=3, stride=1, padding=1, conv_cfg=None, norm_cfg=norm_cfg,
        #                             act_cfg=act_cfg)

    def forward(self, inputs):
        """GSCONV+VOV"""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        p6 = self.p5_to_p6(inputs[-1])
        laterals.append(p6)
        p7 = self.p6_to_p7(p6)
        laterals.append(p7)

        # build top-down path
        used_backbone_levels = len(laterals)

        k = 0
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.upRep[k](laterals[i - 1], laterals[i])
            k += 1

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # feats = []
        # feats.append(inter_outs[0])
        # gather_size = inputs[self.refine_level].size()[2:]
        # for i in range(1, len(laterals)):
        #     h_feat_o = inter_outs[i]
        #     h_feat = F.interpolate(
        #         inter_outs[i], size=gather_size, mode='bilinear', align_corners=False)
        #     flow = self.conv[i - 1](torch.cat([h_feat, inter_outs[0]], 1))
        #     gathered = self.flow_warp(h_feat_o, flow, size=gather_size)
        #
        #     feats.append(gathered)
        #
        # outs_3 = torch.cat(feats, dim=1)
        #
        #
        #
        # #
        # outs_3 = self.conv_out1(outs_3)
        # outs_3 = self.GAP(outs_3)
        # outs_3 = self.conv_out(outs_3)
        # #
        #
        #
        # for i in range(len(laterals)):
        #     inter_outs[i] = inter_outs[i] * outs_3 + inter_outs[i]

        return tuple(inter_outs)

    @staticmethod
    def flow_warp(inputs, flow, size):
        out_h, out_w = size  # 对应高分辨率的low-level feature的特征图尺寸
        n, c, h, w = inputs.size()  # 对应低分辨率的high-level feature的4个输入维度

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)
        # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # 生成w的转置矩阵
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        # 展开后进行合并
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        # grid指定由input空间维度归一化的采样像素位置，其大部分值应该在[ -1, 1]的范围内
        # 如x=-1,y=-1是input的左上角像素，x=1,y=1是input的右下角像素。
        # 具体可以参考《Spatial Transformer Networks》，下方参考文献[2]
        output = F.grid_sample(inputs, grid)
        return output





