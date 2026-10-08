import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import timm
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from einops import rearrange


# CA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()

        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x)

        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(
            y,
            [h, w],
            dim=2
        )

        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(
            self.conv_h(x_h)
        )

        a_w = torch.sigmoid(
            self.conv_w(x_w)
        )

        out = identity * a_h * a_w

        return out


# TLAM
class MCPE(nn.Module):
    def __init__(self, Ch, h, window, path):
        super().__init__()
        if isinstance(window, int):
            window = {window: h}
        self.window = window
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        self.num_path = path

        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch * path,
                cur_head_split * path,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=dilation,
                groups=cur_head_split * path,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, v, size):
        B, h, N, Ch = v.shape
        H, W = size
        v_img = rearrange(v, "B h (H W) Ch -> B h Ch H W", H=H, W=W)
        v_img = rearrange(v_img, "b h Ch H W -> b (h Ch) H W", H=H, W=W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        v_img_list_reshape = [rearrange(x, "(p B) c H W -> B (p c) H W", H=H, W=W, p=self.num_path) for x in v_img_list]
        v_att_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list_reshape)]
        v_img_list_reshape = [rearrange(x, "B (p c) H W -> (p B) c H W", H=H, W=W, p=self.num_path) for x in v_att_list]
        v_att = torch.cat(v_img_list_reshape, dim=1)
        v_att = rearrange(v_att, "B (h Ch) H W -> B h (H W) Ch", h=h)
        return v_att


class TLAM(nn.Module):
    def __init__(self, dim, num_heads, bias, qk_norm=1, path=1, focusing_factor=8):
        super().__init__()
        self.norm = qk_norm
        self.num_heads = num_heads
        self.path = path
        self.focusing_factor = focusing_factor

        self.temperature = nn.Parameter(torch.ones(path, num_heads, 1, 1))
        self.scale = nn.Parameter(torch.ones(path, num_heads, 1, 1))
        self.sigmoid = nn.Sigmoid()

        self.qkv = nn.Conv2d(dim * path, dim * 3 * path, kernel_size=1, groups=path, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3 * path, dim * 3 * path, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3 * path, bias=bias)
        self.project_out = nn.Conv2d(dim * path, dim * path, kernel_size=1, groups=path, bias=bias)

        if num_heads == 8:
            crpe_window = {3: 2, 5: 3, 7: 3}
        elif num_heads == 4:
            crpe_window = {3: 2, 5: 2}
        elif num_heads == 2:
            crpe_window = {3: 2}
        elif num_heads == 1:
            crpe_window = {3: 1}
        self.refine_att = MCPE(Ch=dim // num_heads, h=num_heads, window=crpe_window, path=path)

        # 固定参数：序列长度归一化
        self.one_M = nn.Parameter(torch.ones(1), requires_grad=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        b, c, h, w = x.shape

        N = h * w

        x = rearrange(x, '(p B) c h w -> B (p c) h w', B=b // self.path, p=self.path)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = rearrange(qkv, 'B (p c) h w -> (p B) c h w', B=b // self.path, p=self.path)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q_norm = torch.norm(q, p=2, dim=-1, keepdim=True) + 1e-8
        q_1 = torch.div(q, q_norm)
        k_norm = torch.norm(k, p=2, dim=-2, keepdim=True) + 1e-8
        k_1 = torch.div(k, k_norm)

        q_2 = self.relu(q) ** self.focusing_factor
        k_2 = self.relu(k) ** self.focusing_factor
        q_2 = q_2 / (q_2.norm(dim=-1, keepdim=True) + 1e-8)
        k_2 = k_2 / (k_2.norm(dim=-2, keepdim=True) + 1e-8)

        refine_weight = self.refine_att(v, size=(h, w))
        refine_weight = self.sigmoid(refine_weight)

        attn_1 = k_1 @ v
        attn_2 = k_2 @ v
        scale = self.sigmoid(self.scale).repeat_interleave(b // self.path, 0)

        out_numerator = torch.sum(v, dim=-2).unsqueeze(2) + (q_1 @ attn_1) + scale * (q_2 @ attn_2)
        one_M = self.one_M * N
        target_shape = (N, c // self.num_heads)
        expanded_tensor = one_M.expand(target_shape)
        out_denominator = expanded_tensor + q_1 @ torch.sum(k_1, dim=-1).unsqueeze(3).repeat(1, 1, 1,
                                                                                             c // self.num_heads) + \
                          q_2 @ torch.sum(scale * k_2, dim=-1).unsqueeze(3).repeat(1, 1, 1, c // self.num_heads) + 1e-8

        out = torch.div(out_numerator, out_denominator)
        out = out * (self.temperature.repeat_interleave(b // self.path, 0)) + refine_weight

        out = rearrange(out, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, '(p b) c h w-> b (p c) h w', h=h, w=w, p=self.path)
        out = self.project_out(out)
        out = rearrange(out, 'b (p c) h w-> (p b) c h w', h=h, w=w, p=self.path)

        return out


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class GCAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super(GCAG, self).__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MDWC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MDWC, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class MSIRB(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                 activation='relu6'):
        super(MSIRB, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = True if self.stride == 1 else False
        self.ex_c = int(self.in_c * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = MDWC(self.ex_c, self.kernel_sizes, self.stride, activation,
                                                           dw_parallel=dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_c),
        )

        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add == True:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def msirb_bottleneck(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                      activation='relu6'):
    convs = []
    xx = MSIRB(in_c, out_c, s, expansion_factor=expansion_factor, dw_parallel=dw_parallel,
                                          add=add, kernel_sizes=kernel_sizes, activation=activation)
    convs.append(xx)
    if n > 1:
        for i in range(1, n):
            xx = MSIRB(out_c, out_c, 1, expansion_factor=expansion_factor,
                                                  dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes,
                                                  activation=activation)
            convs.append(xx)
    conv = nn.Sequential(*convs)
    return conv


# ==================== LFEM and feature upsampling ====================

class LFEM(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class LFEMUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.ghost = LFEM(in_ch, out_ch, kernel_size=3, relu=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.ghost(x)
        return x


class MTL_UNet(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, channels=[16, 32, 64, 96, 160], depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5], expansion_factor=2, gag_kernel=3, gdn_heads=8, **kwargs):
        super().__init__()

        self.encoder1 = msirb_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = msirb_bottleneck(channels[0], channels[1], depths[1], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder3 = msirb_bottleneck(channels[1], channels[2], depths[2], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.encoder4 = TLAM(dim=channels[2], num_heads=gdn_heads, bias=True)
        self.encoder4_adapter = msirb_bottleneck(channels[2], channels[3], depths[3], 1,
                                                  expansion_factor=expansion_factor, dw_parallel=True, add=True,
                                                  kernel_sizes=kernel_sizes)
        self.encoder5 = TLAM(dim=channels[3], num_heads=gdn_heads, bias=True)
        self.encoder5_adapter = msirb_bottleneck(channels[3], channels[4], depths[4], 1,
                                                  expansion_factor=expansion_factor, dw_parallel=True, add=True,
                                                  kernel_sizes=kernel_sizes)

        self.AG1 = GCAG(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                        kernel_size=gag_kernel, groups=channels[3] // 2)
        self.AG2 = GCAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                        kernel_size=gag_kernel, groups=channels[2] // 2)
        self.AG3 = GCAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                        kernel_size=gag_kernel, groups=channels[1] // 2)
        self.AG4 = GCAG(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                        kernel_size=gag_kernel, groups=channels[0] // 2)

        self.decoder1 = msirb_bottleneck(channels[4], channels[3], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder2 = msirb_bottleneck(channels[3], channels[2], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder3 = msirb_bottleneck(channels[2], channels[1], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder4 = msirb_bottleneck(channels[1], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder5 = msirb_bottleneck(channels[0], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.up1 = LFEMUpsample(channels[3], channels[3], 2)
        self.up2 = LFEMUpsample(channels[2], channels[2], 2)
        self.up3 = LFEMUpsample(channels[1], channels[1], 2)
        self.up4 = LFEMUpsample(channels[0], channels[0], 2)
        self.up5 = LFEMUpsample(channels[0], channels[0], 2)

        #################################
        # Coordinate Attention (CA) in the existing decoder stages
        #################################
        self.Att1 = CA(channels[4])
        self.Att2 = CA(channels[3])
        self.Att3 = CA(channels[2])
        self.Att4 = CA(channels[1])
        self.Att5 = CA(channels[0])

        # ######## output ########
        # self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        # self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        # self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        # if x.shape[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)

        ######## encoder ########
        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        out = self.encoder4_adapter(out)
        t4 = out
        out = F.max_pool2d(self.encoder5(out), 2, 2)
        out = self.encoder5_adapter(out)

        ######## decoder ########
        out = self.Att1(out)
        out = F.relu(self.up1(self.decoder1(out)))
        t4 = self.AG1(g=out, x=t4)
        out = torch.add(out, t4)
        out = self.Att2(out)
        out = F.relu(self.up2(self.decoder2(out)))
        # p1 = F.interpolate(self.out1(out),scale_factor=(8,8),mode='bilinear')
        t3 = self.AG2(g=out, x=t3)
        out = torch.add(out, t3)
        out = self.Att3(out)
        out = F.relu(self.up3(self.decoder3(out)))
        # p2 = F.interpolate(self.out2(out),scale_factor=(4,4),mode='bilinear')
        t2 = self.AG3(g=out, x=t2)
        out = torch.add(out, t2)
        out = self.Att4(out)
        out = F.relu(self.up4(self.decoder4(out)))
        # p3 = F.interpolate(self.out3(out),scale_factor=(2,2),mode='bilinear')
        t1 = self.AG4(g=out, x=t1)
        out = torch.add(out, t1)
        out = self.Att5(out)
        out = F.relu(self.up5(self.decoder5(out)))

        p4 = self.out4(out)
        return p4

