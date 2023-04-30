import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from copy import deepcopy
from .unet import exists, ResnetBlocWithAttn


def zero_conv1x1(in_channels, out_channels, stride=1):
    conv = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
    nn.init.constant_(conv.weight, 0)
    nn.init.constant_(conv.bias, 0)
    return conv

class ControlNet(nn.Module):
    def __init__(
        self,
        unet,
        in_channel=6,
        inner_channel=32,
        channel_mults=(1, 2, 4, 8, 8),
        res_blocks=3,
        dropout=0,
        image_size=128,
        version='v2'
    ):
        super().__init__()
        self.version = version

        self.locked_unet = unet
        self.trainable_downs = deepcopy(unet.downs)
        self.trainable_mid = deepcopy(unet.mid)

        for param in self.locked_unet.parameters():
            param.requires_grad = False

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        self.zero_stem = zero_conv1x1(in_channel, in_channel)
        zero_downs = [zero_conv1x1(inner_channel, inner_channel)]

        encoder_dropout = 0.0
        print('dropout', dropout, 'encoder dropout', encoder_dropout)
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                zero_downs.append(zero_conv1x1(channel_mult, channel_mult))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                zero_downs.append(zero_conv1x1(pre_channel, pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.zero_downs = nn.ModuleList(zero_downs)
        assert(len(self.zero_downs) == len(self.locked_unet.downs))

        self.zero_mid = zero_conv1x1(pre_channel, pre_channel)

    def forward(self, x, cond, time=None):
        t = self.locked_unet.noise_level_mlp(time) if exists(
            self.locked_unet.noise_level_mlp) else None
        
        feats = [x]
        cond = self.zero_stem(cond) + x
        control_feats = [cond]

        for idx in range(len(self.trainable_downs)):
            locked_layer = self.locked_unet.downs[idx]
            train_layer = self.trainable_downs[idx]
            zero_layer = self.zero_downs[idx]
            if isinstance(locked_layer, ResnetBlocWithAttn):
                x = locked_layer(x, t)
                cond = train_layer(cond, t)
            else:
                x = locked_layer(x)
                cond = train_layer(cond)
            zero_cond = zero_layer(cond)
            print(zero_cond.max())
            feats.append(x)
            control_feats.append(zero_cond)

        for idx in range(len(self.trainable_mid)):
            locked_layer = self.locked_unet.mid[idx]
            train_layer = self.trainable_mid[idx]
            if isinstance(locked_layer, ResnetBlocWithAttn):
                x = locked_layer(x, t)
                cond = locked_layer(cond, t)
            else:
                x = locked_layer(x)
                cond = locked_layer(cond)
        zero_cond = self.zero_mid(cond)
        x += zero_cond

        for locked_layer in self.locked_unet.ups:
            if isinstance(locked_layer, ResnetBlocWithAttn):
                skip = feats.pop()
                zero_cond = control_feats.pop()
                skip += zero_cond
                x = F.interpolate(x, size=skip.shape[-2:])
                x = locked_layer(torch.cat((x, skip), dim=1), t)
            else:
                x = locked_layer(x)

        if self.version == 'v2':
            skip = feats.pop()
            zero_cond = control_feats.pop()
            skip += zero_cond
            noise = self.locked_unet.final_conv1(x, skip)
            noise = self.locked_unet.final_conv2(noise)
        else:
            noise = self.locked_unet.final_conv1(x)

        return noise

