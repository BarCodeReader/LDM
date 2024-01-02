from abc import abstractmethod
from functools import partial
import math
from typing import Iterable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_modules import UpDownSampler, zero_module, ResBlock, TimestepEmbedSequential, timestep_embedding
from attentions import SpatialTransformer


class UNet(nn.Module):
    def __init__(self, inc, base_dim, outc, res_blk_num, use_attn, dropout=0, channel_mult=(1, 2, 4, 8),
                 conv_resample=True, num_heads=-1, use_scale_shift_norm=False, resblock_updown=False,
                 transformer_depth=1, context_dim=None):
        super().__init__()

        self.inc = inc
        self.base_dim = base_dim
        self.outc = outc
        self.res_blk_num = res_blk_num
        self.use_attn = use_attn
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        t_embd = base_dim * 4
        self.time_embed = nn.Sequential(nn.Linear(base_dim, t_embd),
                                        nn.SiLU(),
                                        nn.Linear(t_embd, t_embd))

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(inc, base_dim, kernel_size=3, padding=1))])
        self.output_blocks = nn.ModuleList([])

        # input blocks
        input_block_chans = [base_dim]
        ch = base_dim
        for lvl, (mult, attn) in enumerate(zip(channel_mult, use_attn)):
            # for each lvl, we assemble res_blk_num blocks
            for _ in range(res_blk_num):
                layers = [ResBlock(inc=ch, outc=mult * base_dim,
                                   embd=t_embd, dropout=dropout,
                                   use_conv=False, up_or_down=False,
                                   mode=None, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * base_dim
                if attn:
                    layers += [SpatialTransformer(inc=ch, qkv_dim=ch, n_heads=num_heads,
                                                  depth=transformer_depth, context_dim=context_dim)]
                self.input_blocks += [TimestepEmbedSequential(*layers)]
                input_block_chans.append(ch)
            # add downsamplers
            if lvl != len(channel_mult)-1:
                out_ch = ch
                if resblock_updown:
                    blk = ResBlock(inc=ch, outc=out_ch, embd=t_embd, dropout=dropout, use_conv=False,
                                   up_or_down=True, mode="downsample", use_scale_shift_norm=use_scale_shift_norm)
                else:
                    blk = UpDownSampler(inc=ch, outc=out_ch, mode='downsample', use_conv=conv_resample)
                self.input_blocks.append(TimestepEmbedSequential(blk))

                input_block_chans.append(ch)

        # middle blocks
        self.middle_block = TimestepEmbedSequential(
            ResBlock(inc=ch, outc=ch, embd=t_embd, dropout=dropout, use_conv=False,
                     up_or_down=False, mode=None, use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(inc=ch, qkv_dim=ch, n_heads=num_heads,
                               depth=transformer_depth, context_dim=context_dim),
            ResBlock(inc=ch, outc=ch, embd=t_embd, dropout=dropout, use_conv=False,
                     up_or_down=False, mode=None, use_scale_shift_norm=use_scale_shift_norm)
        )

        # output blocks
        for lvl, (mult, attn) in list(enumerate(zip(channel_mult, use_attn)))[::-1]:
            for i in range(res_blk_num + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(inc=ch+ich, outc=base_dim * mult, embd=t_embd, dropout=dropout, use_conv=False,
                                   up_or_down=False, mode=None, use_scale_shift_norm=use_scale_shift_norm)]
                ch = base_dim * mult
                if attn:
                    layers.append(SpatialTransformer(inc=ch, qkv_dim=ch, n_heads=num_heads,
                                                     depth=transformer_depth, context_dim=context_dim))
                # self.output_blocks.append(TimestepEmbedSequential(*layers))
                # add upsamplers
                if lvl and i == res_blk_num:
                    out_ch = ch
                    if resblock_updown:
                        blk = ResBlock(inc=ch, outc=out_ch, embd=t_embd, dropout=dropout, use_conv=False,
                                       up_or_down=True, mode="upsample", use_scale_shift_norm=use_scale_shift_norm)
                    else:
                        blk = UpDownSampler(inc=ch, outc=out_ch, mode='upsample', use_conv=conv_resample)
                    layers.append(blk)
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=ch),
                                 nn.SiLU(),
                                 zero_module(nn.Conv2d(base_dim, outc, kernel_size=3, padding=1)))

    def forward(self, x, t_step=None, context=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        t_emb = timestep_embedding(t_step, self.base_dim, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x

        # encoder
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        # bridge
        h = self.middle_block(h, emb, context)
        # decoder
        for module in self.output_blocks:
            enc = hs.pop()
            h = torch.cat([h, enc], dim=1)
            h = module(h, emb, context)

        return self.out(h)


if __name__ == "__main__":
    unet = UNet(inc=4, base_dim=64, outc=4, res_blk_num=2, use_attn=[1,1], dropout=0, channel_mult=[1,2],
                conv_resample=True, num_heads=1, use_scale_shift_norm=False, resblock_updown=False,
                transformer_depth=1, context_dim=4)

    k = torch.ones(2,4,64,64)
    ts = torch.tensor([15,])
    ct = torch.zeros(2,4096,4)
    out = unet(k, t_step=ts, context=ct)

    print(">>>final", out.shape)
