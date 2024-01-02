import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attentions import TimestepBlock, TimestepEmbedSequential, zero_module

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class UpDownSampler(nn.Module):
    def __init__(self, inc, outc, mode, use_conv=False, padding=1):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.use_conv = use_conv
        self.mode = mode
        self.stride = 1 if mode == 'upsample' else 2
        if use_conv:
            self.conv = nn.Conv2d(self.inc, self.outc, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.inc, f"{x.shape[1]} and {self.inc} are not the same"

        # up/down sample
        if self.mode == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)

        # use conv after up/down sample
        if self.use_conv:
            x = self.conv(x)
        return x

class ResBlock(TimestepBlock):
    def __init__(self, inc, outc, embd, dropout, mode=None, up_or_down=False,
                 use_conv=False, use_scale_shift_norm=False):
        super().__init__()
        self.inc = inc
        self.embd = embd
        self.dropout = dropout
        self.outc = outc
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(nn.GroupNorm(32, inc), nn.SiLU())
        self.in_conv = nn.Conv2d(inc, outc, kernel_size=3, padding=1)

        self.up_or_down = up_or_down

        if up_or_down is False:
            self.h_upd = self.x_upd = nn.Identity()
        else:
            # mode: upsample or downsample
            self.h_upd = UpDownSampler(inc=inc, outc=outc, mode=mode, use_conv=False)
            self.x_upd = UpDownSampler(inc=inc, outc=outc, mode=mode, use_conv=False)

        self.emb_layers = nn.Sequential(nn.SiLU(),
                                        nn.Linear(embd, 2 * self.outc if use_scale_shift_norm else self.outc))

        self.out_norm = nn.GroupNorm(32, outc)
        self.out_layers = nn.Sequential(nn.SiLU(),
                                        nn.Dropout(p=dropout),
                                        zero_module(nn.Conv2d(outc, outc, kernel_size=3, padding=1)))

        if outc == inc:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(inc, outc, kernel_size=1, padding=0)

    def forward(self, x, emb):
        # in layers process
        h = self.in_layers(x)
        if self.up_or_down:
            h = self.h_upd(h)
            x = self.x_upd(x)
        h = self.in_conv(h)

        # embedding process
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # out layers process
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
        else:
            h = h + emb_out
            h = self.out_norm(h)
        h = self.out_layers(h)

        out = self.skip_connection(x) + h

        return out

if __name__ == '__main__':
    res = ResBlock(inc=64, outc=128, embd=64, dropout=0., mode='downsample', up_or_down=True,
                     use_conv=False, use_scale_shift_norm=False)
    x = torch.ones(2,64,64,64)
    embd = torch.zeros(2,64)

    out = res(x, embd)

    print(out.shape)