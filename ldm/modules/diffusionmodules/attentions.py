from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, dim_out, mult=4, dropout=0.):
        super().__init__()
        qkv_dim = int(dim * mult)
        dim_out = dim_out
        
        self.net = nn.Sequential(nn.Linear(dim, qkv_dim), 
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(qkv_dim, dim_out))

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, qkv_dim, heads=8, dropout=0.):
        super().__init__()

        assert qkv_dim % heads == 0, "qkv_dim should be divisable by heads"
        dim_head = qkv_dim // heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, qkv_dim, bias=False)
        self.to_k = nn.Linear(context_dim, qkv_dim, bias=False)
        self.to_v = nn.Linear(context_dim, qkv_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(qkv_dim, query_dim),
                                    nn.Dropout(dropout))

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
class BasicTransformerBlock(nn.Module):
    def __init__(self, inc, qkv_dim, n_heads, dropout=0., context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=inc, context_dim=inc, qkv_dim=qkv_dim,
                                    heads=n_heads, dropout=dropout)
        self.attn2 = CrossAttention(query_dim=inc, context_dim=context_dim, qkv_dim=qkv_dim,
                                    heads=n_heads, dropout=dropout)
        self.ff = MLP(inc, inc, dropout=dropout)
        self.norm1 = nn.LayerNorm(inc)
        self.norm2 = nn.LayerNorm(inc)
        self.norm3 = nn.LayerNorm(inc)

    def forward(self, x, context):
        x = self.attn1(self.norm1(x), context=x) + x  # self-attn
        x = self.attn2(self.norm2(x), context=context) + x # cross-attn
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, inc, qkv_dim, n_heads, depth=1, dropout=0., context_dim=None):
        super().__init__()

        assert qkv_dim % n_heads == 0, "qkv_dim shoule be divisable by n_heads"
        self.inc = inc
        self.norm = nn.GroupNorm(num_groups=32, num_channels=inc)
        self.proj_in = nn.Conv2d(inc, qkv_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inc=qkv_dim, qkv_dim=qkv_dim, n_heads=n_heads,
                                   dropout=dropout, context_dim=context_dim) for d in range(depth)])

        self.proj_out = zero_module(nn.Conv2d(qkv_dim, inc, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context):
        # note: if no context is given, cross-attention defaults to self-attention
        # context must have a shape of [b, hxw, c]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


if __name__ == '__main__':
    sp_tf = SpatialTransformer(inc=64, qkv_dim=64, n_heads=1, context_dim=4)
    x = torch.ones(2,64,64,64)
    # context can be any BxNxC shape, as long as we indicate in the SpatialTransformer
    ct = torch.zeros(2,1024,4)
    out = sp_tf(x, ct)

    print(out.shape) # same as x