import torch
from torch import nn
from torch.nn import Module, ModuleList, Linear

import torch.nn.functional as F

from einops import einsum
from einops.layers.torch import Rearrange

from x_mlps_pytorch import create_mlp

# ein notation

# b - batch
# h - heads
# g - groups
# n - sequence
# i, j - sequence (source, target)
# d - feature dimension

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor function

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        kv_heads = 2
    ):
        super().__init__()
        dim_q_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads
        dim_context = default(dim_context, dim)

        self.scale = dim_head ** -0.5

        self.to_queries = Linear(dim, dim_q_inner, bias = False)
        self.to_keys_values = Linear(dim_context, dim_kv_inner * 2, bias = False)
        self.to_out = Linear(dim_q_inner, dim, bias = False)

        assert divisible_by(heads, kv_heads)
        groups = heads // kv_heads

        self.split_q_heads = Rearrange('b n (g h d) -> b g h n d', g = groups, d = dim_head)
        self.split_kv_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b g h n d -> b n (g h d)')

    def forward(
        self,
        tokens,
        context = None,
        context_mask = None
    ):
        context = default(context, tokens)

        queries = self.to_queries(tokens)
        keys, values = self.to_keys_values(context).chunk(2, dim = -1)

        queries = self.split_q_heads(queries)
        keys, values = tuple(self.split_kv_heads(t) for t in (keys, values))

        queries = queries * self.scale

        sim = einsum(queries, keys, 'b g h i d, b h j d -> b g h i j')

        if exists(context_mask):
            mask_value = max_neg_value(sim)
            sim = einx.where('b j, b g h i j,', context_mask, sim, mask_value)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, values, 'b g h i j, b h j d -> b g h i d')

        out = self.merge_heads(out)

        return self.to_out(out)

# feedforward

class SwiGLUFeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        expansion_factor = 4.,
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.proj_in = nn.Linear(dim, dim_inner * 2)
        self.proj_out = nn.Linear(dim_inner, dim)

    def forward(self, x):
        x, gates = self.proj_in(x).chunk(2, dim = -1)

        x = x * F.gelu(gates)

        return self.proj_out(x)

# classes

class MimicVideo(Module):
    def __init__(self):
        super().__init__()

    def forward(self, video):
        pass
