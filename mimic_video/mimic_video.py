import torch
from torch import nn, cat, stack, is_tensor, tensor
from torch.nn import Module, ModuleList, Linear

import torch.nn.functional as F

import einx
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange

from x_mlps_pytorch import create_mlp

from torch_einops_utils import (
    pad_left_ndim,
    align_dims_left,
    pad_at_dim,
    pack_with_inverse,
)

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

def cast_tensor(val, device = None):
    return tensor(val, device = device) if not is_tensor(val) else val

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def l2norm(t, eps = 1e-10):
    return F.normalize(t, dim = -1, eps = eps)

# token shift from Peng et al. of RWKV
# cheap way to generate relative positions

def shift_feature_dim(t):
    x, x_shift = t.chunk(2, dim = -1)
    x_shift = pad_at_dim(x_shift, (1, -1), dim = 1)
    return cat((x, x_shift), dim = -1)

# time

# they follow p0's research finding with the beta distribution
# lets stick with 0 noise to 1 data instead of the reverse

def default_sample_time_fn(time, s = 0.999):
    return torch.sqrt(s - time)

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, dim)
        )

        self.proj.requires_grad_(False)

    def forward(self, times):
        rand_proj = self.proj(times)
        return torch.cos(2 * torch.pi * rand_proj)

# adaptive rmsnorm

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_time_cond,
        eps = 1e-6
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps

        self.to_modulation = Linear(dim_time_cond, dim * 3, bias = False)
        self.split_modulation = Rearrange('b (three d) -> three b 1 d', three = 3)

        nn.init.zeros_(self.to_modulation.weight)

    def forward(
        self,
        tokens,
        time_cond
    ):

        if time_cond.ndim == 1:
            time_cond = pad_left_ndim(time_cond, 1)

        modulations = self.to_modulation(time_cond)

        scale, shift, gate = self.split_modulation(modulations)

        normed = l2norm(tokens, self.eps) * self.scale

        adaptive_normed = normed * (scale + 1.) + shift

        return adaptive_normed, gate

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

    def forward(
        self,
        tokens
    ):
        hidden, gates = self.proj_in(tokens).chunk(2, dim = -1)

        out = hidden * F.gelu(gates)

        return self.proj_out(out)

# classes

class MimicVideo(Module):
    def __init__(
        self,
        dim,
        *,
        dim_video_hidden,
        dim_action = 20,
        dim_joint_state = 32,
        depth = 8,
        dim_head = 64,
        heads = 8,
        expansion_factor = 4.,
        dim_time_cond = None,
        sample_time_fn = None
    ):
        super().__init__()

        # flow related

        self.sample_time_fn = default(sample_time_fn, default_sample_time_fn)

        # embed

        self.to_action_tokens = Linear(dim_action, dim)

        dim_time_cond = default(dim_time_cond, dim * 2)

        self.to_fourier_embed = RandomFourierEmbed(dim) # used by deepmind, its fine
        self.to_time_cond = create_mlp(dim_in = dim * 2, dim = dim_time_cond, depth = 2, activation = nn.SiLU())

        self.to_joint_state_token = Linear(dim_joint_state, dim)

        self.video_hidden_norm = nn.RMSNorm(dim_video_hidden)

        # transformer

        layers = []

        for _ in range(depth):
            attn_adanorm = AdaptiveRMSNorm(dim = dim, dim_time_cond = dim_time_cond)

            attn = Attention(dim = dim, dim_head = dim_head, heads = heads)

            cross_attn_adanorm = AdaptiveRMSNorm(dim = dim, dim_time_cond = dim_time_cond)

            cross_attn = Attention(dim = dim, dim_head = dim_head, dim_context = dim_video_hidden, heads = heads)

            ff_adanorm = AdaptiveRMSNorm(dim = dim, dim_time_cond = dim_time_cond)

            ff = SwiGLUFeedForward(dim = dim, expansion_factor = expansion_factor)

            layers.append(ModuleList([
                attn_adanorm,
                attn,
                cross_attn_adanorm,
                cross_attn,
                ff_adanorm,
                ff
            ]))

        self.layers = ModuleList(layers)

        # predictions

        self.to_pred_action_flow = nn.Sequential(
            nn.RMSNorm(dim),
            Linear(dim, dim_action)
        )

    def forward(
        self,
        actions,
        video_hiddens, # they use layer 19 of cosmos predict, at first denoising step. that's all
        *,
        joint_state,
        time = None,
        time_video_denoise = 0., # 0 is noise in the scheme i prefer - default to their optimal choice, but can be changed
        context_mask = None,
    ):
        batch, device = actions.shape[0], actions.device

        is_training = not exists(time)

        # handle flow time conditioning

        if is_training:
            time = torch.rand((batch,), device = device)
            time = self.sample_time_fn(time)

            noise = torch.randn_like(actions)
            flow = actions - noise

            actions, left_aligned_time = align_dims_left((actions, time))

            noised = noise.lerp(actions, left_aligned_time)
        else:
            noised = actions

        if time.ndim == 0:
            time = rearrange(time, '-> b', b = batch)

        # handle the video denoising times

        time_video_denoise = cast_tensor(time_video_denoise)

        if time_video_denoise.ndim == 0:
            time_video_denoise = rearrange(time_video_denoise, '-> 1')

        if time_video_denoise.shape[0] != batch:
            time_video_denoise = repeat(time_video_denoise, '1 -> b', b = batch)

        times = stack((time, time_video_denoise), dim = -1)

        # fourier embed and mlp to time condition

        fourier_embed = self.to_fourier_embed(times)

        fourier_embed = rearrange(fourier_embed, '... times d -> ... (times d)')

        time_cond = self.to_time_cond(fourier_embed)

        # handle video hiddens

        video_hiddens = self.video_hidden_norm(video_hiddens)

        # embed

        tokens = self.to_action_tokens(noised)

        joint_state_token = self.to_joint_state_token(joint_state)

        tokens, inverse_pack = pack_with_inverse((joint_state_token, tokens), 'b * d')

        # transformer layers

        for (
            attn_norm,
            attn,
            cross_attn_norm,
            cross_attn,
            ff_norm,
            ff
        ) in self.layers:

            # cross attention

            residual = tokens

            tokens, gate = cross_attn_norm(tokens, time_cond)

            tokens = residual + cross_attn(tokens, context = video_hiddens, context_mask = context_mask) * gate

            # self attention

            residual = tokens

            tokens, gate = attn_norm(tokens, time_cond)

            tokens = residual + attn(tokens) * gate

            # prepare feedforward

            residual = tokens

            tokens, gate = ff_norm(tokens, time_cond)

            # shift along time for action tokens for cheap relative positioning, which is better than messing with rope with such short action chunks

            joint_state_token, tokens = inverse_pack(tokens)

            tokens = shift_feature_dim(tokens)

            tokens, _ = pack_with_inverse((joint_state_token, tokens), 'b * d')

            # feedforward

            tokens = residual + ff(tokens) * gate

        # remove joint token

        _, tokens = inverse_pack(tokens)

        # prediction

        pred_flow = self.to_pred_action_flow(tokens)

        if not is_training:
            return pred_flow

        # mse flow loss

        flow_loss = F.mse_loss(pred_flow, flow)
        return flow_loss
