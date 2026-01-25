from __future__ import annotations
from functools import partial

import torch
from torch import nn, cat, stack, is_tensor, tensor
from torch.nn import Module, ModuleList, Linear, GRU

import torch.nn.functional as F

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_mlps_pytorch import create_mlp

from tqdm import tqdm

from torch_einops_utils import (
    lens_to_mask,
    pad_right_ndim_to,
    align_dims_left,
    pad_at_dim,
    pack_with_inverse,
    masked_mean,
    tree_map_tensor
)

from hyper_connections.mHCv2 import get_init_and_expand_reduce_stream_functions

# ein notation

# b - batch
# h - heads
# g - groups
# n - sequence
# i, j - sequence (source, target)
# d - feature dimension

# constants

LinearNoBias = partial(Linear, bias = False)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

# wrappers

def eval_no_grad(fn):
    def inner(*args, **kwargs):
        with torch.no_grad():
            fn.eval()
            return fn(*args, **kwargs)

    return inner

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

# action normalization

class Normalizer(Module):
    def __init__(
        self,
        mean,
        std,
        eps = 1e-6
    ):
        super().__init__()
        assert (std > 0.).all(), 'std must be positive'
        self.eps = eps

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def normalize(self, t):
        mean, std = self.mean, self.std
        return (t - mean) / std.clamp_min(self.eps)

    def inverse_normalize(self, t):
        mean, std = self.mean, self.std
        return (t * std) + mean

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
        eps = 1e-6,
        ada_ln_zero_bias = -5.
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps

        self.to_modulation = LinearNoBias(dim_time_cond, dim * 3)
        self.split_modulation = Rearrange('... (three d) -> three ... d', three = 3)

        nn.init.zeros_(self.to_modulation.weight)

        self.ada_ln_zero_bias = ada_ln_zero_bias

    def forward(
        self,
        tokens,
        time_cond
    ):
        if time_cond.ndim == 2:
            time_cond = rearrange(time_cond, 'b d -> b 1 d')

        modulations = self.to_modulation(time_cond)

        scale, shift, gate = self.split_modulation(modulations)

        normed = l2norm(tokens, self.eps) * self.scale

        adaptive_normed = normed * (scale + 1.) + shift

        gate_with_bias = (gate + self.ada_ln_zero_bias).sigmoid()

        return adaptive_normed, gate_with_bias

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        kv_heads = 2,
        attn_gate_value = True,
        norm_context = False
    ):
        super().__init__()
        dim_q_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        dim_context = default(dim_context, dim)
        self.context_norm = nn.RMSNorm(dim_context) if norm_context else nn.Identity()

        self.scale = dim_head ** -0.5

        self.to_queries = LinearNoBias(dim, dim_q_inner)
        self.to_keys_values = LinearNoBias(dim_context, dim_kv_inner * 2)

        self.attn_gate_value = nn.Sequential(LinearNoBias(dim, heads), Rearrange('b n (g h) -> b g h n 1', h = kv_heads))

        self.to_out = LinearNoBias(dim_q_inner, dim)

        assert divisible_by(heads, kv_heads)
        groups = heads // kv_heads

        self.split_q_heads = Rearrange('b n (g h d) -> b g h n d', g = groups, d = dim_head)
        self.split_kv_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b g h n d -> b n (g h d)')

    def forward(
        self,
        tokens,
        context = None,
        context_mask = None,
        kv = None,
        return_kv = False
    ):
        context = default(context, tokens)

        queries = self.to_queries(tokens)
        queries = self.split_q_heads(queries)

        if not exists(kv):
            context = self.context_norm(context)

            keys, values = self.to_keys_values(context).chunk(2, dim = -1)
            keys, values = tuple(self.split_kv_heads(t) for t in (keys, values))
        else:
            keys, values = kv

        queries = queries * self.scale

        sim = einsum(queries, keys, 'b g h i d, b h j d -> b g h i j')

        if exists(context_mask):
            mask_value = max_neg_value(sim)
            sim = einx.where('b j, b g h i j,', context_mask, sim, mask_value)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, values, 'b g h i j, b h j d -> b g h i d')

        # https://openreview.net/forum?id=1b7whO4SfY - should become standard practice

        out = out * self.attn_gate_value(tokens).sigmoid()

        out = self.merge_heads(out)

        out = self.to_out(out)

        if not return_kv:
            return out

        return out, stack((keys, values))

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
        video_predict_wrapper: Module | None = None,
        *,
        dim_video_hidden = None,
        action_chunk_len = 32,
        dim_action = 20,
        dim_joint_state = 32,
        proprio_mask_prob = 0.1,
        depth = 8,
        dim_head = 64,
        heads = 8,
        expansion_factor = 4.,
        ada_ln_zero_bias = -5.,
        dim_time_cond = None,
        sample_time_fn = None,
        train_time_rtc = False,
        train_time_rtc_max_delay = None,
        num_residual_streams = 1,
        mhc_kwargs: dict = dict(),
        action_mean_std: Tensor | None = None,
        joint_mean_std: Tensor | None = None,
        model_output_clean = True,  # https://arxiv.org/abs/2511.13720 - Kaiming He group paper
        num_task_ids = 0,
        num_advantage_ids = 0,
        advantage_cfg_dropout = 0.25,
        extracted_video_layer_indices: list[int] | None = None,
        eps = 1e-5
    ):
        super().__init__()

        self.depth = depth

        # maybe video predict

        self.video_predict_wrapper = video_predict_wrapper

        # action related

        self.action_chunk_len = action_chunk_len
        self.dim_action = dim_action

        self.action_shape = (action_chunk_len, dim_action)

        self.action_normalizer = None

        if exists(action_mean_std):
            assert action_mean_std.shape == (2, dim_action), f'must be in shape of (2 action_dim)'
            self.action_normalizer = Normalizer(*action_mean_std)

        # joint dim

        self.dim_joint_state = dim_joint_state

        dim_video_hidden = default(dim_video_hidden, video_predict_wrapper.dim_latent if exists(video_predict_wrapper) else None)

        assert exists(dim_video_hidden), f'`dim_video_hidden` must be set or `video_predict_wrapper` passed in with `dim_latent`'

        self.dim_video_hidden = dim_video_hidden

        self.joint_normalizer = None

        if exists(joint_mean_std):
            assert joint_mean_std == (2, dim_joint_state)
            self.joint_normalizer = Normalizer(*joint_mean_std)

        # flow related

        self.sample_time_fn = default(sample_time_fn, default_sample_time_fn)

        # embed

        self.to_action_tokens = Linear(dim_action, dim)

        dim_time_cond = default(dim_time_cond, dim * 2)

        self.to_fourier_embed = RandomFourierEmbed(dim) # used by deepmind, its fine
        self.to_time_cond = create_mlp(dim_in = dim * 2, dim = dim_time_cond, depth = 2, activation = nn.SiLU())

        # joint token related

        self.to_joint_state_token = Linear(dim_joint_state, dim)

        self.proprio_mask_prob = proprio_mask_prob
        self.has_proprio_masking = proprio_mask_prob > 0.

        self.proprio_mask_token = nn.Parameter(torch.randn(dim))

        # video norm

        self.video_hidden_norm = nn.RMSNorm(dim_video_hidden)

        # manifold constrained hyper connections (mHC) from bytedance + deepseek

        init_hyper_conn, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, add_stream_embed = True, **mhc_kwargs)

        # rnn

        self.rnn = GRU(dim, dim)

        # transformer

        layers = []

        for _ in range(depth):
            attn_adanorm = AdaptiveRMSNorm(dim = dim, dim_time_cond = dim_time_cond)

            attn = Attention(dim = dim, dim_head = dim_head, heads = heads)

            cross_attn_adanorm = AdaptiveRMSNorm(dim = dim, dim_time_cond = dim_time_cond)

            cross_attn = Attention(dim = dim, dim_head = dim_head, dim_context = dim_video_hidden, heads = heads, norm_context = True)

            ff_adanorm = AdaptiveRMSNorm(dim = dim, dim_time_cond = dim_time_cond, ada_ln_zero_bias = ada_ln_zero_bias)

            ff = SwiGLUFeedForward(dim = dim, expansion_factor = expansion_factor)

            # maybe hyper connect

            attn_residual = init_hyper_conn()
            cross_attn_residual = init_hyper_conn()
            ff_residual = init_hyper_conn()

            layers.append(ModuleList([
                cross_attn_residual,
                cross_attn_adanorm,
                cross_attn,
                attn_residual,
                attn_adanorm,
                attn,
                ff_residual,
                ff_adanorm,
                ff
            ]))

        self.layers = ModuleList(layers)

        # predictions

        self.to_pred = nn.Sequential(
            nn.RMSNorm(dim),
            Linear(dim, dim_action, bias = False)
        )

        # inference related

        # train time RTC related - https://arxiv.org/abs/2512.05964

        self.train_time_rtc = train_time_rtc

        assert not train_time_rtc or exists(train_time_rtc_max_delay)
        self.train_time_rtc_max_delay = train_time_rtc_max_delay

        # condition related

        self.task_embed = nn.Embedding(num_task_ids, dim) if num_task_ids > 0 else None

        self.advantage_embed = nn.Embedding(num_advantage_ids + 1, dim) if num_advantage_ids > 0 else None

        assert advantage_cfg_dropout > 0.

        self.advantage_cfg_dropout = advantage_cfg_dropout

        # allow for researchers to explore beyond just one layer of pretrained
        # we should also open up research into multiple pretrained models eventually

        self.extracted_video_layer_indices = default(extracted_video_layer_indices, (0,) * depth)
        assert len(self.extracted_video_layer_indices) == depth

        # whether output to action transformer tower is flow or x0

        self.model_output_clean = model_output_clean
        self.eps = eps

        # aux loss and device

        self.register_buffer('zero', tensor(0.), persistent = False)

    # only action parameters

    def action_parameters(self):
        video_model_params = set(self.video_predict_wrapper.parameters()) if exists(self.video_predict_wrapper) else {}
        return set(self.parameters()) - video_model_params

    @property
    def device(self):
        return self.zero.device

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        prefix_action_chunk = None,
        disable_progress_bar = False,
        predict_num_future_latents = 1,
        **kwargs
    ):
        assert predict_num_future_latents > 0

        self.eval()

        inpainting = exists(prefix_action_chunk)

        # times

        times = torch.linspace(0., 1., steps + 1, device = self.device)[:-1]
        delta = 1. / steps

        # inpaint

        if inpainting:
            prefix_len = prefix_action_chunk.shape[1]
            assert prefix_len < self.action_chunk_len

            maybe_normed_prefix = prefix_action_chunk

            if exists(self.action_normalizer):
                maybe_normed_prefix = self.action_normalizer.normalize(prefix_action_chunk)

            times = repeat(times, 'steps -> steps b n', b = batch_size, n = self.action_chunk_len).clone()
            times[..., :prefix_len] = 1.

        # noise

        noise = torch.randn((batch_size, *self.action_shape), device = self.device)

        # denoised action starts as noise

        denoised = noise

        cache = None

        # denoise

        for time in tqdm(times, disable = disable_progress_bar):

            if inpainting:
                denoised[:, :prefix_len] = maybe_normed_prefix

            pred_flow, cache = self.forward(actions = denoised, time = time, cache = cache, predict_num_future_latents = predict_num_future_latents, return_cache = True, **kwargs)

            denoised = denoised + delta * pred_flow

        # handle action inverse norm

        if exists(self.action_normalizer):
            denoised = self.action_normalizer.inverse_normalize(denoised)

        # final set, with unnormalized prefix, if inpainting

        if inpainting:
            denoised[:, :prefix_len] = prefix_action_chunk

        return denoised

    def forward(
        self,
        *,
        actions,                        # (b na d)
        joint_state,                    # (b)
        task_ids = None,                # (b)
        advantage_ids = None,           # (b)
        dropout_advantage_ids = False,
        video = None,                   # (b c t h w)
        video_hiddens = None,           # (b nv dv) - they use layer 19 of cosmos predict, at first denoising step. that's all
        context_mask = None,
        time = None,                    # () | (b) | (b n)
        time_video_denoise = None,      # override default logit normal sampling for video denoising time
        predict_num_future_latents = 0,
        prompts: list[str] | None = None,
        prompt_token_ids = None,
        detach_video_hiddens = False,
        no_grad_video_model_forward = False,
        cache = None,
        return_cache = False,
        return_flow = False
    ):
        assert not exists(self.video_predict_wrapper) or (exists(prompts) ^ exists(prompt_token_ids))
        assert actions.shape[-2:] == self.action_shape

        if exists(self.action_normalizer):
            actions = self.action_normalizer.normalize(actions)

        batch, device = actions.shape[0], actions.device
        orig_actions = actions

        is_training = not exists(time) and not return_flow

        # handle the video denoising times

        time_video_denoise = cast_tensor(time_video_denoise, device = device)

        if time_video_denoise.ndim == 0:
            time_video_denoise = rearrange(time_video_denoise, '-> 1')

        if time_video_denoise.shape[0] != batch:
            time_video_denoise = repeat(time_video_denoise, '1 -> b', b = batch)

        if not exists(cache):
            # handle maybe extraction of video hiddens
            # only if cache is not given

            assert exists(video) ^ exists(video_hiddens)

            if not exists(video_hiddens):
                assert exists(self.video_predict_wrapper), f'`video_predict_wrapper` must be passed in if raw video is passed into MimicVideo'

                video_forward_wrap = eval_no_grad if no_grad_video_model_forward else identity
        
                video_hiddens = video_forward_wrap(self.video_predict_wrapper)(
                    video,
                    prompts = prompts,
                    prompt_token_ids = prompt_token_ids,
                    timestep = time_video_denoise,
                    predict_num_future_latents = predict_num_future_latents
                )

                video_hiddens = tree_map_tensor(lambda t: t.to(self.device).float(), video_hiddens) # maybe bfloat to float32

                video_hiddens = tree_map_tensor(lambda t: pack_with_inverse(t, 'b * d')[0], video_hiddens)

            # handle video hiddens

            if detach_video_hiddens:
                video_hiddens = video_hiddens.detach()

            if not isinstance(video_hiddens, list):
                video_hiddens = [video_hiddens]

        # handle caching

        prev_cached_video_hiddens_kv = cache if exists(cache) else ((None,) * self.depth)
        next_cached_video_hiddens_kv = []

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

        # save the action time condition

        action_time = time

        if action_time.ndim == 0:
            action_time = repeat(action_time, ' -> b ', b = batch)

        # maybe train time rtc

        action_loss_mask = None

        if is_training and self.train_time_rtc:

            rand_prefix_len = torch.randint(0, self.train_time_rtc_max_delay, (batch,), device = device)
            action_prefix_mask = lens_to_mask(rand_prefix_len, self.action_chunk_len)

            actions = einx.where('b na, b na d, b na d', action_prefix_mask, orig_actions, actions)
            time = einx.where('b na, , b', action_prefix_mask, 1., time)

            action_loss_mask = ~action_prefix_mask

        if time.ndim == 0:
            time = repeat(time, '-> b', b = batch)

        if time.ndim == 2:
            time_video_denoise = repeat(time_video_denoise, 'b -> b n', n = time.shape[-1])

        times = stack((time, time_video_denoise), dim = -1)

        # embed

        tokens = self.to_action_tokens(noised)

        # setup empty tokens for various packed condition tokens

        empty_token = tokens[:, 0:0]

        # one layer of rnn for actions

        rnn_out, _, = self.rnn(tokens)
        tokens = rnn_out + tokens

        #  mask joint state token for proprioception masking training

        if exists(self.joint_normalizer):
            joint_state = self.joint_normalizer.normalize(joint_state)

        joint_state_token = self.to_joint_state_token(joint_state)

        if self.training and self.has_proprio_masking:
            mask = torch.rand((batch,), device = device) < self.proprio_mask_prob

            joint_state_token = einx.where('b, d, b d', mask, self.proprio_mask_token, joint_state_token)

        # setup task

        task_embed = empty_token

        if exists(task_ids):
            assert exists(self.task_embed)
            task_embed = self.task_embed(task_ids)

        # setup maybe advantage

        advantage_embed = empty_token

        if exists(advantage_ids):
            assert exists(self.advantage_embed)

            advantage_ids = advantage_ids + 1 # 0 for dropout

            dropout_advantage_ids = default(dropout_advantage_ids, self.training)

            if dropout_advantage_ids:
                cfg_dropout = torch.rand_like(advantage_ids.float()) < self.advantage_cfg_dropout

                advantage_ids = einx.where('b, , b', cfg_dropout, 0, advantage_ids)

            advantage_embed = self.advantage_embed(advantage_ids)

        # determine time - need to handle the sequence dimension given train time RTC and various conditioning tokens

        if times.ndim == 3:
            joint_task_advantage_times = 1 + int(exists(advantage_ids)) + int(exists(task_ids))

            times = pad_at_dim(times, (joint_task_advantage_times, 0), dim = 1, value = 1.) # handle joint state token on the action

        # fourier embed and mlp to time condition

        fourier_embed = self.to_fourier_embed(times)

        fourier_embed = rearrange(fourier_embed, '... times d -> ... (times d)')

        time_cond = self.to_time_cond(fourier_embed)

        # pack with action tokens for attention tower

        tokens, inverse_pack = pack_with_inverse((advantage_embed, task_embed, joint_state_token, tokens), 'b * d')

        # maybe expand streams

        tokens = self.expand_stream(tokens)

        # transformer layers

        for ((
            maybe_cross_attn_mhc,
            cross_attn_norm,
            cross_attn,
            maybe_attn_mhc,
            attn_norm,
            attn,
            maybe_ff_mhc,
            ff_norm,
            ff
        ), layer_video_hidden_index, cached_video_kv) in zip(self.layers, self.extracted_video_layer_indices, prev_cached_video_hiddens_kv):

            # cross attention

            tokens, add_residual = maybe_cross_attn_mhc(tokens)

            tokens, gate = cross_attn_norm(tokens, time_cond)

            layer_video_hidden = None

            if exists(video_hiddens):
                layer_video_hidden = video_hiddens[layer_video_hidden_index]

            cross_attn_out, video_kv = cross_attn(tokens, context = layer_video_hidden, context_mask = context_mask, kv = cached_video_kv, return_kv = True)

            tokens = add_residual(cross_attn_out * gate)

            if return_cache:
                next_cached_video_hiddens_kv.append(video_kv)

            # self attention

            tokens, add_residual = maybe_attn_mhc(tokens)

            tokens, gate = attn_norm(tokens, time_cond)

            tokens = add_residual(attn(tokens) * gate)

            # prepare feedforward

            tokens, add_residual = maybe_ff_mhc(tokens)

            tokens, gate = ff_norm(tokens, time_cond)

            # shift along time for action tokens for cheap relative positioning, which is better than messing with rope with such short action chunks

            *non_action_tokens, tokens = inverse_pack(tokens)

            tokens = shift_feature_dim(tokens)

            tokens, _ = pack_with_inverse((*non_action_tokens, tokens), 'b * d')

            # feedforward

            tokens = add_residual(ff(tokens) * gate)

        # maybe reduce streams

        tokens = self.reduce_stream(tokens)

        # remove joint token

        *_, tokens = inverse_pack(tokens)

        # prediction

        pred = self.to_pred(tokens)

        # convert to flow if outputting in x0 space

        if self.model_output_clean:
            pred_flow = (pred - actions) / pad_right_ndim_to(1. - action_time, pred.ndim).clamp_min(self.eps)
        else:
            pred_flow = pred

        # handle maybe loss or returning flow for inference

        if not is_training:
            # flow

            out = pred_flow
        else:
            # mse flow loss

            flow_loss = F.mse_loss(pred_flow, flow, reduction = 'none')

            out = masked_mean(flow_loss, action_loss_mask)

        if not return_cache:
            return out

        # handle returning of cache

        return out, next_cached_video_hiddens_kv
