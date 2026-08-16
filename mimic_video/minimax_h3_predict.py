from __future__ import annotations

import os
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torch import cat, tensor, is_tensor, nn, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from einops import rearrange, repeat
import einx

from torch_einops_utils import shape_with_replace, lens_to_mask, masked_mean, temp_eval

from mimic_video.utils import exists, default, check_import

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# helpers

def cast_tensor(val, device = None):
    return tensor(val, device = device) if not is_tensor(val) else val

def logit_normal_sample(size, mu = 0.0, sigma = 1.0, device = 'cpu'):
    z = torch.randn(size, device = device) * sigma + mu
    return torch.sigmoid(z)

# constants

MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

MINIMAX_H3_TEXT_ENCODER_LAYER = 50

MINIMAX_H3_FPS = 24
MINIMAX_H3_IMAGENET_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_IMAGENET_STD = (0.229, 0.224, 0.225)

MINIMAX_H3_AUDIO_SAMPLING_RATE = 32000
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

TINY_TRANSFORMER_CONFIG = dict(
    num_attention_heads = 2,
    attention_head_dim = 16,
    hidden_size = 32,
    num_refiner_layers = 1,
    ffn_dim = 64,
    in_channels = 24,
    audio_in_channels = 32,
    patch_size = (1, 2, 2),
    text_dim = 32,
    freq_dim = 32,
    time_embed_hidden_dim = 32,
    time_embed_dim = 32,
    rope_freq_dim = 2,
    rope_theta = 10000.,
    norm_eps = 1e-5,
    qk_norm_eps = 1e-5,
    final_norm_eps = 1e-5,
)

TINY_VAE_CONFIG = dict(
    in_channels = 3,
    out_channels = 3,
    latent_channels = 24,
    block_out_channels = (8, 16, 32),
    layers_per_block = 1,
    spatial_downsample_factors = (2, 2, 1),
    temporal_downsample_factors = (1, 2, 1),
    norm_num_groups = 4,
    norm_eps = 1e-6,
    decoder_num_layers = 1,
    decoder_num_attention_heads = 1,
    decoder_attention_head_dim = 8,
    decoder_num_register_tokens = 1,
    decoder_ffn_mult = 2,
    decoder_rope_theta = 100.,
    decoder_rope_dim_ratio = 0.75,
    decoder_norm_eps = 1e-5,
    clip_length = 17,
    token_drop = 3,
    latents_mean = (0.,) * 24,
    latents_std = (1.,) * 24,
)

TINY_T5_CONFIG = dict(
    vocab_size = 32128,
    d_model = 32,
    d_kv = 8,
    d_ff = 64,
    num_layers = 1,
    num_heads = 1,
)

TINY_AUDIO_VAE_CONFIG = dict(
    encoder_dim = 8,
    encoder_rates = (2, 4, 4, 5, 5),
    latent_dim = 32,
    latent_channels = 32,
    num_attention_heads = 1,
    decoder_dim = 128,
    decoder_rates = (5, 5, 2, 2, 2, 2, 2),
    decoder_kernel_sizes = (9, 9, 4, 4, 4, 4, 4),
    resblock_kernel_sizes = (3, 7, 11),
    resblock_dilation_sizes = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    sampling_rate = 32000,
    latents_mean = [0.] * 32,
    latents_std = [1.] * 32,
)

REAL_T5_CONFIG = dict(
    vocab_size = 32128,
    d_model = 1024,
    d_kv = 64,
    d_ff = 2048,
    num_layers = 4,
    num_heads = 16,
)

DEFAULT_LORA_CONFIG = dict(
    r = 8,
    lora_alpha = 16,
    target_modules = ['to_q', 'to_k', 'to_v', 'to_out.0'],
    lora_dropout = 0.05,
    bias = 'none',
)

def imagenet_normalize(t):
    mean = torch.tensor(MINIMAX_H3_IMAGENET_MEAN, device = t.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(MINIMAX_H3_IMAGENET_STD, device = t.device).view(1, 3, 1, 1, 1)
    return (t - mean) / std

# main class

class MiniMaxH3PredictWrapper(Module):
    def __init__(
        self,
        model_name: str = 'MiniMaxAI/MiniMax-H3',
        extract_layers: int | list[int] | None = None,
        random_weights: bool = False,
        tiny: bool = False,
        normalize = imagenet_normalize,
        extract_layer: int | None = None,
        lora_path: str | None = None,
        dtype: torch.dtype | None = None,
        video_time_sample_mu: float = 0.,
        video_time_sample_sigma: float = 1.,
        train_fixed_video_prefix: bool = False,
        train_fixed_video_prefix_max_delay: int | None = None
    ):
        super().__init__()
        extract_layers = default(default(extract_layers, extract_layer), 24)
        self.extract_layers = [extract_layers] if isinstance(extract_layers, int) else extract_layers
        self.return_list = isinstance(extract_layers, list)

        self.hook_handles = []
        self.cached_hidden_states = []
        self._current_extract_indices = None

        self.random_weights = random_weights

        check_import('diffusers', (0, 32, 0), '`diffusers` must be installed from main branch for MiniMax H3 wrapper - `pip install "diffusers@git+https://github.com/huggingface/diffusers.git"`')
        check_import('diffusers.models.transformers.transformer_minimax_h3', None, '`diffusers` must be installed from main branch for MiniMax H3 wrapper - `pip install "diffusers@git+https://github.com/huggingface/diffusers.git"`')
        check_import('diffusers.models.autoencoders.autoencoder_kl_minimax_h3', None, '`diffusers` must be installed from main branch for MiniMax H3 wrapper - `pip install "diffusers@git+https://github.com/huggingface/diffusers.git"`')
        check_import('diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio', None, '`diffusers` must be installed from main branch for MiniMax H3 wrapper - `pip install "diffusers@git+https://github.com/huggingface/diffusers.git"`')

        if random_weights:
            self._init_random_weights(tiny = tiny)
        else:
            self._init_pretrained(model_name, dtype = dtype)

        self.normalize = normalize
        self.dim_latent = self.transformer.config.hidden_size
        self.patch_size = tuple(self.transformer.config.patch_size)

        self.vae_latent_channels = self.vae.config.latent_channels
        self.vae_spatial_compression_ratio = self.vae.spatial_compression_ratio
        self.vae_temporal_compression_ratio = self.vae.temporal_compression_ratio
        self.audio_latent_channels = self.transformer.config.audio_in_channels

        self.canvas_multiple = self.vae_spatial_compression_ratio * self.patch_size[2]

        self.video_tag = MINIMAX_H3_VIDEO_TAG
        self.text_tag = MINIMAX_H3_TEXT_TAG
        self.audio_tag = MINIMAX_H3_AUDIO_TAG

        self.audio_channels = MINIMAX_H3_AUDIO_CHANNELS

        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

        if random_weights:
            self.audio_vae = AutoencoderKLMiniMaxH3Audio(**TINY_AUDIO_VAE_CONFIG) if tiny else None
        else:
            self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(model_name, subfolder = 'audio_vae', **self._load_kwargs)

        self.audio_sampling_rate = self.audio_vae.config.sampling_rate if exists(self.audio_vae) else MINIMAX_H3_AUDIO_SAMPLING_RATE
        self.audio_hop_length = self.audio_vae.hop_length if exists(self.audio_vae) else None

        if exists(self.audio_vae):
            audio_mean_list = default(self.audio_vae.config.latents_mean, (0.,) * self.audio_latent_channels)
            audio_std_list = default(self.audio_vae.config.latents_std, (1.,) * self.audio_latent_channels)

            self.audio_latents_mean = rearrange(torch.tensor(audio_mean_list), 'c -> 1 c 1')
            self.audio_latents_std = rearrange(torch.tensor(audio_std_list), 'c -> 1 c 1')

        if exists(lora_path):
            self.load_lora(lora_path)

        self.video_time_sample_mu = video_time_sample_mu
        self.video_time_sample_sigma = video_time_sample_sigma

        self.train_fixed_video_prefix = train_fixed_video_prefix
        self.train_fixed_video_prefix_max_delay = train_fixed_video_prefix_max_delay

        if random_weights:
            self.latents_mean = torch.zeros(1, self.vae_latent_channels, 1, 1, 1)
            self.latents_std = torch.ones(1, self.vae_latent_channels, 1, 1, 1)
        else:
            mean_list = self.vae.config.latents_mean[:self.vae_latent_channels]
            std_list = self.vae.config.latents_std[:self.vae_latent_channels]

            self.latents_mean = rearrange(torch.tensor(mean_list), 'c -> 1 c 1 1 1')
            self.latents_std = rearrange(torch.tensor(std_list), 'c -> 1 c 1 1 1')

        self._register_hook()

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_pretrained(self, model_name: str, dtype: torch.dtype | None = None):
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
        from transformers import Qwen3VLForConditionalGeneration, Qwen2TokenizerFast, Qwen3VLProcessor

        check_import('transformers', (4, 55, 0), '`transformers` must be recent enough for Qwen3-VL conditioner of MiniMax H3 wrapper - `pip install -U transformers`')

        self._load_kwargs = dict(torch_dtype = dtype) if exists(dtype) else dict()

        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(model_name, subfolder = 'vae', **self._load_kwargs)
        self.transformer = MiniMaxH3Transformer3DModel.from_pretrained(model_name, subfolder = 'transformer', **self._load_kwargs)
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(model_name, subfolder = 'text_encoder', **self._load_kwargs)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(model_name, subfolder = 'tokenizer')
        self.processor = Qwen3VLProcessor.from_pretrained(model_name, subfolder = 'processor')

        self.text_encoder_layer = MINIMAX_H3_TEXT_ENCODER_LAYER
        self.text_proj = nn.Identity()

        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        assert num_layers > self.text_encoder_layer, f'MiniMax-H3 conditions on layer {self.text_encoder_layer} of Qwen3-VL conditioner, but loaded model only has {num_layers} layers'

    def _init_random_weights(self, tiny: bool = False):
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
        from transformers import T5EncoderModel, T5TokenizerFast, T5Config

        self._load_kwargs = dict()

        config_t = TINY_TRANSFORMER_CONFIG if tiny else dict()
        config_v = TINY_VAE_CONFIG if tiny else dict()

        num_layers = max(50 if not tiny else 4, *[layer + 1 for layer in self.extract_layers])

        self.transformer = MiniMaxH3Transformer3DModel(num_layers = num_layers, **config_t)
        self.vae = AutoencoderKLMiniMaxH3(**config_v)
        self.text_encoder = T5EncoderModel(T5Config(**(TINY_T5_CONFIG if tiny else REAL_T5_CONFIG)))
        self.text_proj = nn.Linear(self.text_encoder.config.d_model, self.transformer.config.text_dim)
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.processor = None
        self.text_encoder_layer = None

    def _register_hook(self):
        for layer_index in self.extract_layers:
            target = self.transformer.transformer_blocks[layer_index]
            self.hook_handles.append(target.register_forward_hook(lambda m, i, o: self.cached_hidden_states.append(o[:, self._current_extract_indices].detach().cpu())))

    def load_lora(self, lora_path: str):
        from peft import PeftModel
        if isinstance(self.transformer, PeftModel):
            self.transformer.load_adapter(lora_path, adapter_name = 'default')
        else:
            self.transformer = PeftModel.from_pretrained(self.transformer, lora_path)

    # text encoding

    def _encode_text(self, prompts: list[str] | None = None, prompt_token_ids: Tensor | None = None):
        device = self.device

        if self.random_weights:
            if exists(prompt_token_ids):
                input_ids = prompt_token_ids.to(device)
            else:
                text_inputs = self.tokenizer(prompts, return_tensors = 'pt', padding = True, truncation = True, max_length = 512).to(device)
                input_ids = text_inputs.input_ids

            hiddens = self.text_encoder(input_ids = input_ids).last_hidden_state
            hiddens = self.text_proj(hiddens)
            return hiddens, torch.ones(hiddens.shape[1], dtype = torch.long, device = device)

        if exists(prompt_token_ids):
            ids_list = [ids.tolist() for ids in prompt_token_ids]
            input_ids = prompt_token_ids.to(device)
        else:
            ids_list = [self.tokenizer(prompt, add_special_tokens = False)['input_ids'] for prompt in prompts]
            pad_id = default(self.tokenizer.pad_token_id, 151643)

            max_len = max(len(ids) for ids in ids_list)
            ids_list = [ids if len(ids) > 0 else [pad_id] for ids in ids_list]
            ids_list = [ids + [pad_id] * (max_len - len(ids)) for ids in ids_list]
            input_ids = tensor(ids_list, dtype = torch.long, device = device)

        if hasattr(self.processor, 'create_mm_token_type_ids'):
            mm_token_type_ids = tensor(self.processor.create_mm_token_type_ids(ids_list), dtype = torch.long, device = device)
        else:
            mm_token_type_ids = torch.zeros_like(input_ids)

        outputs = self.text_encoder.model(
            input_ids = input_ids,
            attention_mask = torch.ones_like(input_ids),
            mm_token_type_ids = mm_token_type_ids,
            use_cache = False,
            output_hidden_states = True
        )

        hiddens = outputs.hidden_states[self.text_encoder_layer]
        return hiddens, torch.ones(hiddens.shape[1], dtype = torch.long, device = device)

    # audio encoding

    def _audio_latent_num_frames(self, num_frames: int):
        return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))

    def _encode_audio(self, audio: Tensor, num_audio_latents: int):
        assert exists(self.audio_vae), 'audio given but no audio vae available'

        if audio.ndim == 2:
            audio = rearrange(audio, 'b s -> b 1 s')

        batch, channels, samples = audio.shape
        assert channels == self.audio_channels, f'audio must have {self.audio_channels} channels, but got {channels} - convert mono to stereo (e.g. duplicate the channel) before passing into the wrapper'

        num_samples = num_audio_latents * self.audio_hop_length
        if samples < num_samples:
            audio = F.pad(audio, (0, num_samples - samples))
        else:
            audio = audio[..., :num_samples]

        posterior = self.audio_vae.encode(audio.reshape(batch * channels, 1, num_samples), return_dict = False)[0]
        latents = posterior.mode().float()
        latents = (latents - self.audio_latents_mean.to(latents.device)) / self.audio_latents_std.to(latents.device)

        return rearrange(latents, '(b ch) c n -> b (ch n) c', b = batch, ch = channels)

    # position grid

    def _spatial_position_grid(self, dim: int, patch: int, sqrt_area: float):
        ratio = dim / sqrt_area
        left = (1.0 - ratio) / 2.0
        grid = np.linspace(left, left + ratio, dim // patch, endpoint = False) * _ROPE_SPATIAL_SCALE
        return torch.from_numpy(grid).to(torch.float64)

    def _temporal_position_grid(self, num_latent_frames: int, origin: float):
        spans = torch.tensor(
            [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)] for i in range(num_latent_frames)],
            dtype = torch.float64
        )
        return origin + cat((torch.zeros(1, dtype = torch.float64), spans[:-1].cumsum(0)))

    def _frame_position_grid(self, latent_height: int, latent_width: int, patch_h: int, patch_w: int):
        sqrt_area = np.sqrt(latent_height * latent_width)
        height_grid = self._spatial_position_grid(latent_height, patch_h, sqrt_area)
        width_grid = self._spatial_position_grid(latent_width, patch_w, sqrt_area)
        grids = torch.meshgrid(height_grid, width_grid, indexing = 'ij')
        return torch.stack([grid.reshape(-1) for grid in grids], dim = -1), width_grid

    # layout

    def _build_layout(self, text_token_tags: Tensor, num_latent_frames: int, latent_height: int, latent_width: int, num_audio_latents: int = 0):
        device = self.device
        patch_h, patch_w = self.patch_size[1], self.patch_size[2]
        rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
        num_text_tokens = text_token_tags.shape[0]
        num_video_rows = num_latent_frames * rows_per_frame
        num_audio_rows = num_audio_latents * self.audio_channels
        sequence_length = num_text_tokens + num_audio_rows + num_video_rows

        position_ids = torch.zeros(sequence_length, 3, dtype = torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype = torch.float64)

        frame_grid, width_grid = self._frame_position_grid(latent_height, latent_width, patch_h, patch_w)

        audio_start = num_text_tokens
        video_start = audio_start + num_audio_rows

        if num_audio_rows > 0:
            audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype = torch.float64)
            audio_slice = slice(audio_start, video_start)
            position_ids[audio_slice, 0] = audio_time.repeat(self.audio_channels)
            position_ids[audio_slice, 2] = torch.cat((
                torch.full((num_audio_latents,), float(width_grid[0]), dtype = torch.float64),
                torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype = torch.float64)
            ))

        video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype = torch.float64)
        video_position_ids[:, :, 0] = self._temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
        video_position_ids[:, :, 1:] = frame_grid[None]
        position_ids[video_start:] = video_position_ids.reshape(-1, 3)

        video_indices = torch.arange(video_start, sequence_length)
        audio_indices = torch.arange(audio_start, video_start)
        text_indices = torch.arange(num_text_tokens)

        token_tags = torch.empty(sequence_length, dtype = torch.long)
        token_tags[text_indices] = text_token_tags.to(torch.device('cpu')).to(torch.long)
        token_tags[audio_indices] = self.audio_tag
        token_tags[video_indices] = self.video_tag

        position_ids = position_ids.to(device, dtype = torch.float32 if device.type == 'mps' else None)
        token_tags = token_tags.to(device)
        video_indices = video_indices.to(device)
        audio_indices = audio_indices.to(device)
        text_indices = text_indices.to(device)

        return position_ids, token_tags, video_indices, audio_indices, text_indices

    # patchify / unpatchify

    def _patchify_video_latents(self, latents: Tensor):
        patch_t, patch_h, patch_w = self.patch_size
        return rearrange(latents, 'b c (f pt) (h ph) (w pw) -> b (f h w) (c pt ph pw)', pt = patch_t, ph = patch_h, pw = patch_w).contiguous()

    def _unpatchify_video_latents(self, rows: Tensor, num_latent_frames: int, latent_height: int, latent_width: int):
        patch_t, patch_h, patch_w = self.patch_size
        h, w = latent_height // patch_h, latent_width // patch_w
        return rearrange(rows, 'b (f h w) (c pt ph pw) -> b c (f pt) (h ph) (w pw)', f = num_latent_frames, h = h, w = w, pt = patch_t, ph = patch_h, pw = patch_w)

    # transformer forward

    def _transformer_forward(
        self,
        latents: Tensor,
        encoder_states: Tensor,
        timestep: Tensor,
        text_token_tags: Tensor,
        audio_rows: Tensor | None = None,
        num_prefix_audio: int = 0
    ):
        batch, _, num_latent_frames, latent_height, latent_width = latents.shape

        has_audio = exists(audio_rows)
        num_audio_latents = audio_rows.shape[1] // self.audio_channels if has_audio else 0

        position_ids, token_tags, video_indices, audio_indices, text_indices = self._build_layout(text_token_tags, num_latent_frames, latent_height, latent_width, num_audio_latents)

        self._current_extract_indices = cat((audio_indices, video_indices)) if has_audio else video_indices

        patch_h, patch_w = self.patch_size[1], self.patch_size[2]
        rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)

        video_rows = self._patchify_video_latents(latents)

        per_row_ts = repeat(timestep[:, 0, :, 0, 0], 'b f -> b (f r)', r = rows_per_frame)

        base_ts = per_row_ts[0, -1]
        num_text_tokens = text_token_tags.shape[0]
        num_audio_rows = audio_indices.shape[0]

        num_video_rows = video_indices.shape[0]

        row_timesteps = torch.full((num_text_tokens + num_audio_rows + num_video_rows,), base_ts, dtype = torch.float32, device = self.device)

        if has_audio:
            audio_ts = torch.full((num_audio_rows,), base_ts, dtype = torch.float32, device = self.device)
            audio_ts[:num_prefix_audio * self.audio_channels] = 1.0
            row_timesteps[num_text_tokens:num_text_tokens + num_audio_rows] = audio_ts

        row_timesteps[num_text_tokens + num_audio_rows:] = per_row_ts[0]

        unique_ts, timestep_indices = torch.unique(row_timesteps, sorted = True, return_inverse = True)

        audio_hidden_states = default(audio_rows, torch.zeros((batch, 0, self.audio_latent_channels), device = self.device))

        video_out, audio_out = self.transformer(
            hidden_states = video_rows,
            audio_hidden_states = audio_hidden_states,
            encoder_hidden_states = encoder_states,
            timestep = unique_ts,
            timestep_indices = timestep_indices,
            token_tags = token_tags,
            position_ids = position_ids,
            video_indices = video_indices,
            audio_indices = audio_indices,
            text_indices = text_indices,
            return_dict = False
        )

        return video_out, audio_out

    # sampling

    @torch.no_grad()
    def sample_flow_trajectory(
        self,
        latents: Tensor,
        encoder_states: Tensor,
        text_token_tags: Tensor,
        target_tau: float = 1.0,
        steps: int = 10,
        num_prefix_frames: int = 0,
        audio_rows: Tensor | None = None,
        num_prefix_audio: int = 0
    ) -> None:

        batch_size, _, total_frames, latent_height, latent_width = latents.shape
        has_audio = exists(audio_rows)

        def get_dense_time(t_value):
            ts = torch.full(
                (batch_size, 1, total_frames, 1, 1),
                t_value,
                device = self.device,
                dtype = latents.dtype
            )
            ts[:, :, :num_prefix_frames] = 1.0
            return ts

        if target_tau >= 1.0 - 1e-4:
            ts = get_dense_time(0.0)
            self.cached_hidden_states.clear()
            self._transformer_forward(latents, encoder_states, ts, text_token_tags, audio_rows = audio_rows, num_prefix_audio = num_prefix_audio)
            return

        timesteps = torch.linspace(0.0, 1.0 - target_tau, steps + 1, device = self.device)
        curr_latents = latents.clone()

        if has_audio:
            audio_noise = torch.randn_like(audio_rows)
            curr_audio = audio_rows.clone()
            curr_audio[:, num_prefix_audio * self.audio_channels:] = audio_noise[:, num_prefix_audio * self.audio_channels:]
        else:
            curr_audio = None

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            ts = get_dense_time(t_curr)
            self.cached_hidden_states.clear()

            video_velocity, audio_velocity = self._transformer_forward(curr_latents, encoder_states, ts, text_token_tags, audio_rows = curr_audio, num_prefix_audio = num_prefix_audio)
            video_velocity = self._unpatchify_video_latents(video_velocity, total_frames, latent_height, latent_width)

            curr_latents[:, :, num_prefix_frames:] += video_velocity[:, :, num_prefix_frames:] * dt

            if has_audio:
                curr_audio[:, num_prefix_audio * self.audio_channels:] += audio_velocity[:, num_prefix_audio * self.audio_channels:] * dt

        self.cached_hidden_states.clear()
        final_ts = get_dense_time(1.0 - target_tau)
        self._transformer_forward(curr_latents, encoder_states, final_ts, text_token_tags, audio_rows = curr_audio, num_prefix_audio = num_prefix_audio)

    # train forward item

    def _train_forward_item(
        self,
        latents: Tensor,
        noise: Tensor,
        encoder_states: Tensor,
        text_token_tags: Tensor,
        timestep: Tensor,
        use_fixed_prefix: bool = False,
        audio_rows: Tensor | None = None,
        audio_noise: Tensor | None = None
    ):
        self.cached_hidden_states.clear()

        batch, _, frames, _, _ = latents.shape
        padded_timestep = repeat(timestep, 'b -> b 1 f 1 1', f = frames)

        if use_fixed_prefix:
            rand_prefix_len = torch.randint(0, self.train_fixed_video_prefix_max_delay, (batch,), device = self.device)
            fixed_prefix_mask = lens_to_mask(rand_prefix_len, frames)
            fixed_prefix_mask = rearrange(fixed_prefix_mask, 'b f -> b 1 f 1 1')
            padded_timestep = einx.where('b 1 f 1 1, , b 1 f 1 1 -> b 1 f 1 1', fixed_prefix_mask, 0., padded_timestep)

        noisy_latents = torch.lerp(latents, noise, padded_timestep)
        transformer_timestep = 1.0 - padded_timestep

        noised_audio = None

        if exists(audio_rows):
            audio_ts = repeat(timestep, 'b -> b n 1', n = audio_rows.shape[1])
            noised_audio = torch.lerp(audio_rows, audio_noise, audio_ts)

        self._transformer_forward(noisy_latents, encoder_states, transformer_timestep, text_token_tags, audio_rows = noised_audio)

        return self.cached_hidden_states[:len(self.extract_layers)]

    # forward

    @torch.no_grad()
    @temp_eval
    def forward(
        self,
        videos: Tensor,
        prompts: str | list[str] | None = None,
        prompt_token_ids: Tensor | None = None,
        audio: Tensor | None = None,
        timestep: float | Tensor | None = None,
        predict_num_future_latents = 0,
        video_flow_target_tau: float = 1.0,
        inference_steps: int = 10
    ) -> Tensor | list[Tensor]:

        batch = videos.shape[0]
        if isinstance(prompts, str): prompts = [prompts] * batch

        self.cached_hidden_states.clear()

        videos = rearrange(videos, 'b t c h w -> b c t h w').to(self.device)
        videos = self.normalize(videos)

        _, _, num_frames, height, width = videos.shape
        pad_h = (-height) % self.canvas_multiple
        pad_w = (-width) % self.canvas_multiple

        if pad_h or pad_w:
            videos = F.pad(videos, (0, pad_w, 0, pad_h), mode = 'constant', value = 0.)

        if not self.random_weights and num_frames % 17 != 5:
            warnings.warn(f'MiniMax H3 video vae encodes 17n + 5 frames at a time, but {num_frames} frames were given - the last chunk will be padded')

        encoder_states, text_token_tags = self._encode_text(default(prompts, [''] * batch), prompt_token_ids = prompt_token_ids)

        latents = self.vae.encode(videos).latent_dist.sample()
        latents = (latents - self.latents_mean.to(latents.device)) / self.latents_std.to(latents.device)

        is_inference = predict_num_future_latents > 0

        if exists(audio):
            if is_inference:
                total_frames = num_frames + predict_num_future_latents * self.vae_temporal_compression_ratio
                num_audio_latents = self._audio_latent_num_frames(total_frames)
            else:
                num_audio_latents = self._audio_latent_num_frames(num_frames)

            audio_rows = self._encode_audio(audio, num_audio_latents)
        else:
            audio_rows = None

        if not is_inference:
            if exists(timestep):
                timestep = cast_tensor(timestep, device = self.device)
                if timestep.ndim == 0:
                    timestep = rearrange(timestep, '-> 1')
                num_timesteps = timestep.shape[0]

                if num_timesteps != batch:
                    timestep = repeat(timestep, '1 -> b', b = batch)
            else:
                timestep = torch.zeros(batch, device = self.device)

            use_fixed_prefix = self.training and self.train_fixed_video_prefix

            noise = torch.randn_like(latents)
            audio_noise = torch.randn_like(audio_rows) if exists(audio_rows) else None

            all_same_timestep = (timestep == timestep[0]).all()

            if batch == 1 or (all_same_timestep and not use_fixed_prefix):
                hiddens = self._train_forward_item(latents, noise, encoder_states, text_token_tags, timestep, use_fixed_prefix, audio_rows = audio_rows, audio_noise = audio_noise)
            else:
                per_item_hiddens = [
                    self._train_forward_item(latents[i:i+1], noise[i:i+1], encoder_states[i:i+1], text_token_tags, timestep[i:i+1], use_fixed_prefix, audio_rows = audio_rows[i:i+1] if exists(audio_rows) else None, audio_noise = audio_noise[i:i+1] if exists(audio_noise) else None)
                    for i in range(batch)
                ]

                hiddens = [cat([per_item_hiddens[j][k] for j in range(batch)], dim = 0) for k in range(len(self.extract_layers))]

        else:
            num_prefix_frames = latents.shape[2]

            pred_shape = shape_with_replace(latents, {2: predict_num_future_latents})
            future_noise = torch.randn(pred_shape, device = latents.device)

            starting_latents = cat((latents, future_noise), dim = 2)
            num_prefix_audio = self._audio_latent_num_frames(num_frames)

            self.sample_flow_trajectory(
                latents = starting_latents,
                encoder_states = encoder_states,
                text_token_tags = text_token_tags,
                target_tau = video_flow_target_tau,
                steps = inference_steps,
                num_prefix_frames = num_prefix_frames,
                audio_rows = audio_rows,
                num_prefix_audio = num_prefix_audio
            )

            hiddens = self.cached_hidden_states[:len(self.extract_layers)]

        return hiddens if self.return_list else hiddens[0]

    # finetuning

    def finetune(
        self,
        dataset,
        save_path: str = 'minimax-h3-lora-adapter',
        batch_size: int = 1,
        lr: float = 1e-4,
        epochs: int = 1,
        mu: float = 0.0,
        sigma: float = 1.0,
        lora_config = None,
        accelerator = None,
        unfreeze_transformer: bool = False,
        train_fixed_video_prefix_max_delay: int = 0
    ):
        from peft import LoraConfig, get_peft_model, PeftModel
        from accelerate import Accelerator

        if not exists(accelerator):
            accelerator = Accelerator()

        device = accelerator.device

        if not isinstance(self.transformer, PeftModel):
            lora_config = default(lora_config, DEFAULT_LORA_CONFIG)
            if isinstance(lora_config, dict): lora_config = LoraConfig(**lora_config)
            self.transformer = get_peft_model(self.transformer, lora_config)

        self.transformer.train()
        if unfreeze_transformer:
            for p in self.transformer.parameters(): p.requires_grad = True

        self.vae.eval()
        self.text_encoder.eval()
        self.text_proj.eval()

        if exists(self.audio_vae):
            self.audio_vae.eval()

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr = lr)

        prepared_modules = [self.transformer, self.vae, self.text_encoder, self.text_proj]
        if exists(self.audio_vae):
            prepared_modules.append(self.audio_vae)

        self.transformer, self.vae, self.text_encoder, self.text_proj, *audio_vae_prepared, optimizer, dataloader = accelerator.prepare(*prepared_modules, optimizer, dataloader)

        if exists(self.audio_vae):
            self.audio_vae = audio_vae_prepared[0]

        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc = f'Epoch {epoch}', disable = not accelerator.is_local_main_process)
            for batch_items in pbar:
                self.cached_hidden_states.clear()

                has_audio = len(batch_items) == 3

                if has_audio:
                    videos, audios, texts = batch_items
                else:
                    videos, texts = batch_items
                    audios = None

                batch = videos.shape[0]
                videos = rearrange(videos, 'b t c h w -> b c t h w').to(device)
                videos = self.normalize(videos)

                _, _, num_frames, height, width = videos.shape
                pad_h = (-height) % self.canvas_multiple
                pad_w = (-width) % self.canvas_multiple

                if pad_h or pad_w:
                    videos = F.pad(videos, (0, pad_w, 0, pad_h), mode = 'constant', value = 0.)

                with torch.no_grad():
                    latents = self.vae.encode(videos).latent_dist.sample()
                    latents = (latents - self.latents_mean.to(latents.device)) / self.latents_std.to(latents.device)
                    if isinstance(texts, (list, tuple)) and isinstance(texts[0], str):
                        encoder_states, text_token_tags = self._encode_text(list(texts))
                    else:
                        encoder_states, text_token_tags = self._encode_text(prompt_token_ids = texts)

                    num_audio_latents = self._audio_latent_num_frames(num_frames) if has_audio else 0
                    audio_rows = self._encode_audio(audios, num_audio_latents) if has_audio else None

                ts = logit_normal_sample((batch,), mu = mu, sigma = sigma, device = device)

                noise = torch.randn_like(latents)
                audio_noise = torch.randn_like(audio_rows) if exists(audio_rows) else None

                frames = latents.shape[2]
                use_fixed_prefix = train_fixed_video_prefix_max_delay > 0
                total_loss = torch.tensor(0., device = device)

                for i in range(batch):
                    latents_i = latents[i:i+1]
                    noise_i = noise[i:i+1]

                    padded_ts = repeat(ts[i:i+1], '1 -> 1 1 f 1 1', f = frames)
                    loss_mask = None

                    if use_fixed_prefix:
                        rand_prefix_len = torch.randint(0, train_fixed_video_prefix_max_delay, (1,), device = device)
                        fixed_prefix_mask = lens_to_mask(rand_prefix_len, frames)
                        fixed_prefix_mask = rearrange(fixed_prefix_mask, 'b f -> b 1 f 1 1')
                        padded_ts = einx.where('b 1 f 1 1, , b 1 f 1 1 -> b 1 f 1 1', fixed_prefix_mask, 0., padded_ts)
                        loss_mask = ~fixed_prefix_mask

                    noisy_latents = torch.lerp(latents_i, noise_i, padded_ts)
                    flow_target = latents_i - noise_i

                    audio_rows_i = audio_noise_i = None

                    if exists(audio_rows):
                        audio_ts = repeat(ts[i:i+1], '1 -> 1 n 1', n = audio_rows.shape[1])
                        audio_rows_i = torch.lerp(audio_rows[i:i+1], audio_noise[i:i+1], audio_ts)
                        audio_noise_i = audio_noise[i:i+1]

                    video_pred, audio_pred = self._transformer_forward(noisy_latents, encoder_states[i:i+1], 1.0 - padded_ts, text_token_tags, audio_rows = audio_rows_i)
                    video_pred = self._unpatchify_video_latents(video_pred, frames, latents_i.shape[3], latents_i.shape[4])

                    loss = F.mse_loss(video_pred, flow_target, reduction = 'none')

                    if exists(loss_mask):
                        loss_mask = loss_mask.broadcast_to(loss.shape)
                        loss = masked_mean(loss, loss_mask)
                    else:
                        loss = loss.mean()

                    if exists(audio_rows_i):
                        audio_loss = F.mse_loss(audio_pred, audio_rows[i:i+1] - audio_noise_i)
                        loss = loss + audio_loss

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += loss.detach()

                self.cached_hidden_states.clear()
                pbar.set_postfix(loss = (total_loss / batch).item())

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.unwrap_model(self.transformer).save_pretrained(save_path)
