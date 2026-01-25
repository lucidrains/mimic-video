from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch import cat, nn, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from einops import rearrange, repeat

from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from diffusers.models.autoencoders.autoencoder_kl_cosmos import AutoencoderKLCosmos
from transformers import T5EncoderModel, T5TokenizerFast, T5Config

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def logit_normal_sample(size, mu = 0.0, sigma = 1.0, device = 'cpu'):
    z = torch.randn(size, device = device) * sigma + mu
    return torch.sigmoid(z)

# constants

TINY_TRANSFORMER_CONFIG = dict(
    in_channels = 16,
    out_channels = 16,
    num_attention_heads = 1,
    attention_head_dim = 16,
    mlp_ratio = 1.0,
    text_embed_dim = 32,
    adaln_lora_dim = 32,
    patch_size = (1, 2, 2),
    max_size = (4, 16, 16),
    extra_pos_embed_type = None,
    concat_padding_mask = False,
)

TINY_VAE_CONFIG = dict(
    in_channels = 3,
    out_channels = 3,
    latent_channels = 16,
    encoder_block_out_channels = (8, 16),
    decode_block_out_channels = (8, 16),
    temporal_compression_ratio = 4,
    spatial_compression_ratio = 4,
    num_layers = 1,
    attention_resolutions = (),
    resolution = 64,
)

TINY_T5_CONFIG = dict(
    vocab_size = 32128,
    d_model = 32,
    d_kv = 8,
    d_ff = 64,
    num_layers = 1,
    num_heads = 1,
)

REAL_TRANSFORMER_CONFIG = dict(
    in_channels = 16,
    out_channels = 16,
    num_attention_heads = 32,
    attention_head_dim = 128,
    mlp_ratio = 4.0,
    text_embed_dim = 1024,
    patch_size = (1, 2, 2),
    max_size = (128, 240, 240),
    extra_pos_embed_type = "learnable",
    concat_padding_mask = True,
)

REAL_VAE_CONFIG = dict(
    in_channels = 3,
    out_channels = 3,
    latent_channels = 16,
    encoder_block_out_channels = (128, 256, 512, 512),
    decode_block_out_channels = (256, 512, 512, 512),
    temporal_compression_ratio = 8,
    spatial_compression_ratio = 8,
)

REAL_T5_CONFIG = dict(
    vocab_size = 32128,
    d_model = 1024,
    d_kv = 64,
    d_ff = 2048,
    num_layers = 12,
    num_heads = 16,
)

DEFAULT_LORA_CONFIG = dict(
    r = 8,
    lora_alpha = 16,
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout = 0.05,
    bias = "none",
)

# main class

class CosmosPredictWrapper(Module):
    def __init__(
        self,
        model_name: str = 'nvidia/Cosmos-1.0-Diffusion-7B-Video2World',
        extract_layers: int | list[int] | None = None,
        random_weights: bool = False,
        tiny: bool = False,
        normalize = lambda t: (t - 0.5) * 2.0,
        extract_layer: int | None = None,
        lora_path: str | None = None,
        video_time_sample_mu: float = 0.,
        video_time_sample_sigma: float = 1.
    ):
        super().__init__()
        extract_layers = default(default(extract_layers, extract_layer), 19)
        self.extract_layers = [extract_layers] if isinstance(extract_layers, int) else extract_layers
        self.return_list = isinstance(extract_layers, list)

        self.hook_handles = []
        self.cached_hidden_states = []
        
        if random_weights:
            self._init_random_weights(tiny = tiny)
        else:
            self._init_pretrained(model_name)

        self.dim_latent = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.normalize = normalize

        if exists(lora_path):
            self.load_lora(lora_path)

        self.video_time_sample_mu = video_time_sample_mu
        self.video_time_sample_sigma = video_time_sample_sigma

        self._register_hook()

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_pretrained(self, model_name: str):
        from diffusers import CosmosVideoToWorldPipeline
        pipeline = CosmosVideoToWorldPipeline.from_pretrained(model_name)
        self.vae, self.transformer, self.text_encoder, self.tokenizer = pipeline.vae, pipeline.transformer, pipeline.text_encoder, pipeline.tokenizer
        self.vae_temporal_compression_ratio = self.vae.config.temporal_compression_ratio
        self.vae_spatial_compression_ratio = self.vae.config.spatial_compression_ratio
        self.vae_latent_channels = self.vae.config.latent_channels
        del pipeline

    def _init_random_weights(self, tiny: bool = False):
        config_t = TINY_TRANSFORMER_CONFIG if tiny else REAL_TRANSFORMER_CONFIG
        config_v = TINY_VAE_CONFIG if tiny else REAL_VAE_CONFIG
        config_5 = TINY_T5_CONFIG if tiny else REAL_T5_CONFIG

        num_layers = max(28 if not tiny else 2, *[layer + 1 for layer in self.extract_layers])
        
        self.transformer = CosmosTransformer3DModel(num_layers = num_layers, **config_t)
        self.vae = AutoencoderKLCosmos(**config_v)
        self.text_encoder = T5EncoderModel(T5Config(**config_5))
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        
        self.vae_temporal_compression_ratio = config_v['temporal_compression_ratio']
        self.vae_spatial_compression_ratio = config_v['spatial_compression_ratio']
        self.vae_latent_channels = config_v['latent_channels']

    def __del__(self):
        for handle in getattr(self, 'hook_handles', []): handle.remove()

    def _register_hook(self):
        for layer_index in self.extract_layers:
            target = self.transformer.transformer_blocks[layer_index]
            self.hook_handles.append(target.register_forward_hook(lambda m, i, o: self.cached_hidden_states.append(o.detach().cpu())))

    def load_lora(self, lora_path: str):
        from peft import PeftModel
        if isinstance(self.transformer, PeftModel):
            self.transformer.load_adapter(lora_path, adapter_name = "default")
        else:
            self.transformer = PeftModel.from_pretrained(self.transformer, lora_path)

    def forward(
        self,
        videos: Tensor,
        prompts: str | list[str] | None = None,
        prompt_token_ids: Tensor | None = None,
        timestep: float | Tensor | None = None,
        predict_num_future_latents = 0 # number of future frames to predict at inference - presumably the given video will be fixed at T = 0, with the future predicted frames at T = 1
    ) -> Tensor | list[Tensor]:

        batch = videos.shape[0]
        videos = self.normalize(videos)

        if isinstance(prompts, str): prompts = [prompts] * batch

        self.cached_hidden_states.clear()

        videos = rearrange(videos, 'b t c h w -> b c t h w').to(self.device)

        if exists(prompt_token_ids):
            text_inputs = dict(input_ids = prompt_token_ids.to(self.device))
        else:
            text_inputs = self.tokenizer(default(prompts, [""] * batch), return_tensors = "pt", padding = True, truncation = True, max_length = 512).to(self.device)

        encoder_states = self.text_encoder(**text_inputs)
        if hasattr(encoder_states, 'last_hidden_state'):
            encoder_states = encoder_states.last_hidden_state

        latents = self.vae.encode(videos).latent_dist.sample()

        # use the `predict_num_future_latents` to differentiate between training, where MimicVideo is exposed to varying times for video denoising
        # vs inference where the prefix is fixed at T = 1 and some future number of latents T = 0 done for one step

        is_inference = predict_num_future_latents > 0

        if not is_inference:
            if not exists(timestep):
                timestep = logit_normal_sample((batch,), self.video_time_sample_mu, self.video_time_sample_sigma, device = self.device)

            if timestep.ndim == 0:
                timestep = repeat(timestep, '-> b', b = batch)

            noise = torch.randn_like(latents)

            frames = latents.shape[2]
            padded_timestep = repeat(timestep, 'b -> b 1 f 1 1', f = frames)

            noisy_latents = torch.lerp(latents, noise, padded_timestep)

            self.transformer(
                hidden_states = noisy_latents,
                encoder_hidden_states = encoder_states,
                timestep = padded_timestep * 1000,
                return_dict = False
            )

        else:
            # conditioning on time=0 for prefix (clean), and time=999 for future (noise)
            
            num_prefix_frames = latents.shape[2]

            # timesteps

            total_frames = num_prefix_frames + predict_num_future_latents

            timestep = torch.zeros((batch, 1, total_frames, 1, 1), device = self.device)
            timestep[:, :, num_prefix_frames:] = 999.

            # get the future latents

            shape = list(latents.shape)
            shape[2] = predict_num_future_latents
            future_latents = torch.randn(shape, device = latents.device)

            # concat clean prefix with noised future

            model_input = cat((latents, future_latents), dim = 2)

            self.transformer(
                hidden_states = model_input,
                encoder_hidden_states = encoder_states,
                timestep = timestep,
                return_dict = False
            )

        hiddens = self.cached_hidden_states[:len(self.extract_layers)]

        return hiddens if self.return_list else hiddens[0]

    def finetune(
        self,
        dataset,
        save_path: str = "cosmos-lora-adapter",
        batch_size: int = 1,
        lr: float = 1e-4,
        epochs: int = 1,
        mu: float = 0.0,
        sigma: float = 1.0,
        lora_config = None,
        accelerator = None,
        unfreeze_transformer: bool = False
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

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        optimizer = torch.optim.AdamW(self.transformer.parameters(), lr = lr)
        
        self.transformer, self.vae, self.text_encoder, optimizer, dataloader = accelerator.prepare(
            self.transformer, self.vae, self.text_encoder, optimizer, dataloader
        )

        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc = f"Epoch {epoch}", disable = not accelerator.is_local_main_process)
            for videos, texts in pbar:
                batch = videos.shape[0]
                videos = self.normalize(videos)
                videos = rearrange(videos, 'b t c h w -> b c t h w').to(device)

                with torch.no_grad():
                    latents = self.vae.encode(videos).latent_dist.sample()
                    if isinstance(texts, list) and isinstance(texts[0], str):
                        text_inputs = self.tokenizer(texts, return_tensors = "pt", padding = True, truncation = True, max_length = 512).to(device)
                        encoder_states = self.text_encoder(**text_inputs).last_hidden_state
                    else:
                        encoder_states = self.text_encoder(texts.to(device)).last_hidden_state

                ts = logit_normal_sample((batch,), mu = mu, sigma = sigma, device = device)

                # noise and flow matching logic

                noise = torch.randn_like(latents)

                frames = latents.shape[2]
                padded_ts = repeat(ts, 'b -> b 1 f 1 1', f = frames)

                noisy_latents = torch.lerp(latents, noise, padded_ts)

                flow_target = noise - latents

                pred = self.transformer(hidden_states = noisy_latents, encoder_hidden_states = encoder_states, timestep = ts * 1000, return_dict = False)[0]

                loss = F.mse_loss(pred, flow_target)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(loss = loss.item())

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.unwrap_model(self.transformer).save_pretrained(save_path)
            print(f"saved to {save_path}")
