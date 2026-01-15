from __future__ import annotations

import torch
from torch import nn, Tensor
from einops import rearrange

from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from diffusers.models.autoencoders.autoencoder_kl_cosmos import AutoencoderKLCosmos
from diffusers.schedulers.scheduling_edm_euler import EDMEulerScheduler
from transformers import T5EncoderModel, T5TokenizerFast, T5Config

# helpers

def exists(v):
    return v is not None

def identity(t):
    return t

def default(v, d):
    return v if exists(v) else d

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

# main class

class CosmosPredictWrapper(nn.Module):
    """
    Wraps Cosmos VAE + DiT for extracting hidden states from a video.
    Supports proper EDM Euler denoising steps.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-1.0-Diffusion-7B-Video2World",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        extract_layer: int = 19,
        random_weights: bool = False,
        tiny: bool = False,
        normalize = lambda t: (t - 0.5) * 2.0
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.extract_layer = extract_layer
        self.hook_handle = None
        self.cached_hidden_states: list[Tensor] = []
        
        if random_weights:
            self._init_random_weights(tiny = tiny)
        else:
            self._init_pretrained(model_name)
        
        # Initialize scheduler
        self.scheduler = EDMEulerScheduler()
        
        # store hidden dim for consumers
        self.dim_latent = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        # maybe normalize
        self.normalize = normalize

        self._register_hook()

    def _init_pretrained(self, model_name: str):
        """Load pretrained weights from HuggingFace"""
        from diffusers import CosmosVideoToWorldPipeline
        
        pipeline = CosmosVideoToWorldPipeline.from_pretrained(
            model_name, 
            torch_dtype = self.torch_dtype
        )
        
        # Extract components we need
        self.vae = pipeline.vae.to(self.device)
        self.transformer = pipeline.transformer.to(self.device)
        self.text_encoder = pipeline.text_encoder.to(self.device)
        self.tokenizer = pipeline.tokenizer

        # Clean up pipeline
        del pipeline

    def _init_random_weights(self, tiny: bool = False):
        """Initialize with random weights for testing"""
        
        transformer_config = TINY_TRANSFORMER_CONFIG if tiny else REAL_TRANSFORMER_CONFIG
        vae_config = TINY_VAE_CONFIG if tiny else REAL_VAE_CONFIG
        t5_config_dict = TINY_T5_CONFIG if tiny else REAL_T5_CONFIG

        num_layers = max(2, self.extract_layer + 1)
        if not tiny:
            num_layers = max(28, num_layers)
        
        self.transformer = CosmosTransformer3DModel(
            num_layers = num_layers,
            **transformer_config
        ).to(self.device)
        
        self.vae = AutoencoderKLCosmos(**vae_config).to(self.device)
        
        t5_config = T5Config(**t5_config_dict)
        self.text_encoder = T5EncoderModel(t5_config).to(self.device)
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    def __del__(self):
        if exists(self.hook_handle):
            self.hook_handle.remove()

    def _register_hook(self):
        assert hasattr(self.transformer, 'transformer_blocks'), 'transformer must have transformer_blocks'
        assert len(self.transformer.transformer_blocks) > self.extract_layer, f'layer {self.extract_layer} out of bounds'

        target_layer = self.transformer.transformer_blocks[self.extract_layer]

        def hook_fn(module, inp, out):
            self.cached_hidden_states.append(out.detach().cpu())

        self.hook_handle = target_layer.register_forward_hook(hook_fn)

    def forward(
        self,
        videos: Tensor,
        prompts: str | list[str] | None = None,
        num_inference_steps: int = 1,
    ) -> Tensor:
        """
        videos: (batch, frames, channels, height, width) in [0, 1]
        num_inference_steps: number of denoising steps to run
        returns: hidden states tensor from the specified transformer layer (from first step)
        """
        b, t, c, h, w = videos.shape

        # Scale videos from [0, 1] to [-1, 1] for Cosmos VAE

        videos = self.normalize(videos)
        
        prompts = default(prompts, [""] * b)
        if isinstance(prompts, str):
            prompts = [prompts] * b

        self.cached_hidden_states.clear()
        
        # Move video to device and rearrange for VAE: (B, T, C, H, W) -> (B, C, T, H, W)
        videos = rearrange(videos, 'b t c h w -> b c t h w')
        videos = videos.to(device=self.device, dtype=self.torch_dtype)
        
        with torch.inference_mode():
            # 1. Encode video to latents via VAE
            latents = self.vae.encode(videos).latent_dist.sample()
            
            # 2. Encode text prompts
            text_inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            encoder_hidden_states = self.text_encoder(**text_inputs).last_hidden_state
            encoder_hidden_states = encoder_hidden_states.to(dtype=self.torch_dtype)
            
            # 3. Setup scheduler timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps
            
            # 4. Add noise to latents (start from pure noise scaled by initial sigma)
            noise = torch.randn_like(latents)
            latents = latents + noise * self.scheduler.init_noise_sigma
            
            # 5. Denoising loop
            for i, timestep in enumerate(timesteps):
                # Scale model input
                latent_model_input = self.scheduler.scale_model_input(latents, timestep)
                
                # Predict noise residual
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep.expand(b),
                    return_dict=False
                )[0]
                
                # Compute previous noisy sample
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        assert len(self.cached_hidden_states) > 0, 'hidden states not captured'
        
        # Return hidden states from the first denoising step
        hidden = self.cached_hidden_states[0]
        
        assert hidden.shape[-1] == self.dim_latent, f'hidden dim mismatch: expected {self.dim_latent_hidden}, got {hidden.shape[-1]}'
        
        return hidden
