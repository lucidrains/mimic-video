from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module

from mimic_video.mimic_video import MimicVideo, Actor
from mimic_video.flow_steering import FlowSteering

from torch_einops_utils import temp_eval

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class FlowDagger(Module):
    """
    FlowDAgger - https://arxiv.org/abs/2607.08877

    trains the latent noise policy - the same actor built by `FlowSteering` - by regressing
    it onto expert actions inverted back through the frozen flow sampler's ODE.
    the base policy weights are never touched, only its initial noise is steered.

    if a `FlowSteering` is passed in, its actor is reused, so latent space RL and flow dagger
    can be interleaved on a single network - as in UniSteer - https://arxiv.org/abs/2605.10821
    """

    def __init__(
        self,
        model: MimicVideo,
        *,
        steering: FlowSteering | None = None,
        actor: Actor | None = None,
        inversion_steps = 16,
        inversion_fixed_point_steps = 5
    ):
        super().__init__()

        self.inversion_steps = inversion_steps
        self.inversion_fixed_point_steps = inversion_fixed_point_steps

        self.action_flow_model = model

        # reuse the actor from flow steering, or build one onto the frozen model

        self.actor = steering.actor if exists(steering) else default(actor, model.create_actor_from())

        # share the video wrapper across all models

        video_wrapper = model.video_predict_wrapper

        if exists(video_wrapper):
            for m in (self.action_flow_model, self.actor.model):
                m.video_predict_wrapper = video_wrapper

    def actor_forward(self, *args, **kwargs):
        mean, logvar = self.actor(*args, **kwargs).unbind(dim = -1)
        std = (0.5 * logvar).exp()

        return mean, std

    @torch.no_grad()
    @temp_eval
    def predict_noise_latents(self, *args, **kwargs):
        mean, _ = self.actor_forward(*args, **kwargs)
        return mean

    def forward(
        self,
        *args,
        video = None,
        joint_state,
        actions = None,            # (b na d) expert actions, inverted on the fly
        noise_latents = None,      # (b na d) already inverted targets, from the replay buffer
        steps = None,
        inversion_fixed_point_steps = None,
        disable_progress_bar = False,
        **kwargs
    ):
        if not exists(noise_latents):
            assert exists(actions), 'either expert `actions` or inverted `noise_latents` must be given'

            # expert actions -> noise latents, via the inverse of the flow sampler

            noise_latents = self.action_flow_model.action_to_noise_latents(
                actions,
                steps = default(steps, self.inversion_steps),
                inversion_fixed_point_steps = default(inversion_fixed_point_steps, self.inversion_fixed_point_steps),
                disable_progress_bar = disable_progress_bar,
                joint_state = joint_state,
                video = video,
                **kwargs
            )

        mean, _ = self.actor_forward(video = video, joint_state = joint_state, **kwargs)

        return F.mse_loss(mean, noise_latents.detach())

    @torch.no_grad()
    @temp_eval
    def sample(
        self,
        *args,
        video = None,
        joint_state,
        steps = 16,
        **kwargs
    ):
        # deterministic latent at deployment, as in the paper

        noise_latents = self.predict_noise_latents(video = video, joint_state = joint_state, **kwargs)

        actions = self.action_flow_model.sample(
            *args,
            steps = steps,
            noise_latents = noise_latents,
            video = video,
            joint_state = joint_state,
            **kwargs
        )

        return actions, noise_latents
