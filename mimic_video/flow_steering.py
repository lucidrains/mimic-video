from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.nn import Module

from mimic_video.mimic_video import MimicVideo

from ema_pytorch import EMA

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class FlowSteering(Module):
    def __init__(
        self,
        model: MimicVideo,
        *,
        discount_factor = 0.999,
        ema_decay = 0.999,
        actor_loss_weight = 1.,
        outer_critic_loss_weight = 1.,
        inner_critic_loss_weight = 1.,
        ema_kwargs: dict | None = None
    ):
        super().__init__()
        ema_kwargs = default(ema_kwargs, dict())

        self.discount_factor = discount_factor

        self.actor_loss_weight = actor_loss_weight
        self.outer_critic_loss_weight = outer_critic_loss_weight
        self.inner_critic_loss_weight = inner_critic_loss_weight

        self.action_flow_model = model
        self.actor = model.create_actor_from()

        self.outer_critic = model.create_critic_from()
        self.inner_critic = model.create_critic_from()

        self.ema_outer_critic = EMA(self.outer_critic, beta = ema_decay, **ema_kwargs)
        self.ema_inner_critic = EMA(self.inner_critic, beta = ema_decay, **ema_kwargs)

        # share the video wrapper across all models - EMA deepcopy creates broken copies with stale hooks

        video_wrapper = model.video_predict_wrapper

        for m in (self.action_flow_model, self.actor.model, self.outer_critic.model, self.inner_critic.model, self.ema_outer_critic.ema_model.model, self.ema_inner_critic.ema_model.model):
            m.video_predict_wrapper = video_wrapper

    def actor_forward(
        self,
        *args,
        actions = None,
        sample_noise_latent = True,
        **kwargs
    ):
        actor_out = self.actor(*args, **kwargs)
        mean, logvar = actor_out.unbind(dim = -1)
        std = (0.5 * logvar).exp()

        if not sample_noise_latent:
            return mean, std

        return torch.normal(mean, std)

    def update_critic_ema(self):
        self.ema_outer_critic.update()
        self.ema_inner_critic.update()

    def sample(
        self,
        *args,
        actions = None,
        exploration_noise = None,
        exploration_noise_std = None,
        **kwargs
    ):
        noise_latents = self.actor_forward(*args, **kwargs)

        # maybe exploration noise for the noise latent actions

        assert not (exists(exploration_noise) and exists(exploration_noise_std))

        if exists(exploration_noise_std):
            exploration_noise = torch.randn_like(noise_latents) * exploration_noise_std

        if exists(exploration_noise):
            noise_latents = noise_latents + exploration_noise

        with torch.no_grad():
            self.action_flow_model.eval()
            actions = self.action_flow_model.sample(*args, noise_latents = noise_latents, **kwargs)

        return actions, noise_latents

    def forward(
        self,
        *args,
        video,
        joint_state,
        actions,
        noise_latents,
        next_video,
        next_joint_state,
        rewards, # (b)
        **kwargs,
    ):
        next_actions, next_noise_latents = self.sample(*args, video = next_video, joint_state = next_joint_state, **kwargs)

        # actor loss

        actor_loss = -self.inner_critic(*args, video = video, joint_state = joint_state, actions = noise_latents, **kwargs).mean()

        # bellman for outer critic

        target_q = rewards + self.discount_factor * self.ema_outer_critic(*args, video = next_video, joint_state = next_joint_state, actions = next_actions, **kwargs)

        outer_critic_loss = F.mse_loss(
            self.outer_critic(*args, video = video, joint_state = joint_state, actions = actions, **kwargs),
            target_q.detach()
        )

        # tether the inner critic to the outer critic q estimation
        # main contribution of the paper

        inner_target_q = self.ema_outer_critic(*args, video = video, joint_state = joint_state, actions = actions, **kwargs)

        inner_critic_loss = F.mse_loss(
            self.inner_critic(*args, video = video, joint_state = joint_state, actions = noise_latents.detach(), **kwargs),
            inner_target_q.detach()
        )

        # total loss

        total_loss = (
            actor_loss * self.actor_loss_weight +
            outer_critic_loss * self.outer_critic_loss_weight +
            inner_critic_loss * self.inner_critic_loss_weight
        )

        loss_breakdown = (actor_loss, outer_critic_loss, inner_critic_loss)

        return total_loss, loss_breakdown
