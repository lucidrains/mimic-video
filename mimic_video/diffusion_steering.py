from __future__ import annotations

import torch
from torch.nn import Module

from mimic_video.mimic_video import MimicVideo

from ema_pytorch import EMA

# functions

def exists(v):
    return v is not None

# classes

class DiffusionSteering(Module):
    def __init__(
        self,
        model: MimicVideo
    ):
        super().__init__()

        self.model = model

    def sample(
        self,
        *args,
        **kwargs
    ):
        return self.model.sample(*args, **kwargs)

    def forward(
        self,
        *args,
        next_video,
        next_joint_state,
        **kwargs,
    ):
        return self.model(*args, **kwargs)
