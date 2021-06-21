from dataclasses import dataclass
from enum import Enum

import torch

FrameType = Enum("FrameType", "START MID END")


@dataclass
class Frame:
    frame_type: FrameType
    observation: torch.Tensor  # 1-d vector
    action_logp: torch.Tensor  # 1-d scalar
    action: torch.LongTensor  # 1-d int64 scalar
    value: torch.Tensor  # scalar
    value_t1: torch.Tensor  # scalar
    reward: torch.Tensor  # scalar
    empret: torch.Tensor = None  # scalar, empirical return
    advantage: torch.Tensor = None  # scalar


class FrameBatch:
    def __init__(self, frames: list[Frame]):
        self.frame_types = [f.frame_type for f in frames]
        self.observations = torch.stack([f.observation for f in frames])
        self.action_logps = torch.stack([f.action_logp for f in frames])
        self.actions = torch.stack([f.action for f in frames])
        self.emprets = torch.stack([f.empret for f in frames])
        self.advantages = torch.stack([f.advantage for f in frames])

    def pin_memory(self):
        self.observations = self.observations.pin_memory()
        self.action_logps = self.action_logps.pin_memory()
        self.actions = self.actions.pin_memory()
        self.emprets = self.emprets.pin_memory()
        self.advantages = self.advantages.pin_memory()

        return self
