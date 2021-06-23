from dataclasses import dataclass
from enum import Enum
from typing import List

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
    legal_moves: List[int] = None


class FrameBatch:
    def __init__(self, frames: List[Frame]):
        self.frame_types = [f.frame_type for f in frames]
        self.observations = torch.stack([f.observation for f in frames])
        self.action_logps = torch.stack([f.action_logp for f in frames])
        self.actions = torch.stack([f.action for f in frames])
        self.emprets = torch.stack([f.empret for f in frames])
        self.advantages = torch.stack([f.advantage for f in frames])
        self.legal_moves = [f.legal_moves for f in frames]

    def to(self, device):
        self.observations = self.observations.to(device)
        self.action_logps = self.action_logps.to(device)
        self.actions = self.actions.to(device)
        self.emprets = self.emprets.to(device)
        self.advantages = self.advantages.to(device)

        return self
