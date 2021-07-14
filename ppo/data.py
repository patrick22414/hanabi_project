from dataclasses import dataclass
from typing import List

import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence


@dataclass
class Frame:
    observation: torch.Tensor  # [obs_size]
    illegal_mask: torch.BoolTensor  # [num_actions]
    action_logp: torch.Tensor  # [1]
    action: torch.LongTensor  # [1]
    reward: torch.Tensor  # []
    value: torch.Tensor  # []
    value_t1: torch.Tensor  # []
    empret: torch.Tensor = None  # [], empirical return
    advantage: torch.Tensor = None  # []


class FrameBatch:
    def __init__(self, frames: List[Frame]):
        self.observations = torch.stack([f.observation for f in frames])
        self.illegal_mask = torch.stack([f.illegal_mask for f in frames])
        self.action_logps = torch.stack([f.action_logp for f in frames])
        self.actions = torch.stack([f.action for f in frames])
        self.emprets = torch.stack([f.empret for f in frames])
        self.advantages = torch.stack([f.advantage for f in frames])

    def to(self, device):
        self.observations = self.observations.to(device)
        self.illegal_mask = self.illegal_mask.to(device)
        self.action_logps = self.action_logps.to(device)
        self.actions = self.actions.to(device)
        self.emprets = self.emprets.to(device)
        self.advantages = self.advantages.to(device)

        return self


# Turn-based trajectory
@dataclass
class Trajectory:
    observations: torch.Tensor  # [L, enc_size]
    action_mask: torch.BoolTensor  # [L]
    illegal_mask: torch.BoolTensor  # [LT, num_actions]
    action_logps: torch.Tensor  # [LT, 1]
    actions: torch.LongTensor  # [LT, 1]
    advantages: torch.Tensor  # [LT]
    # rewards: torch.Tensor  # [L]
    # values: torch.Tensor  # [L]
    # values_t1: torch.Tensor  # [L]
    emprets: torch.Tensor  # [L]


class TrajectoryBatch:
    def __init__(self, trajs: List[Trajectory]):
        self.observations = pack_sequence(
            [t.observations for t in trajs], enforce_sorted=False
        )

        # reorder trajs according to their order in PackedSequence
        trajs = [trajs[i] for i in self.observations.sorted_indices]

        self.action_mask = _pack_tensor([t.action_mask for t in trajs])
        self.illegal_mask = _pack_tensor([t.illegal_mask for t in trajs])
        self.action_logps = _pack_tensor([t.action_logps for t in trajs])
        self.actions = _pack_tensor([t.actions for t in trajs])
        self.advantages = _pack_tensor([t.advantages for t in trajs])

        self.emprets = _pack_tensor([t.emprets for t in trajs])  # [sum(L)]

        # print(self.observations.data.shape)
        # print(self.action_mask.shape)
        # print(self.illegal_mask.shape)
        # print(self.action_logps.shape)
        # print(self.actions.shape)
        # print(self.advantages.shape)
        # print(self.emprets.shape)

        # __import__("ipdb").set_trace()


def _pack_tensor(tensors) -> torch.Tensor:
    padded = pad_sequence(tensors)
    padded_mask = pad_sequence([torch.ones(len(x), dtype=torch.bool) for x in tensors])
    return padded[padded_mask]
