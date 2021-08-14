from dataclasses import dataclass
from typing import List

import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence


@dataclass
class Frame:
    observation: torch.Tensor  # [obs_size]
    full_observation: torch.Tensor  # [num_players * enc_size]
    illegal_mask: torch.BoolTensor  # [num_actions]
    action_logp: torch.Tensor  # [1]
    action: torch.LongTensor  # [1]
    reward: torch.Tensor  # []
    value: torch.Tensor  # []
    empret: torch.Tensor = None  # [], empirical return
    advantage: torch.Tensor = None  # []


class FrameBatch:
    def __init__(self, frames: List[Frame]):
        self.observations = torch.stack([f.observation for f in frames])
        self.full_observations = torch.stack([f.full_observation for f in frames])
        self.illegal_masks = torch.stack([f.illegal_mask for f in frames])
        self.action_logps = torch.stack([f.action_logp for f in frames])
        self.actions = torch.stack([f.action for f in frames])
        self.emprets = torch.stack([f.empret for f in frames])
        self.advantages = torch.stack([f.advantage for f in frames])


# Turn-based trajectory
@dataclass
class Trajectory:
    observations: torch.Tensor  # [L, enc_size]
    full_observations: torch.Tensor  # [L, num_players * enc_size]
    illegal_masks: torch.BoolTensor  # [L, num_actions]
    action_logps: torch.Tensor  # [L, 1]
    actions: torch.LongTensor  # [L, 1]
    emprets: torch.Tensor  # [L]
    advantages: torch.Tensor  # [L]
    # rewards: torch.Tensor  # [L]
    # values: torch.Tensor  # [L]
    # values_t1: torch.Tensor  # [L]
    # action_mask: torch.BoolTensor = None  # [L]


class TrajectoryBatch:
    def __init__(self, trajs: List[Trajectory]):
        self.observations = pack_sequence(
            [t.observations for t in trajs], enforce_sorted=False
        )

        # reorder trajs according to their order in PackedSequence
        trajs = [trajs[i] for i in self.observations.sorted_indices]

        self.full_observations = _pack_tensor([t.full_observations for t in trajs])
        self.illegal_masks = _pack_tensor([t.illegal_masks for t in trajs])
        self.action_logps = _pack_tensor([t.action_logps for t in trajs])
        self.actions = _pack_tensor([t.actions for t in trajs])
        self.emprets = _pack_tensor([t.emprets for t in trajs])
        self.advantages = _pack_tensor([t.advantages for t in trajs])

        # __import__("ipdb").set_trace()


def _pack_tensor(tensors) -> torch.Tensor:
    padded = pad_sequence(tensors)
    padded_mask = pad_sequence([torch.ones(len(x), dtype=torch.bool) for x in tensors])
    return padded[padded_mask]
