import torch

from .env import Frame


class FrameBatch:
    def __init__(self, frames: list[Frame]):
        self.frame_types = [f.frame_type for f in frames]
        self.observations = torch.stack([f.observation for f in frames])
        self.action_logps = torch.stack([f.action_logp for f in frames])
        self.actions = torch.stack([f.action for f in frames])
        self.advantages = torch.stack([f.advantage for f in frames])

    def pin_memory(self):
        self.observations = self.observations.pin_memory()
        self.action_logps = self.action_logps.pin_memory()
        self.actions = self.actions.pin_memory()
        self.advantages = self.advantages.pin_memory()

        return self
