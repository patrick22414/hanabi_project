import torch
from torch.nn import Module, ReLU, Sequential, Linear, Identity
from torch.nn import functional as F


class MLPAgent(Module):
    def __init__(self, obs_size, num_actions, hidden_size, num_layers):
        super().__init__()

        self.layers = Sequential(
            *[
                Sequential(
                    Linear(
                        obs_size if i == 0 else hidden_size,
                        hidden_size if i < num_layers else num_actions,
                    ),
                    ReLU() if i < num_layers else Identity(),
                )
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, obs, logp=True):
        logits = self.layers(obs)

        if logp:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
