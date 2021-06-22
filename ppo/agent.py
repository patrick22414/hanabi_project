from dataclasses import dataclass

import torch
from torch.nn import Identity, Linear, Module, ReLU, Sequential
from torch.nn import functional as F


# @dataclass
# class PPOAgent:
#     policy_fn: Module
#     value_fn: Module

#     def to(self, device):
#         self.policy_fn.to(device)
#         self.value_fn.to(device)

#     def state_dict(self):
#         return {
#             "policy_fn": self.policy_fn.state_dict(),
#             "value_fn": self.value_fn.state_dict(),
#         }

#     def load_state_dict(self, state_dict):
#         self.policy_fn.load_state_dict(state_dict["policy_fn"])
#         self.value_fn.load_state_dict(state_dict["value_fn"])


class PPOAgent(Module):
    policy_fn: Module
    value_fn: Module

    def __init__(self, policy_fn, value_fn):
        super().__init__()

        self.add_module("policy_fn", policy_fn)
        self.add_module("value_fn", value_fn)


class MLPPolicy(Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()

        self.layers = Sequential(
            *[
                Sequential(
                    Linear(
                        input_size if i == 0 else hidden_size,
                        hidden_size if i < num_layers else output_size,
                    ),
                    ReLU() if i < num_layers else Identity(),
                )
                for i in range(num_layers + 1)
            ]
        )

        for p in self.layers[-1].parameters():
            torch.nn.init.zeros_(p)

    def forward(self, obs, logp=True):
        logits = self.layers(obs)

        if logp:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class MLPValueFunction(Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.layers = Sequential(
            *[
                Sequential(
                    Linear(
                        input_size if i == 0 else hidden_size,
                        hidden_size if i < num_layers else 1,
                    ),
                    ReLU() if i < num_layers else Identity(),
                )
                for i in range(num_layers + 1)
            ]
        )

        for p in self.layers[-1].parameters():
            torch.nn.init.zeros_(p)

    def forward(self, obs):
        value = self.layers(obs).squeeze()
        return value
