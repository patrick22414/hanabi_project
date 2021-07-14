from typing import Callable, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


class RNNPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()

        self.policy_pre = _linear_tanh(input_size, hidden_size)
        self.policy_rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.policy_post = _linear_tanh(hidden_size, output_size, tanh=False)

        # for p in self.parameters():
        #     torch.nn.init.normal_(p, std=1e-6)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def new_memory(self, batch_size: int):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def forward(
        self, x: Union[torch.Tensor, PackedSequence], illegal_mask=None, h_0=None
    ):
        if self.training:
            # `PackedSequence`s are used for training
            assert isinstance(x, PackedSequence)

            if h_0 is None:
                h_0 = self.new_memory(len(x.sorted_indices))

            x = _apply_to_packed_sequence(self.policy_pre, x)
            logits, h_t = self.policy_rnn(x, h_0)
            logits = _apply_to_packed_sequence(self.policy_post, logits)

            return logits, h_t

        else:
            assert h_0 is not None
            assert isinstance(x, torch.Tensor)

            x = self.policy_pre(x)
            logits, h_t = self.policy_rnn(x, h_0)

            if illegal_mask is not None:
                logits = self.policy_post(logits)

                # sample actions from the last output
                action, action_logp, ent = _action_sampling(logits[-1], illegal_mask)
                return action, action_logp, ent, h_t
            else:
                return h_t


class MLPPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                _linear_tanh(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers + 1)
            ],
            _linear_tanh(hidden_size, output_size, tanh=False),
        )

        # for p in self.parameters():
        #     torch.nn.init.normal_(p, 1e-6)

    def forward(self, x: torch.Tensor, illegal_mask=None):
        logits = self.layers(x)

        if self.training:
            return logits
        else:
            return _action_sampling(logits, illegal_mask)


class MLPValueFn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                _linear_tanh(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers + 1)
            ],
            _linear_tanh(hidden_size, 1, tanh=False),
        )

        # for p in self.parameters():
        #     torch.nn.init.normal_(p, std=1e-6)

    def forward(self, x):
        value = self.layers(x).squeeze()
        return value


class PPOAgent(nn.Module):
    policy: Union[MLPPolicy, RNNPolicy]
    value_fn: MLPValueFn

    def __init__(self, policy, value_fn):
        super().__init__()

        self.add_module("policy", policy)
        self.add_module("value_fn", value_fn)


def _linear_tanh(input_size, output_size, tanh=True):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.Tanh() if tanh else nn.Identity(),
    )


def _apply_to_packed_sequence(fn: Callable, ps: PackedSequence):
    """Apply a function to the `data` in a PackedSequence, circumventing the
    non-assignability of NamedTuple
    """
    return PackedSequence(
        fn(ps.data), ps.batch_sizes, ps.sorted_indices, ps.unsorted_indices
    )


def _action_sampling(logits: torch.Tensor, illegal_mask: torch.BoolTensor):
    """
    logits: [batch_size, num_actions] or [num_actions]
    illegal_mask: same as logits
    """
    logits[illegal_mask] = float("-inf")
    logp = F.log_softmax(logits, dim=-1)
    prob = torch.exp(logp)
    prob[illegal_mask] = 0.0  # exp(-inf) should be 0.0, but just to make sure

    action = torch.multinomial(prob.view(-1, prob.shape[-1]), num_samples=1)

    if logits.ndim == 1:
        action = action.view(1)
    else:
        action = action.view(*logits.shape[:-1], 1)
    action_logp = torch.gather(logp, dim=-1, index=action)

    logp[illegal_mask] = 0.0
    entropy = -torch.sum(prob * logp, dim=-1).mean()

    return action, action_logp, entropy
