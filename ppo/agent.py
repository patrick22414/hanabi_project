from typing import Callable, Union
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


class PPOAgent(nn.Module):
    policy: nn.Module
    value_fn: nn.Module

    def __init__(self, policy, value_fn):
        super().__init__()

        self.add_module("policy", policy)
        self.add_module("value_fn", value_fn)


class RNNPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()

        self.policy_pre = _linear_tanh(input_size, hidden_size)
        self.policy_rnn = nn.GRU(hidden_size, hidden_size)
        self.policy_post = _linear_tanh(hidden_size, output_size, tanh=False)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, illegal_mask=None, h_0=None, return_logits=False):
        if isinstance(x, PackedSequence):
            # `PackedSequence`s are used for training, representing a batch of full trajectories
            assert h_0 is None
            h_0 = torch.zeros(
                self.num_layers,
                len(x.sorted_indices),
                self.hidden_size,
                device=x.data.device,
            )

            # x.data: [sum(L*), input_size]
            x = _apply_to_packed_sequence(self.policy_pre, x)
            logits, h_0 = self.policy_rnn(x, h_0)
            logits = _apply_to_packed_sequence(self.policy_post, logits)
            # logits.data: [sum(L*), output_size]

        else:
            assert h_0 is not None

            # x: [L_seq, N_batch, input_size]
            x = self.policy_pre(x)
            logits, h_0 = self.policy_rnn(x, h_0)
            logits = self.policy_post(logits)
            # y: [L_seq, N_batch, output_size]

        if return_logits:
            return logits, h_0
        else:
            action, action_logp = _action_sampling(logits, illegal_mask)
            return action, action_logp, h_0


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

        for p in self.layers[-1].parameters():
            torch.nn.init.zeros_(p)

    def forward(self, x, illegal_mask):
        logits = self.layers(x)
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

        for p in self.layers[-1].parameters():
            torch.nn.init.zeros_(p)

    def forward(self, x):
        value = self.layers(x).squeeze()
        return value


def _linear_tanh(input_size, output_size, tanh=True):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.Tanh() if tanh else nn.Identity(),
    )


def _apply_to_packed_sequence(fn: Callable, ps: PackedSequence):
    return PackedSequence(
        fn(ps.data), ps.batch_sizes, ps.sorted_indices, ps.unsorted_indices
    )


def _action_sampling(
    logits: Union[torch.Tensor, PackedSequence], illegal_mask: torch.BoolTensor
):
    """
    logits: [*, num_actions]
    illegal_mask: same as logits
    """
    if isinstance(logits, PackedSequence):
        raise NotImplementedError

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

    return action, action_logp
