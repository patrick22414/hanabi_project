from typing import Callable, Union, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


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


@torch.no_grad()
def _action_sampling(logits: Tensor, illegal_mask: Tensor):
    """
    logits: [batch_size, num_actions] or [num_actions]
    illegal_mask: same as logits
    """
    logits[illegal_mask] = float("-inf")
    logps = F.log_softmax(logits, dim=-1)
    probs = torch.exp(logps)
    probs[illegal_mask] = 0.0  # exp(-inf) should be 0.0, but just to make sure

    actions = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)

    if logits.ndim == 1:
        actions = actions.view(1)
    else:
        actions = actions.view(*logits.shape[:-1], 1)
    action_logps = torch.gather(logps, dim=-1, index=actions)

    entropy = -torch.sum(probs * logps.masked_fill(illegal_mask, 0.0), dim=-1)

    return actions, action_logps, entropy


class _RNNBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        use_preprocessing: bool,
    ):
        super().__init__()

        if use_preprocessing:
            self.pre = _linear_tanh(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.pre = None
            self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.post = _linear_tanh(hidden_size, output_size, tanh=False)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        for name, param in self.named_parameters():
            if "bias" in name or "post" in name:
                torch.nn.init.zeros_(param)

    def new_states(self, batch_size: int, extra_dims=None):
        if extra_dims is None:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            return torch.zeros(
                *extra_dims, self.num_layers, batch_size, self.hidden_size
            )


class RNNPolicy(_RNNBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        use_preprocessing: bool = True,
    ):
        super().__init__(
            input_size, output_size, hidden_size, num_layers, use_preprocessing
        )
        self._args = input_size, output_size, hidden_size, num_layers, use_preprocessing

    def forward(
        self,
        x: Union[Tensor, PackedSequence],
        h_0: Tensor = None,
        illegal_mask: Tensor = None,
    ):
        if self.training:
            assert isinstance(x, PackedSequence)

            if h_0 is None:
                h_0 = self.new_states(len(x.sorted_indices))

            if self.pre:
                x = _apply_to_packed_sequence(self.pre, x)
            logits, h_t = self.gru(x, h_0)
            logits = _apply_to_packed_sequence(self.post, logits)

            return logits, h_t
        else:
            assert isinstance(x, Tensor)
            assert h_0 is not None

            if self.pre:
                x = self.pre(x)
            logits, h_t = self.gru(x, h_0)

            # non-acting case, only new states are needed
            if illegal_mask is None:
                return h_t

            # sample actions from the last output
            logits = self.post(logits)
            action, action_logp, ent = _action_sampling(logits[-1], illegal_mask)
            return action, action_logp, ent, h_t


class RNNValueFn(_RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        use_preprocessing: bool = True,
    ):
        super().__init__(input_size, 1, hidden_size, num_layers, use_preprocessing)
        self._args = input_size, hidden_size, num_layers, use_preprocessing

    def forward(self, x: Union[Tensor, PackedSequence], h_0: Tensor = None):
        if self.training:
            assert isinstance(x, PackedSequence)
            if h_0 is None:
                h_0 = self.new_states(len(x.sorted_indices))

            if self.pre:
                x = _apply_to_packed_sequence(self.pre, x)
            values, h_t = self.gru(x, h_0)
            values = _apply_to_packed_sequence(self.post, values)

            return values, h_t
        else:
            assert isinstance(x, Tensor)
            assert h_0 is not None

            if self.pre:
                x = self.pre(x)
            values, h_t = self.gru(x, h_0)
            values = self.post(values)

            return values, h_t


class _MLPBase(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, num_layers: int
    ):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                _linear_tanh(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ],
            _linear_tanh(hidden_size, output_size, tanh=False),
        )

        for name, param in self.named_parameters():
            if "bias" in name or f"layers.{num_layers}" in name:
                torch.nn.init.zeros_(param)

        self._args = input_size, output_size, hidden_size, num_layers


class MLPPolicy(_MLPBase):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, num_layers: int
    ):
        super().__init__(input_size, output_size, hidden_size, num_layers)
        self._args = input_size, output_size, hidden_size, num_layers

    def forward(self, x: Tensor, illegal_mask: Tensor = None):
        logits = self.layers(x)

        if self.training:
            return logits
        else:
            return _action_sampling(logits, illegal_mask)


class MLPValueFn(_MLPBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__(input_size, 1, hidden_size, num_layers)
        self._args = input_size, hidden_size, num_layers

    def forward(self, x):
        value = self.layers(x).squeeze()
        return value


class PPOAgent(nn.Module):
    policy: Union[MLPPolicy, RNNPolicy]
    value_fn: Union[MLPValueFn, RNNValueFn]

    @overload
    def __init__(self, policy: nn.Module, value_fn: nn.Module):
        ...

    @overload
    def __init__(self, state_dict: dict):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__()

        if len(args) == 2:
            policy, value_fn = args
            self.add_module("policy", policy)
            self.add_module("value_fn", value_fn)
        elif len(args) == 1:
            state_dict: dict = args[0]

            policy_type = state_dict.pop("policy.type")
            policy_args = state_dict.pop("policy.args")
            if policy_type == MLPPolicy.__name__:
                self.policy = MLPPolicy(*policy_args)
            elif policy_type == RNNPolicy.__name__:
                self.policy = RNNPolicy(*policy_args)

            value_fn_type = state_dict.pop("value_fn.type")
            value_fn_args = state_dict.pop("value_fn.args")
            if value_fn_type == MLPValueFn.__name__:
                self.value_fn = MLPValueFn(*value_fn_args)
            elif value_fn_type == RNNValueFn.__name__:
                self.value_fn = RNNValueFn(*value_fn_args)

            self.load_state_dict(state_dict)
        else:
            raise ValueError

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["policy.type"] = self.policy.__class__.__name__
        state_dict["policy.args"] = self.policy._args
        state_dict["value_fn.type"] = self.value_fn.__class__.__name__
        state_dict["value_fn.args"] = self.value_fn._args
        return state_dict
