import logging
from time import perf_counter
from typing import Iterator, Union

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_value_

from ppo.agent import PPOAgent, RNNPolicy
from ppo.data import FrameBatch, TrajectoryBatch

LOG_TRAIN = logging.getLogger("ppo.train")


def train(
    data: Iterator[Union[FrameBatch, TrajectoryBatch]],
    agent: PPOAgent,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    ppo_clip: float,
    entropy_coef: float,
    value_fn_coef: float,
):
    start = perf_counter()
    is_rnn_policy = isinstance(agent.policy, RNNPolicy)

    for i_epoch in range(1, epochs + 1):
        losses_ppo = []
        losses_ent = []
        losses_vf = []

        for batch in data:
            # batch = batch.to(device)
            optimizer.zero_grad()

            # update RNN policy
            if is_rnn_policy:
                logits, _ = agent.policy(batch.observations)
                logits = logits.data.masked_fill(batch.illegal_masks, float("-inf"))
            else:
                logits = agent.policy(batch.observations)
                logits = logits.masked_fill(batch.illegal_masks, float("-inf"))

            logps = F.log_softmax(logits, dim=-1)

            if entropy_coef != 0:
                # loss_ent is actually (coeff * -entropy), since we want to max entropy
                loss_ent = logps.exp() * logps.masked_fill(batch.illegal_masks, 0.0)
                loss_ent = entropy_coef * torch.mean(torch.sum(loss_ent, dim=-1))

            action_logps = torch.gather(logps, dim=1, index=batch.actions)
            ratio = torch.exp(action_logps - batch.action_logps).squeeze()
            surr_1 = ratio * batch.advantages
            surr_2 = torch.clip(ratio, 1 - ppo_clip, 1 + ppo_clip) * batch.advantages
            loss_ppo = -torch.mean(torch.minimum(surr_1, surr_2))

            # update value function
            values = agent.value_fn(batch.full_observations)
            loss_vf = value_fn_coef * F.smooth_l1_loss(values, batch.emprets)

            # backward
            if entropy_coef != 0:
                (loss_ppo + loss_vf + loss_ent).backward()
            else:
                (loss_ppo + loss_vf).backward()
            clip_grad_value_(agent.parameters(), 1.0)
            optimizer.step()

            losses_ppo.append(loss_ppo.item())
            losses_vf.append(loss_vf.item())
            if entropy_coef != 0:
                losses_ent.append(loss_ent.item())

        avg_loss_ppo = np.mean(losses_ppo)
        avg_loss_vf = np.mean(losses_vf)
        avg_loss_ent = np.mean(losses_ent) if entropy_coef != 0 else 0

        if i_epoch == 1 or i_epoch == epochs:
            LOG_TRAIN.info(
                f"Training epoch {i_epoch:{int(np.ceil(np.log10(epochs + 1)))}d}: "
                f"average loss_ppo={avg_loss_ppo:.4f}, "
                f"loss_ent={avg_loss_ent:.4f}, "
                f"loss_vf={avg_loss_vf:.4f}"
            )

    LOG_TRAIN.info(f"Training done in {perf_counter() - start:.2f} s ")
