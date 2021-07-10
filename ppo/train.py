from typing import Iterator, Union

import numpy as np
import torch
from torch.nn import functional as F

from ppo.agent import PPOAgent, RNNPolicy
from ppo.data import TrajectoryBatch, FrameBatch
from ppo.log import log_train


def train(
    data: Iterator[Union[FrameBatch, TrajectoryBatch]],
    agent: PPOAgent,
    policy_optimizer,
    value_fn_optimizer,
    ppo_clip: float,
    entropy_coeff: float,
    epochs: int,
):
    is_rnn = isinstance(agent.policy, RNNPolicy)

    for i_epoch in range(1, epochs + 1):
        losses_ppo = []
        losses_ent = []
        losses_vf = []

        for batch in data:
            # batch = batch.to(device)

            # update RNN policy
            policy_optimizer.zero_grad()

            if is_rnn:
                logits, _ = agent.policy(batch.observations, return_logits=True)
            else:
                logits = agent.policy(batch.observations)

            logits = logits.data.masked_fill(batch.illegal_mask, float("-inf"))
            logps = F.log_softmax(logits, dim=-1)

            # loss_ent is actually (coeff * -entropy), since we want to max entropy
            loss_ent = logps.exp() * logps.masked_fill(batch.illegal_mask, 0.0)
            loss_ent = entropy_coeff * torch.mean(torch.sum(loss_ent, dim=-1))

            action_logps = torch.gather(logps, dim=1, index=batch.actions)
            ratio = torch.exp(action_logps - batch.action_logps).squeeze()
            surr_1 = ratio * batch.advantages
            surr_2 = torch.clip(ratio, 1 - ppo_clip, 1 + ppo_clip) * batch.advantages
            loss_ppo = -torch.mean(torch.minimum(surr_1, surr_2))

            (loss_ppo + loss_ent).backward()
            policy_optimizer.step()

            # update value function
            value_fn_optimizer.zero_grad()

            if is_rnn:
                values = agent.value_fn(batch.observations.data)
            else:
                values = agent.value_fn(batch.observations)
            loss_vf = F.smooth_l1_loss(values, batch.emprets)

            loss_vf.backward()
            value_fn_optimizer.step()

            losses_ppo.append(loss_ppo.item())
            losses_ent.append(loss_ent.item())
            losses_vf.append(loss_vf.item())

        avg_loss_ppo = np.mean(losses_ppo)
        avg_loss_ent = np.mean(losses_ent)
        avg_loss_vf = np.mean(losses_vf)

        log_train.info(
            f"Training epoch {i_epoch}: "
            f"average loss_ppo={avg_loss_ppo:.4f}, "
            f"loss_ent={avg_loss_ent:.4f}, "
            f"loss_vf={avg_loss_vf:.4f}"
        )
