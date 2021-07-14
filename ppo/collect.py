from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Union

import torch

from ppo.agent import MLPPolicy, PPOAgent, RNNPolicy
from ppo.data import Frame, Trajectory
from ppo.env import PPOEnv
from ppo.log import log_collect


@torch.no_grad()
def collect(
    collection_type: str,
    collection_size: int,
    collect_workers: int,  # NOT USED!
    env: PPOEnv,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    if collection_type == "frame":
        return _collect_frames(
            collection_size,
            env,
            agent,
            gae_gamma,
            gae_lambda,
        )

    elif collection_type == "traj":
        return _collect_trajectories(
            collection_size,
            env,
            agent,
            gae_gamma,
            gae_lambda,
        )

    else:
        raise ValueError('`collection_type` must be "frame" or "traj"')


def _collect_frames(
    max_step: int,
    env: PPOEnv,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    assert isinstance(agent.policy, MLPPolicy)

    collection: List[Frame] = []
    total_steps = 0
    total_entropy = 0.0

    gae_time_cost = 0.0
    start = perf_counter()

    while total_steps < max_step:
        frames = []

        env.reset()
        is_terminal = False

        while not is_terminal:
            p = env.cur_player

            obs = torch.tensor(env.observation(p), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, action_logp, entropy = agent.policy(obs, illegal_mask)
            total_entropy += entropy.item()

            # env update
            reward, is_terminal = env.step(action.item())
            reward = obs.new_tensor(reward)

            # value estimation
            value = agent.value_fn(obs)
            if is_terminal:
                value_t1 = value.new_tensor(0.0)
            else:
                obs_t1 = obs.new_tensor(env.observation(p))
                value_t1 = agent.value_fn(obs_t1)

            frames.append(
                Frame(
                    observation=obs,
                    illegal_mask=illegal_mask,
                    action_logp=action_logp,
                    action=action,
                    reward=reward,
                    value=value,
                    value_t1=value_t1,
                )
            )

            total_steps += 1

        gae_start = perf_counter()
        frames = _gae_frames(frames, gae_gamma, gae_lambda)
        collection.extend(frames)
        gae_time_cost += perf_counter() - gae_start

    log_collect.info(f"Collection size: {len(collection)} frames")
    log_collect.info(f"Collection avg_entropy: {total_entropy / total_steps:.4f}")
    log_collect.info(
        f"Collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection


def _collect_trajectories(
    max_step: int,
    env: PPOEnv,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    assert isinstance(agent.policy, RNNPolicy)

    players = env.players

    collection: List[Trajectory] = []
    total_steps = 0

    gae_time_cost = 0.0
    start = perf_counter()

    while total_steps < max_step:
        # we can collect `players` trajectories in one episode
        trajs = [_IncompleteTrajectory() for _ in range(players)]

        # RNN hidden states, one for each player
        memories = [agent.policy.new_memory(1) for _ in range(players)]

        env.reset()
        is_terminal = False

        while not is_terminal:
            # make turn-based observation vector
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            # obs and mask are shaped as 1-length and 1-batch
            for p in range(players):
                obs = torch.tensor(env.observation(p), dtype=torch.float)
                trajs[p].observations.append(obs)

                if p == env.cur_player:
                    action, action_logp, entropy, h_t = agent.policy(
                        obs.view(1, 1, -1),
                        illegal_mask=illegal_mask.view(1, -1),
                        h_0=memories[p],
                    )
                    trajs[p].illegal_mask.append(illegal_mask)
                    trajs[p].action_logps.append(action_logp.view(1))
                    trajs[p].actions.append(action.view(1))
                else:
                    h_t = agent.policy(
                        obs.view(1, 1, -1), illegal_mask=None, h_0=memories[p]
                    )

                memories[p] = h_t

                trajs[p].values.append(agent.value_fn(obs))

            # env update
            reward, is_terminal = env.step(action.item())
            reward = torch.tensor(reward, dtype=torch.float)
            for p in range(players):
                trajs[p].rewards.append(reward)

            # estimate value_t1
            if is_terminal:
                for p in range(players):
                    trajs[p].values_t1.append(torch.zeros(()))
            else:
                for p in range(players):
                    obs_t1 = torch.tensor(env.observation(p), dtype=torch.float)
                    trajs[p].values_t1.append(agent.value_fn(obs_t1))

            total_steps += 1

        gae_start = perf_counter()
        for p, t in enumerate(trajs):
            collection.append(_gae_traj(t, p, players, gae_gamma, gae_lambda))
        gae_time_cost += perf_counter() - gae_start

    log_collect.info(
        f"Collection size: {len(collection)} trajectories including {total_steps} steps"
    )
    log_collect.info(
        f"Collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection


@dataclass
class _IncompleteTrajectory:
    """Internal class used for collecting trajectories"""

    observations: List[torch.Tensor] = field(default_factory=list)
    illegal_mask: List[torch.Tensor] = field(default_factory=list)
    action_logps: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    rewards: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    values_t1: List[torch.Tensor] = field(default_factory=list)


def _gae_frames(frames: List[Frame], gamma: float, lam: float):
    """Generalised Advantage Estimation over one episode"""
    length = len(frames)

    exponents = torch.arange(length)
    glexp = torch.tensor(gamma * lam).pow(exponents)  # exponentials of (gamma * lam)
    gammaexp = torch.tensor(gamma).pow(exponents)  # exponentials of gamma

    rewards = torch.stack([f.reward for f in frames])
    values = torch.stack([f.value for f in frames])
    values_t1 = torch.stack([f.value_t1 for f in frames])

    deltas = rewards + gamma * values_t1 - values

    # empirical return is a discounted sum of all future returns
    # advantage is a discounted sum of all future deltas
    empret = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    for t in range(length):
        empret[t] = torch.sum(gammaexp[: length - t] * rewards[t:])
        advantages[t] = torch.sum(glexp[: length - t] * deltas[t:])

    for f, r, adv in zip(frames, empret, advantages):
        f.empret = r
        f.advantage = adv

    return frames


def _gae_traj(
    traj: _IncompleteTrajectory, player_id, players, gamma: float, lam: float
):
    """Generalised Advantage Estimation for trajectories"""
    length = len(traj.observations)

    exponents = torch.arange(length)
    glexp = torch.tensor(gamma * lam).pow(exponents)  # exponentials of (gamma * lam)
    gammaexp = torch.tensor(gamma).pow(exponents)  # exponentials of gamma

    rewards = torch.stack(traj.rewards)
    values = torch.stack(traj.values)
    values_t1 = torch.stack(traj.values_t1)

    deltas = rewards + gamma * values_t1 - values

    emprets = torch.zeros_like(deltas)
    advantages = torch.zeros_like(deltas)
    for t in range(length):
        emprets[t] = torch.sum(gammaexp[: length - t] * rewards[t:])
        advantages[t] = torch.sum(glexp[: length - t] * deltas[t:])  # TODO: check math

    # __import__("ipdb").set_trace()

    observations = torch.stack(traj.observations)
    action_mask = torch.zeros(len(observations), dtype=torch.bool)
    action_mask[player_id::players] = True
    illegal_mask = torch.stack(traj.illegal_mask)
    action_logps = torch.stack(traj.action_logps)
    actions = torch.stack(traj.actions)
    advantages = advantages[player_id::players]

    return Trajectory(
        observations=observations,
        action_mask=action_mask,
        illegal_mask=illegal_mask,
        action_logps=action_logps,
        actions=actions,
        advantages=advantages,
        emprets=emprets,
    )
