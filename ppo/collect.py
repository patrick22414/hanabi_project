from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Union

import torch

from ppo.agent import PPOAgent, RNNPolicy
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

    elif collection_type.startswith("seg"):
        raise NotImplementedError

    else:
        raise ValueError(
            '`collection_type` must be one of "frame", "seg n", or "traj" ("n" is an integer)'
        )


def _collect_frames(
    collection_size: int,
    env: PPOEnv,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    assert not isinstance(agent.policy, RNNPolicy), "Cannot use RNNPolicy with frames"

    collection: List[Frame] = []

    gae_time_cost = 0.0
    start = perf_counter()

    while len(collection) < collection_size:
        frames = []

        env.reset()
        is_terminal = False

        while not is_terminal:
            p = env.cur_player

            obs = torch.tensor(env.observations[p], dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, action_logp = agent.policy(obs, illegal_mask)

            # env update
            reward, is_terminal = env.step(action.item())
            reward = obs.new_tensor(reward)

            # value estimation
            value = agent.value_fn(obs)
            if is_terminal:
                value_t1 = value.new_tensor(0.0)
            else:
                obs_t1 = obs.new_tensor(env.observations[p])
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

        gae_start = perf_counter()
        frames = _gae(frames, gae_gamma, gae_lambda)
        gae_time_cost += perf_counter() - gae_start

        collection.extend(frames)

    log_collect.info(
        f"Collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection[:collection_size]


def _collect_trajectories(
    collection_size: int,
    env: PPOEnv,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    num_players = env.players
    enc_size = env.enc_size
    is_rnn = isinstance(agent.policy, RNNPolicy)

    collection: List[Trajectory] = []

    gae_time_cost = 0.0
    start = perf_counter()

    while len(collection) < collection_size:
        # we can collect `num_players` trajectories in one episode
        trajs = [_IncompleteTrajectory() for _ in range(num_players)]

        # turn-based observations
        env.reset()
        obs_buffer = deque(
            [torch.zeros(num_players, enc_size) for _ in range(num_players - 1)],
            maxlen=num_players,
        )
        obs_buffer.append(torch.tensor(env.observations, dtype=torch.float))

        # RNN hidden states (h_0); one for each player
        if is_rnn:
            memories = torch.zeros(
                num_players,
                agent.policy.num_layers,
                1,
                agent.policy.hidden_size,
            )

        is_terminal = False

        while not is_terminal:
            p = env.cur_player

            # make turn-based observation vector
            obs = torch.cat([obs[p] for obs in obs_buffer])
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            if is_rnn:
                # obs and mask are shaped as 1-length and 1-batch
                action, action_logp, new_mem = agent.policy(
                    obs.view(1, 1, -1), illegal_mask.view(1, 1, -1), memories[p]
                )
                memories[p] = new_mem
            else:
                action, action_logp = agent.policy(obs, illegal_mask)

            action = action.view(1)
            action_logp = action_logp.view(1)

            # env update
            reward, is_terminal = env.step(action.item())
            reward = obs.new_tensor(reward)

            obs_buffer.append(obs.new_tensor(env.observations))

            # value estimation
            value = agent.value_fn(obs)
            if is_terminal:
                value_t1 = value.new_tensor(0.0)
            else:
                obs_t1 = torch.cat([obs[p] for obs in obs_buffer])
                value_t1 = agent.value_fn(obs_t1)

            traj = trajs[p]
            traj.observations.append(obs)
            traj.illegal_mask.append(illegal_mask)
            traj.action_logps.append(action_logp)
            traj.actions.append(action)
            traj.rewards.append(reward)
            traj.values.append(value)
            traj.values_t1.append(value_t1)

        gae_start = perf_counter()
        trajs = [_gae(traj.seal(), gae_gamma, gae_lambda) for traj in trajs]
        gae_time_cost += perf_counter() - gae_start

        collection.extend(trajs)

    log_collect.info(
        f"Collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection[:collection_size]


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

    def seal(self):
        return Trajectory(
            observations=torch.stack(self.observations),
            illegal_mask=torch.stack(self.illegal_mask),
            action_logps=torch.stack(self.action_logps),
            actions=torch.stack(self.actions),
            rewards=torch.stack(self.rewards),
            values=torch.stack(self.values),
            values_t1=torch.stack(self.values_t1),
        )


def _gae(episode: Union[List[Frame], Trajectory], gamma: float, lam: float):
    """Generalised Advantage Estimation over one episode"""
    length = len(episode)

    exponents = torch.arange(length)
    glexp = torch.tensor(gamma * lam).pow(exponents)  # exponentials of (gamma * lam)
    gammaexp = torch.tensor(gamma).pow(exponents)  # exponentials of gamma

    if isinstance(episode, list):
        rewards = torch.stack([f.reward for f in episode])
        values = torch.stack([f.value for f in episode])
        values_t1 = torch.stack([f.value_t1 for f in episode])

        deltas = rewards + gamma * values_t1 - values

        # empirical return is a discounted sum of all future returns
        # advantage is a discounted sum of all future deltas
        empret = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        for t in range(length):
            empret[t] = torch.sum(gammaexp[: length - t] * rewards[t:])
            advantages[t] = torch.sum(glexp[: length - t] * deltas[t:])

        for f, r, adv in zip(episode, empret, advantages):
            f.empret = r
            f.advantage = adv

        return episode

    elif isinstance(episode, Trajectory):
        deltas = episode.rewards + gamma * episode.values_t1 - episode.values

        # empirical return is a discounted sum of all future returns
        # advantage is a discounted sum of all future deltas
        episode.emprets = torch.zeros_like(deltas)
        episode.advantages = torch.zeros_like(deltas)
        for t in range(length):
            episode.emprets[t] = torch.sum(gammaexp[: length - t] * episode.rewards[t:])
            episode.advantages[t] = torch.sum(glexp[: length - t] * deltas[t:])

        return episode

    else:
        raise TypeError
