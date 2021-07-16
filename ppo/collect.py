from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Union

import torch

from ppo.agent import MLPPolicy, MLPValueFn, PPOAgent, RNNPolicy
from ppo.data import Frame, Trajectory
from ppo.env import PPOEnv
from ppo.log import log_collect


@torch.no_grad()
def collect(
    collection_type: str,
    collection_size: int,
    parallel: int,
    env_or_envs: Union[PPOEnv, List[PPOEnv]],
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
    use_same_player_for_next_value=False,
):
    if collection_type == "frame":
        if parallel > 1:
            return _parallel_collect_frames(
                collection_size,
                env_or_envs,
                agent,
                gae_gamma,
                gae_lambda,
                use_same_player_for_next_value,
            )
        else:
            return _collect_frames(
                collection_size,
                env_or_envs,
                agent,
                gae_gamma,
                gae_lambda,
                use_same_player_for_next_value,
            )

    elif collection_type == "traj":
        return _collect_trajectories(
            collection_size, env_or_envs, agent, gae_gamma, gae_lambda
        )

    else:
        raise ValueError('`collection_type` must be "frame" or "traj"')


def _collect_frames(
    collection_size: int,
    env: PPOEnv,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
    use_same_player_for_next_value: bool,
):
    assert isinstance(agent.policy, MLPPolicy)
    assert isinstance(agent.value_fn, MLPValueFn)

    collection: List[Frame] = []
    total_entropy = 0.0

    gae_time_cost = 0.0
    start = perf_counter()

    while len(collection) < collection_size:
        frames: List[Frame] = []

        env.reset()
        is_terminal = False

        while not is_terminal:
            p = env.cur_player

            obs = torch.tensor(env.observation(p), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, action_logp, entropy = agent.policy(obs, illegal_mask)
            total_entropy += entropy.sum().item()

            # env update
            reward, is_terminal = env.step(action.item())
            reward = obs.new_tensor(reward)

            # value estimation
            value = agent.value_fn(obs)
            if use_same_player_for_next_value:
                if is_terminal:
                    value_t1 = value.new_tensor(0.0)
                else:
                    obs_t1 = obs.new_tensor(env.observation(p))
                    value_t1 = agent.value_fn(obs_t1)
            else:
                # give value to the prev frame, and assign 0.0 to this frame
                if len(frames) > 0:
                    frames[-1].value_t1 = value
                value_t1 = value.new_tensor(0.0)

            # __import__("ipdb").set_trace()

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
        frames = _gae_frames(frames, gae_gamma, gae_lambda)
        gae_time_cost += perf_counter() - gae_start

        collection.extend(frames)

    log_collect.info(f"Collection size: {len(collection)} frames")
    log_collect.info(f"Collection avg_entropy: {total_entropy / len(collection):.4f}")
    log_collect.info(
        f"Collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection


def _parallel_collect_frames(
    collection_size: int,
    envs: List[PPOEnv],
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
    use_same_player_for_next_value: bool,
):
    assert isinstance(agent.policy, MLPPolicy)
    assert isinstance(agent.value_fn, MLPValueFn)

    collection: List[Frame] = []

    frames: List[List[Frame]] = [[] for _ in envs]
    is_terminal_ls = [True] * len(envs)
    total_entropy = 0.0

    gae_time_cost = 0.0
    start = perf_counter()

    while len(collection) < collection_size:
        for is_terminal, env in zip(is_terminal_ls, envs):
            if is_terminal:
                env.reset()

        ps = [env.cur_player for env in envs]

        obs = torch.stack(
            [
                torch.tensor(env.observation(p), dtype=torch.float)
                for p, env in zip(ps, envs)
            ]
        )
        illegal_mask = torch.stack([torch.tensor(env.illegal_mask) for env in envs])

        actions, action_logps, entropy = agent.policy(obs, illegal_mask)
        total_entropy += entropy.sum().item()

        rewards, is_terminal_ls = tuple(
            zip(*[env.step(a.item()) for a, env in zip(actions, envs)])
        )
        rewards = [obs.new_tensor(r) for r in rewards]

        values = agent.value_fn(obs)
        if use_same_player_for_next_value:
            obs_t1 = torch.stack(
                [
                    torch.tensor(env.observation(p), dtype=torch.float)
                    for p, env in zip(ps, envs)
                ]
            )
            values_t1 = agent.value_fn(obs_t1)
            values_t1[is_terminal_ls] = 0.0
        else:
            # give value to the prev frame, and assign 0.0 to this frame
            for fs, v in zip(frames, values):
                if len(fs) > 0:
                    fs[-1].value_t1 = v
            values_t1 = torch.zeros_like(values)

        # __import__("ipdb").set_trace()

        for i, fs in enumerate(frames):
            fs.append(
                Frame(
                    observation=obs[i],
                    illegal_mask=illegal_mask[i],
                    action_logp=action_logps[i],
                    action=actions[i],
                    reward=rewards[i],
                    value=values[i],
                    value_t1=values_t1[i],
                )
            )

        for is_terminal, fs in zip(is_terminal_ls, frames):
            if is_terminal:
                gae_start = perf_counter()
                collection.extend(_gae_frames(fs, gae_gamma, gae_lambda))
                fs.clear()
                gae_time_cost += perf_counter() - gae_start

    log_collect.info(f"Collection size: {len(collection)} frames")
    log_collect.info(
        f"Collection approximate avg_entropy: {total_entropy / len(collection):.4f}"
    )
    log_collect.info(
        f"Parallel collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
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
