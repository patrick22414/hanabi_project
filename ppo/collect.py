import logging
from time import perf_counter
from typing import List, Union

import torch

from ppo.agent import MLPPolicy, MLPValueFn, PPOAgent, RNNPolicy, RNNValueFn
from ppo.data import Frame, Trajectory
from ppo.env import PPOEnv

LOG_COLLECT = logging.getLogger("ppo.collect")


def _gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float):
    """Generalised Advantage Estimation over one episode"""
    length = len(rewards)

    exponents = torch.arange(length)
    glexp = torch.tensor(gamma * lam).pow(exponents)  # exponentials of (gamma * lam)
    gammaexp = torch.tensor(gamma).pow(exponents)  # exponentials of gamma

    values_t1 = torch.cat([values[1:], torch.zeros(1)])
    deltas = rewards + gamma * values_t1 - values

    # empirical return is a discounted sum of all future returns
    # advantage is a discounted sum of all future deltas
    emprets = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    for t in range(length):
        emprets[t] = torch.sum(gammaexp[: length - t] * rewards[t:])
        advantages[t] = torch.sum(glexp[: length - t] * deltas[t:])

    return emprets, advantages


def _collect_frames(
    collection_size: int,
    env: PPOEnv,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
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
            total_entropy += entropy.item()

            # value estimation
            full_obs = obs.new_tensor(env.full_observation())
            value = agent.value_fn(full_obs)

            # env step
            reward, is_terminal = env.step(action.item())
            reward = obs.new_tensor(reward)

            # __import__("ipdb").set_trace()

            frames.append(
                Frame(
                    observation=obs,
                    full_observation=full_obs,
                    illegal_mask=illegal_mask,
                    action_logp=action_logp,
                    action=action,
                    reward=reward,
                    value=value,
                )
            )

        gae_start = perf_counter()
        gae_rewards = torch.stack([f.reward for f in frames])
        gae_values = torch.stack([f.value for f in frames])
        emprets, advantages = _gae(gae_rewards, gae_values, gae_gamma, gae_lambda)
        gae_time_cost += perf_counter() - gae_start

        for f, empret, adv in zip(frames, emprets, advantages):
            f.empret = empret
            f.advantage = adv
        collection.extend(frames)

    LOG_COLLECT.info(
        f"Collection size: {len(collection)} frames; avg_entropy: {total_entropy / len(collection):.4f}"
    )
    LOG_COLLECT.info(
        f"Collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection


def _parallel_collect_frames(
    collection_size: int,
    envs: List[PPOEnv],
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    assert isinstance(agent.policy, MLPPolicy)
    assert isinstance(agent.value_fn, MLPValueFn)

    collection: List[Frame] = []

    frames_ls: List[List[Frame]] = [[] for _ in envs]
    is_terminal_ls = [True] * len(envs)
    total_entropy = 0.0

    gae_time_cost = 0.0
    start = perf_counter()

    while len(collection) < collection_size:
        for is_terminal, env in zip(is_terminal_ls, envs):
            if is_terminal:
                env.reset()

        cur_players = [env.cur_player for env in envs]

        obs = torch.stack(
            [
                torch.tensor(env.observation(p), dtype=torch.float)
                for p, env in zip(cur_players, envs)
            ]
        )
        illegal_mask = torch.stack([torch.tensor(env.illegal_mask) for env in envs])

        # action selection
        actions, action_logps, entropy = agent.policy(obs, illegal_mask)
        total_entropy += entropy.sum().item()

        # value estimation
        full_obs = torch.stack([obs.new_tensor(env.full_observation()) for env in envs])
        values = agent.value_fn(full_obs)

        # env step
        rewards, is_terminal_ls = tuple(
            zip(*[env.step(a.item()) for a, env in zip(actions, envs)])
        )
        rewards = obs.new_tensor(rewards)

        for i, frames in enumerate(frames_ls):
            frames.append(
                Frame(
                    observation=obs[i],
                    full_observation=full_obs[i],
                    illegal_mask=illegal_mask[i],
                    action_logp=action_logps[i],
                    action=actions[i],
                    reward=rewards[i],
                    value=values[i],
                )
            )

        # __import__("ipdb").set_trace()

        for is_terminal, frames in zip(is_terminal_ls, frames_ls):
            if is_terminal:
                gae_start = perf_counter()
                gae_rewards = torch.stack([f.reward for f in frames])
                gae_values = torch.stack([f.value for f in frames])
                emprets, advantages = _gae(
                    gae_rewards, gae_values, gae_gamma, gae_lambda
                )
                gae_time_cost += perf_counter() - gae_start

                for f, empret, adv in zip(frames, emprets, advantages):
                    f.empret = empret
                    f.advantage = adv
                collection.extend(frames)

                frames.clear()

    LOG_COLLECT.info(
        f"Collection size: {len(collection)} frames; avg_entropy: {total_entropy / len(collection):.4f}"
    )
    LOG_COLLECT.info(
        f"Parallel collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection


def _collect_trajectories(
    collection_size: int,
    env: PPOEnv,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    assert isinstance(agent.policy, RNNPolicy)
    assert isinstance(agent.value_fn, MLPValueFn)  # for now

    collection: List[Trajectory] = []
    total_steps = 0
    total_entropy = 0.0

    gae_time_cost = 0.0
    start = perf_counter()

    while total_steps < collection_size:
        frames: List[Frame] = []

        pl_states = agent.policy.new_states(1, extra_dims=(env.players,))

        env.reset()
        is_terminal = False

        while not is_terminal:
            obs = torch.tensor(env.observation(env.cur_player), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, action_logp, entropy, h_t = agent.policy(
                obs.view(1, 1, -1), pl_states[env.cur_player], illegal_mask.view(1, -1)
            )
            total_entropy += entropy.item()

            pl_states[env.cur_player] = h_t

            # value estimation
            full_obs = obs.new_tensor(env.full_observation())
            value = agent.value_fn(full_obs)

            # env update
            reward, is_terminal = env.step(action.item())
            reward = obs.new_tensor(reward)

            frames.append(
                Frame(
                    observation=obs,
                    full_observation=full_obs,
                    illegal_mask=illegal_mask,
                    action_logp=action_logp,
                    action=action,
                    reward=reward,
                    value=value,
                )
            )

        gae_start = perf_counter()
        gae_rewards = torch.stack([f.reward for f in frames])
        gae_values = torch.stack([f.value for f in frames])
        emprets, advantages = _gae(gae_rewards, gae_values, gae_gamma, gae_lambda)
        gae_time_cost += perf_counter() - gae_start

        # player may have 0 move before the game ends
        for p in range(min(env.players, len(frames))):
            subset = frames[p :: env.players]
            collection.append(
                Trajectory(
                    observations=torch.stack([f.observation for f in subset]),
                    full_observations=torch.stack([f.full_observation for f in subset]),
                    illegal_masks=torch.stack([f.illegal_mask for f in subset]),
                    action_logps=torch.stack([f.action_logp for f in subset]),
                    actions=torch.stack([f.action for f in subset]),
                    emprets=emprets[p :: env.players],
                    advantages=advantages[p :: env.players],
                )
            )
            total_steps += len(subset)

    LOG_COLLECT.info(
        f"Collection size: {total_steps} frames in {len(collection)} trajectories; avg_entropy: {total_entropy / total_steps:.4f}"
    )
    LOG_COLLECT.info(
        f"Collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection


def _parallel_collect_trajectories(
    collection_size: int,
    envs: List[PPOEnv],
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    assert isinstance(agent.policy, RNNPolicy)
    assert isinstance(agent.value_fn, MLPValueFn)  # for now

    num_envs = len(envs)
    num_players = envs[0].players
    collection: List[Trajectory] = []

    frames_ls: List[List[Frame]] = [[] for _ in envs]
    is_terminal_ls = [True] * len(envs)
    total_steps = 0
    total_entropy = 0.0

    pl_states = agent.policy.new_states(1, extra_dims=(num_envs, num_players))
    if isinstance(agent.value_fn, RNNValueFn):
        vf_states = agent.value_fn.new_states(1, extra_dims=(num_envs, num_players))
    else:
        vf_states = None

    gae_time_cost = 0.0
    start = perf_counter()

    while total_steps < collection_size:
        for is_terminal, (i, env) in zip(is_terminal_ls, enumerate(envs)):
            if is_terminal:
                env.reset()
                pl_states[i] = agent.policy.new_states(1, extra_dims=(num_players,))
                if vf_states:
                    vf_states[i] = agent.value_fn.new_states(
                        1, extra_dims=(num_players,)
                    )

        cur_players = [env.cur_player for env in envs]

        obs = torch.stack(
            [
                torch.tensor(env.observation(p), dtype=torch.float)
                for p, env in zip(cur_players, envs)
            ]
        )
        h_0 = torch.cat([st[p] for p, st in zip(cur_players, pl_states)], dim=1)
        illegal_mask = torch.stack([torch.tensor(env.illegal_mask) for env in envs])

        # action selection
        actions, action_logps, entropy, h_t = agent.policy(
            obs.unsqueeze(0), h_0, illegal_mask
        )
        total_entropy += entropy.sum().item()

        for i, p in enumerate(cur_players):
            pl_states[i, p] = h_t[:, i].unsqueeze(1)

        # value estimation
        full_obs = torch.stack([obs.new_tensor(env.full_observation()) for env in envs])
        if vf_states:
            raise NotImplementedError
        else:
            values = agent.value_fn(full_obs)

        # env step
        rewards, is_terminal_ls = tuple(
            zip(*[env.step(a.item()) for a, env in zip(actions, envs)])
        )
        rewards = obs.new_tensor(rewards)

        for i, frames in enumerate(frames_ls):
            frames.append(
                Frame(
                    observation=obs[i],
                    full_observation=full_obs[i],
                    illegal_mask=illegal_mask[i],
                    action_logp=action_logps[i],
                    action=actions[i],
                    reward=rewards[i],
                    value=values[i],
                )
            )

        # __import__("ipdb").set_trace()

        for is_terminal, frames, env in zip(is_terminal_ls, frames_ls, envs):
            if is_terminal:
                gae_start = perf_counter()
                gae_rewards = torch.stack([f.reward for f in frames])
                gae_values = torch.stack([f.value for f in frames])
                emprets, advantages = _gae(
                    gae_rewards, gae_values, gae_gamma, gae_lambda
                )
                gae_time_cost += perf_counter() - gae_start

                # player may have 0 move before the game ends
                for p in range(min(env.players, len(frames))):
                    subset = frames[p :: env.players]
                    collection.append(
                        Trajectory(
                            observations=torch.stack([f.observation for f in subset]),
                            full_observations=torch.stack(
                                [f.full_observation for f in subset]
                            ),
                            illegal_masks=torch.stack([f.illegal_mask for f in subset]),
                            action_logps=torch.stack([f.action_logp for f in subset]),
                            actions=torch.stack([f.action for f in subset]),
                            emprets=emprets[p :: env.players],
                            advantages=advantages[p :: env.players],
                        )
                    )
                    total_steps += len(subset)

                frames.clear()

    LOG_COLLECT.info(
        f"Collection size: {total_steps} frames in {len(collection)} trajectories; avg_entropy: {total_entropy / total_steps:.4f}"
    )
    LOG_COLLECT.info(
        f"Parallel collection done in {perf_counter() - start:.2f} s, in which GAE cost {gae_time_cost:.2f} s"
    )

    return collection


@torch.no_grad()
def collect(
    collection_type: str,
    collection_size: int,
    parallel: int,
    env_or_envs: Union[PPOEnv, List[PPOEnv]],
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
):
    if collection_type == "frame":
        if parallel > 1:
            return _parallel_collect_frames(
                collection_size, env_or_envs, agent, gae_gamma, gae_lambda
            )
        else:
            return _collect_frames(
                collection_size, env_or_envs, agent, gae_gamma, gae_lambda
            )

    elif collection_type == "traj":
        if parallel > 1:
            return _parallel_collect_trajectories(
                collection_size, env_or_envs, agent, gae_gamma, gae_lambda
            )
        else:
            return _collect_trajectories(
                collection_size, env_or_envs, agent, gae_gamma, gae_lambda
            )

    else:
        raise ValueError('`collection_type` must be "frame" or "traj"')
