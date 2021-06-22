import argparse
import json
import random
from time import perf_counter
from typing import Iterator, List

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ppo.agent import MLPPolicy, MLPValueFunction, PPOAgent
from ppo.data import Frame, FrameBatch, FrameType
from ppo.env import PPOEnvironment
from ppo.log import log_collect, log_eval, log_main, log_train
from ppo.utils import action_histogram


@torch.no_grad()
def collect(
    env: PPOEnvironment,
    agent: PPOAgent,
    gae_gamma: float,
    gae_lambda: float,
    collection_size: int,
):
    """Collection phase in PG iteration

    It does:
        1. Run the env for `collection-size` frames, reset if necessary
        2. Compute advantages over each episode
        3. Return a list of frames
    """

    # maximum length of a hanabi game is about 88
    exponents = torch.arange(100)
    # exponentials of (gamma * lambda), ie [1.0, (gl)^1, (gl)^2, ...]
    glexp = torch.tensor(gae_gamma * gae_lambda).pow(exponents)
    # exponentials of gamma
    gammaexp = torch.tensor(gae_gamma).pow(exponents)

    def _gae(frames: List[Frame]):
        """Nested function for Generalised Advantage Estimation"""
        episode_length = len(frames)

        rewards = torch.stack([f.reward for f in frames])
        values = torch.stack([f.value for f in frames])
        values_t1 = torch.stack([f.value_t1 for f in frames])

        deltas = rewards + gae_gamma * values_t1 - values

        # empirical return is a discounted sum of all future returns
        # advantage is a discounted sum of all future deltas
        empret = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        for t in range(episode_length):
            empret[t] = torch.sum(gammaexp[: episode_length - t] * rewards[t:])
            advantages[t] = torch.sum(glexp[: episode_length - t] * deltas[t:])

        for f, r, adv in zip(frames, empret, advantages):
            f.empret = r
            f.advantage = adv

        return frames

    data: List[Frame] = []

    while len(data) < collection_size:
        env.reset()
        episode_frames = []
        frame_type = FrameType.START
        is_terminal = False

        while not is_terminal:
            obs = torch.tensor(env.observation, dtype=torch.float)

            # action selection
            logp = agent.policy_fn(obs)
            prob = torch.exp(logp)

            illegal_mask = torch.ones_like(prob, dtype=torch.bool)
            illegal_mask[env.legal_moves] = False
            prob[illegal_mask] = 0.0

            action = torch.multinomial(prob, 1)

            # env update
            obs_t1, reward, is_terminal = env.step(action.item())
            obs_t1 = obs.new_tensor(obs_t1)
            reward = obs.new_tensor(reward)

            # value estimation
            value = agent.value_fn(obs)
            value_t1 = (
                agent.value_fn(obs_t1) if not is_terminal else value.new_tensor(0.0)
            )

            if is_terminal:
                frame_type = FrameType.END

            frame = Frame(
                frame_type=frame_type,
                observation=obs,
                action_logp=logp[action],
                action=action,
                value=value,
                value_t1=value_t1,
                reward=reward,
                # empret and advantage default to None and will be computed in GAE
            )

            if frame_type is FrameType.START:
                frame_type = FrameType.MID

            episode_frames.append(frame)

        # compute advantages and empirical returns in GAE
        episode_frames = _gae(episode_frames)

        data.extend(episode_frames)

    return data[:collection_size]


def mp_collect(env_config, agent, gae_gamma, gae_lambda, collection_size):
    env = PPOEnvironment(**env_config)
    return collect(env, agent, gae_gamma, gae_lambda, collection_size)


def train(
    data: Iterator[FrameBatch],
    agent: PPOAgent,
    policy_fn_optimizer: torch.optim.Optimizer,
    value_fn_optimizer: torch.optim.Optimizer,
    epsilon: float,
    epochs: int,
    device: torch.device,
):
    for i_epoch in range(epochs):
        for batch in data:
            batch = batch.to(device)

            # update policy
            policy_fn_optimizer.zero_grad()
            logps = agent.policy_fn(batch.observations)
            logps = torch.gather(logps, dim=1, index=batch.actions)

            # policy surrogate objective
            ratio = torch.exp(logps - batch.action_logps).squeeze()
            surr_1 = ratio * batch.advantages
            surr_2 = torch.clip(surr_1, 1 - epsilon, 1 + epsilon)
            surr_ppo = -torch.mean(torch.minimum(surr_1, surr_2))

            surr_ppo.backward()
            policy_fn_optimizer.step()

            # update value function
            value_fn_optimizer.zero_grad()
            values = agent.value_fn(batch.observations)
            loss_vf = F.smooth_l1_loss(values, batch.emprets)

            loss_vf.backward()
            value_fn_optimizer.step()


@torch.no_grad()
def evaluate(env, agent, episodes=100):
    all_lengths = []
    all_rewards = []
    all_actions = []
    all_entropy = []

    for _ in range(episodes):
        env.reset()

        is_terminal = False
        episode_length = 0
        episode_reward = 0.0

        while not is_terminal:
            obs = torch.tensor(env.observation, dtype=torch.float)

            logp = agent.policy_fn(obs)
            prob = torch.exp(logp)

            illegal_mask = torch.ones_like(prob, dtype=torch.bool)
            illegal_mask[env.legal_moves] = False
            prob[illegal_mask] = 0.0

            action = torch.multinomial(prob, 1).item()

            _, reward, is_terminal = env.step(action)  # discarding obs_t1

            episode_length += 1
            episode_reward += reward
            all_actions.append(env.get_move(action).type().name)
            all_entropy.append(-torch.sum(prob * logp).item())

        all_lengths.append(episode_length)
        all_rewards.append(episode_reward)

    avg_length = np.mean(all_lengths)
    avg_reward = np.mean(all_rewards)
    avg_entropy = np.mean(all_entropy)
    action_hist = "\n" + action_histogram(all_actions)

    log_eval.info(
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"avg_entropy={avg_entropy:.4f}, "
        f"action_histogram={action_hist}"
    )


def main(
    env_players: int,
    # collection config
    iterations: int,
    collection_size: int,
    gae_gamma: float,
    gae_lambda: float,
    # neural network config
    hidden_size: int,
    num_layers: int,
    # training config
    batch_size: int,
    dataloader_workers: int,
    epochs: int,
    learning_rate: float,
    ppo_epsilon: float,
    # eval_config
    eval_every: int,  # every n iterations
    eval_episodes: int,
    # misc
    seed: int,
    device: torch.device,
):
    env = PPOEnvironment(env_players, seed)

    learn_agent = PPOAgent(
        MLPPolicy(env.obs_size, env.num_actions, hidden_size, num_layers),
        MLPValueFunction(env.obs_size, hidden_size, num_layers),
    )
    actor_agent = PPOAgent(
        MLPPolicy(env.obs_size, env.num_actions, hidden_size, num_layers),
        MLPValueFunction(env.obs_size, hidden_size, num_layers),
    )
    actor_agent.load_state_dict(learn_agent.state_dict())

    learn_agent.to(device)

    policy_fn_optimizer = torch.optim.Adam(
        learn_agent.policy_fn.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    value_fn_optimizer = torch.optim.Adam(
        learn_agent.value_fn.parameters(), lr=learning_rate, weight_decay=1e-6
    )

    for i_iter in range(1, iterations + 1):
        log_main.info(f"====== Iteration {i_iter}/{iterations} ======")

        start = perf_counter()
        data = collect(env, actor_agent, gae_gamma, gae_lambda, collection_size)

        data = DataLoader(
            data,
            batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
            collate_fn=FrameBatch,
        )

        log_collect.info(f"collected in {perf_counter() - start:.2f} s")

        start = perf_counter()
        train(
            data,
            learn_agent,
            policy_fn_optimizer,
            value_fn_optimizer,
            ppo_epsilon,
            epochs,
            device,
        )

        log_train.info(f"trained in {perf_counter() - start:.2f} s")

        actor_agent.load_state_dict(learn_agent.state_dict())
        if i_iter % eval_every == 0:
            evaluate(env, actor_agent, eval_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-d", "--device", type=int, default=0)

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as fi:
        config = json.load(fi)

    if "collect_workers" in config:
        log_main.warning(
            "You are not running main_mp. `collect_workers` is multiplied onto `collection_size` instead"
        )
        config["collection_size"] *= config["collect_workers"]
        del config["collect_workers"]

    if "seed" not in config or config["seed"] < 0:
        config["seed"] = random.randint(0, 999)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    print(">>>", config)

    # choose device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    main(**config, device=device)
