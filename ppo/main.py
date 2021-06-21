from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ppo.data import FrameBatch
from ppo.env import Frame, FrameType, PPOEnvironment
from ppo.module import MLPPolicy
from ppo.utils import action_histogram


@torch.no_grad()
def collect(
    env: PPOEnvironment,
    policy_fn: nn.Module,
    value_fn: nn.Module,
    gae_gamma: float,
    gae_lambda: float,
    collection_size: int,
    device: torch.device,
):
    """Collection phase in PG iteration

    It does:
        1. Run the env for `collection-size` frames, reset if necessary
        2. Compute advantages over each episode
        3. Return a list of frames
    """

    # exponentials of (gamma * lambda), ie [1.0, (gl)^1, (gl)^2, ...]
    glexp = torch.tensor(gae_gamma * gae_lambda, device=device)
    glexp = glexp.pow(torch.arange(100))  # max length of hanabi is about 88

    gammaexp = torch.tensor(gae_gamma, device=device).pow(torch.arange(100))

    def _gae(frames: list[Frame]):
        """Nested function for Generalised Advantage Estimation"""
        episode_length = len(frame)

        rewards = torch.stack([f.reward for f in frames])
        values = torch.stack([f.value for f in frames])
        values_t1 = torch.stack([f.value_t1 for f in frames])

        deltas = rewards + gae_gamma * values_t1 - values

        empret = torch.zeros_like(rewards)  # empirical returns
        advantages = torch.zeros_like(rewards)
        for t in range(episode_length):
            empret[t] = torch.sum(gammaexp[: episode_length - t] * rewards[t:])
            advantages[t] = torch.sum(glexp[: episode_length - t] * deltas[t:])

        for f, r, adv in zip(frames, empret, advantages):
            f.empret = r
            f.advantage = adv

        return frames

    data: list[Frame] = []

    while len(data) < collection_size:
        env.reset()
        is_terminal = False
        episode_frames = []

        while not is_terminal:
            frame_type = env.frame_type
            obs = torch.tensor(env.observation, dtype=torch.float, device=device)

            # policy_fn(observation) -> action
            logp = policy_fn(obs)
            prob = torch.exp(logp)

            illegal_mask = torch.ones_like(prob, dtype=torch.bool)
            illegal_mask[env.legal_moves] = False
            prob[illegal_mask] = 0.0

            action = torch.multinomial(prob, 1)

            # env update
            obs_t1, reward, is_terminal = env.step(action.item())
            obs_t1 = obs.new_tensor(obs_t1)
            reward = obs.new_tensor(reward)

            # value_fn(observation) -> value_estimate
            value = value_fn(obs)
            value_t1 = value_fn(obs_t1) if not is_terminal else value.new_tensor(0.0)

            frame = Frame(
                frame_type=frame_type,
                observation=obs,
                observation_t1=obs_t1,
                action_logp=logp,
                action=action,
                value=value,
                value_t1=value_t1,
                reward=reward,
                # advantage defaults to None and will be computed in GAE
            )

            episode_frames.append(frame)

        episode_frames = _gae(episode_frames)
        data.extend(episode_frames)

    return data[:collection_size]


def train(
    data: Iterator[FrameBatch],
    policy_fn,
    policy_fn_optimizer,
    value_fn,
    value_fn_optimizer,
    ppo_epsilon,
    epochs,
):
    for i_epoch in range(epochs):
        for batch in data:
            policy_fn_optimizer.zero_grad()

            # update policy
            logps = policy_fn(batch.observations)
            logps = torch.gather(logps, dim=1, index=batch.actions)

            ratio = torch.exp(logps - batch.action_logps)
            loss_1 = ratio * batch.advantages
            loss_2 = torch.clip(loss_1, 1 - ppo_epsilon, 1 + ppo_epsilon)
            loss = torch.mean(torch.minimum(loss_1, loss_2))

            loss.backward()
            policy_fn_optimizer.step()

            # update value function
            value_fn_optimizer.zero_grad()

            values = value_fn(batch.observations)
            loss_vf = F.smooth_l1_loss(values, batch.empret)

            loss_vf.backward()
            value_fn_optimizer.step()


@torch.no_grad()
def evaluate(env, policy_fn, episodes=100, device="cpu"):
    all_lengths = []
    all_rewards = []
    all_actions = []
    all_entropy = []

    for _ in range(episodes):
        env.reset()

        episode_length = 0
        episode_reward = 0.0

        while env.frame_type is not FrameType.END:
            obs = torch.tensor(env.observation, dtype=torch.float, device=device)
            legal_moves = env.legal_moves

            logp = policy_fn(obs)
            prob = torch.exp(logp)

            illegal_mask = torch.ones_like(prob, dtype=torch.bool)
            illegal_mask[legal_moves] = False
            prob[illegal_mask] = 0.0

            action = torch.multinomial(prob, 1).item()

            reward = env.step(action)

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

    print(
        "Evaluate:",
        f"avg_length={avg_length:.2f},",
        f"avg_reward={avg_reward:.2f},",
        f"avg_entropy={avg_entropy:.4f},",
        f"action_histogram={action_hist}",
    )


def main(
    env_players=2,
    # collection config
    iterations=1000,
    collection_size=1000,
    gae_gamma=0.99,
    gae_lambda=0.99,
    # neural network config
    hidden_size=512,
    num_layers=2,
    # training config
    batch_size=1000,
    dataloader_workers=0,
    epochs=1,
    learning_rate=1e-4,
    ppo_epsilon=0.2,
    # eval_config
    eval_every=1,  # every n iterations
    eval_episodes=100,
    # misc
    seed=-1,
    device="cpu",
):
    env = PPOEnvironment(env_players, seed)

    agent = MLPPolicy(env.obs_size, env.num_actions, hidden_size, num_layers)
    for p in agent.layers[-1].parameters():
        torch.nn.init.zeros_(p)

    optimizer = torch.optim.Adam(
        agent.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=1e-6
    )

    for i_iter in range(1, iterations + 1):
        data = collect(env, agent, collection_size, device)

        data = DataLoader(
            data,
            batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
            collate_fn=FrameBatch,
            pin_memory=True,
            drop_last=True,
        )

        train(data, agent, optimizer, ppo_epsilon, epochs)

        schedular.step()

        print(f"Iteration {i_iter}")

        if i_iter % eval_every == 0:
            evaluate(env, agent, eval_episodes, device)


if __name__ == "__main__":
    main()
