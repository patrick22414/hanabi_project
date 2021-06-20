from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import action_histogram

from .data import FrameBatch
from .env import Frame, FrameType, PPOEnvironment
from .module import MLPAgent


@torch.no_grad()
def collect(env: PPOEnvironment, agent, collection_size, device):
    data: list[Frame] = []
    episodes = 0
    total_reward = 0.0

    while len(data) < collection_size:
        env.reset()
        episode_frame_types = []
        episode_observations = []
        episode_action_logps = []
        episode_actions = []
        episode_rewards = []

        frame_type = env.frame_type

        while frame_type is not FrameType.END:
            frame_type = env.frame_type
            obs = torch.tensor(env.observation, dtype=torch.float, device=device)
            legal_moves = env.legal_moves

            logp = agent(obs)
            prob = torch.exp(logp)

            illegal_mask = torch.ones_like(prob, dtype=torch.bool)
            illegal_mask[legal_moves] = False
            prob[illegal_mask] = 0.0

            action = torch.multinomial(prob, 1)

            reward = env.step(action.item())

            episode_frame_types.append(frame_type)
            episode_observations.append(obs)
            episode_action_logps.append(logp)
            episode_actions.append(action)
            episode_rewards.append(reward)

        # simple advantage = sum(all future rewards)
        episode_rewards = torch.tensor(episode_rewards, device=device).view(-1, 1)
        advantages = episode_rewards.sum().expand_as(episode_rewards)
        # advantages = episode_rewards.flipud().cumsum().flipud().view(-1, 1)

        frames = [
            Frame(*args)
            for args in zip(
                episode_frame_types,
                episode_observations,
                episode_action_logps,
                episode_actions,
                episode_rewards,
                advantages,
            )
        ]

        data.extend(frames)
        episodes += 1
        total_reward += episode_rewards.sum()

    baseline = total_reward / episodes
    print(">>> baseline:", baseline)
    for f in data:
        f.advantage.sub_(baseline)

    return data


def train(data: Iterator[FrameBatch], agent, optimizer, epsilon, epochs):
    for i_epoch in range(epochs):
        for batch in data:
            optimizer.zero_grad()

            logps = agent(batch.observations)
            logps = torch.gather(logps, dim=1, index=batch.actions)

            ratio = torch.exp(logps - batch.action_logps)
            loss_1 = ratio * batch.advantages
            loss_2 = torch.clip(loss_1, 1 - epsilon, 1 + epsilon)
            loss = torch.mean(torch.minimum(loss_1, loss_2))

            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(env, agent, episodes=100, device="cpu"):
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

            logp = agent(obs)
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
    # neural network config
    hidden_size=512,
    num_layers=2,
    # training config
    batch_size=1000,
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

    agent = MLPAgent(env.obs_size, env.num_actions, hidden_size, num_layers)
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
