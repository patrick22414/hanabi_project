import argparse
import json
import random
from collections import deque
from timeit import default_timer

import numpy as np
import torch
from torch import nn

from .env import HanabiEnvironment, Transition


class EpisodicStats(object):
    def __init__(self, maxlen=100) -> None:
        super().__init__()

        self.steps = deque(maxlen=maxlen)  # item: episode length
        self.reward = deque(maxlen=maxlen)  # item: episode cumulative reward
        self.avg_q_max = deque(maxlen=maxlen)  # item: avg q_max per episode
        self.actions = []

    def push(self, steps, reward, avg_q_max, actions):
        self.steps.append(steps)
        self.reward.append(reward)
        self.avg_q_max.append(avg_q_max)
        self.actions.extend(actions)

    def __str__(self) -> str:
        steps = np.mean(self.steps)
        reward = np.mean(self.reward)
        avg_q_max = np.mean(self.avg_q_max)

        def _tabulate(values, counts):
            line_1, line_2 = "\t", "\t"

            for v, c in zip(map(str, values), map(str, counts)):
                size = max(len(v), len(c)) + 2
                line_1 += v.ljust(size, " ")
                line_2 += c.ljust(size, " ")

            return line_1 + "\n" + line_2

        action_hist = _tabulate(*np.unique(self.actions, return_counts=True))

        return (
            f"Average over {len(self.steps)} episodes: "
            f"steps={steps:.2f}, "
            f"reward={reward:.2f}, "
            f"avg_q_max={avg_q_max:.2f}, "
            f"action_histogram=\n{action_hist}"
        )


class DuelingDQN(nn.Module):
    def __init__(self, obs_size, num_actions, hidden_size, num_layers):
        super().__init__()

        self.features = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(obs_size if i == 0 else hidden_size, hidden_size),
                    nn.ReLU(),
                )
                for i in range(num_layers)
            ]
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions, bias=False),
        )

    def forward(self, x, illegal_mask=None):
        x = self.features(x)

        v = self.value(x)
        a = self.advantage(x)

        q = v + a - a.mean()

        if illegal_mask is not None:
            q[illegal_mask] = float("-inf")

        return q


class ReplayMemory(object):
    def __init__(self, capacity, device):
        super().__init__()

        self.capacity = capacity
        self.device = device
        self.position = 0
        self.memory = []
        self.priorities = []

    def push(self, transition: Transition):
        transition = Transition(
            transition.obs_t0.to(device=self.device),
            transition.action.to(device=self.device),
            transition.reward.to(device=self.device),
            transition.obs_t1.to(device=self.device),
            transition.illegal_mask_t1.to(device=self.device),
            torch.tensor(transition.is_terminal, device=self.device).view(1),
        )

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        return random.sample(self.memory, batch_size)

    def stats(self):
        return f"(size={len(self)})"

    def __len__(self):
        return len(self.memory)


@torch.no_grad()
def evaluate(env, policy_net, episodes=100, max_episode_steps=300):
    is_training = policy_net.training
    policy_net.eval()

    stats = EpisodicStats(maxlen=episodes)

    for _ in range(episodes):
        env.reset()

        steps = 0
        total_reward = 0.0
        total_q_max = 0.0
        actions = []

        is_terminal = False

        for _ in range(max_episode_steps):
            player = env.state.cur_player()

            q = policy_net(env.observations[player], env.illegal_mask)
            q_max, action = q.max(dim=0)

            reward, is_terminal = env.quick_step(action)

            steps += 1
            total_reward += reward
            total_q_max += q_max.item()
            actions.append(env.game.get_move(action.item()).type().name)

            if is_terminal:
                break

        stats.push(steps, total_reward, total_q_max / steps, actions)

    policy_net.train(is_training)

    return stats


def main(
    # env config
    env_preset="small",
    env_players=2,
    reward_scheme=None,
    # replay config
    replay_capacity=10_000,
    replay_init_size=1_000,
    # neural network config
    batch_size=16,
    hidden_size=512,
    num_layers=2,
    learning_rate=1e-4,
    # q-learning config
    max_steps=10_000_000,
    max_episode_steps=100,
    policy_sync_every=1_000,
    discount=0.99,
    eval_every=10_000,
    eval_episodes=1000,
    # other config
    seed=-1,
    device="cpu",
):

    env = HanabiEnvironment(env_preset, env_players, device=device)

    replay_memory = ReplayMemory(capacity=replay_capacity, device=device)

    policy_net = DuelingDQN(env.obs_shape, env.max_moves, hidden_size, num_layers)
    policy_net = policy_net.to(device)

    target_net = DuelingDQN(env.obs_shape, env.max_moves, hidden_size, num_layers)
    target_net = target_net.to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.requires_grad_(False)
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=learning_rate / 10
    )
    criterion = torch.nn.SmoothL1Loss()

    print(policy_net)

    epsilons = torch.cat(
        (
            torch.linspace(1.0, 0.1, max_steps // 10),
            torch.full((max_steps // 10 * 9,), 0.1),
        )
    )

    # fill some memory
    env_reset = True
    while len(replay_memory) < replay_init_size:
        if env_reset:
            env.reset()

        action = random.choice(
            [env.game.get_move_uid(m) for m in env.state.legal_moves()]
        )

        transition = env.step(action)
        replay_memory.push(transition)

        env_reset = transition.is_terminal

    # start training
    env_reset = True
    avg_loss = 0.0
    time_start = default_timer()
    for i_step in range(max_steps):
        if env_reset:
            env.reset()
            episode_steps = 0

        player = env.state.cur_player()

        if random.random() > epsilons[i_step]:
            # take policy action
            with torch.no_grad():
                action = policy_net(env.observations[player], env.illegal_mask).argmax()
        else:
            # take random action
            action = random.choice(
                [env.game.get_move_uid(m) for m in env.state.legal_moves()]
            )

        transition = env.step(action)
        replay_memory.push(transition)

        episode_steps += 1
        env_reset = transition.is_terminal or episode_steps >= max_episode_steps

        # learning
        optimizer.zero_grad()

        batch = replay_memory.sample(batch_size)
        batch_obs_t0 = torch.stack([t.obs_t0 for t in batch])
        batch_action = torch.stack([t.action for t in batch])
        batch_reward = torch.stack([t.reward for t in batch])
        batch_obs_t1 = torch.stack([t.obs_t1 for t in batch])
        batch_illegal_mask_t1 = torch.stack([t.illegal_mask_t1 for t in batch])
        batch_is_terminal = torch.stack([t.is_terminal for t in batch])

        # Double DQN
        q_policy = policy_net(batch_obs_t0)
        q_policy = torch.gather(q_policy, dim=1, index=batch_action)

        # target = R + discount * Q'(obs_t1, argmax_a(Q(obs_t1)))
        with torch.no_grad():
            action_t1 = policy_net(batch_obs_t1, batch_illegal_mask_t1)
            action_t1 = action_t1.argmax(dim=1).view(-1, 1)

            q_target = target_net(batch_obs_t1)
            q_target = torch.gather(q_target, dim=1, index=action_t1)

            q_target.masked_fill_(batch_is_terminal, 0.0)
            q_target = batch_reward + discount * q_target

        loss = criterion(q_policy, q_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        avg_loss = avg_loss * 0.9 + loss.item() * 0.1

        if (i_step + 1) % (eval_every // 10) == 0:
            print(f"Step {(i_step + 1)}: avg_loss={avg_loss:.6f}")

        if (i_step + 1) % policy_sync_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Policy synchronized")

        if (i_step + 1) % eval_every == 0:
            print(evaluate(env, policy_net, eval_episodes, max_episode_steps))

        if (i_step + 1) % (max_steps // 100) == 0:
            percentage = (i_step + 1) * 100 // max_steps
            torch.save(
                policy_net.state_dict(), f"checkpoints/hanabi_dqn-{percentage}%.pt"
            )

            print(
                f"Checkpoint {percentage}%: "
                f"epsilon={epsilons[i_step]:.2f}, "
                f"learning_rate={scheduler.get_last_lr()[0]:.2e}, "
                f"replay={replay_memory.stats()}, "
                f"total_time={(default_timer() - time_start) / 60:.2f}min"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-d", "--device", type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    with open(args.config, "r") as fi:
        config = json.load(fi)

    config["device"] = device

    main(**config)
