import random
from collections import deque
from timeit import default_timer

import numpy as np
import torch
from torch import nn

from env_dqn import ActionIsIllegal, HanabiEnvironment, Transition


class EpisodicStats(object):
    def __init__(self, maxlen=100) -> None:
        super().__init__()

        self.steps = deque(maxlen=maxlen)
        self.score = deque(maxlen=maxlen)
        self.fireworks = deque(maxlen=maxlen)
        self.q_max = deque(maxlen=maxlen)

    def push(self, steps=None, score=None, fireworks=None, q_max=None):
        if steps is not None:
            self.steps.append(steps)
        if score is not None:
            self.score.append(score)
        if fireworks is not None:
            self.fireworks.append(fireworks)
        if q_max is not None:
            self.q_max.append(q_max)

    def __str__(self) -> str:
        steps = np.mean(self.steps) if len(self.steps) != 0 else -1
        score = np.mean(self.score) if len(self.score) != 0 else -1
        fireworks = np.mean(self.fireworks) if len(self.fireworks) != 0 else -1
        q_max = np.mean(self.q_max) if len(self.q_max) != 0 else -1

        return (
            f"Average over {len(self.steps)} episodes: "
            f"steps={steps:.2f}, "
            f"score={score:.2f}, "
            f"fireworks={fireworks:.2f}, "
            f"q_max={q_max:.2f}"
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
            transition.obs_t0.to(device=self.device, copy=True),
            transition.action.to(device=self.device, copy=True),
            transition.reward.to(device=self.device, copy=True),
            transition.obs_t1.to(device=self.device, copy=True),
            transition.illegal_mask_t1.to(device=self.device, copy=True),
            transition.is_terminal,
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
    moves = []

    for _ in range(episodes):
        env.reset()

        steps = 0
        q_max_list = []

        is_terminal = False

        for _ in range(max_episode_steps):
            player = env.state.cur_player()

            q = policy_net(env.observations[player], env.illegal_mask)
            q_max, action = q.max(dim=0)

            q_max_list.append(q_max)

            moves.append(int(action.item()))

            is_terminal = env.quick_step(action)
            steps += 1

            if is_terminal:
                break

        stats.push(
            steps,
            env.score,
            env.fireworks,
            np.mean(q_max_list),
        )

    print(
        "Action histogram:",
        [(m, c) for m, c in zip(*np.unique(moves, return_counts=True))],
    )

    policy_net.train(is_training)

    return stats


def main(
    # env config
    env_preset="full",
    env_players=2,
    # replay config
    replay_capacity=10_000,
    replay_init_size=1_000,
    # neural network config
    batch_size=16,
    hidden_size=512,
    num_layers=2,
    learning_rate=1e-4,
    # q-learning config
    max_steps=100_000,
    max_episode_steps=100,
    policy_sync_every=5_000,
    discount=0.99,
    eval_every=5_000,
    eval_episodes=100,
    # other config
    seed=-1,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    env = HanabiEnvironment(env_preset, env_players)

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

        action = random.randint(0, env.max_moves - 1)
        try:
            transition = env.step(action)
        except ActionIsIllegal:
            continue

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
        batch_action = torch.stack([t.action for t in batch]).view(-1, 1)
        batch_reward = torch.stack([t.reward for t in batch]).view(-1, 1)
        batch_obs_t1 = torch.stack([t.obs_t1 for t in batch])
        batch_illegal_mask_t1 = torch.stack([t.illegal_mask_t1 for t in batch])
        batch_is_terminal = (
            torch.tensor([t.is_terminal for t in batch]).to(device).view(-1, 1)
        )

        # Double DQN
        q_policy = policy_net(batch_obs_t0)
        q_policy = torch.gather(q_policy, dim=1, index=batch_action)

        # target = R + discount * Q'(obs_t1, argmax_a(Q(obs_t1)))
        with torch.no_grad():
            action_t1 = (
                policy_net(batch_obs_t1, batch_illegal_mask_t1)
                .argmax(dim=1)
                .view(-1, 1)
            )

            q_target = target_net(batch_obs_t1)
            q_target = torch.gather(q_target, dim=1, index=action_t1)

            q_target.masked_fill_(batch_is_terminal, 0.0)
            q_target = batch_reward + discount * q_target

        loss = criterion(q_policy, q_target)
        loss.backward()

        optimizer.step()
        scheduler.step()

        avg_loss = avg_loss * 0.9 + loss.item() * 0.1

        if (i_step + 1) % (max_steps // 100) == 0:
            print(f"Step {(i_step + 1)}: avg_loss={avg_loss:.6f}")

        if (i_step + 1) % policy_sync_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Policy synchronized")

        if (i_step + 1) % eval_every == 0:
            print(evaluate(env, policy_net, eval_episodes, max_episode_steps))

        if (i_step + 1) % (max_steps // 10) == 0:
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
    main()
