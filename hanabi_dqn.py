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
        self.q_avg = deque(maxlen=maxlen)
        self.steps_ill = deque(maxlen=maxlen)

    def push(
        self,
        steps=None,
        score=None,
        fireworks=None,
        q_max=None,
        q_avg=None,
        steps_ill=None,
    ):
        if steps is not None:
            self.steps.append(steps)
        if score is not None:
            self.score.append(score)
        if fireworks is not None:
            self.fireworks.append(fireworks)
        if q_max is not None:
            self.q_max.append(q_max)
        if q_avg is not None:
            self.q_avg.append(q_avg)
        if steps_ill is not None:
            self.steps_ill.append(steps_ill)

    def __str__(self) -> str:
        steps = np.mean(self.steps) if len(self.steps) != 0 else -1
        score = np.mean(self.score) if len(self.score) != 0 else -1
        fireworks = np.mean(self.fireworks) if len(self.fireworks) != 0 else -1
        q_max = np.mean(self.q_max) if len(self.q_max) != 0 else -1
        q_avg = np.mean(self.q_avg) if len(self.q_avg) != 0 else -1
        steps_ill = np.mean(self.steps_ill) if len(self.steps_ill) != 0 else -1

        return (
            f"Average over {len(self.steps)} episodes: "
            f"steps={steps:.2f}, "
            f"score={score:.2f}, "
            f"fireworks={fireworks:.2f}, "
            f"q_avg|max={q_avg:.2f}|{q_max:.2f}, "
            f"steps_illegal={steps_ill:.2f}"
        )


class DuelingDQNAgent(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()

        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                    nn.ReLU(),
                )
            )

        self.layers.append(nn.Linear(hidden_size, output_size, bias=False))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


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
            transition.is_terminal,
        )

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=1):
        return random.sample(self.memory, batch_size)

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
        steps_illegal = 0
        q_max_list = []
        q_avg_list = []

        is_terminal = False

        for _ in range(max_episode_steps):
            player = env.state.cur_player()
            q = policy_net(env.observations[player])
            q_avg = q.mean()
            q_max, action = q.max(dim=0)

            q_max_list.append(q_max)
            q_avg_list.append(q_avg)

            try:
                is_terminal = env.quick_step(action)
            except ActionIsIllegal:
                steps_illegal += 1
            else:
                steps += 1

            if is_terminal:
                break

        stats.push(
            steps,
            env.score,
            env.fireworks,
            np.mean(q_max_list),
            np.mean(q_avg_list),
            steps_illegal,
        )

    policy_net.train(is_training)

    # print(">>>", stats.steps_ill)

    return stats


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    env = HanabiEnvironment("small")

    replay_memory = ReplayMemory(capacity=100_000, device=device)

    batch_size = 16
    max_steps = 1_000_000
    max_episode_steps = 100
    policy_sync_every = 10_000

    hidden_size = 256
    num_layers = 4

    policy_net = DuelingDQNAgent(
        env.obs_shape, env.max_moves, hidden_size, num_layers
    ).to(device)
    target_net = DuelingDQNAgent(
        env.obs_shape, env.max_moves, hidden_size, num_layers
    ).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.requires_grad_(False)
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    criterion = torch.nn.SmoothL1Loss()

    # print(policy_net)

    discount = 0.95
    epsilons = torch.cat(
        (torch.linspace(1.0, 1e-2, max_steps // 2), torch.full((max_steps // 2,), 1e-2))
    )

    # fill some memory
    env_reset = True
    bootstrap_memory = 1000
    while len(replay_memory) < bootstrap_memory:
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
                action = policy_net(env.observations[player]).argmax()
        else:
            # take random action
            action = random.randint(0, env.max_moves - 1)

        episode_steps += 1

        try:
            transition = env.step(action)
        except ActionIsIllegal:
            env_reset = episode_steps >= max_episode_steps
        else:
            replay_memory.push(transition)
            env_reset = transition.is_terminal or episode_steps >= max_episode_steps

        # learning
        optimizer.zero_grad()

        batch = replay_memory.sample(batch_size)
        batch_obs_t0 = torch.stack([t.obs_t0 for t in batch])
        batch_action = torch.stack([t.action for t in batch]).view(-1, 1)
        batch_reward = torch.stack([t.reward for t in batch]).view(-1, 1)
        batch_obs_t1 = torch.stack([t.obs_t1 for t in batch])
        batch_is_terminal = torch.tensor(
            [t.is_terminal for t in batch], device=device
        ).view(-1, 1)

        # Double DQN
        q_policy = policy_net(batch_obs_t0)
        q_policy = torch.gather(q_policy, dim=1, index=batch_action)

        # target = R + discount * Q'(obs_t1, argmax_a(Q(obs_t1)))
        with torch.no_grad():
            action_t1 = policy_net(batch_obs_t1).argmax(dim=1).view(-1, 1)

            q_target = target_net(batch_obs_t1)
            q_target = torch.gather(q_target, dim=1, index=action_t1)

            q_target.masked_fill_(batch_is_terminal, 0.0)
            q_target = batch_reward + discount * q_target

        loss = criterion(q_policy, q_target)
        loss.backward()

        optimizer.step()

        avg_loss = avg_loss * 0.9 + loss.item() * 0.1

        if (i_step + 1) % (max_steps // 1000) == 0:
            print(f"Step {(i_step + 1)}: avg_loss={avg_loss:.6f}")

        if (i_step + 1) % policy_sync_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Policy synchronized")

        if (i_step + 1) % (max_steps // 100) == 0:
            percentage = (i_step + 1) * 100 // max_steps
            torch.save(
                policy_net.state_dict(), f"checkpoints/hanabi_dqn-{percentage}%.pt"
            )

            print(
                f"Checkpoint {percentage}% "
                f"epsilon={epsilons[i_step]:.2f}, "
                f"memory_size={len(replay_memory)}, "
                f"total_time={(default_timer() - time_start) / 60:.2f}min"
            )

            print(evaluate(env, policy_net, 100, max_episode_steps))


if __name__ == "__main__":
    main()
