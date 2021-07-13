import numpy as np
import torch

from ppo.agent import PPOAgent, RNNPolicy
from ppo.env import PPOEnv
from ppo.log import log_eval
from ppo.utils import action_histogram


def evaluate(collection_type, env, agent, episodes=100):
    if collection_type == "frame":
        _evaluate_frames(env, agent, episodes)

    elif collection_type == "traj":
        _evaluate_trajectories(env, agent, episodes)

    else:
        raise ValueError


def _evaluate_frames(env: PPOEnv, agent: PPOAgent, episodes=100):
    assert not isinstance(agent.policy, RNNPolicy), "Cannot use RNNPolicy with frames"

    all_lengths = []
    all_rewards = []
    all_actions = []

    for _ in range(episodes):
        env.reset()
        is_terminal = False
        episode_length = 0
        episode_reward = 0.0

        while not is_terminal:
            p = env.cur_player
            obs = torch.tensor(env.observation(p), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, action_logp = agent.policy(obs, illegal_mask)

            reward, is_terminal = env.step(action.item())

            episode_length += 1
            episode_reward += reward
            all_actions.append(env.get_move(action).type().name)

        all_lengths.append(episode_length)
        all_rewards.append(episode_reward)

    avg_length = np.mean(all_lengths)
    avg_reward = np.mean(all_rewards)
    action_hist = "\n" + action_histogram(all_actions)

    log_eval.info(
        "Evaluation done: "
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"action_histogram={action_hist}"
    )


def _evaluate_trajectories(env: PPOEnv, agent: PPOAgent, episodes=100):
    assert isinstance(agent.policy, RNNPolicy)

    players = env.players

    all_lengths = []
    all_rewards = []
    all_actions = []

    for _ in range(episodes):
        episode_length = 0
        episode_reward = 0.0

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

                if p == env.cur_player:
                    action, _, h_t = agent.policy(
                        obs.view(1, 1, -1), illegal_mask.view(1, -1), h_0=memories[p]
                    )
                else:
                    h_t = agent.policy(
                        obs.view(1, 1, -1), illegal_mask=None, h_0=memories[p]
                    )

                memories[p] = h_t

            # env update
            reward, is_terminal = env.step(action.item())

            episode_length += 1
            episode_reward += reward
            all_actions.append(env.get_move(action).type().name)

        all_lengths.append(episode_length)
        all_rewards.append(episode_reward)

    avg_length = np.mean(all_lengths)
    avg_reward = np.mean(all_rewards)
    action_hist = "\n" + action_histogram(all_actions)

    log_eval.info(
        "Evaluation done: "
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"action_histogram={action_hist}"
    )
