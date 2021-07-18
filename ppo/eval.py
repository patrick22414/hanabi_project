import argparse
import itertools
import json
import random
from typing import List

import torch

from ppo.agent import PPOAgent, RNNPolicy
from ppo.env import PPOEnv
from ppo.log import log_eval
from ppo.utils import action_histogram


@torch.no_grad()
def evaluate(collection_type, env, agent, episodes=100):
    if collection_type == "frame":
        _evaluate_frames(env, agent, episodes)

    elif collection_type == "traj":
        _evaluate_trajectories(env, agent, episodes)

    else:
        raise ValueError


def _evaluate_frames(env: PPOEnv, agent: PPOAgent, episodes=100):
    assert not isinstance(agent.policy, RNNPolicy), "Cannot use RNNPolicy with frames"

    total_steps = 0
    total_reward = 0.0
    total_entropy = 0.0
    all_actions = []

    for _ in range(episodes):
        env.reset()
        is_terminal = False

        while not is_terminal:
            p = env.cur_player
            obs = torch.tensor(env.observation(p), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, action_logp, entropy = [
                x.item() for x in agent.policy(obs, illegal_mask)
            ]

            reward, is_terminal = env.step(action)

            total_steps += 1
            total_reward += reward
            total_entropy += entropy
            all_actions.append(env.get_move(action).type().name)

    avg_length = total_steps / episodes
    avg_reward = total_reward / episodes
    avg_entropy = total_entropy / total_steps
    action_hist = "\n" + action_histogram(all_actions)

    log_eval.info(
        "Evaluation done: "
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"avg_ent={avg_entropy:.2f}, "
        f"action_histogram={action_hist}"
    )


def _evaluate_trajectories(env: PPOEnv, agent: PPOAgent, episodes=100):
    assert isinstance(agent.policy, RNNPolicy)

    players = env.players

    total_steps = 0
    total_reward = 0.0
    total_entropy = 0.0
    all_actions = []

    for _ in range(episodes):
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
                    action, action_logp, entropy, h_t = agent.policy(
                        obs.view(1, 1, -1), illegal_mask.view(1, -1), h_0=memories[p]
                    )
                else:
                    h_t = agent.policy(
                        obs.view(1, 1, -1), illegal_mask=None, h_0=memories[p]
                    )

                memories[p] = h_t

            # env update
            reward, is_terminal = env.step(action.item())

            total_steps += 1
            total_reward += reward
            total_entropy += entropy.item()
            all_actions.append(env.get_move(action.item()).type().name)

    avg_length = total_steps / episodes
    avg_reward = total_reward / episodes
    avg_entropy = total_entropy / total_steps
    action_hist = "\n" + action_histogram(all_actions)

    log_eval.info(
        "Evaluation done: "
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"avg_ent={avg_entropy:.2f}, "
        f"action_histogram={action_hist}"
    )


@torch.no_grad()
def main(
    env_config: dict,
    agents: List[PPOAgent],
    episodes: int,
    random_first_player=True,
):
    env = PPOEnv(**env_config)
    assert len(agents) == 1 or len(agents) == env.players

    total_steps = 0
    total_reward = 0.0
    all_actions = []

    for _ in range(episodes):
        env.reset()
        is_terminal = False

        if random_first_player:
            random.shuffle(agents)

        for agent in itertools.cycle(agents):
            obs = torch.tensor(env.observation(env.cur_player), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, _, _ = [x.item() for x in agent.policy(obs, illegal_mask)]

            reward, is_terminal = env.step(action)

            total_steps += 1
            total_reward += reward
            all_actions.append(env.get_move(action).type().name)
            if is_terminal:
                break

    avg_length = total_steps / episodes
    avg_reward = total_reward / episodes
    action_hist = "\n" + action_histogram(all_actions)

    log_eval.info(f"Evaluation done for {episodes} episodes")
    log_eval.info(
        "Performace: "
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"action_histogram={action_hist}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-a", "--agents", type=str, nargs="+", required=True)
    parser.add_argument("-l", "--logfile", type=str)
    parser.add_argument("-n", "--episodes", type=int, default=1000)

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as fi:
        config = json.load(fi)

    env_config = config["env_config"]

    state_dicts = [torch.load(f, map_location="cpu") for f in args.agents]
    agents = [PPOAgent(state_dict) for state_dict in state_dicts]
    for agent in agents:
        agent.eval()

    main(env_config, agents, args.episodes)
