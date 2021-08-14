import argparse
import itertools
import json
import logging
import os
import random
from datetime import datetime
from typing import List

import torch

from ppo.agent import MLPPolicy, PPOAgent, RNNPolicy
from ppo.env import PPOEnv
from ppo.utils import action_histogram, set_logfile

LOG_EVAL = logging.getLogger("ppo.eval")


def _evaluate_trajectories(env: PPOEnv, agent: PPOAgent, episodes=100):
    assert isinstance(agent.policy, RNNPolicy)

    total_steps = 0
    total_reward = 0.0
    total_entropy = 0.0
    all_actions = []

    for _ in range(episodes):
        # RNN hidden states, one for each player
        states = agent.policy.new_states(1, extra_dims=(env.players,))

        env.reset()
        is_terminal = False

        while not is_terminal:
            obs = torch.tensor(env.observation(env.cur_player), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, action_logp, entropy, h_t = agent.policy(
                obs.view(1, 1, -1), states[env.cur_player], illegal_mask.view(1, -1)
            )

            states[env.cur_player] = h_t

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

    LOG_EVAL.info(
        "Evaluation done: "
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"avg_ent={avg_entropy:.2f}, "
        f"action_histogram={action_hist}"
    )


def _evaluate_frames(
    env: PPOEnv,
    agents: List[PPOAgent],
    episodes: int,
    random_first_player=True,
    save_record_file=False,
):
    assert len(agents) == 1 or len(agents) == env.players
    assert all(isinstance(a.policy, MLPPolicy) for a in agents)

    records = {}
    total_length = 0
    total_reward = 0.0
    perfect_games = 0
    total_entropy = 0.0
    all_actions = []

    for i in range(episodes):
        env.reset()
        is_terminal = False

        episode_record = {"reward": 0.0, "length": 0, "history": []}

        if random_first_player:
            random.shuffle(agents)

        for agent in itertools.cycle(agents):
            obs = torch.tensor(env.observation(env.cur_player), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            action, _, entropy = [x.item() for x in agent.policy(obs, illegal_mask)]

            reward, is_terminal = env.step(action)

            move = env.get_move(action)
            episode_record["length"] += 1
            episode_record["reward"] += reward
            episode_record["history"].append(f"{int(reward):+} {move}")
            total_entropy += entropy
            all_actions.append(move.type().name)

            if is_terminal:
                break

        records[f"episode {i + 1}"] = episode_record
        total_length += episode_record["length"]
        total_reward += episode_record["reward"]
        if episode_record["reward"] == env.max_score:
            perfect_games += 1

    avg_length = total_length / episodes
    avg_reward = total_reward / episodes
    perfect_games = perfect_games / episodes
    avg_entropy = total_entropy / total_length
    action_hist = action_histogram(all_actions)

    LOG_EVAL.info(f"Evaluation done for {episodes} episodes")
    LOG_EVAL.info(
        "Performace: "
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"perfect={perfect_games:.2%}, "
        f"avg_entropy={avg_entropy:.2f}"
    )
    LOG_EVAL.info(f"Action histogram:\n{action_hist}")

    if save_record_file:
        filename = f"records/{datetime.utcnow().strftime('%Y-%m-%dT%H%M%SZ')}.json"
        with open(filename, "w") as fo:
            json.dump(records, fo, indent=4)
        LOG_EVAL.info(f"Records saved to {filename}")


@torch.no_grad()
def evaluate(collection_type, env, agent, episodes=100):
    if collection_type == "frame":
        _evaluate_frames(env, [agent], episodes)

    elif collection_type == "traj":
        _evaluate_trajectories(env, agent, episodes)

    else:
        raise ValueError


@torch.no_grad()
def main(env_config: dict, agents: List[str], episodes: int):
    env = PPOEnv(**env_config)

    state_dicts = [torch.load(f, map_location="cpu") for f in agents]
    agents = [PPOAgent(state_dict) for state_dict in state_dicts]
    for agent in agents:
        agent.eval()

    os.makedirs("records", exist_ok=True)

    _evaluate_frames(env, agents, episodes, save_record_file=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-a", "--agents", type=str, nargs="+", required=True)
    parser.add_argument("-l", "--logfile", type=str)
    parser.add_argument("-n", "--episodes", type=int, default=1000)

    args = parser.parse_args()

    if args.logfile:
        if args.logfile == "0":
            set_logfile("none")

    # parse config file
    with open(args.config, "r") as fi:
        config = json.load(fi)

    env_config = config["env_config"]

    main(env_config, args.agents, args.episodes)
