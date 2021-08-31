import argparse
import json
import logging
import os
import random
from typing import List
import numpy as np

import torch

from ppo.agent import PPOAgent, RNNPolicy
from ppo.env import PPOEnv
from ppo.utils import action_histogram, score_histogram, set_logfile

LOG_EVAL = logging.getLogger("ppo.eval")


@torch.no_grad()
def evaluate(
    env: PPOEnv,
    agents: List[PPOAgent],
    episodes: int,
    deterministic=False,
    random_first_player=True,
    save_record_file=False,
):
    assert len(agents) == 1 or len(agents) == env.players
    if len(agents) == 1:
        agents = agents * env.players

    all_records = {}
    all_actions = []
    perfect_games = 0
    total_entropy = 0.0

    for i in range(episodes):
        if random_first_player:
            random.shuffle(agents)

        states = [
            a.policy.new_states(1) if isinstance(a.policy, RNNPolicy) else None
            for a in agents
        ]

        episode_record = {"reward": 0.0, "length": 0, "history": []}
        env.reset()
        is_terminal = False

        while not is_terminal:
            p = env.cur_player
            obs = torch.tensor(env.observation(p), dtype=torch.float)
            illegal_mask = torch.tensor(env.illegal_mask)

            # action selection
            if states[p] is not None:
                action, _, entropy, h_t = agents[p].policy(
                    obs.view(1, 1, -1), states[p], illegal_mask.view(1, -1)
                )
                states[p] = h_t
            else:
                action, _, entropy = agents[p].policy(obs, illegal_mask)

            action = action.item()
            entropy = entropy.item()
            reward, is_terminal = env.step(action)

            move = env.get_move(action)
            episode_record["length"] += 1
            episode_record["reward"] += reward
            episode_record["history"].append(f"{int(reward):+} {move}")
            total_entropy += entropy
            all_actions.append(move.type().name)

        all_records[f"episode {i + 1}"] = episode_record
        # total_length += episode_record["length"]
        # total_reward += episode_record["reward"]
        if episode_record["reward"] == env.max_score:
            perfect_games += 1

    all_lengths = np.array([rec["length"] for rec in all_records.values()])
    all_rewards = np.array([rec["reward"] for rec in all_records.values()])

    avg_length = np.mean(all_lengths)
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    perfect_games = perfect_games / episodes
    avg_entropy = total_entropy / np.sum(all_lengths)

    score_hist = score_histogram(all_rewards.astype(int))
    action_hist = action_histogram(all_actions)

    LOG_EVAL.info(f"Evaluation done for {episodes} episodes")
    LOG_EVAL.info(
        "Performace: "
        f"avg_length={avg_length:.2f}, "
        f"avg_reward={avg_reward:.2f}, "
        f"std_reward={std_reward:.2f}, "
        # f"perfect={perfect_games:.2%}, "
        f"avg_entropy={avg_entropy:.2f}"
    )
    LOG_EVAL.info(f"Score histogram:\n{score_hist}")
    LOG_EVAL.info(f"Action histogram:\n{action_hist}")

    if save_record_file:
        filename = f"records/{os.path.splitext(os.path.basename(LOG_EVAL.handlers[-1].baseFilename))[0]}.json"
        with open(filename, "w") as fo:
            json.dump(all_records, fo, indent=4)
        LOG_EVAL.info(f"Records saved to {filename}")


@torch.no_grad()
def main_selfplay(env_config: dict, agent_files: List[str], episodes: int):
    env = PPOEnv(**env_config)

    agents = [PPOAgent(torch.load(f, map_location="cpu")) for f in agent_files]
    for a, f in zip(agents, agent_files):
        LOG_EVAL.info(f"=== Self-play evaluation of {os.path.basename(f)} ===")

        a.eval()
        evaluate(env, [a], episodes)

    # os.makedirs("records", exist_ok=True)


@torch.no_grad()
def main_adhoc(env_config: dict, agents: List[str], episodes: int):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("game", type=str, choices=["selfplay", "adhoc"])
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-a", "--agents", type=str, nargs="+", required=True)
    parser.add_argument("-l", "--logfile", type=str)
    parser.add_argument("-n", "--episodes", type=int, default=1000)

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as fi:
        config = json.load(fi)

    env_config = config["env_config"]
    prefix = f"{env_config['preset']}{env_config['players']}p"

    if args.logfile:
        if args.logfile == "0":
            logfile = set_logfile(prefix, "none")
        else:
            logfile = set_logfile(prefix, args.logfile)
    else:
        logfile = set_logfile(prefix)

    if args.game == "selfplay":
        main_selfplay(env_config, args.agents, args.episodes)
    else:  # args.game == "adhoc"
        main_adhoc(env_config, args.agents, args.episodes)
