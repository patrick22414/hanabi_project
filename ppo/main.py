import argparse
import json
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ppo.agent import MLPPolicy, MLPValueFn, PPOAgent, RNNPolicy
from ppo.collect import collect
from ppo.data import FrameBatch, TrajectoryBatch
from ppo.env import PPOEnv
from ppo.log import log_eval, log_main
from ppo.train import train
from ppo.utils import action_histogram


@torch.no_grad()
def evaluate(env, agent, episodes=100, device="cpu"):
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
            obs = torch.tensor(env.observation, dtype=torch.float, device=device)
            legal_moves = env.legal_moves

            logits = agent.policy_fn(obs)

            ent_logps = F.log_softmax(logits, dim=-1)
            all_entropy.append(-torch.sum(ent_logps.exp() * ent_logps).item())

            logits = logits[legal_moves]
            logp = F.log_softmax(logits, dim=-1)
            prob = torch.exp(logp)

            action = torch.multinomial(prob, 1).item()
            action = legal_moves[action]

            _, reward, is_terminal = env.step(action)  # discarding obs_t1

            episode_length += 1
            episode_reward += reward
            all_actions.append(env.get_move(action).type().name)

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
    seed: int,
    iterations: int,
    env_config: dict,
    agent_config: dict,
    collect_config: dict,
    train_config: dict,
    eval_config: dict,
):
    env = PPOEnv(**env_config, seed=seed)

    if agent_config["policy"]["type"] == "RNN":
        input_size = env.players * env.enc_size
        output_size = env.num_actions
        policy_cls = RNNPolicy
    elif agent_config["policy"]["type"] == "MLP":
        input_size = env.enc_size
        output_size = env.num_actions
        policy_cls = MLPPolicy
    else:
        raise TypeError

    if agent_config["value_fn"]["type"] == "MLP":
        value_fn_cls = MLPValueFn
    else:
        raise TypeError

    agent = PPOAgent(
        policy_cls(
            input_size,
            output_size,
            agent_config["policy"]["hidden_size"],
            agent_config["policy"]["num_layers"],
        ),
        value_fn_cls(
            input_size,
            agent_config["value_fn"]["hidden_size"],
            agent_config["value_fn"]["num_layers"],
        ),
    )

    policy_optimizer = torch.optim.Adam(
        agent.policy.parameters(), **train_config["policy_optimizer"]
    )
    value_fn_optimizer = torch.optim.Adam(
        agent.value_fn.parameters(), **train_config["value_fn_optimizer"]
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(policy_optimizer)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_fn_optimizer)

    collection_type = collect_config["collection_type"]

    for i in range(iterations):
        log_main.info(f"====== Iteration {i}/{iterations} ======")

        collection = collect(env=env, agent=agent, **collect_config)

        dataloader = DataLoader(
            collection,
            train_config["batch_size"],
            shuffle=True,
            collate_fn=TrajectoryBatch if collection_type == "traj" else FrameBatch,
        )

        train(
            dataloader,
            agent,
            policy_optimizer,
            value_fn_optimizer,
            ppo_clip=train_config["ppo_clip"],
            entropy_coeff=train_config["entropy_coeff"],
            epochs=train_config["epochs"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as fi:
        config = json.load(fi)

    if "seed" not in config or config["seed"] < 0:
        config["seed"] = random.randint(0, 999)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    log_main.info("JSON config:\n" + json.dumps(config, indent=4))

    # choose device
    # if torch.cuda.is_available() and args.device > 0:
    #     device = torch.device(f"cuda:{args.device}")
    # else:
    #     device = torch.device("cpu")

    # if torch.cuda.is_available() and args.actor_device > 0:
    #     actor_device = torch.device(f"cuda:{args.device}")
    # else:
    #     actor_device = torch.device("cpu")

    # log_main.info(f"learn_agent device: {device}, actor_agent device: {actor_device}")

    main(**config)
