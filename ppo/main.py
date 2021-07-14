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
from ppo.eval import evaluate
from ppo.log import log_eval, log_main
from ppo.train import train
from ppo.utils import action_histogram


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

    collection_type = collect_config["collection_type"]

    if agent_config["policy"]["type"] == "RNN":
        policy_cls = RNNPolicy
    elif agent_config["policy"]["type"] == "MLP":
        policy_cls = MLPPolicy
    else:
        raise ValueError

    if agent_config["value_fn"]["type"] == "MLP":
        value_fn_cls = MLPValueFn
    else:
        raise ValueError

    agent = PPOAgent(
        policy_cls(
            env.obs_size,
            env.num_actions,
            agent_config["policy"]["hidden_size"],
            agent_config["policy"]["num_layers"],
        ),
        value_fn_cls(
            env.obs_size,
            agent_config["value_fn"]["hidden_size"],
            agent_config["value_fn"]["num_layers"],
        ),
    )

    log_main.info(f"Agent:\n{agent}")

    policy_optimizer = torch.optim.Adam(
        agent.policy.parameters(), **train_config["policy_optimizer"]
    )
    value_fn_optimizer = torch.optim.Adam(
        agent.value_fn.parameters(), **train_config["value_fn_optimizer"]
    )

    policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer,
        T_max=iterations,
        eta_min=train_config["policy_optimizer"]["lr"] / 10,
    )
    value_fn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        value_fn_optimizer,
        T_max=iterations,
        eta_min=train_config["value_fn_optimizer"]["lr"] / 10,
    )

    for i in range(1, iterations + 1):
        log_main.info(f"====== Iteration {i}/{iterations} ======")

        agent.eval()
        collection = collect(env=env, agent=agent, **collect_config)

        dataloader = DataLoader(
            collection,
            train_config["batch_size"],
            shuffle=True,
            collate_fn=TrajectoryBatch if collection_type == "traj" else FrameBatch,
        )

        agent.train()
        train(
            dataloader,
            agent,
            policy_optimizer,
            value_fn_optimizer,
            ppo_clip=train_config["ppo_clip"],
            entropy_coeff=train_config["entropy_coeff"],
            epochs=train_config["epochs"],
        )

        policy_scheduler.step()
        value_fn_scheduler.step()

        if i % eval_config["eval_every"] == 0:
            agent.eval()
            evaluate(collection_type, env, agent, eval_config["episodes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-l", "--log-file", type=str)

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
