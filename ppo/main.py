import argparse
import json
import logging
import os
import random

import torch
from torch.utils.data import DataLoader

from ppo.agent import MLPPolicy, MLPValueFn, PPOAgent, RNNPolicy, RNNValueFn
from ppo.collect import collect
from ppo.data import FrameBatch, TrajectoryBatch
from ppo.env import PPOEnv
from ppo.eval import evaluate
from ppo.train import train
from ppo.utils import checkpoint, linear_decay, set_logfile

LOG_MAIN = logging.getLogger("ppo.main")


def main(
    seed: int,
    iterations: int,
    env_config: dict,
    agent_config: dict,
    collect_config: dict,
    train_config: dict,
    eval_config: dict,
    checkpoints: int,
):
    env = PPOEnv(**env_config, seed=seed)

    collection_type = collect_config["collection_type"]
    parallel = collect_config["parallel"]

    if parallel > 1:
        envs = [PPOEnv(**env_config, seed=seed + i + 1) for i in range(parallel)]

    if agent_config["policy"]["type"] == "RNN":
        policy_cls = RNNPolicy
    elif agent_config["policy"]["type"] == "MLP":
        policy_cls = MLPPolicy
    else:
        raise ValueError

    if agent_config["value_fn"]["type"] == "RNN":
        value_fn_cls = RNNValueFn
    elif agent_config["value_fn"]["type"] == "MLP":
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
            env.full_obs_size,
            agent_config["value_fn"]["hidden_size"],
            agent_config["value_fn"]["num_layers"],
        ),
    )

    LOG_MAIN.info(f"Observation size: {env.obs_size}. Num actions: {env.num_actions}")

    optimizer = torch.optim.Adam(agent.parameters(), **train_config["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, **train_config["scheduler"]
    )

    ppo_clip = linear_decay(*train_config["ppo_clip"], iterations)
    entropy_coef = linear_decay(*train_config["entropy_coef"], iterations)

    LOG_MAIN.info("Initial evaluation")
    agent.eval()
    evaluate(env, [agent], eval_config["episodes"])

    for i in range(1, iterations + 1):
        LOG_MAIN.info(f"====== Iteration {i}/{iterations} ======")

        agent.eval()
        if parallel > 1:
            collection = collect(env_or_envs=envs, agent=agent, **collect_config)
        else:
            collection = collect(env_or_envs=env, agent=agent, **collect_config)

        collate_fn = TrajectoryBatch if collection_type == "traj" else FrameBatch
        dataloader = DataLoader(
            collection,
            train_config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )

        agent.train()
        train(
            dataloader,
            agent,
            optimizer,
            epochs=train_config["epochs"],
            ppo_clip=ppo_clip[i - 1],
            entropy_coef=entropy_coef[i - 1],
            value_fn_coef=train_config["value_fn_coef"],
        )

        if i % eval_config["eval_every"] == 0:
            agent.eval()
            evaluate(env, [agent], eval_config["episodes"])

        if i % (iterations // checkpoints) == 0:
            checkpoint(
                i,
                iterations,
                checkpoints,
                agent.state_dict(),
                scheduler.get_last_lr()[0],
                ppo_clip[i - 1],
                entropy_coef[i - 1],
            )

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-l", "--logfile", type=str)
    parser.add_argument("-k", "--checkpoints", type=int, default=10)

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as fi:
        config = json.load(fi)

    prefix = os.path.splitext(os.path.basename(args.config))[0]

    if args.logfile:
        if args.logfile == "0":
            logfile = set_logfile(prefix, "none")
            LOG_MAIN.info("Not using logfile")
        else:
            logfile = set_logfile(prefix, args.logfile)
            LOG_MAIN.info(f"Using logfile {logfile}")
    else:
        logfile = set_logfile(prefix)
        LOG_MAIN.info(f"Using default logfile {logfile}")

    if "seed" not in config or config["seed"] < 0:
        config["seed"] = random.randint(0, 999)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    LOG_MAIN.info("JSON config:\n" + json.dumps(config, indent=4))

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

    main(**config, checkpoints=args.checkpoints)
