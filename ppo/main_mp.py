import argparse
import json
import multiprocessing as mp
import random
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from ppo.agent import MLPPolicy, MLPValueFunction, PPOAgent
from ppo.data import FrameBatch
from ppo.env import PPOEnvironment
from ppo.log import log_collect, log_main, log_train
from ppo.main import collect, evaluate, train


def mp_collect(env_config, agent, gae_gamma, gae_lambda, collection_size):
    env = PPOEnvironment(**env_config)
    return collect(env, agent, gae_gamma, gae_lambda, collection_size)


def main_mp(
    env_players: int,
    # collection config
    iterations: int,
    collection_size: int,
    collect_workers: int,
    gae_gamma: float,
    gae_lambda: float,
    # neural network config
    hidden_size: int,
    num_layers: int,
    # training config
    batch_size: int,
    dataloader_workers: int,
    epochs: int,
    learning_rate: float,
    ppo_epsilon: float,
    # eval_config
    eval_every: int,  # every n iterations
    eval_episodes: int,
    # misc
    seed: int,
    device: torch.device,
):
    # this env is only used for initialising agents and evaluation
    env = PPOEnvironment(env_players, seed)
    seed += 1

    learn_agent = PPOAgent(
        MLPPolicy(env.obs_size, env.num_actions, hidden_size, num_layers),
        MLPValueFunction(env.obs_size, hidden_size, num_layers),
    )
    actor_agent = PPOAgent(
        MLPPolicy(env.obs_size, env.num_actions, hidden_size, num_layers),
        MLPValueFunction(env.obs_size, hidden_size, num_layers),
    )
    actor_agent.load_state_dict(learn_agent.state_dict())

    learn_agent.to(device)

    policy_fn_optimizer = torch.optim.Adam(
        learn_agent.policy_fn.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    value_fn_optimizer = torch.optim.Adam(
        learn_agent.value_fn.parameters(), lr=learning_rate, weight_decay=1e-6
    )

    with mp.Pool(collect_workers) as pool:
        for i_iter in range(1, iterations + 1):
            log_main.info(f"====== Iteration {i_iter}/{iterations} ======")

            start = perf_counter()
            data = pool.starmap(
                mp_collect,
                [
                    (
                        {"players": env_players, "seed": seed + i},
                        actor_agent,
                        gae_gamma,
                        gae_lambda,
                        collection_size,
                    )
                    for i in range(collect_workers)
                ],
            )
            data = sum(data, [])
            # data = collect(env, actor_agent, gae_gamma, gae_lambda, collection_size)

            data = DataLoader(
                data,
                batch_size,
                shuffle=True,
                num_workers=dataloader_workers,
                collate_fn=FrameBatch,
            )

            log_collect.info(f"collected in {perf_counter() - start:.2f} s")

            start = perf_counter()
            train(
                data,
                learn_agent,
                policy_fn_optimizer,
                value_fn_optimizer,
                ppo_epsilon,
                epochs,
                device,
            )

            log_train.info(f"trained in {perf_counter() - start:.2f} s")

            actor_agent.load_state_dict(learn_agent.state_dict())
            if i_iter % eval_every == 0:
                evaluate(env, actor_agent, eval_episodes)

            seed += collect_workers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-d", "--device", type=int, default=0)

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as fi:
        config = json.load(fi)

    if "seed" not in config or config["seed"] < 0:
        config["seed"] = random.randint(0, 999)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    print(">>>", config)

    # choose device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    main_mp(**config, device=device)
