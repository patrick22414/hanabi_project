import argparse
import json
import random
from time import perf_counter
from typing import Iterator, List

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ppo.agent import MLPPolicy, MLPValueFunction, PPOAgent
from ppo.data import Frame, FrameBatch, FrameType
from ppo.env import PPOEnvironment
from ppo.log import log_collect, log_eval, log_main, log_train
from ppo.utils import action_histogram


@torch.no_grad()
def collect(env, agent, gae_gamma, gae_lambda, collection_size, device):
    """Collection phase in PG iteration

    It does:
        1. Run the env for `collection-size` frames, reset if necessary
        2. Compute advantages over each episode
        3. Return a list of frames
    """

    # maximum length of a hanabi game is about 88
    exponents = torch.arange(100, device=device)
    # exponentials of (gamma * lambda), ie [1.0, (gl)^1, (gl)^2, ...]
    glexp = torch.tensor(gae_gamma * gae_lambda, device=device).pow(exponents)
    # exponentials of gamma
    gammaexp = torch.tensor(gae_gamma, device=device).pow(exponents)

    def _gae(frames: List[Frame]):
        """Nested function for Generalised Advantage Estimation"""
        episode_length = len(frames)

        rewards = torch.stack([f.reward for f in frames])
        values = torch.stack([f.value for f in frames])
        values_t1 = torch.stack([f.value_t1 for f in frames])

        deltas = rewards + gae_gamma * values_t1 - values

        # empirical return is a discounted sum of all future returns
        # advantage is a discounted sum of all future deltas
        empret = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        for t in range(episode_length):
            empret[t] = torch.sum(gammaexp[: episode_length - t] * rewards[t:])
            advantages[t] = torch.sum(glexp[: episode_length - t] * deltas[t:])

        for f, r, adv in zip(frames, empret, advantages):
            f.empret = r
            f.advantage = adv

        return frames

    data: List[Frame] = []

    while len(data) < collection_size:
        env.reset()

        is_terminal = False
        episode_frames = []
        frame_type = FrameType.START

        while not is_terminal:
            obs = torch.tensor(env.observation, dtype=torch.float, device=device)
            legal_moves = env.legal_moves

            # action selection
            logits = agent.policy_fn(obs)
            logits = logits[legal_moves]
            logp = F.log_softmax(logits, dim=-1)

            action = torch.multinomial(torch.exp(logp), 1).item()
            action_logp = logp[action]
            action = legal_moves[action]

            # env update
            obs_t1, reward, is_terminal = env.step(action)
            obs_t1 = obs.new_tensor(obs_t1)
            reward = obs.new_tensor(reward)

            # value estimation
            value = agent.value_fn(obs)
            value_t1 = (
                agent.value_fn(obs_t1) if not is_terminal else value.new_tensor(0.0)
            )

            if is_terminal:
                frame_type = FrameType.END

            frame = Frame(
                frame_type=frame_type,
                observation=obs,
                action_logp=action_logp.view(1),
                action=torch.tensor(action).view(1),
                value=value,
                value_t1=value_t1,
                reward=reward,
                legal_moves=legal_moves,
                # empret and advantage default to None and will be computed in GAE
            )

            if frame_type is FrameType.START:
                frame_type = FrameType.MID

            episode_frames.append(frame)

        # compute advantages and empirical returns in GAE
        episode_frames = _gae(episode_frames)

        data.extend(episode_frames)

    return data[:collection_size]


def train(
    data,
    agent,
    policy_fn_optimizer,
    value_fn_optimizer,
    epsilon,
    entropy_coeff,
    epochs,
    device,
):
    for i_epoch in range(1, epochs + 1):
        losses_ppo = []
        losses_ent = []
        losses_vf = []

        for batch in data:
            batch = batch.to(device)

            # update policy
            policy_fn_optimizer.zero_grad()
            logits = agent.policy_fn(batch.observations)

            # compute entropy
            # to encourage entropy, we are minimising -entropy
            ent_logps = F.log_softmax(logits, dim=-1)
            loss_ent = torch.mean(torch.sum(ent_logps.exp() * ent_logps, dim=-1))

            illegal_mask = torch.ones_like(logits, dtype=torch.bool)
            for x, idx in zip(illegal_mask, batch.legal_moves):
                x[idx] = False

            masked_logits = logits.masked_fill(illegal_mask, float("-inf"))
            logps = F.log_softmax(masked_logits, dim=-1)
            logps = torch.gather(logps, dim=1, index=batch.actions)

            # to maximise the surrogate objective, we minimise loss_ppo = -surrogate
            ratio = torch.exp(logps - batch.action_logps).squeeze()
            surr_1 = ratio * batch.advantages
            surr_2 = torch.clip(ratio, 1 - epsilon, 1 + epsilon) * batch.advantages
            loss_ppo = -torch.mean(torch.minimum(surr_1, surr_2))

            (loss_ppo + entropy_coeff * loss_ent).backward()
            policy_fn_optimizer.step()

            # update value function
            value_fn_optimizer.zero_grad()
            values = agent.value_fn(batch.observations)
            loss_vf = F.smooth_l1_loss(values, batch.emprets)

            loss_vf.backward()
            value_fn_optimizer.step()

            losses_ppo.append(loss_ppo.item())
            losses_ent.append(loss_ent.item())
            losses_vf.append(loss_vf.item())

        avg_loss_ppo = np.mean(losses_ppo)
        avg_loss_ent = np.mean(losses_ent)
        avg_loss_vf = np.mean(losses_vf)

        log_train.info(
            f"Training epoch {i_epoch}: "
            f"average loss_ppo={avg_loss_ppo:.4f}, "
            f"loss_ent={avg_loss_ent:.4f}, "
            f"loss_vf={avg_loss_vf:.4f}"
        )


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
    env_preset: str,
    env_players: int,
    # collection config
    iterations: int,
    collection_size: int,
    gae_gamma: float,
    gae_lambda: float,
    # neural network config
    hidden_size: int,
    num_layers: int,
    # training config
    batch_size: int,
    dataloader_workers: int,
    epochs: int,
    policy_fn_optim: dict,
    value_fn_optim: dict,
    ppo_epsilon: float,
    entropy_coeff: float,
    # eval_config
    eval_every: int,  # every n iterations
    eval_episodes: int,
    # misc
    seed: int,
    device: torch.device,
    actor_device: torch.device,
):
    env = PPOEnvironment(env_preset, env_players, seed)

    learn_agent = PPOAgent(
        MLPPolicy(env.obs_size, env.num_actions, hidden_size, num_layers),
        MLPValueFunction(env.obs_size, hidden_size, num_layers),
    ).to(device)
    policy_fn_optimizer = torch.optim.Adam(
        learn_agent.policy_fn.parameters(), **policy_fn_optim
    )
    value_fn_optimizer = torch.optim.Adam(
        learn_agent.value_fn.parameters(), **value_fn_optim
    )

    actor_agent = PPOAgent(
        MLPPolicy(env.obs_size, env.num_actions, hidden_size, num_layers),
        MLPValueFunction(env.obs_size, hidden_size, num_layers),
    ).to(actor_device)
    actor_agent.load_state_dict(learn_agent.state_dict())

    for i_iter in range(1, iterations + 1):
        log_main.info(f"====== Iteration {i_iter}/{iterations} ======")

        start = perf_counter()
        data = collect(
            env, actor_agent, gae_gamma, gae_lambda, collection_size, actor_device
        )
        dataloader = DataLoader(
            data,
            batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
            collate_fn=FrameBatch,
        )

        log_collect.info(f"Collection done in {perf_counter() - start:.2f} s")

        start = perf_counter()
        train(
            data=dataloader,
            agent=learn_agent,
            policy_fn_optimizer=policy_fn_optimizer,
            value_fn_optimizer=value_fn_optimizer,
            epsilon=ppo_epsilon,
            entropy_coeff=entropy_coeff,
            epochs=epochs,
            device=device,
        )

        log_train.info(f"Training done in {perf_counter() - start:.2f} s")

        actor_agent.load_state_dict(learn_agent.state_dict())

        if i_iter % eval_every == 0:
            evaluate(env, actor_agent, eval_episodes, actor_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-ad", "--actor-device", type=int, default=-1)

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as fi:
        config = json.load(fi)

    if "collect_workers" in config:
        log_main.warning(
            "No multiprocessing yet. `collect_workers` is multiplied onto `collection_size`"
        )
        config["collection_size"] *= config["collect_workers"]
        del config["collect_workers"]

    if "seed" not in config or config["seed"] < 0:
        config["seed"] = random.randint(0, 999)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    log_main.info("JSON config:\n" + json.dumps(config, indent=4))

    # choose device
    if torch.cuda.is_available() and args.device > 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available() and args.actor_device > 0:
        actor_device = torch.device(f"cuda:{args.device}")
    else:
        actor_device = torch.device("cpu")

    log_main.info(f"learn_agent device: {device}, actor_agent device: {actor_device}")

    main(**config, device=device, actor_device=actor_device)
