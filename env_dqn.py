from typing import NamedTuple, Union

import torch
from hanabi_learning_environment.pyhanabi import *


class ActionIsIllegal(Exception):
    pass


class Transition(NamedTuple):
    obs_t0: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    obs_t1: torch.Tensor
    is_terminal: bool


class HanabiEnvironment(object):
    def __init__(self, preset="small", players=2, card_knowledge=True, seed=-1):
        super().__init__()

        if card_knowledge:
            observation_type = AgentObservationType.CARD_KNOWLEDGE.value
        else:
            observation_type = AgentObservationType.MINIMAL.value

        if preset == "full":
            self.game = HanabiGame(
                {
                    "players": players,
                    "observation_type": observation_type,
                    "seed": seed,
                }
            )
        elif preset == "small":
            self.game = HanabiGame(
                {
                    "players": players,
                    "colors": 2,
                    "rank": 5,
                    "hand_size": 2,
                    "max_information_tokens": 3,
                    "max_life_tokens": 1,
                    "observation_type": observation_type,
                    "seed": seed,
                }
            )
        else:
            raise ValueError('`preset` must be "full" or "small"')

        self.encoder = ObservationEncoder(self.game)

        self.players = players
        self.max_moves = self.game.max_moves()
        self.obs_shape = self.encoder.shape()[0]

    def reset(self):
        self.state = self.game.new_initial_state()
        while self.state.cur_player() == CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        self.score = self.state.score()
        self.fireworks = sum(self.state.fireworks())
        self.observations = self._make_all_observations()

    def step(self, action: Union[torch.Tensor, int]):
        player = self.state.cur_player()

        obs_t0 = self.observations[player]

        if isinstance(action, int):
            action = torch.tensor(action)

        move = self.game.get_move(int(action.item()))
        if self.state.move_is_legal(move):
            self.state.apply_move(move)
            while self.state.cur_player() == CHANCE_PLAYER_ID:
                self.state.deal_random_card()

            new_score = self.state.score()
            reward = torch.tensor(new_score - self.score) * 10

            self.score = new_score
            self.fireworks = sum(self.state.fireworks())
            self.observations = self._make_all_observations()
        else:
            raise ActionIsIllegal

        obs_t1 = self.observations[player]
        is_terminal = self.state.is_terminal()

        return Transition(obs_t0, action, reward, obs_t1, is_terminal)

    def quick_step(self, action: Union[torch.Tensor, int]):
        if isinstance(action, int):
            move = self.game.get_move()
        else:
            move = self.game.get_move(int(action.item()))

        if self.state.move_is_legal(move):
            self.state.apply_move(move)
            while self.state.cur_player() == CHANCE_PLAYER_ID:
                self.state.deal_random_card()
        else:
            raise ActionIsIllegal

        return self.state.is_terminal()

    def _make_all_observations(self) -> torch.Tensor:
        every_obs = torch.zeros((self.players, self.obs_shape))
        for p in range(self.players):
            every_obs[p] = torch.tensor(self.encoder.encode(self.state.observation(p)))

        return every_obs
