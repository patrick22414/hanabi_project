from enum import Enum
from dataclasses import dataclass

import torch
from hanabi_learning_environment.pyhanabi import *


class ActionIsIllegal(RuntimeError):
    pass


FrameType = Enum("FrameType", "START MID END")


@dataclass
class Frame:
    frame_type: FrameType
    observation: torch.Tensor  # 1-d vector
    observation_t1: torch.Tensor  # 1-d vector
    action_logp: torch.Tensor  # 1-d vector
    action: torch.Tensor  # 1-d scalar
    value: torch.Tensor  # scalar
    value_t1: torch.Tensor  # scalar
    reward: torch.Tensor  # scalar
    empret: torch.Tensor  # scalar, empirical return
    advantage: torch.Tensor = None  # scalar


class PPOEnvironment(object):
    def __init__(self, players, seed=-1):
        super().__init__()

        self._game = HanabiGame({"players": players, "seed": seed})
        self._encoder = ObservationEncoder(self._game)

        self.players = players
        self.num_actions = self._game.max_moves()
        self.obs_size = self._encoder.shape()[0]

        self.get_move = self._game.get_move
        self.get_move_uid = self._game.get_move_uid

    def reset(self):
        self._state = self._game.new_initial_state()
        while self._state.cur_player() == CHANCE_PLAYER_ID:
            self._state.deal_random_card()

        self.frame_type = FrameType.START
        self.observation = self._make_observation()
        self.legal_moves = self._make_legal_moves()

        self.score = self._state.score()

    def step(self, action: int):
        player = self._state.cur_player()
        move = self._game.get_move(action)

        if not self._state.move_is_legal(move):
            raise ActionIsIllegal

        self._state.apply_move(move)
        while self._state.cur_player() == CHANCE_PLAYER_ID:
            self._state.deal_random_card()

        new_score = self._state.score()
        if new_score > self.score:
            reward = 1.0
        else:
            reward = 0.0

        obs_t1 = self._make_observation(player)  # obs_t1 is from the old player

        # prepare for next step
        is_terminal = self._state.is_terminal()
        if is_terminal:
            self.frame_type = FrameType.END
        else:
            self.frame_type = FrameType.MID
            self.observation = self._make_observation()
            self.legal_moves = self._make_legal_moves()
            self.score = new_score

        return obs_t1, reward, is_terminal

    def _make_observation(self, player=None):
        if player is None:
            player = self._state.cur_player()

        obs = self._encoder.encode(self._state.observation(player))
        return obs

    def _make_legal_moves(self) -> list[int]:
        legal_moves = [self.get_move_uid(m) for m in self._state.legal_moves()]
        return legal_moves
