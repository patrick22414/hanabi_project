from collections import deque
from typing import Tuple

import numpy as np
from hanabi_learning_environment.pyhanabi import *


class ActionIsIllegal(RuntimeError):
    pass


class PPOEnv:
    def __init__(self, preset, players, buffer_size=1, idle_reward=0.0, seed=-1):
        super().__init__()

        if preset == "full":
            self._game = HanabiGame(
                {
                    "players": players,
                    "seed": seed,
                }
            )
        elif preset == "small":
            self._game = HanabiGame(
                {
                    "players": players,
                    "colors": 2,
                    "rank": 5,
                    "hand_size": 2,
                    "max_information_tokens": 3,
                    "max_life_tokens": 1,
                    "seed": seed,
                }
            )

        self._encoder = ObservationEncoder(self._game)

        self.players = players
        self.num_actions = self._game.max_moves()
        self.max_score = self._game.num_colors() * self._game.num_ranks()
        self.idle_reward = idle_reward

        self.enc_size = self._encoder.shape()[0]
        self.buf_size = buffer_size
        self.enc_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

        self.obs_size = self.buf_size * self.enc_size
        self.full_obs_size = self.players * self.enc_size

        self.get_move = self._game.get_move
        self.get_move_uid = self._game.get_move_uid

    def reset(self):
        self._state = self._game.new_initial_state()
        while self._state.cur_player() == CHANCE_PLAYER_ID:
            self._state.deal_random_card()

        for _ in range(self.buf_size - 1):
            self.enc_buffer.append(np.zeros((self.players, self.enc_size), dtype=int))
        self._update()

    def step(self, action: int) -> Tuple[float, bool]:
        move = self.get_move(action)
        if not self._state.move_is_legal(move):
            raise ActionIsIllegal

        # apply move and resolve randomness
        self._state.apply_move(move)
        while self._state.cur_player() == CHANCE_PLAYER_ID:
            self._state.deal_random_card()

        new_score = self._state.score()
        reward = 1.0 if new_score > self.score else self.idle_reward

        is_terminal = self._state.is_terminal()

        # prepare for next step
        self._update()

        return reward, is_terminal

    def observation(self, player: int) -> np.ndarray:
        if self.buf_size == 1:
            return self.enc_buffer[0][player]
        else:
            return np.concatenate([obs[player] for obs in self.enc_buffer])

    def full_observation(self) -> np.ndarray:
        # return self.enc_buffer[-1].flatten()
        return np.roll(self.enc_buffer[-1], -self.cur_player, axis=0).flatten()

    def _update(self):
        self.score = self._state.score()
        self.cur_player = self._state.cur_player()
        self.illegal_mask = self._make_illegal_mask()
        self.enc_buffer.append(self._make_all_encodings())

    def _make_all_encodings(self):
        all_enc = np.stack(
            [
                np.array(self._encoder.encode(self._state.observation(p)))
                for p in range(self.players)
            ]
        )
        return all_enc

    # def _make_legal_moves(self):
    #     legal_moves = [self.get_move_uid(m) for m in self._state.legal_moves()]
    #     return legal_moves

    def _make_illegal_mask(self):
        illegal_mask = np.ones(self.num_actions, dtype=bool)
        for move in self._state.legal_moves():
            illegal_mask[self.get_move_uid(move)] = False
        return illegal_mask
