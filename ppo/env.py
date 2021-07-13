from collections import deque
from typing import List, Tuple

import numpy as np
from hanabi_learning_environment.pyhanabi import *


class ActionIsIllegal(RuntimeError):
    pass


class PPOEnv:
    def __init__(self, preset, players, enc_type, buffer_size=1, seed=-1):
        super().__init__()

        if preset == "full":
            self._game = HanabiGame(
                {
                    "players": players,
                    "observation_type": AgentObservationType.CARD_KNOWLEDGE.value,
                    "seed": seed,
                }
            )
        elif preset == "small":
            self._game = HanabiGame(
                {
                    "players": players,
                    "colors": 2,
                    "rank": 5,
                    "hand_size": 3,
                    "max_information_tokens": 5,
                    "max_life_tokens": 3,
                    "observation_type": AgentObservationType.CARD_KNOWLEDGE.value,
                    "seed": seed,
                }
            )
        self._encoder = ObservationEncoder(self._game)

        self.players = players
        self.num_actions = self._game.max_moves()
        self.enc_size = self._encoder.shape()[0]
        self.buf_size = buffer_size
        self.obs_size = self.buf_size * self.enc_size
        self.enc_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

        self.get_move = self._game.get_move
        self.get_move_uid = self._game.get_move_uid

    def reset(self):
        self._state = self._game.new_initial_state()
        while self._state.cur_player() == CHANCE_PLAYER_ID:
            self._state.deal_random_card()

        self.score = self._state.score()
        self.cur_player = self._state.cur_player()
        self.illegal_mask = self._make_illegal_mask()
        for _ in range(self.buf_size - 1):
            self.enc_buffer.append(np.zeros((self.players, self.enc_size), dtype=int))
        self.enc_buffer.append(self._make_all_encodings())

    def step(self, action: int) -> Tuple[float, bool]:
        move = self.get_move(action)
        if not self._state.move_is_legal(move):
            raise ActionIsIllegal

        # apply move and resolve randomness
        self._state.apply_move(move)
        while self._state.cur_player() == CHANCE_PLAYER_ID:
            self._state.deal_random_card()

        new_score = self._state.score()
        reward = 1.0 if new_score > self.score else 0.0

        # prepare for next step
        self.score = new_score
        self.cur_player = self._state.cur_player()
        self.illegal_mask = self._make_illegal_mask()
        self.enc_buffer.append(self._make_all_encodings())

        is_terminal = self._state.is_terminal()

        return reward, is_terminal

    def observation(self, player: int) -> np.ndarray:
        if self.buf_size == 1:
            return self.enc_buffer[0][player]
        else:
            return np.concatenate([obs[player] for obs in self.enc_buffer])

    def _encoding(self, player: int):
        return np.array(self._encoder.encode(self._state.observation(player)))

    def _make_all_encodings(self):
        all_obs = np.stack([self._encoding(p) for p in range(self.players)])
        return all_obs

    # def _make_legal_moves(self):
    #     legal_moves = [self.get_move_uid(m) for m in self._state.legal_moves()]
    #     return legal_moves

    def _make_illegal_mask(self):
        illegal_mask = np.ones(self.num_actions, dtype=bool)
        for move in self._state.legal_moves():
            illegal_mask[self.get_move_uid(move)] = False
        return illegal_mask
