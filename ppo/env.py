from typing import List, Tuple

from hanabi_learning_environment.pyhanabi import *


class ActionIsIllegal(RuntimeError):
    pass


class PPOEnv:
    def __init__(self, preset, players, seed=-1):
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

        self.get_move = self._game.get_move
        self.get_move_uid = self._game.get_move_uid

    def reset(self):
        self._state = self._game.new_initial_state()
        while self._state.cur_player() == CHANCE_PLAYER_ID:
            self._state.deal_random_card()

        self.cur_player = self._state.cur_player()
        self.observations = self._make_all_observations()
        self.illegal_mask = self._make_illegal_mask()
        self.score = self._state.score()

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
        self.cur_player = self._state.cur_player()
        self.observations = self._make_all_observations()
        self.illegal_mask = self._make_illegal_mask()
        self.score = new_score

        is_terminal = self._state.is_terminal()

        return reward, is_terminal

    def _make_all_observations(self):
        all_obs = [
            self._encoder.encode(self._state.observation(p))
            for p in range(self.players)
        ]

        return all_obs

    # def _make_legal_moves(self):
    #     legal_moves = [self.get_move_uid(m) for m in self._state.legal_moves()]
    #     return legal_moves

    def _make_illegal_mask(self) -> List[bool]:
        illegal_mask = [True] * self.num_actions
        for move in self._state.legal_moves():
            illegal_mask[self.get_move_uid(move)] = False
        return illegal_mask
