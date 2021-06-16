import dm_env
from dm_env import specs
from hanabi_learning_environment.pyhanabi import *
import numpy as np


class HanabiEnvironment(dm_env.Environment):
    def __init__(self, num_players=2, seed=0):
        self.game = HanabiGame(
            {
                "players": num_players,
                "seed": seed,
            }
        )

        self.state = self.game.new_initial_state()

        self.observation_encoder = ObservationEncoder(self.game)

    def __repr__(self) -> str:
        return (
            f"Cur player: {self.state.cur_player()}\n"
            f"Score: {self.state.score()}\n"
            f"{self.state}"
        )

    def reset(self) -> dm_env.TimeStep:
        self.state = self.game.new_initial_state()
        while self.state.cur_player() == CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        return dm_env.restart(
            observation=self._observation_array(self.state.cur_player())
        )

    def step(self, action) -> dm_env.TimeStep:
        action_player = self.state.cur_player()
        old_score = self.state.score()

        # apply action and update game state
        game_move = self.game.get_move(action)
        if self.state.move_is_legal(game_move):
            self.state.apply_move(game_move)
        else:
            # raise RuntimeError("Move is illegal")
            pass

        # deal new cards if needed
        while self.state.cur_player() == CHANCE_PLAYER_ID:
            self.state.deal_random_card()

        # return a TimeStep object
        reward = self.state.score() - old_score
        observation = self._observation_array(action_player)

        if self.state.is_terminal():
            time_step = dm_env.termination(
                reward=reward,
                observation=observation,
            )
        else:
            time_step = dm_env.transition(
                reward=reward,
                observation=observation,
            )

        return time_step

    def observation_spec(self):
        return specs.BoundedArray(
            self.observation_encoder.shape(), dtype=np.float32, minimum=0, maximum=1
        )

    def action_spec(self):
        return specs.DiscreteArray(self.game.max_moves())

    def _observation_array(self, player):
        """Make a new observation from `self.state`."""
        return np.array(
            self.observation_encoder.encode(self.state.observation(player)),
            dtype=np.float32,
        )


if __name__ == "__main__":
    env = HanabiEnvironment()
