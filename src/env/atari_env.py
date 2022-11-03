import gym

from src.utils.types import GameSet


# pip install gym[accept-rom-license]
class AtariEnv(gym.Wrapper):
    def __init__(self, game=""):
        env_id = GameSet.env_id(game)
        self.env = gym.make(env_id, full_action_space=True)
        super().__init__(self.env)
