import gym


# pip install gym[accept-rom-license]
class AtariEnv(gym.Env):
    def __init__(self, game=""):
        # TODO game -> gym env
        self.env = gym.make("ALE/MsPacman-v5", full_action_space=True)
        # super().__init__(self.env)
