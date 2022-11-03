from collections import namedtuple
from enum import Enum

GameInfo = namedtuple("gameinfo", "env_id action_space")


class ErrMsg(str, Enum):
    TypeNotSupported = "Type not supported"
    ParseFailure = "Parse failed"
    InvalidParam = "Invalid parameter"
    NotImplemented = "Not implemented"
    UncaughtException = "Uncaught exception"
    ForkFailure = "Fork failed"
    InitFailure = "Initialization failed"
    InvalidResponse = "Invalid response"


class GameName(str, Enum):
    ATARI_MSPACMAN = "mspacman"


class GameSet:
    SET = {GameName.ATARI_MSPACMAN: GameInfo("ALE/MsPacman-v5", 18)}

    def get_env_id(self, game):
        return self.SET[game]["env_id"]

    def get_action_space_dim(self, game):
        return self.SET[game]["action_space"]


class Constants(int, Enum):
    CHECKPOINT_FREQUENCY = 50
