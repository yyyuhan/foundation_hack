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


SET = {GameName.ATARI_MSPACMAN: GameInfo("ALE/MsPacman-v5", 18)}


class GameSet:
    @staticmethod
    def env_id(game):
        return SET[game].env_id

    @staticmethod
    def action_space_dim(game):
        return SET[game].action_space


class Constants(int, Enum):
    CHECKPOINT_FREQUENCY = 50
