from enum import Enum, IntEnum


class ErrMsg(str, Enum):
    TypeNotSupported = "Type not supported"
    ParseFailure = "Parse failed"
    InvalidParam = "Invalid parameter"
    NotImplemented = "Not implemented"
    UncaughtException = "Uncaught exception"
    ForkFailure = "Fork failed"
    InitFailure = "Initialization failed"
    InvalidResponse = "Invalid response"
