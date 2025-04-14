from enum import IntEnum


class EpochEvent(IntEnum):
    RESTING_START = 1
    RESTING_MIDDLE = 2
    MOVEMENT_ADDITIONAL = 4
    MOVEMENT_START = 5

    def __str__(self):
        return str(self.value)
