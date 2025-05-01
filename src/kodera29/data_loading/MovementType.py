from enum import Enum

from src.kodera29.data_loading.EpochEvent import EpochEvent


class MovementType(Enum):
    UNKNOWN = ("", "", -1)
    RESTING = ("", "", EpochEvent.RESTING_MIDDLE)
    LEFT = ("lh", "leva", EpochEvent.MOVEMENT_START)
    RIGHT = ("rh", "prava", 6)

    def get_epoch_event(self) -> int:
        return self.value[2]

    @staticmethod
    def get_type(name: str) -> 'MovementType':
        movement_type = MovementType.UNKNOWN
        for t in MovementType:
            if t.value[0] == name or t.value[1] == name:
                movement_type = t
                break
        return movement_type

    # @staticmethod
    # def get_display_labels() -> list:
    #     movement_types = [MovementType.RESTING, MovementType.LEFT]
    #     if config.classification_type == ClassificationType.MULTICLASS:
    #         movement_types.append(MovementType.RIGHT)
    #
    #     return [MovementType.label_to_readable_str(movement_type.get_epoch_event()) for movement_type in movement_types]

    # @staticmethod
    # def label_to_readable_str(label: int) -> str:
    #     if label == MovementType.RESTING.get_epoch_event():
    #         return 'Resting'
    #     if label == MovementType.LEFT.get_epoch_event():
    #         if config.classification_type == ClassificationType.BINARY:
    #             return 'Movement'
    #         elif config.classification_type == ClassificationType.MULTICLASS:
    #             return 'Left movement'
    #     if label == MovementType.RIGHT.get_epoch_event():
    #         return 'Right movement'
    #
    #     return f"{label}"

