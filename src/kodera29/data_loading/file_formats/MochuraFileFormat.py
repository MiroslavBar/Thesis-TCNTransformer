import os
import re

from src.kodera29.data_loading.MovementType import MovementType
from src.kodera29.data_loading.file_formats.FileFormat import FileFormat


class MochuraFileFormat(FileFormat):
    file_name_pattern = r"(\d+)([mz])(\d{8})([rl]h)(\d+).vhdr"

    def __init__(self, file_path: str, ID: int,  gender: int, date: str, movement_type: MovementType,
                 trial_order_number: int):
        super().__init__()
        self._file_path = file_path
        self.ID = ID
        self.gender = gender
        self.date = date
        self._movement_type = movement_type
        self.trial_order_number = trial_order_number

    @property
    def movement_type(self) -> MovementType:
        return self._movement_type

    @movement_type.setter
    def movement_type(self, value: MovementType) -> None:
        self._movement_type = value

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value

    @staticmethod
    def process(file_path: str):
        file_name = os.path.basename(file_path)
        matcher = re.search(MochuraFileFormat.file_name_pattern, file_name)
        if not matcher:
            return None

        ID = matcher.group(1)
        gender = matcher.group(2)
        date = matcher.group(3)
        movement_type = MovementType.get_type(matcher.group(4))
        trial_order_number = matcher.group(5)

        return MochuraFileFormat(file_path, ID, gender, date, movement_type, trial_order_number)

    def same_person(self, other: FileFormat) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.gender == other.gender and self.ID == other.ID and self.date == other.date
