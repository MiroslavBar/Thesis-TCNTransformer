import logging as log
import warnings
from abc import ABC, abstractmethod

import mne
from mne.io import Raw

from tcn.kodera29.data_loading.MovementType import MovementType


class FileFormat(ABC):

    def __init__(self):
        self._raw = None

    @property
    @abstractmethod
    def movement_type(self) -> MovementType:
        pass

    @property
    @abstractmethod
    def file_path(self) -> str:
        pass

    @property
    def raw(self) -> Raw:
        return self._raw

    @staticmethod
    @abstractmethod
    def process(file_path) -> 'FileFormat':
        pass

    def read_raw(self) -> Raw:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Online software filter detected. Using software filter '
                                                          'settings and ignoring hardware values')
                self._raw = mne.io.read_raw_brainvision(self.file_path, verbose=False)
        except FileNotFoundError:
            log.warning(f"Unable to read file {self.file_path}.")

        return self.raw

    @abstractmethod
    def same_person(self, other: 'FileFormat') -> bool:
        pass
