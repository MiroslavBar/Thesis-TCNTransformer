import random
import sys

import mne
import numpy as np
from mne import Epochs

from tcn.zcu.data_loading.EpochEvent import EpochEvent
from tcn.zcu.data_loading.MovementType import MovementType
from tcn.zcu.data_loading.file_formats import FileFormat


def find_min_sampling_frequency(files_per_person: list[list[FileFormat]]) -> int:
    min_sampling_frequency = sys.float_info.max
    for person_files in files_per_person:
        for file in person_files:
            if file.raw is not None:
                min_sampling_frequency = min(file.raw.info['sfreq'], min_sampling_frequency)

    return min_sampling_frequency



def _equalize_epoch_events(epochs: Epochs, movement_start_side_marker: int) -> None:
    events = epochs.events
    events_to_keep = []
    indices_to_drop = []
    for i in range(len(events)):
        keep = False
        marker = events[i][2]
        last_marker = None if len(events_to_keep) == 0 else events_to_keep[len(events_to_keep) - 1][2]

        # Only keep the resting epoch if there was a movement epoch between this epoch and the last resting epoch
        if marker == EpochEvent.RESTING_MIDDLE and marker != last_marker:
            keep = True
        # Only keep the first movement epoch between two resting epochs
        elif (marker == EpochEvent.MOVEMENT_START or marker == EpochEvent.MOVEMENT_ADDITIONAL) and \
                (last_marker == EpochEvent.RESTING_MIDDLE):
            epochs.events[i][2] = movement_start_side_marker
            keep = True

        if keep:
            events_to_keep.append(events[i])
        else:
            indices_to_drop.append(i)

    epochs.event_id = {f"{MovementType.RESTING.get_epoch_event()}": int(MovementType.RESTING.get_epoch_event()),
                       f"{movement_start_side_marker}": movement_start_side_marker}
    epochs.drop(indices_to_drop, reason='equalize epoch events')


def get_epochs(files: list[FileFormat], movement_event: int, sample_frequency: int) \
        -> tuple[None, None] or tuple[Epochs, np.ndarray]:
    raws = [file.raw for file in files if file.raw is not None]

    if not raws:
        return None, None

    raw = mne.concatenate_raws(raws)

    events, _ = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw,
                        tmin=-3.5, tmax=0.5,
                        events=events,
                        picks=['Cz', 'C3', 'C4'],
                        preload=True,
                        verbose=False,
                        baseline=(None, -3.5 + 0.5))
    # Cropping the time because when the sampling frequency is e.g. 500 and trying to get 1 sec epoch,
    # the constructor returns 501 samples, by calling crop with include_tmax set to False we get the expected 500
    # samples
    epochs.crop(include_tmax=False)
    epochs.resample(sample_frequency)
    epochs.filter(8, 30, verbose=False)
    epochs.drop_bad(reject={'eeg': 100e-6}, verbose=False)

    _equalize_epoch_events(epochs, movement_event)

    return epochs, epochs.events[:, 2]

def drop_half_resting(epochs: Epochs) -> None:
    epoch_events = epochs.events[:, 2]
    resting_indices = [i for i, event in enumerate(epoch_events) if event == MovementType.RESTING.get_epoch_event()]

    amount_to_delete = len(resting_indices) // 2
    # Randomly pick half of the resting indices to drop
    resting_indices_to_remove = random.sample(resting_indices, amount_to_delete)
    epochs.drop(resting_indices_to_remove, reason='half resting')


def transform_data_representation(epochs: Epochs) -> np.ndarray:
    return epochs.get_data()

