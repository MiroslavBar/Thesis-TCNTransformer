import mne
import numpy as np
import scipy.io
import os


def preprocess_competition(data_path: str, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)

    preprocess_data_test(data_path, save_path)
    preprocess_data_train(data_path, save_path)


def preprocess_data_test(data_path: str, save_path: str) -> None:
    data_files = ['A0' + str(i) + 'E.gdf' for i in range(1, 10)]
    label_files = ['A0' + str(i) + 'E.mat' for i in range(1, 10)]

    event_description = {'783': "CueUnknown"}

    for file in data_files:
        raw_data = mne.io.read_raw_gdf(os.path.join(data_path, file), preload=True, verbose=False)

        raw_events, all_event_id = mne.events_from_annotations(raw_data)

        raw_data = mne.io.RawArray(raw_data.get_data() * 1e6, raw_data.info)

        raw_data.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

        test_picks = mne.pick_types(raw_data.info, eeg=True, exclude='bads')

        tmin, tmax = 0, 4

        # Unknown = 783
        # event_id = dict({'783':7})
        event_id = dict()
        for event in all_event_id:
            if event in event_description:
                event_id[event] = all_event_id[event]

        raw_epochs = mne.Epochs(raw_data, raw_events, event_id, tmin, tmax, proj=True, picks=test_picks, baseline=None,
                                preload=True)

        # print(test_epochs)

        data = raw_epochs.get_data()  # [n_epochs, n_channels, n_times]
        # print(data.shape)
        data = data[:, :, :-1]

        np.save(os.path.join(save_path, file[:-4] + '_data.npy'), data)

    for file in label_files:
        true_label = scipy.io.loadmat(os.path.join(data_path, file))
        label = true_label['classlabel']
        np.save(os.path.join(save_path, file[:-4] + '_label.npy'), label)


def preprocess_data_train(data_path: str, save_path: str) -> None:
    data_files = ['A0' + str(i) + 'T.gdf' for i in range(1, 10)]

    event_description = {'769': "CueLeft", '770': "CueRight", '771': "CueFoot", '772': "CueTongue"}

    for file in data_files:
        raw_data = mne.io.read_raw_gdf(os.path.join(data_path, file), preload=True, verbose=False)

        raw_events, all_event_id = mne.events_from_annotations(raw_data)
        raw_data = mne.io.RawArray(raw_data.get_data() * 1e6, raw_data.info)

        raw_data.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

        picks = mne.pick_types(raw_data.info, eeg=True, exclude='bads')

        tmin, tmax = 0, 4

        event_id = dict()

        for event in all_event_id:
            if event in event_description:
                event_id[event_description[event]] = all_event_id[event]

        raw_epochs = mne.Epochs(raw_data, raw_events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None,
                                preload=True)

        true_labels = raw_epochs.events[:, -1] - event_id['CueLeft'] + 1
        data = raw_epochs.get_data()  # [n_epochs, n_channels, n_times]
        data = data[:, :, :-1]

        np.save(os.path.join(save_path, file[:-4] + '_data.npy'), data)
        np.save(os.path.join(save_path, file[:-4] + '_label.npy'), true_labels)

