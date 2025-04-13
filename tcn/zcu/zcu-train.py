import argparse

import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
from typing import Dict, Optional, Tuple, Any

from tcn.common_utils import (
    load_config,
    configure_gpu,
    build_tcn_transformer_model,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix
)
from tcn.zcu.data_loading import file_manager
from tcn.zcu.data_loading.EpochEvent import EpochEvent
from tcn.zcu.data_loading.MovementType import MovementType
from tcn.zcu.data_loading.utils import find_min_sampling_frequency, get_epochs, drop_half_resting, \
    transform_data_representation


def get_data_path(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if config['network_args']['num_classes'] == 2:
        return config['data']['data_file_binary'], config['data']['labels_file_binary']
    elif config['network_args']['num_classes'] == 3:
        return config['data']['data_file_multiclass'], config['data']['labels_file_multiclass']
    else:
        print("Invalid number of classes in configuration.")
        return None, None




def load_data(config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads the input data, either the already saved and preprocessed data if the file exits
    and config.save_load_preprocessed data has been set to true, otherwise reads the raw data signals from data folder.

    Each raw data are grouped together by person that the data belongs to. The raw signals are then preprocessed
    to the desired data representation set in config.data_representation.

    The shape of the returned array data array is:
        time_series: n_people, n_samples, n_channels, n_times

    The shape of the returned labels array is:
        n_people, n_samples

    :return: a tuple of 2 elements, where the first element is the preprocessed data
     and the second is labels for each data sample
    """
    data = []
    labels = []

    preprocessed_data = file_manager.load_preprocessed_data(config)
    if preprocessed_data[0] is not None:
        return preprocessed_data

    files_per_person = file_manager.group_input_files_per_person(config)

    sampling_frequency = find_min_sampling_frequency(files_per_person)
    personal_epochs = []
    for i, person_files in enumerate(files_per_person):
        if config['network_args']['num_classes'] == 3:
            left = [left for left in person_files if left.movement_type is MovementType.LEFT]
            right = [right for right in person_files if right.movement_type is MovementType.RIGHT]

            left_epochs, left_labels = get_epochs(left, MovementType.LEFT.get_epoch_event(), sampling_frequency)
            right_epochs, right_labels = get_epochs(right, MovementType.RIGHT.get_epoch_event(), sampling_frequency)

            if left_epochs is None or right_epochs is None:
                continue

            # Dropping half of the epochs representing the resting state of the patient from each set, in order to
            # try to maintain a balanced overall dataset where 1/3 is resting 1/3 is left movement and 1/3 is right
            # movement, otherwise the resting state would be much larger than the movements
            drop_half_resting(left_epochs)
            left_labels = left_epochs.events[:, 2]
            drop_half_resting(right_epochs)
            right_labels = right_epochs.events[:, 2]

            personal_epochs.append(mne.concatenate_epochs([left_epochs, right_epochs]))

            left_data = transform_data_representation(left_epochs)
            right_data = transform_data_representation(right_epochs)

            data.append(np.concatenate((left_data, right_data)))
            labels.append(np.concatenate((left_labels, right_labels)))

        elif config['network_args']['num_classes'] == 2:
            epochs, epochs_labels = get_epochs(person_files, EpochEvent.MOVEMENT_START, sampling_frequency)

            if epochs is None:
                continue

            personal_epochs.append(epochs)

            epochs_data = transform_data_representation(epochs)
            data.append(epochs_data)
            labels.append(epochs_labels)

    data = np.array(data, dtype=object)
    labels = np.array(labels, dtype=object)


    file_manager.save_preprocessed_data(data, labels, config)

    return data, labels






def transform_labels(labels: np.ndarray) -> np.ndarray:
    """
    Transform original labels to a zero-based indexing system.

    Original label mapping:
    - 2 -> 0
    - 5 -> 1
    - 6 -> 2

    Args:
        labels: Original labels array.

    Returns:
        np.ndarray: Transformed labels array.
    """
    transformed_labels = labels.copy()
    transformed_labels[transformed_labels == 2] = 0
    transformed_labels[transformed_labels == 5] = 1
    transformed_labels[transformed_labels == 6] = 2
    return transformed_labels


def preprocess_data_for_tf(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Preprocess the EEG dataset for TCN-Transformer model.

    Args:
        X: Input data.
        y: Input labels.
        config: Configuration dictionary.

    Returns:
        Tuple containing train, validation, and test sets for X and y, and number of classes
    """
    print(f"Original data shape: {X.shape}")

    # Get number of classes from config
    num_classes = config['network_args']['num_classes']
    print(f"Dataset has {num_classes} classes")

    # Extract preprocessing parameters from config
    test_size = config['data']['test_size']
    val_size = config['data']['validation_size']

    # Normalize each channel separately
    X_normalized = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[1]):  # Iterate over channels
        channel_data = X[:, i, :]
        scaler = StandardScaler()
        X_normalized[:, i, :] = scaler.fit_transform(channel_data)

    # Transpose data to shape [n_samples, n_times, n_channels]
    X_normalized = np.transpose(X_normalized, (0, 2, 1))

    # Stratified splitting to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=y_train
    )

    # Convert to int32
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Print distribution to verify
    print("Train set distribution:", np.bincount(y_train))
    print("Val set distribution:", np.bincount(y_val))
    print("Test set distribution:", np.bincount(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def train_and_test_model(data: np.ndarray, labels: np.ndarray, config: Dict[str, Any]) -> None:
    """
    Train a global model using combined data from all subjects.

    Args:
        data: Combined dataset.
        labels: Labels for the dataset.
        config: Configuration dictionary.
    """
    # Concatenate all data and train a global model
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = transform_labels(labels)

    # Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess_data_for_tf(data, labels, config)

    # Get model dimensions
    timesteps, num_features = X_train.shape[1], X_train.shape[2]

    # Build the model with configuration
    model = build_tcn_transformer_model(timesteps, num_features, num_classes, config)

    # Train the model
    history = train_model(model, X_train, X_val, y_train, y_val, config)

    # Evaluate the model
    acc, report, cm, y_pred = evaluate_model(model, X_test, y_test)

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    plot_training_history(history, config)
    plot_confusion_matrix(cm, np.arange(num_classes), config)

    # Save the model
    model.save(config['logging']['model_path'])
    print("Model saved")


def main() -> None:
    parser = argparse.ArgumentParser(description='EEG Classification Model')
    parser.add_argument('--config', default='zcu-config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Configure GPU based on configuration
    config = load_config(args.config)
    configure_gpu(config)

    # Load data
    data, labels = load_data(config)

    if data is not None and labels is not None:
        # Train and test the model
        train_and_test_model(data, labels, config)


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    main()