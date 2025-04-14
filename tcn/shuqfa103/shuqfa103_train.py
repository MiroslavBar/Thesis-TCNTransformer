import argparse
import matplotlib
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
from typing import Dict, List, Optional, Callable, Tuple, Any

from tcn.common_utils import (
    load_config,
    configure_gpu,
    build_tcn_transformer_model,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix
)


def label_conversion_map(conversion_type: str) -> Optional[Callable[[int], Optional[int]]]:
    """
    Returns the appropriate label conversion function based on configuration.

    Args:
        conversion_type: The type of label conversion to apply.

    Returns:
        Optional function to convert labels or None if not found.
    """
    conversion_functions = {
        'two_class_ME': two_class_ME_labels,
        'two_class_MI': two_class_MI_labels,
        'four_class_ME': four_class_ME_labels,
        'four_class_MI': four_class_MI_labels,
        'two_class_labels': two_class_labels,
        'four_class_labels': four_class_labels
    }
    return conversion_functions.get(conversion_type)


def two_class_labels(label: int) -> Optional[int]:
    if label in [2, 3, 5, 6, 8, 9, 11, 12]: return 0  # Movement
    if label in [1, 4, 7, 10]: return 1  # Relax
    return None


def two_class_MI_labels(label: int) -> Optional[int]:
    if label in [5, 6, 11, 12]: return 0  # MI Movement
    if label in [4, 10]: return 1  # MI Relax
    return None


def two_class_ME_labels(label: int) -> Optional[int]:
    if label in [2, 3, 8, 9]: return 0  # ME Movement
    if label in [1, 7]: return 1  # ME Relax
    return None


def four_class_MI_labels(label: int) -> Optional[int]:
    if label in [5]: return 0  # MI left fist movement
    if label in [6]: return 1  # MI right first movement
    if label in [12]: return 2  # MI both feet movement
    if label in [4, 10]:
        if random.randint(1, 4) == 4:  # Reducing the amount of rest trials to balance the data
            return 3  # MI relax
    return None


def four_class_ME_labels(label: int) -> Optional[int]:
    if label in [2]: return 0  # ME left fist movement
    if label in [3]: return 1  # ME right first movement
    if label in [9]: return 2  # ME both feet movement
    if label in [1, 7]:
        if random.randint(1, 4) == 4:  # Reducing the amount of rest trials to balance the data
            return 3  # ME relax
    return None


def four_class_labels(label: int) -> Optional[int]:
    if label in [2, 5]: return 0  # Left first movement
    if label in [3, 6]: return 1  # Right first movement
    if label in [9, 12]: return 2  # Both feet movement
    if label in [1, 4, 7, 10]:
        if random.randint(1, 4) == 4:  # Reducing the amount of rest trials to balance the data
            return 3  # Relax
    return None

def load_csv_data(file_path: str) -> np.ndarray:
    return np.loadtxt(file_path, delimiter=',')


def get_subjects(csv_dir: str) -> List[str]:
    subject_ids = set()
    for file in os.listdir(csv_dir):
        if 'SIG' in file:
            subject_id = file.split('_')[1]
            subject_ids.add(subject_id)
    return sorted(subject_ids)


def preprocess_subject(
    csv_dir: str,
    subject_id: str,
    config: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Preprocess data for a specific subject.

    Args:
        csv_dir: Directory containing CSV files
        subject_id: Subject identifier
        config: Configuration dictionary

    Returns:
        Tuple of preprocessed data and labels, or (None, None) if no valid data
    """
    num_samples = config['data']['num_samples']
    conversion_type = config['data']['label_conversion']
    convert_label = label_conversion_map(conversion_type)

    signal_files = sorted([f for f in os.listdir(csv_dir) if f'SUB_{subject_id}_SIG' in f])
    annotation_files = sorted([f for f in os.listdir(csv_dir) if f'SUB_{subject_id}_ANN' in f])

    assert len(signal_files) == len(annotation_files), f"Mismatch for subject {subject_id}"

    all_data, all_labels = [], []

    for signal_file, annotation_file in zip(signal_files, annotation_files):
        signal_data = load_csv_data(os.path.join(csv_dir, signal_file))
        annotation_data = load_csv_data(os.path.join(csv_dir, annotation_file))

        for trial_idx in range(annotation_data.shape[0]):
            trial_label = int(annotation_data[trial_idx, 0])
            converted_label = convert_label(trial_label)
            if converted_label is None:
                continue

            start_idx = int(annotation_data[trial_idx, 3]) - 1
            end_idx = int(annotation_data[trial_idx, 4]) - 1
            trial_signal = signal_data[start_idx:end_idx, :]

            # Ensure shape consistency
            if trial_signal.shape[0] < num_samples:
                trial_signal = np.pad(trial_signal, ((0, num_samples - trial_signal.shape[0]), (0, 0)), mode='constant')
            elif trial_signal.shape[0] > num_samples:
                trial_signal = trial_signal[:num_samples, :]

            # Normalize
            mean = np.mean(trial_signal, axis=0, keepdims=True)
            std = np.std(trial_signal, axis=0, keepdims=True)
            std[std == 0] = 1
            trial_signal = ((trial_signal - mean) / std).astype(np.float32)

            # Transpose to [num_channels, num_samples]
            trial_signal = trial_signal.T

            all_data.append(trial_signal)
            all_labels.append(converted_label)

    if not all_data:  # Skip subjects with no valid trials
        return None, None

    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    return all_data, all_labels


def prepare_global_dataset(config: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Combine all subjects' data and prepare a global dataset.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of combined data and labels, or (None, None) if no valid data
    """
    csv_dir = config['data']['csv_directory']

    all_data, all_labels = [], []

    subjects = get_subjects(csv_dir)
    for subject_id in subjects:
        data, labels = preprocess_subject(csv_dir, subject_id, config)
        if data is not None and labels is not None:
            all_data.append(data)
            all_labels.append(labels)

    if not all_data:
        print("No valid data found for global evaluation.")
        return None, None

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_data, all_labels


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

    test_size = config['data']['test_size']
    val_size = config['data']['validation_size']

    print(f"Original data shape: {X.shape}")

    # Get number of classes
    num_classes = len(np.unique(y))
    print(f"Dataset has {num_classes} classes")

    # Transpose data to shape [n_samples, n_times, n_channels]
    # TCN expects input shape: (batch_size, timesteps, features)
    X = np.transpose(X, (0, 2, 1))
    print(f"Transposed data to: {X.shape} (samples, timesteps, channels)")

    # Split data into train, validation, and test sets
    # Using stratified splitting to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=y_train
    )

    # Convert labels to integers
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def train_and_test_model(data: np.ndarray, labels: np.ndarray, config: Dict[str, Any]) -> None:
    """
    Train a global model using combined data from all subjects.

    Args:
        data: Combined dataset.
        labels: Labels for the dataset.
        config: Configuration dictionary.
    """

    # Preprocess data for TensorFlow model
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess_data_for_tf(data, labels, config)

    # Get model dimensions
    timesteps, num_features = X_train.shape[1], X_train.shape[2]

    # Build model
    model = build_tcn_transformer_model(timesteps, num_features, num_classes, config)

    # Train model
    history = train_model(model, X_train, X_val, y_train, y_val, config)

    # Evaluate model
    acc, report, cm, y_pred = evaluate_model(model, X_test, y_test)

    # Print evaluation results
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Plot results
    plot_training_history(history, config)
    plot_confusion_matrix(cm, np.arange(num_classes), config)

    # Save the model
    model.save('tcn_transformer_eeg_model.h5')
    print("Model saved to tcn_transformer_eeg_model.h5")


def main() -> None:
    parser = argparse.ArgumentParser(description='EEG Classification Model')
    parser.add_argument('--config', default='shuqfa103_config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Configure GPU based on configuration
    config = load_config(args.config)
    configure_gpu(config)

    print("Loading data...")
    data, labels = prepare_global_dataset(config)

    if data is not None and labels is not None:
        # Train and test the model
        train_and_test_model(data, labels, config)


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    main()