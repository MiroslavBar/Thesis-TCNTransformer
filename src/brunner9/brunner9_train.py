import argparse
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib
from typing import Dict, List, Tuple, Any


from src.common_utils import (
    load_config,
    configure_gpu,
    build_tcn_transformer_model,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix
)
from src.brunner9.preprocess_data import preprocess_competition

np.random.seed(42)
tf.random.set_seed(42)


def load_and_combine_data(
    data_dir: str,
    subjects: List[int] =None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine EEG data from multiple subjects in the BCI IV 2a dataset.

    This function reads data and label files for specified subjects from a given directory,
    combines them into a single dataset, and adjusts class labels to be 0-indexed.

    Args:
        data_dir: Path to the directory containing dataset files
        subjects: List of subject numbers to include in the dataset (default: 1-9)

    Returns:
        A tuple containing:
        - X: Combined EEG data with shape [n_samples, n_channels, n_times]
        - y: Combined labels with shape [n_samples]
    """
    if subjects is None:
        subjects = list(range(1, 10))

    X_all = []
    y_all = []

    preprocessed_data_path = "preprocessed-data-brunner9"

    if not os.path.exists(preprocessed_data_path):
        print("Preprocessing data...")
        preprocess_competition(data_dir, preprocessed_data_path)

    for subject in subjects:
        # Load data and labels for this subject
        data_file = os.path.join(preprocessed_data_path, f'A0{subject}E_data.npy')
        label_file = os.path.join(preprocessed_data_path, f'A0{subject}E_label.npy')

        if os.path.exists(data_file) and os.path.exists(label_file):
            X_subject = np.load(data_file)
            y_subject = np.load(label_file).ravel()  # Flatten to 1D array

            print(f"Subject {subject}: X shape {X_subject.shape}, y shape {y_subject.shape}")

            X_all.append(X_subject)
            y_all.append(y_subject)

    # Combine data from all subjects
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # Adjust class labels to be 0-indexed if they aren't already
    y = y - 1 if np.min(y) == 1 else y

    return X, y


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

    test_size = config['data']['test_size']
    val_size = config['data']['validation_size']

    # Get number of classes
    num_classes = len(np.unique(y))
    print(f"Dataset has {num_classes} classes")

    # Transpose data to shape [n_samples, n_times, n_channels]
    # TCN expects input shape: (batch_size, timesteps, features)
    X = np.transpose(X, (0, 2, 1))
    print(f"Transposed data to: {X.shape} (samples, timesteps, channels)")

    # Normalize the features using StandardScaler
    orig_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    # Reshape back to original 3D shape
    X = X_scaled.reshape(orig_shape)

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
    parser.add_argument('--config', default='brunner9_config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Configure GPU based on configuration
    config = load_config(args.config)
    configure_gpu(config)

    # Load and combine data from all subjects
    data, labels = load_and_combine_data(config['data']['data_dir'])

    if data is not None and labels is not None:
        # Train and test the model
        train_and_test_model(data, labels, config)

if __name__ == "__main__":
    matplotlib.use('TkAgg')
    main()