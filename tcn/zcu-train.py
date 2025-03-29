import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
from typing import Dict, Optional, Tuple, Any


from common_utils import (
    load_config,
    configure_gpu,
    build_tcn_transformer_model,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix
)

def get_data_path(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if config['network_args']['num_classes'] == 2:
        return config['data']['data_file_binary'], config['data']['labels_file_binary']
    elif config['network_args']['num_classes'] == 3:
        return config['data']['data_file_multiclass'], config['data']['labels_file_multiclass']
    else:
        print("Invalid number of classes in configuration.")
        return None, None


def load_npy_files(data_file: str, labels_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        data = np.load(data_file, allow_pickle=True)
        print(f"Data loaded successfully from {data_file}. Shape: {data.shape}")

        labels = np.load(labels_file, allow_pickle=True)
        print(f"Labels loaded successfully from {labels_file}. Shape: {labels.shape}")

        return data, labels
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


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

    # Get the appropriate data files based on number of classes
    data_file, labels_file = get_data_path(config)

    # Load data
    data, labels = load_npy_files(data_file, labels_file)

    if data is not None and labels is not None:
        # Train and test the model
        train_and_test_model(data, labels, config)


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    main()