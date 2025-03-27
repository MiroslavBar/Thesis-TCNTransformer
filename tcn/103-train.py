import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Add, Dropout


import yaml
import sys

# Import required modules from TCN and Transformer implementations
# Assuming these are available in your project
from tcn import compiled_tcn
from transformer import transformer_encoder_block

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# torch.manual_seed(42)


def configure_gpu(use_gpu=True, memory_growth=True, mixed_precision=True):
    """
    Configures TensorFlow to use the GPU efficiently.
    """
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode
        print("GPU disabled. Running on CPU.")
        return

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f" - {gpu.name}")

        if memory_growth:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth enabled")
            except Exception as e:
                print(f"⚠️ Error setting memory growth: {e}")

        if mixed_precision:
            try:
                from tensorflow.keras.mixed_precision import set_global_policy
                set_global_policy('mixed_float16')
                print("✅ Mixed precision training enabled (float16 for faster computation)")
            except Exception as e:
                print(f"⚠️ Error enabling mixed precision: {e}")
    else:
        print("⚠️ No GPU found. Running on CPU.")


# Load CSV data functions from 103-train.py
def load_csv_data(file_path):
    return np.loadtxt(file_path, delimiter=',')


def convert_label(label):
    # return two_class_labels(label)
    return four_class_labels(label)

def two_class_labels(label):
    # 2class movement, relax
    if label in [2, 3, 5, 6, 8, 9, 11, 12]: return 0
    if label in [1, 4, 7, 10]: return 1
    return None


def four_class_labels(label):
    # 4class right hand, left hand, both feet, relax
    if label in [2, 5]: return 0
    if label in [3, 6]: return 1
    if label in [9, 12]: return 2
    if label in [1, 4, 7, 10]:
        import random
        if random.randint(1, 4) == 4:
            return 3
    return None  # Exclude other labels


def get_subjects(csv_dir):
    """Extracts unique subject IDs from filenames."""
    subject_ids = set()
    for file in os.listdir(csv_dir):
        if 'SIG' in file:
            subject_id = file.split('_')[1]  # Extracts "001" from "SUB_001_SIG_01"
            subject_ids.add(subject_id)
    # return ("001", "002")
    return sorted(subject_ids)


def preprocess_subject(csv_dir, subject_id, num_samples=700):
    """Preprocess and save subject data to .npy files."""
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


def prepare_global_dataset(csv_dir, num_samples=700):
    """Combine all subjects' data and prepare a global dataset."""
    all_data, all_labels = [], []

    subjects = get_subjects(csv_dir)
    for subject_id in subjects:
        data, labels = preprocess_subject(csv_dir, subject_id, num_samples)
        if data is not None and labels is not None:
            all_data.append(data)
            all_labels.append(labels)

    if not all_data:
        print("No valid data found for global evaluation.")
        return None, None

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_data, all_labels


def preprocess_data_for_tf(X, y, test_size=0.2, val_size=0.1):
    """
    Preprocess the EEG dataset for TCN-Transformer model.

    Args:
        X: Data in shape [n_samples, n_channels, n_times]
        y: Labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes
    """
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
        test_size=val_size / (1 - test_size),  # Adjust val_size to be relative to remaining data
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


def build_tcn_transformer_model(timesteps, num_features, num_classes):
    """
    Build the TCN model with a Transformer encoder for EEG classification.
    """
    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(timesteps, num_features))

    # Define TCN model parameters
    tcn_layer = compiled_tcn(
        num_feat=num_features,
        num_classes=num_classes,
        nb_filters=64,
        kernel_size=2,
        dilations=[1, 2, 4, 8],
        nb_stacks=1,
        max_len=timesteps,
        use_skip_connections=True,
        return_sequences=True,
        regression=False,
        dropout_rate=0.3,
        activation='relu',
        opt='adam',
        lr=0.0001
    )

    # Extract the TCN layer from the compiled model
    tcn_output = tcn_layer.layers[1](input_layer)

    # Add an interface layer to match dimensions
    interface_layer = tf.keras.layers.Dense(128, activation='relu')(tcn_output)

    # Add transformer encoder block
    transformer_output = transformer_encoder_block(
        inputs=interface_layer,
        num_heads=4,
        key_dim=128,
        dropout_rate=0.2
    )

    # Global Average Pooling for sequence aggregation
    gap_output = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)

    # Final classification layer
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(gap_output)

    # Create the combined model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Model summary
    model.summary()

    return model


def train_model(model, X_train, X_val, y_train, y_val, batch_size=64, epochs=100):
    """
    Train the TCN-Transformer model with class balancing and callbacks.
    """
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.0001
    )

    # Add learning rate scheduler for better convergence
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * (0.85 ** (epoch // 5))
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, lr_scheduler],
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
        model: Trained TCN-Transformer model
        X_test: Test data
        y_test: Test labels

    Returns:
        Test accuracy, classification report, and confusion matrix
    """
    # Predict classes
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    # Generate classification report
    report = classification_report(y_true, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return acc, report, cm, y_pred


def plot_training_history(history):
    """
    Plot training and validation loss/accuracy.

    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('tcn_transformer_training_history.png')
    plt.show()


def plot_confusion_matrix(cm, classes):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        classes: Class labels
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('tcn_transformer_confusion_matrix.png')
    plt.show()


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found.")
        sys.exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit()


def train_global_model(config):
    """Train a global model using combined data from all subjects."""
    csv_dir = config["csv_files"]
    num_samples = config.get("num_samples", 700)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 100)

    # Load and combine data from all subjects
    print("Loading global dataset...")
    X, y = prepare_global_dataset(csv_dir, num_samples=num_samples)

    if X is None or y is None:
        print("No valid data found for global evaluation.")
        return

    # Preprocess data for TensorFlow model
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess_data_for_tf(X, y)

    # Get model dimensions
    timesteps, num_features = X_train.shape[1], X_train.shape[2]

    # Build model
    print("Building TCN-Transformer model...")
    model = build_tcn_transformer_model(timesteps, num_features, num_classes)

    # Train model
    print("Training model...")
    history = train_model(model, X_train, X_val, y_train, y_val, batch_size, epochs)

    # Evaluate model
    print("Evaluating model...")
    acc, report, cm, y_pred = evaluate_model(model, X_test, y_test)

    # Print evaluation results
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(cm, np.arange(num_classes))

    # Save the model
    model.save('tcn_transformer_eeg_model.h5')
    print("Model saved to tcn_transformer_eeg_model.h5")


def train_per_subject(config):
    """Train a separate model for each subject."""
    csv_dir = config["csv_files"]
    num_samples = config.get("num_samples", 700)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 100)

    subjects = get_subjects(csv_dir)
    results = {}

    for subject_id in subjects:
        print(f"\nProcessing subject {subject_id}...\n")

        # Load subject data
        X, y = preprocess_subject(csv_dir, subject_id, num_samples=num_samples)

        if X is None or y is None:
            print(f"No valid data found for subject {subject_id}.")
            continue

        # Preprocess data for TensorFlow model
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess_data_for_tf(X, y)

        # Get model dimensions
        timesteps, num_features = X_train.shape[1], X_train.shape[2]

        # Build model
        print(f"Building TCN-Transformer model for subject {subject_id}...")
        model = build_tcn_transformer_model(timesteps, num_features, num_classes)

        # Train model
        print(f"Training model for subject {subject_id}...")
        history = train_model(model, X_train, X_val, y_train, y_val, batch_size, epochs)

        # Evaluate model
        print(f"Evaluating model for subject {subject_id}...")
        acc, report, cm, y_pred = evaluate_model(model, X_test, y_test)

        # Print evaluation results
        print(f"\nSubject {subject_id} Test Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(report)

        # Save results
        results[subject_id] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': cm
        }

        # Save the model
        model.save(f'tcn_transformer_eeg_subject_{subject_id}_model.h5')
        print(f"Model saved to tcn_transformer_eeg_subject_{subject_id}_model.h5")

    # Print summary of results
    print("\n===== Summary of Results =====")
    for subject_id, result in results.items():
        print(f"Subject {subject_id}: Accuracy = {result['accuracy']:.4f}")


def main():
    # Configure GPU
    configure_gpu(use_gpu=True, memory_growth=False, mixed_precision=False)

    # Load configuration
    config_path = "103-config.yaml"
    if not os.path.exists(config_path):
        # Create a default config if not exists
        config = {
            "csv_files": "./data",  # Update this with your CSV directory
            "num_samples": 700,
            "batch_size": 32,
            "epochs": 100,
            "strategy": "global"  # "global" or "per_subject"
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Created default config at {config_path}. Please update it with your settings.")
        return

    config = load_config(config_path)

    # Train model based on strategy
    strategy = config.get("strategy", "global")
    if strategy == "global":
        train_global_model(config)
    elif strategy == "per_subject" or strategy == "per_person":
        train_per_subject(config)
    else:
        print(f"Invalid strategy: {strategy}. Use 'global' or 'per_subject'.")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    # configure_gpu()
    main()