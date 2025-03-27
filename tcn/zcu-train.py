import sys
import os
import numpy as np
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import required modules from TCN and Transformer implementations
from tcn import compiled_tcn
from transformer import transformer_encoder_block

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


def transform_labels(labels):
    """
    Transform labels to match the format expected by the model:
    Original labels: 2 -> 0, 5 -> 1, 6 -> 2
    """

    transformed_labels = labels.copy()
    transformed_labels[transformed_labels == 2] = 0
    transformed_labels[transformed_labels == 5] = 1
    transformed_labels[transformed_labels == 6] = 2
    return transformed_labels


def load_npy_files(data_file, labels_file):
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


def get_data_path(config):
    if config['network_args']['num_classes'] == 2:
        return config['data_file_binary'], config['labels_file_binary']
    elif config['network_args']['num_classes'] == 3:
        return config['data_file_multiclass'], config['labels_file_multiclass']
    else:
        print("Invalid number of classes in configuration.")
        return None, None


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
        nb_filters=64,  # Reduced from 256
        kernel_size=3,  # Increased kernel size
        dilations=[1, 2, 4, 8],  # Reduced dilation complexity
        nb_stacks=1,  # Reduced stacks
        max_len=timesteps,
        use_skip_connections=True,
        return_sequences=True,
        regression=False,
        dropout_rate=0.2,  # Reduced dropout
        activation='relu',
        opt='adam',
        lr=0.001
    )

    # Extract the TCN layer from the compiled model
    tcn_output = tcn_layer.layers[1](input_layer)

    # Add an interface layer to match dimensions
    interface_layer = tf.keras.layers.Dense(64, activation='relu')(tcn_output)

    # Add transformer encoder block
    transformer_output = transformer_encoder_block(
        inputs=interface_layer,
        num_heads=4,
        key_dim=64,
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
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Model summary
    model.summary()

    return model


def train_model(model, X_train, X_val, y_train, y_val, batch_size=32, epochs=100):
    """
    Train the TCN-Transformer model with class balancing and callbacks.
    """
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=40,
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


def plot_training_history(history, filename='training_history.png'):
    """
    Plot training and validation loss/accuracy.
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
    plt.savefig(filename)
    plt.show()


def plot_confusion_matrix(cm, classes, filename='confusion_matrix.png'):
    """
    Plot confusion matrix.
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
    plt.savefig(filename)
    plt.show()


def train_and_test_model(data, labels, config):
    """
    Train and test the TCN-Transformer model on the entire dataset or per-person
    """
    # Transform labels to match the expected format

    # Split data by training strategy
    # Concatenate all data and train a global model
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = transform_labels(labels)

    # Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess_data_for_tf(data, labels)

    # Get model dimensions
    timesteps, num_features = X_train.shape[1], X_train.shape[2]

    # Build and train the model
    model = build_tcn_transformer_model(timesteps, num_features, num_classes)

    # Train the model
    history = train_model(model, X_train, X_val, y_train, y_val,
                          batch_size=config.get('batch_size', 32),
                          epochs=config.get('epochs', 100))

    # Evaluate the model
    acc, report, cm, y_pred = evaluate_model(model, X_test, y_test)

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Plot results
    plot_training_history(history, 'global_training_history.png')
    plot_confusion_matrix(cm, np.arange(num_classes), 'global_confusion_matrix.png')

    # Save the model
    model.save('tcn_transformer_eeg_global_model.h5')
    print("Model saved to tcn_transformer_eeg_global_model.h5")



def main():
    # Configure GPU
    configure_gpu(use_gpu=True, memory_growth=True, mixed_precision=True)

    # Load configuration
    config = load_config("zcu-config.yaml")

    # Get the appropriate data files based on number of classes
    data_file, labels_file = get_data_path(config)

    if data_file is None or labels_file is None:
        print("Invalid data paths in configuration.")
        sys.exit()

    # Load data
    data, labels = load_npy_files(data_file, labels_file)

    if data is not None and labels is not None:
        # Train and test the model
        train_and_test_model(data, labels, config)


if __name__ == "__main__":
    # Use TkAgg backend for matplotlib to ensure plot display
    import matplotlib

    matplotlib.use('TkAgg')
    main()