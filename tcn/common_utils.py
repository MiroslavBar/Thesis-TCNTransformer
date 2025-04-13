import numpy as np
import os
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import yaml
import sys

from tcn import compiled_tcn
from tcn.transformer import transformer_encoder_block

def load_config(config_path='config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def configure_gpu(config: Dict[str, Any]) -> None:
    """
    Configure TensorFlow GPU settings based on the provided configuration.

    Handles GPU selection, memory growth, and mixed precision training.

    Args:
        config: Configuration dictionary
    """
    gpu_config = config.get('gpu', {})
    use_gpu = gpu_config.get('enable', True)
    memory_growth = gpu_config.get('memory_growth', False)
    mixed_precision = gpu_config.get('mixed_precision', False)

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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


def build_tcn_transformer_model(timesteps: int, num_features: int, num_classes: int, config: Dict[str, Any]) -> tf.keras.Model:
    """
    Build a hybrid TCN-Transformer model for EEG classification.

    Combines Temporal Convolutional Network (TCN) with Transformer encoder
    for sequence classification.

    Args:
        timesteps: Number of time steps in input sequence
        num_features: Number of features per time step
        num_classes: Number of output classes
        config: Model configuration dictionary

    Returns:
        tf.keras.Model: Compiled TCN-Transformer classification model
    """
    tcn_config = config['model']['tcn']
    transformer_config = config['model']['transformer']

    # Input layer
    input_layer = tf.keras.layers.Input(shape=(timesteps, num_features))

    # Define TCN model parameters
    tcn_layer = compiled_tcn(
        num_feat=num_features,
        num_classes=num_classes,
        nb_filters=tcn_config['num_filters'],
        kernel_size=tcn_config['kernel_size'],
        dilations=tcn_config['dilations'],
        nb_stacks=tcn_config['num_stacks'],
        max_len=timesteps,
        use_skip_connections=tcn_config['use_skip_connections'],
        return_sequences=True,
        regression=False,
        dropout_rate=tcn_config['dropout_rate'],
        activation=tcn_config['activation'],
        opt='adam',
        lr=config['training']['learning_rate']
    )

    # Extract the TCN layer from the compiled model
    tcn_output = tcn_layer.layers[1](input_layer)

    # Add an interface layer to match dimensions
    interface_layer = tf.keras.layers.Dense(transformer_config['key_dim'], activation='relu')(tcn_output)

    # Add transformer encoder block
    transformer_output = transformer_encoder_block(
        inputs=interface_layer,
        num_heads=transformer_config['num_heads'],
        key_dim=transformer_config['key_dim'],
        dropout_rate=transformer_config['dropout_rate']
    )

    # Global Average Pooling for sequence aggregation
    gap_output = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)

    # Final classification layer
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(gap_output)

    # Create the combined model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> tf.keras.callbacks.History:
    """
    Train the model with advanced training configurations.

    Implements early stopping, learning rate scheduling, and adaptive
    learning rate reduction.

    Args:
        model: Model to train
        X_train: Training data
        X_val: Validation data
        y_train: Training labels
        y_val: Validation labels
        config: Training configuration dictionary

    Returns:
        History of model training
    """
    training_args = config['training']

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=training_args['early_stopping_patience'],
        restore_best_weights=True
    )

    # Learning rate scheduler for better convergence
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: training_args['learning_rate'] * (0.85 ** (epoch // 5))
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=training_args['reduce_lr_patience'],
        min_lr=training_args['min_learning_rate']
    )

    history = model.fit(
        X_train, y_train,
        batch_size=training_args['batch_size'],
        epochs=training_args['epochs'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, lr_scheduler],
        verbose=1
    )

    return history


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[float, str, np.ndarray, np.ndarray]:
    """
    Comprehensively evaluate the trained model.

    Generates predictions, calculates accuracy,
    classification report, and confusion matrix.

    Args:
        model: Trained classification model
        X_test: Test data
        y_test: Test labels

    Returns:
        Tuple containing:
        - Accuracy score
        - Classification report
        - Confusion matrix
        - Predicted labels
    """
    # Predict classes
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    # Generate classification report
    report = classification_report(y_true, y_pred, digits=4)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return acc, report, cm, y_pred


def plot_training_history(
    history: tf.keras.callbacks.History,
    config: Dict[str, Any]
) -> None:
    """
    Visualize model training performance.

    Plots training and validation loss and accuracy across epochs.

    Args:
        history: Training history object
        config: Configuration dictionary
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(config['logging']['training_history_plot'])
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: np.ndarray[Any, np.dtype[np.signedinteger]],
    config: Dict[str, Any]
) -> None:
    """
    Visualize model's classification performance.

    Creates a color-coded confusion matrix with numeric annotations.

    Args:
        cm: Confusion matrix
        classes: List of class labels
        config: Configuration dictionary for saving plot
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
    plt.savefig(config['logging']['confusion_matrix_plot'])
    plt.show()
