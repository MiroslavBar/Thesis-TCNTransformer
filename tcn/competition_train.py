import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Add, Dropout
from sklearn.preprocessing import RobustScaler

from tcn import compiled_tcn
from transformer import transformer_encoder_block

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_and_combine_data(data_dir, subjects=range(1, 10)):
    """
    Load and combine data from multiple subjects in the BCI IV 2a dataset.

    Args:
        data_dir: Directory containing the dataset files
        subjects: List of subject numbers to include

    Returns:
        X: Combined data from all subjects [n_samples, n_channels, n_times]
        y: Combined labels from all subjects [n_samples]
    """
    X_all = []
    y_all = []

    for subject in subjects:
        # Load data and labels for this subject
        data_file = os.path.join(data_dir, f'A0{subject}E_data.npy')
        label_file = os.path.join(data_dir, f'A0{subject}E_label.npy')

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


def preprocess_data(X, y, test_size=0.2, val_size=0.1):
    """
    Preprocess the BCI IV 2a dataset for TCN model.

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

    # Normalize the features using StandardScaler
    # We need to reshape to 2D for scaling (combine samples*timesteps, features)
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
        test_size=val_size / (1 - test_size),  # Adjust val_size to be relative to remaining data
        random_state=42,
        stratify=y_train
    )

    # Convert labels to integers instead of float32
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def build_and_train_model(X_train, X_val, y_train, y_val, num_classes, batch_size=64, epochs=100):
    """
    Build and train the TCN model with a Transformer encoder.

    Args:
        X_train: Training data
        X_val: Validation data
        y_train: Training labels (one-hot encoded)
        y_val: Validation labels (one-hot encoded)
        num_classes: Number of classes
        batch_size: Batch size for training
        epochs: Number of epochs to train

    Returns:
        Trained model and training history
    """
    # Get model dimensions
    timesteps, num_features = X_train.shape[1], X_train.shape[2]

    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(timesteps, num_features))

    # Define TCN model parameters
    tcn_layer = compiled_tcn(
        num_feat=num_features,
        num_classes=num_classes,
        nb_filters=32,  # Reduced from 256
        kernel_size=2,  # Increased kernel size
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
    interface_layer = Dense(128, activation='relu')(tcn_output)

    # Add transformer encoder block
    transformer_output = transformer_encoder_block(
        inputs=interface_layer,
        num_heads=4,  # Number of attention heads 4
        key_dim=128,  # Dimension of the key (same as TCN filters)
        dropout_rate=0.2  # Dropout rate for transformer 2
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

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30, #30
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15, #15
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

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
        model: Trained TCN model
        X_test: Test data
        y_test: Test labels (one-hot encoded)

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



def configure_gpu(use_gpu=True, memory_growth=True, mixed_precision=True):
    """
    Configures TensorFlow to use the GPU efficiently.

    Args:
        use_gpu (bool): Whether to use the GPU or force CPU usage.
        memory_growth (bool): Whether to enable memory growth for GPUs.
        mixed_precision (bool): Whether to enable mixed precision training.
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





def main():
    # Set parameters
    data_dir = "D:\mb\competition_dataset\\bci_iv_2a"  # Directory with the preprocessed data
    batch_size = 64
    epochs = 100

    # Load and combine data from all subjects
    X, y = load_and_combine_data(data_dir)

    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess_data(X, y)

    # Build and train model
    model, history = build_and_train_model(X_train, X_val, y_train, y_val, num_classes, batch_size, epochs)

    # Evaluate model
    acc, report, cm, y_pred = evaluate_model(model, X_test, y_test)

    # Print evaluation results
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(cm, np.arange(num_classes))

    # Save the model
    model.save('tcn_transformer_bci_iv_2a_model.h5')
    print("Model saved to tcn_transformer_bci_iv_2a_model.h5")



if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')

    configure_gpu(use_gpu=True, memory_growth=True, mixed_precision=True)

    main()