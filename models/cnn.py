import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv1D,
    MaxPooling1D,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def load_and_preprocess_data():
    print("Starting data loading and preprocessing...")
    print("Downloading dataset (if not already cached)...")
    try:
        dataset_path_object = kagglehub.Model.download("andradaolteanu/gtzan-dataset-music-genre-classification/tensorFlow2/feature-extraction")
        dataset_download_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
        csv_file_path = os.path.join(dataset_download_path, "Data", "features_3_sec.csv")
        print(f"Dataset downloaded to: {dataset_download_path}")
        print(f"Using CSV file: {csv_file_path}")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have Kaggle API credentials configured or try downloading manually.")
        print("Attempting to use a potential local cache path for development...")
        home_dir = os.path.expanduser("~")
        potential_cache_path = os.path.join(home_dir, ".cache", "kagglehub", "datasets", "andradaolteanu", "gtzan-dataset-music-genre-classification", "versions", "1", "Data", "features_3_sec.csv")
        if os.path.exists(potential_cache_path):
            csv_file_path = potential_cache_path
            print(f"Using local cached CSV: {csv_file_path}")
        else:
            print(f"Local cache not found at {potential_cache_path}. Exiting.")
            return None, None, None, None, None, None


    df = pd.read_csv(csv_file_path)
    print("Dataset loaded successfully.")
    print(f"Initial dataset shape: {df.shape}")

    # Preprocessing steps from the notebook
    df = df.drop(labels=["filename", "length"], axis=1)
    print(f"Shape after dropping 'filename' and 'length': {df.shape}")

    X = df.drop(['label'], axis=1)
    y = df['label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    print(f"Labels encoded. Number of classes: {num_classes}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.array(X, dtype=float))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    # Reshape data for Conv1D
    X_train_reshaped = X_train[:, :, np.newaxis]
    X_test_reshaped = X_test[:, :, np.newaxis]
    input_shape = (X_train_reshaped.shape[1], 1)

    print("Data preprocessing complete.")
    print(f"X_train shape: {X_train_reshaped.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test_reshaped.shape}, y_test shape: {y_test.shape}")
    print(f"Input shape for model: {input_shape}")

    return X_train_reshaped, y_train, X_test_reshaped, y_test, input_shape, num_classes, encoder

def build_and_train_model(X_train, y_train, X_val, y_val, input_shape, num_classes):
    print("Building CNN model...")
    model = Sequential([
        Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        BatchNormalization(),

        Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        BatchNormalization(),

        Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        BatchNormalization(),
        Dropout(0.3), # Original notebook had 0.3

        Flatten(),
        Dense(64, activation='relu'), # Original notebook had 64 units
        Dropout(0.3), # Original notebook had 0.3
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001) # As per notebook
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001) # As per notebook

    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10, 
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )
    print("Model training complete.")
    return model, history

def main():
    # 1. Load and preprocess data
    (X_train_reshaped, y_train, X_test_reshaped, y_test,
     input_shape, num_classes, encoder) = load_and_preprocess_data()

    if X_train_reshaped is None: # Handle case where data loading failed
        return

    # 2. Build and train the model
    model, history = build_and_train_model(
        X_train_reshaped, y_train, X_test_reshaped, y_test,
        input_shape, num_classes
    )

    # 3. Evaluate the model (optional, but good practice)
    print("Evaluating model on test data...")
    loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # 4. Export the model
    model_filename = "music_genre_cnn_model.h5"
    model.save(model_filename)
    print(f"Trained model exported as {model_filename}")


if __name__ == "__main__":
    main()