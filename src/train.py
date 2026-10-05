import os
import numpy as np
import tensorflow as tf


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    # Load processed training and validation data
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    print("Training data loaded.")
    print(f"Training data: {x_train.shape}")
    print(f"Validation data: {x_val.shape}")

    # Build ANN
    model = build_model()

    # Display model architecture
    model.summary()

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=128,
        verbose=1
    )

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save trained model
    model.save("models/model.h5")

    # Save training history
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    np.save("models/history.npy", history_data, allow_pickle=True)

    print("\nTraining completed successfully.")
    print("Model saved to: models/model.h5")
    print("Training history saved to: models/history.npy")


if __name__ == "__main__":
    main()