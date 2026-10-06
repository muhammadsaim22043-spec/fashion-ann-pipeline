import os
import numpy as np
import tensorflow as tf
import yaml


def build_model(hidden_units, dropout):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    epochs = train_params["epochs"]
    batch_size = train_params["batch_size"]
    hidden_units = train_params["hidden_units"]
    dropout = train_params["dropout"]

    # Load processed data
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    # Build model
    model = build_model(hidden_units, dropout)
    model.summary()

    # Train model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    # Save training history
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    np.save(
        "models/history.npy",
        history_data,
        allow_pickle=True
    )

    print("Training completed.")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden units: {hidden_units}")
    print(f"Dropout: {dropout}")


if __name__ == "__main__":
    main()