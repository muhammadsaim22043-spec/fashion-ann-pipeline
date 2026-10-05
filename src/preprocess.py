import os
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Load raw data
    x_train = np.load("data/raw/x_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Split training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    # Create processed data directory
    os.makedirs("data/processed", exist_ok=True)

    # Save processed datasets
    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)

    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)

    np.save("data/processed/x_test.npy", x_test)
    np.save("data/processed/y_test.npy", y_test)

    print("Preprocessing completed successfully.")
    print(f"Training data:   {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Validation data: {x_val.shape}")
    print(f"Validation labels: {y_val.shape}")
    print(f"Test data:       {x_test.shape}")
    print(f"Test labels:     {y_test.shape}")

    print(f"Training pixel range: {x_train.min():.1f} - {x_train.max():.1f}")
    print(f"Test pixel range:     {x_test.min():.1f} - {x_test.max():.1f}")


if __name__ == "__main__":
    main()