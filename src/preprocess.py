import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]

    # Load raw data
    x_train = np.load("data/raw/x_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    # Normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Create training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train
    )

    # Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test)
    np.save("data/processed/y_test.npy", y_test)

    print("Preprocessing completed.")
    print(f"Training data: {x_train.shape}")
    print(f"Validation data: {x_val.shape}")
    print(f"Test data: {x_test.shape}")
    print(f"Test size: {test_size}")
    print(f"Random state: {random_state}")


if __name__ == "__main__":
    main()