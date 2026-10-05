import os
import numpy as np
import tensorflow as tf


def main():
    # Load Fashion-MNIST dataset
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    
    os.makedirs("data/raw", exist_ok=True)

    
    np.save("data/raw/x_train.npy", x_train)
    np.save("data/raw/y_train.npy", y_train)
    np.save("data/raw/x_test.npy", x_test)
    np.save("data/raw/y_test.npy", y_test)

    print("Fashion-MNIST dataset prepared successfully.")
    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")


if __name__ == "__main__":
    main()