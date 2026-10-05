import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def main():
    # Load processed test data
    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Load trained model
    model = tf.keras.models.load_model("models/model.h5")

    print("Test data loaded.")
    print(f"Test data: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")

    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    # Generate predictions
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print("\nEvaluation Results")
    print("------------------")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Accuracy score: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Create metrics directory
    os.makedirs("metrics", exist_ok=True)

    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    plt.xticks(range(10))
    plt.yticks(range(10))

    # Add values to each cell
    for i in range(10):
        for j in range(10):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig("metrics/confusion_matrix.png")
    plt.close()

    print("\nEvaluation completed successfully.")
    print("Metrics saved to: metrics/metrics.json")
    print("Confusion matrix saved to: metrics/confusion_matrix.png")


if __name__ == "__main__":
    main()