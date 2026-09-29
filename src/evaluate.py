"""
B4: Load the trained model and processed test set, compute test loss and
accuracy, generate a confusion matrix image, and write metrics to
metrics.json at the project root.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/model.h5"
METRICS_PATH = "metrics.json"
CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    x_test = np.load(f"{PROCESSED_DIR}/x_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

    model = load_model(MODEL_PATH)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Fashion-MNIST ANN")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
