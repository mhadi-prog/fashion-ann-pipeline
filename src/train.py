"""
B3: Build a Sequential ANN (Flatten -> Dense(ReLU) -> Dropout -> Dense(10, Softmax)),
compile with Adam + sparse_categorical_crossentropy, train on data/processed/,
save the model to models/model.h5 and training history to models/history.csv.
All hyperparameters come from params.yaml.
"""
import os
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras import layers, models, optimizers

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def build_model(dense_units, dropout_rate, learning_rate):
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    os.makedirs(MODELS_DIR, exist_ok=True)

    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    x_val = np.load(os.path.join(PROCESSED_DIR, "x_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    model = build_model(
        dense_units=params["dense_units"],
        dropout_rate=params["dropout_rate"],
        learning_rate=params["learning_rate"],
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
    )

    model.save(os.path.join(MODELS_DIR, "model.h5"))
    pd.DataFrame(history.history).to_csv(
        os.path.join(MODELS_DIR, "history.csv"), index=False
    )

    print(f"Saved model to {MODELS_DIR}/model.h5")
    print(f"Saved training history to {MODELS_DIR}/history.csv")


if __name__ == "__main__":
    main()
