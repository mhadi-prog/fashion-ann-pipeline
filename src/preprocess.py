"""
B2: Load raw arrays, normalize pixel values to [0, 1], split a validation
set out of the training data, and save train/val/test arrays under
data/processed/. test_size and seed come from params.yaml.
"""
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    x_train_full = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train_full = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    x_train_full = x_train_full.astype("float32")
    x_test = x_test.astype("float32")
    x_train_full = (x_train_full / 127.5) - 1.0
    x_test = (x_test / 127.5) - 1.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train_full,
    )

    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"), x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"Saved processed data to {PROCESSED_DIR}/")
    print(f"  train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")

if __name__ == "__main__":
    main()
