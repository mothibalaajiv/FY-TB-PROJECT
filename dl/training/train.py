import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from config import PROJECT_ROOT
from dataset import prepare_dataset_directory, extract_features_and_log
from model import build_model, get_callbacks


def main():
    # 1. Prepare dataset
    permanent_extract_path = prepare_dataset_directory()

    # 2. Extract audio features
    X, y = extract_features_and_log(permanent_extract_path)

    if len(X) == 0:
        raise ValueError(
            "No audio samples were loaded. "
            "Please verify the folder names are 'tb' and 'notb'."
        )

    X = X[..., np.newaxis]

    print(f"\nFeatures matrices mapped: {X.shape}")
    print(f"Target labels mapped: {y.shape}")

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 4. Build CNN model
    input_shape = (
        X_train.shape[1],
        X_train.shape[2],
        1
    )

    model = build_model(input_shape)

    print("\n=== MODEL SUMMARY ===")
    model.summary()

    # 5. Train model
    EPOCHS = 30
    BATCH_SIZE = 32

    print("\n=== STARTING MODEL TRAINING ===")

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=get_callbacks(),
        verbose=1
    )

    # 6. Evaluate model
    test_loss, test_acc, test_recall, test_precision = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\n=== TEST EVALUATION SUMMARY ===")
    print(f"Accuracy:  {test_acc * 100:.2f}%")
    print(f"Recall:    {test_recall * 100:.2f}%")
    print(f"Precision: {test_precision * 100:.2f}%")

    # 7. Save model
    model_directory = PROJECT_ROOT / "backend" / "models"
    model_directory.mkdir(parents=True, exist_ok=True)

    output_model_path = (
        model_directory / "tb_cough_base_model.h5"
    )

    model.save(output_model_path)

    print(
        f"\nModel successfully saved to:\n"
        f" -> {output_model_path}"
    )

    # 8. Plot training results
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(
        history.history["accuracy"],
        label="Train Accuracy"
    )
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(
        history.history["loss"],
        label="Train Loss"
    )
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()