"""
CodeAlpha Internship Project 3: Music Generation with AI
Training Pipeline Module
Executes model training with checkpointing, learning rate decay,
early stopping, resume-training support, and loss curve visualization.
"""

import os
import argparse
from pathlib import Path
from typing import Optional

try:
    from . import config
    from . import preprocess
    from . import model as music_model
except ImportError:
    import config
    import preprocess
    import model as music_model

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
        CSVLogger
    )
    import matplotlib.pyplot as plt
    HAS_TRAIN_LIBS = True
except ImportError:
    HAS_TRAIN_LIBS = False


def plot_and_save_history(history, output_path: Path = None):
    """Plot training and validation loss curves and save as image."""
    out_file = Path(output_path or config.LOSS_PLOT_PATH)
    os.makedirs(out_file.parent, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    plt.title("LSTM Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="Train Acc", color="#2ca02c", linewidth=2)
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Acc", color="#d62728", linewidth=2)
    plt.title("Model Prediction Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Loss curves saved to {out_file}")


def train(
    epochs: int = None,
    batch_size: int = None,
    resume_checkpoint: Optional[str] = None
):
    """
    Main training routine.
    Loads preprocessed sequences, builds the model, trains with callbacks,
    and saves weights and plots.
    """
    if not HAS_TRAIN_LIBS:
        raise ImportError(
            "TensorFlow and NumPy are required for training. "
            "Please run training in Google Colab (with free GPU acceleration) "
            "or install requirements locally."
        )

    # 1. Load or run preprocessing
    try:
        data = preprocess.load_preprocessed_data()
    except FileNotFoundError:
        print("Preprocessed data not found. Running preprocessing pipeline now...")
        preprocess.run_preprocessing_pipeline()
        data = preprocess.load_preprocessed_data()

    X = np.array(data["network_input"], dtype=np.int32)
    y = np.array(data["network_output"], dtype=np.int32)
    vocab_size = data["vocab_size"]

    print(f"Training Dataset Prepared:")
    print(f"  Input Shape X: {X.shape}")
    print(f"  Target Shape y: {y.shape}")
    print(f"  Vocabulary Size: {vocab_size}")

    num_epochs = epochs or config.EPOCHS
    b_size = batch_size or config.BATCH_SIZE
    ckpt_path = str(config.CHECKPOINT_MODEL_PATH)
    final_path = str(config.FINAL_MODEL_PATH)
    csv_log_path = str(config.TRAIN_HISTORY_CSV)

    # 2. Build or resume model
    model = music_model.build_music_lstm_model(vocab_size=vocab_size)
    
    # Check for resume weights
    resume_file = resume_checkpoint or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume_file and os.path.exists(resume_file):
        print(f"Resuming training: loading weights from {resume_file}...")
        try:
            model.load_weights(resume_file)
            print("  -> Checkpoint successfully loaded.")
        except Exception as e:
            print(f"  -> Warning: Could not load weights ({e}). Starting fresh training.")

    # 3. Setup Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss" if config.VALIDATION_SPLIT > 0 else "loss",
            verbose=1,
            save_best_only=True,
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss" if config.VALIDATION_SPLIT > 0 else "loss",
            patience=config.PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss" if config.VALIDATION_SPLIT > 0 else "loss",
            factor=0.5,
            patience=config.PATIENCE_REDUCE_LR,
            min_lr=1e-5,
            verbose=1
        ),
        CSVLogger(csv_log_path, append=True)
    ]

    # 4. Train Model
    print(f"\nStarting LSTM Training for {num_epochs} epochs (Batch size: {b_size})...")
    history = model.fit(
        X, y,
        epochs=num_epochs,
        batch_size=b_size,
        validation_split=config.VALIDATION_SPLIT if len(X) > 20 else 0.0,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Save Final Model
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    print(f"Best checkpoint saved to: {ckpt_path}")

    # 6. Plot Loss Curve
    plot_and_save_history(history)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Music Generation LSTM")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint weights to resume from")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, resume_checkpoint=args.resume)
