"""
CodeAlpha Internship Project 3: Music Generation with AI
Configuration Module
Defines all global parameters, paths, and hyperparameters for preprocessing,
training, and generation.
"""

import os
from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
for directory in [RAW_MIDI_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Preprocessing Configuration
SEQUENCE_LENGTH = 32         # Number of notes/chords in each input sequence window
STEP_SIZE = 1                # Stride step for sliding window
QUANTIZATION_BEATS = 0.25    # Round durations to nearest 16th note (0.25 quarter length)
MIN_NOTE_FREQUENCY = 1       # Filter rare tokens if dataset is large (1 keeps all)

# Vocabulary and Data File Paths
NOTES_PICKLE_PATH = PROCESSED_DATA_DIR / "notes_data.pkl"
VOCAB_MAPPING_PATH = PROCESSED_DATA_DIR / "vocab_mapping.json"
TRAIN_HISTORY_CSV = MODELS_DIR / "training_history.csv"
LOSS_PLOT_PATH = OUTPUT_DIR / "training_loss_curve.png"

# Model Architecture Hyperparameters
EMBEDDING_DIM = 128
LSTM_UNITS = [256, 256]      # 2 stacked LSTM layers
DROPOUT_RATE = 0.3
DENSE_UNITS = 256
LEARNING_RATE = 0.001

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
VALIDATION_SPLIT = 0.1
PATIENCE_EARLY_STOPPING = 10
PATIENCE_REDUCE_LR = 4
CHECKPOINT_MODEL_PATH = MODELS_DIR / "lstm_music_model_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "lstm_music_model_final.keras"

# Generation Hyperparameters
GENERATION_LENGTH = 150      # Number of notes/chords to generate
DEFAULT_TEMPERATURE = 0.8    # 0.2 = conservative/repetitive, 0.8 = balanced, 1.2 = adventurous
OUTPUT_MIDI_PATH = OUTPUT_DIR / "generated_music.mid"
OUTPUT_WAV_PATH = OUTPUT_DIR / "generated_music.wav"
DEFAULT_TEMPO_BPM = 100
