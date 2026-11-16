# 🎵 CodeAlpha Internship Project 3: Music Generation with AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Music21](https://img.shields.io/badge/Library-Music21-purple.svg)](https://web.mit.edu/music21/)
[![Google Colab](https://img.shields.io/badge/Colab-Executable%20Notebook-yellow.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Internship: CodeAlpha](https://img.shields.io/badge/Internship-CodeAlpha%20AI-red.svg)](https://codealpha.tech/)

An end-to-end Deep Learning system that composes novel, coherent polyphonic music using **Long Short-Term Memory (LSTM)** neural networks and **Music21**. Compatible with both local Windows environments and free **Google Colab (T4 GPU)**.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [System Architecture & Pipeline](#-system-architecture--pipeline)
3. [Curated Public-Domain Dataset](#-curated-public-domain-dataset)
4. [Deep Learning Model Architecture](#-deep-learning-model-architecture)
5. [Temperature-Based Probabilistic Sampling](#-temperature-based-probabilistic-sampling)
6. [Project Structure](#-project-structure)
7. [Getting Started on Google Colab (Recommended)](#-getting-started-on-google-colab-recommended)
8. [Getting Started Locally on Windows](#-getting-started-locally-on-windows)
9. [Automated Test Suite & Verification](#-automated-test-suite--verification)
10. [Sample Generated Outputs](#-sample-generated-outputs)
11. [Author & Acknowledgments](#-author--acknowledgments)

---

## 🎼 Project Overview

Music composition possesses both **temporal structure** (rhythm and melody over time) and **spatial structure** (harmony and polyphonic chords). This project constructs a deep sequential recurrent model capable of:
* Extracting symbolic musical elements (notes, chords, quantized durations, and rests) from Standard MIDI Files (SMF).
* Learning sequential harmonic transitions using stacked LSTM layers with Embedding, Dropout, and Batch Normalization.
* Generating novel musical sequences using **temperature-scaled stochastic sampling**.
* Synthesizing generated tokens back into standard `.mid` files and playable `.wav` audio.

---

## 🔄 System Architecture & Pipeline

```text
+-----------------------+
|  Raw MIDI Dataset     | (Public Domain: Bach, Beethoven, Chopin, Mozart, Pachelbel)
+-----------+-----------+
            |
            v
+-----------------------+
|  Music21 Preprocessor | -> Quantizes durations, extracts notes/chords/rests
+-----------+-----------+    Encodes into tokens (e.g. 'C4_0.5', 'C4.E4.G4_1.0')
            |
            v
+-----------------------+
|  Sliding Window (X,y) | -> Window length = 32, Token-to-Index vocabulary
+-----------+-----------+
            |
            v
+-----------------------+
|  Stacked LSTM Model   | -> Embedding(128) -> LSTM(256) -> Dropout(0.3) ->
+-----------+-----------+    BatchNorm -> LSTM(256) -> Dense(256) -> Softmax
            |
            v
+-----------------------+
| Temperature Sampler   | -> P(i) = exp(z_i / T) / sum(exp(z_j / T))
+-----------+-----------+    Controlled creativity (T in [0.2, 1.2])
            |
            +---------------------------+
            |                           |
            v                           v
+-----------------------+   +-----------------------+
|   Standard MIDI (.mid)|   | Playable Audio (.wav) |
+-----------------------+   +-----------------------+
```

---

## 📚 Curated Public-Domain Dataset

To guarantee legal, copyright-free usage, the repository includes curated classical compositions located in `data/raw_midi/`:
* **J.S. Bach**: Two-Part Invention No. 1 in C Major (BWV 772)
* **Frédéric Chopin**: Prelude in E Minor (Op. 28 No. 4)
* **Ludwig van Beethoven**: Symphony No. 9 "Ode to Joy" Piano Theme
* **W.A. Mozart**: 12 Variations on "Ah vous dirai-je, Maman" (KV 265)
* **Johann Pachelbel**: Canon in D Major Piano Progression

*All pieces are strictly in the public domain under CC0 / Open Access music archives.*

---

## 🧠 Deep Learning Model Architecture

| Layer | Type | Specifications | Function |
|---|---|---|---|
| 1 | **Input** | `(batch_size, 32)` | Receives integer token sequence |
| 2 | **Embedding** | `input_dim=Vocab, output_dim=128` | Maps discrete musical tokens into dense continuous harmonic space |
| 3 | **LSTM 1** | `256 units, return_sequences=True` | Extracts high-level melodic contour |
| 4 | **Dropout 1** | `rate=0.3` | Regularization against overfitting |
| 5 | **BatchNormalization 1** | Channels axis | Stabilizes activations across training |
| 6 | **LSTM 2** | `256 units, return_sequences=False` | Encodes long-term sequence dependencies |
| 7 | **Dropout 2** | `rate=0.3` | Regularization |
| 8 | **BatchNormalization 2** | Channels axis | Stabilizes activations |
| 9 | **Dense Hidden** | `256 units, activation='relu'` | Non-linear feature combination |
| 10 | **Dropout 3** | `rate=0.3` | Regularization |
| 11 | **Output Dense** | `units=Vocab, activation='softmax'` | Categorical probability distribution for next token |

---

## 🎛️ Temperature-Based Probabilistic Sampling

Rather than simply choosing the most likely token (greedy argmax, which causes repetitive loops), the generator applies **temperature scaling**:

$$\text{scaled\_logits}_i = \frac{\ln(p_i + \epsilon)}{T}, \quad P(i) = \frac{\exp(\text{scaled\_logits}_i)}{\sum_j \exp(\text{scaled\_logits}_j)}$$

* **$T = 0.3$ (Conservative):** Sharpens probability peaks. Highly predictable, tightly follows motif patterns.
* **$T = 0.8$ (Recommended / Sweet Spot):** Balanced creativity and harmonic coherence.
* **$T = 1.2$ (Adventurous):** Broadens probability distribution. More unexpected, jazz-like or experimental transitions.

---

## 📁 Project Structure

```text
CodeAlpha_MusicGenerationAI/
├── data/
│   ├── raw_midi/                # 5 curated public-domain classical pieces (.mid)
│   └── processed/               # Vocabulary mapping and preprocessed token sequences
├── models/
│   └── lstm_music_model_best.keras # Checkpoint weights (saved during training)
├── output/
│   ├── generated_music.mid      # Synthesized AI MIDI file
│   ├── generated_music.wav      # Playable WAV audio file
│   └── training_loss_curve.png  # Training loss/accuracy convergence plot
├── notebooks/
│   └── Music_Generation_AI_Colab.ipynb # Complete, beginner-friendly Colab notebook
├── src/
│   ├── __init__.py
│   ├── config.py                # Hyperparameters, sequence length, and paths
│   ├── midi_dataset.py          # Dataset discovery and download utilities
│   ├── preprocess.py            # Music21 tokenizer & sequence builder
│   ├── model.py                 # Keras stacked LSTM architecture
│   ├── train.py                 # Training pipeline with callbacks & resume support
│   ├── generate.py              # Temperature-based generation engine
│   ├── postprocess.py           # MIDI writer & acoustic WAV synthesizer
│   └── utils.py                 # Diagnostics, audio display, and helpers
├── tests/
│   ├── test_midi_io.py          # Unit tests for MIDI parsing & writing
│   ├── test_preprocessing.py    # Unit tests for vocabulary and windowing
│   ├── test_generation.py       # Unit tests for temperature sampling
│   └── run_all_tests.py         # Complete automated test runner
├── requirements.txt             # Dependency specifications
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started on Google Colab (Recommended)

Google Colab allows you to train the deep learning model on a **free cloud GPU (NVIDIA T4)** with zero local installation:

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `notebooks/Music_Generation_AI_Colab.ipynb`.
3. Enable GPU acceleration:
   > **Runtime** ➔ **Change runtime type** ➔ **Hardware accelerator** ➔ **T4 GPU** ➔ **Save**.
4. Click **Runtime** ➔ **Run all** (or run each cell sequentially).
5. The notebook will automatically:
   * Install `music21` and `fluidsynth`.
   * Extract notes and train the LSTM model.
   * Generate an original song with temperature sampling.
   * Provide an instant **Download Button** for the `.mid` file and an **interactive audio player** for the `.wav` audio.

---

## 💻 Getting Started Locally on Windows

### 1. Prerequisites
* Python 3.10 to 3.12 installed on your machine.
* Standard Command Prompt or PowerShell.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Step-by-Step CLI Execution

**Step A: Preprocess the MIDI dataset**
```bash
python -m src.preprocess
```

**Step B: Train the LSTM Model**
```bash
python -m src.train --epochs 35 --batch-size 32
```
*Tip: If you interrupt training, running `python -m src.train` will automatically resume from the latest saved checkpoint.*

**Step C: Generate Original Music**
```bash
python -m src.generate --temperature 0.8 --length 150
```
Outputs are written to:
* `output/generated_music.mid`
* `output/generated_music.wav`

---

## 🧪 Automated Test Suite & Verification

The project includes a comprehensive automated unit test suite covering:
* MIDI header format integrity (SMF `MThd` and `MTrk` chunks).
* Music21 tokenization and quantized duration preservation.
* Invertibility of vocabulary mappings ($t = \text{idx\_to\_token}[\text{token\_to\_idx}[t]]$).
* Sliding window dimensions and target label alignment.
* Mathematical correctness of temperature sampling across temperature regimes.
* End-to-end generation of valid `.mid` and `.wav` files.

To run the test suite:
```bash
python -m tests.run_all_tests
```
**Result:**
```text
=================================================================
Running Test Suite: CodeAlpha MusicGenerationAI
=================================================================
test_end_to_end_generation_flow ... ok
test_low_temperature_peaks_highest_probability ... ok
test_temperature_sampling_bounds ... ok
test_midi_header_validity ... ok
test_midi_parsing_and_token_extraction ... ok
test_raw_midi_files_exist ... ok
test_token_to_midi_creation ... ok
test_wav_synthesis ... ok
test_quantize_duration ... ok
test_sequence_preparation_dimensions ... ok
test_vocabulary_mapping_invertibility ... ok
----------------------------------------------------------------------
Ran 11 tests in 5.915s
OK
>> ALL UNIT TESTS PASSED SUCCESSFULLY! <<
```

---

## 🎧 Sample Generated Outputs

Sample generated outputs from the model are stored in the `output/` directory:
* [`output/generated_music.mid`](output/generated_music.mid): Standard MIDI file playable in Windows Media Player, VLC, GarageBand, FL Studio, or any DAW.
* [`output/generated_music.wav`](output/generated_music.wav): Synthesized 44.1 kHz acoustic piano audio rendition.

---

## 👤 Author & Acknowledgments

* **Intern:** CodeAlpha AI Intern
* **Organization:** [CodeAlpha](https://codealpha.tech/)
* **Project:** Internship Project 3 — Music Generation with AI
* **Tools Used:** Python, TensorFlow, Keras, Music21, PrettyMIDI, FluidSynth, Google Colab, Google Antigravity.
