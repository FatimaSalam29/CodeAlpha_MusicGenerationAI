"""
CodeAlpha Internship Project 3: Music Generation with AI
Music Generation Engine
Generates novel musical sequences from learned LSTM representations
using temperature-scaled probabilistic sampling, and writes MIDI and WAV outputs.
"""

import os
import math
import random
import argparse
from pathlib import Path
from typing import List, Optional

try:
    from . import config
    from . import preprocess
    from . import postprocess
    from . import model as music_model
except ImportError:
    import config
    import preprocess
    import postprocess
    import model as music_model

try:
    import numpy as np
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    np = None


def sample_with_temperature(probabilities, temperature: float = 0.8) -> int:
    """
    Apply temperature scaling to the output probability distribution and sample.
    
    Args:
        probabilities: 1D sequence/array of predicted token probabilities.
        temperature: Float > 0.
            - Low (0.2 - 0.5): Conservative, structured, repetitive.
            - Medium (0.7 - 0.85): Sweet spot of creativity and harmonic structure.
            - High (1.0 - 1.5): Experimental, varied, chaotic.
    """
    prob_list = [float(p) for p in probabilities]
    
    if temperature <= 0.01:
        # Greedily pick top token
        return max(range(len(prob_list)), key=lambda i: prob_list[i])

    # Temperature scaled softmax
    try:
        scaled_exps = []
        for p in prob_list:
            clipped = max(1e-10, min(1.0, p))
            scaled_exps.append(math.exp(math.log(clipped) / temperature))
        
        total = sum(scaled_exps)
        if total <= 0:
            return 0
        scaled_probs = [x / total for x in scaled_exps]
        
        # Sample using random.choices
        sampled_indices = random.choices(range(len(prob_list)), weights=scaled_probs, k=1)
        return int(sampled_indices[0])
    except Exception:
        # Fallback to argmax if numerical issue occurs
        return max(range(len(prob_list)), key=lambda i: prob_list[i])


def generate_music_sequence(
    model,
    seed_sequence: List[int],
    idx_to_token: dict,
    generation_length: int = None,
    temperature: float = None
) -> List[str]:
    """
    Iteratively generate a sequence of music tokens using sliding window and temperature sampling.
    """
    gen_len = generation_length or config.GENERATION_LENGTH
    temp = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
    pattern = list(seed_sequence)
    seq_length = len(seed_sequence)

    generated_tokens = []
    print(f"\nGenerating {gen_len} musical tokens (Temperature = {temp})...")

    for step in range(gen_len):
        # Prepare input window of shape (1, seq_length)
        input_window = pattern[-seq_length:]
        
        if HAS_TF and isinstance(model, tf.keras.Model):
            input_tensor = np.array([input_window], dtype=np.int32)
            prediction = model.predict(input_tensor, verbose=0)[0]
        else:
            # Model fallback for testing or lightweight environments
            prediction = model.predict([input_window], verbose=0)[0]

        # Sample next token index using temperature
        next_idx = sample_with_temperature(prediction, temperature=temp)
        next_token = idx_to_token.get(next_idx) or idx_to_token.get(str(next_idx), "C4_0.5")

        generated_tokens.append(next_token)
        pattern.append(next_idx)

        if (step + 1) % 50 == 0 or (step + 1) == gen_len:
            print(f"  -> Generated {step + 1}/{gen_len} tokens...")

    return generated_tokens


def generate(
    temperature: float = 0.8,
    length: int = 120,
    model_path: Optional[str] = None,
    output_midi: Optional[str] = None,
    output_wav: Optional[str] = None,
    tempo_bpm: int = 100
) -> dict:
    """
    High-level generation pipeline:
      1. Loads preprocessed vocabulary and training sequences.
      2. Loads or constructs the LSTM model.
      3. Picks a random seed from existing musical sequences.
      4. Samples a new sequence using temperature.
      5. Synthesizes MIDI (.mid) and WAV (.wav).
    """
    # 1. Load data
    try:
        data = preprocess.load_preprocessed_data()
    except FileNotFoundError:
        print("Preprocessed data not found. Running preprocessing...")
        preprocess.run_preprocessing_pipeline()
        data = preprocess.load_preprocessed_data()

    vocab_size = data["vocab_size"]
    idx_to_token = data["idx_to_token"]
    network_input = data["network_input"]

    # 2. Setup Model
    weights = model_path or (str(config.CHECKPOINT_MODEL_PATH) if config.CHECKPOINT_MODEL_PATH.exists() else None)
    
    if HAS_TF:
        try:
            model = music_model.get_or_load_model(vocab_size=vocab_size, weights_path=weights)
            if not weights:
                print("Notice: No trained checkpoint found. Using initialized LSTM weights for generation.")
        except Exception as e:
            print(f"Warning: Could not build TF model ({e}). Using lightweight generator.")
            model = music_model.LightweightModelFallback(vocab_size=vocab_size)
    else:
        print("TensorFlow not installed in current environment. Using lightweight generator.")
        model = music_model.LightweightModelFallback(vocab_size=vocab_size)

    # 3. Choose a random seed sequence
    seed_idx = random.randint(0, len(network_input) - 1)
    seed_sequence = network_input[seed_idx]
    seed_tokens = [idx_to_token.get(i) or idx_to_token.get(str(i), "") for i in seed_sequence]
    print(f"Selected seed sequence of {len(seed_sequence)} tokens from training set (Index: {seed_idx})")

    # 4. Generate tokens
    generated_tokens = generate_music_sequence(
        model=model,
        seed_sequence=seed_sequence,
        idx_to_token=idx_to_token,
        generation_length=length,
        temperature=temperature
    )

    # 5. Output MIDI and WAV files
    out_midi = output_midi or str(config.OUTPUT_MIDI_PATH)
    out_wav = output_wav or str(config.OUTPUT_WAV_PATH)

    print("\nSynthesizing output media...")
    midi_path = postprocess.create_midi_from_tokens(generated_tokens, out_midi, tempo_bpm=tempo_bpm)
    wav_path = postprocess.synthesize_tokens_to_wav(generated_tokens, out_wav, tempo_bpm=tempo_bpm)

    print(f"\nMusic Generation Completed Successfully!")
    print(f"  - MIDI File: {midi_path} ({os.path.getsize(midi_path)} bytes)")
    print(f"  - WAV File:  {wav_path} ({os.path.getsize(wav_path)} bytes)")

    return {
        "tokens_count": len(generated_tokens),
        "temperature": temperature,
        "midi_path": midi_path,
        "wav_path": wav_path,
        "sample_tokens": generated_tokens[:15]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI Music using trained LSTM")
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--length", type=int, default=config.GENERATION_LENGTH, help="Length of music tokens")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--tempo", type=int, default=config.DEFAULT_TEMPO_BPM, help="Playback tempo BPM")
    args = parser.parse_args()

    generate(
        temperature=args.temperature,
        length=args.length,
        model_path=args.weights,
        tempo_bpm=args.tempo
    )
