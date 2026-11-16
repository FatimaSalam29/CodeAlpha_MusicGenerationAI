"""
CodeAlpha Internship Project 3: Music Generation with AI
Preprocessing Module
Extracts note, chord, duration, and rest tokens from MIDI files using music21,
builds token vocabularies, and constructs sliding-window training sequences.
"""

import os
import glob
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    from . import config
except ImportError:
    import config

# Try importing music21; fallback parser is provided for environments without music21
try:
    import music21
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False


def midi_note_number_to_name(note_num: int) -> str:
    """Convert MIDI pitch number (e.g. 60) to pitch name (e.g. 'C4')."""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_num // 12) - 1
    pitch = names[note_num % 12]
    return f"{pitch}{octave}"


def quantize_duration(quarter_length: float, step: float = 0.25) -> float:
    """Quantize duration to the nearest musical step (default 0.25 = 16th note)."""
    quantized = round(quarter_length / step) * step
    return max(step, round(quantized, 3))


def parse_midi_file_music21(file_path: str) -> List[str]:
    """
    Parse a MIDI file using music21 and return a sequence of tokens in the format:
    '<pitch_or_chord>_<duration_quarter_length>'.
    """
    if not HAS_MUSIC21:
        raise ImportError("music21 is not installed. Use fallback parser or install music21.")
    
    tokens = []
    try:
        midi = music21.converter.parse(file_path)
        # Attempt to isolate piano or primary instrumental parts
        parts = music21.instrument.partitionByInstrument(midi)
        elements_to_parse = parts.parts[0].recurse() if parts else midi.flat.notesAndRests

        for element in elements_to_parse:
            dur = quantize_duration(float(element.duration.quarterLength))
            if isinstance(element, music21.note.Note):
                pitch_str = str(element.pitch)
                tokens.append(f"{pitch_str}_{dur}")
            elif isinstance(element, music21.chord.Chord):
                # Represent chord as sorted pitch names joined by '.'
                chord_pitches = sorted([str(n.pitch) for n in element.notes])
                chord_str = ".".join(chord_pitches)
                tokens.append(f"{chord_str}_{dur}")
            elif isinstance(element, music21.note.Rest):
                tokens.append(f"rest_{dur}")
    except Exception as e:
        print(f"Warning: Could not parse {file_path} with music21: {e}")
    return tokens


def parse_midi_file_builtin(file_path: str) -> List[str]:
    """
    Built-in Standard MIDI File (SMF) parser using Python standard library.
    Ensures the pipeline can run and be tested anywhere without external dependencies.
    """
    import struct
    tokens = []
    with open(file_path, "rb") as f:
        data = f.read()

    if data[:4] != b"MThd":
        return tokens

    format_type, num_tracks, division = struct.unpack(">HHH", data[8:14])
    idx = 14
    
    # Process tracks
    while idx < len(data):
        chunk_type = data[idx:idx+4]
        length = struct.unpack(">I", data[idx+4:idx+8])[0]
        idx += 8
        if chunk_type == b"MTrk":
            track_bytes = data[idx:idx+length]
            t_idx = 0
            current_time = 0
            active_notes = {}  # pitch -> start_time

            def read_vlq(b, ptr):
                val = 0
                while ptr < len(b):
                    byte = b[ptr]
                    ptr += 1
                    val = (val << 7) | (byte & 0x7F)
                    if not (byte & 0x80):
                        break
                return val, ptr

            running_status = 0
            while t_idx < len(track_bytes):
                delta, t_idx = read_vlq(track_bytes, t_idx)
                current_time += delta
                if t_idx >= len(track_bytes):
                    break
                
                status_byte = track_bytes[t_idx]
                if status_byte & 0x80:
                    running_status = status_byte
                    t_idx += 1
                else:
                    # Running status in effect
                    pass

                cmd = running_status & 0xF0
                
                if cmd == 0x90:  # Note On
                    pitch = track_bytes[t_idx]
                    vel = track_bytes[t_idx+1]
                    t_idx += 2
                    if vel > 0:
                        active_notes[pitch] = current_time
                    else:
                        # Velocity 0 is Note Off
                        if pitch in active_notes:
                            start_t = active_notes.pop(pitch)
                            dur_beats = max(0.25, round((current_time - start_t) / division, 2))
                            dur_q = quantize_duration(dur_beats)
                            tokens.append(f"{midi_note_number_to_name(pitch)}_{dur_q}")
                elif cmd == 0x80:  # Note Off
                    pitch = track_bytes[t_idx]
                    vel = track_bytes[t_idx+1]
                    t_idx += 2
                    if pitch in active_notes:
                        start_t = active_notes.pop(pitch)
                        dur_beats = max(0.25, round((current_time - start_t) / division, 2))
                        dur_q = quantize_duration(dur_beats)
                        tokens.append(f"{midi_note_number_to_name(pitch)}_{dur_q}")
                elif cmd in (0xB0, 0xE0, 0xA0):
                    t_idx += 2
                elif cmd in (0xC0, 0xD0):
                    t_idx += 1
                elif running_status == 0xFF:  # Meta event
                    meta_type = track_bytes[t_idx]
                    t_idx += 1
                    meta_len, t_idx = read_vlq(track_bytes, t_idx)
                    t_idx += meta_len
                elif running_status in (0xF0, 0xF7):  # Sysex
                    sysex_len, t_idx = read_vlq(track_bytes, t_idx)
                    t_idx += sysex_len
                else:
                    t_idx += 1
        idx += length
    return tokens


def parse_midi_file(file_path: str) -> List[str]:
    """Universal parser: uses music21 if available, falls back to built-in SMF parser."""
    if HAS_MUSIC21:
        tokens = parse_midi_file_music21(file_path)
        if tokens:
            return tokens
    return parse_midi_file_builtin(file_path)


def load_dataset_tokens(raw_midi_dir: Path = None) -> List[str]:
    """Parse all MIDI files in raw_midi directory and concatenate tokens."""
    target_dir = Path(raw_midi_dir or config.RAW_MIDI_DIR)
    pattern1 = str(target_dir / "*.mid")
    pattern2 = str(target_dir / "*.midi")
    midi_files = sorted(glob.glob(pattern1) + glob.glob(pattern2))
    
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {target_dir}. Please place .mid files there.")
    
    all_tokens = []
    print(f"Preprocessing {len(midi_files)} MIDI files from {target_dir}...")
    for fpath in midi_files:
        tokens = parse_midi_file(fpath)
        print(f"  - {os.path.basename(fpath)}: {len(tokens)} note/chord events")
        all_tokens.extend(tokens)
        
    print(f"Total extracted tokens: {len(all_tokens)}")
    return all_tokens


def build_vocabulary(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build token-to-index and index-to-token dictionaries sorted deterministically."""
    unique_tokens = sorted(list(set(tokens)))
    token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
    idx_to_token = {idx: token for idx, token in enumerate(unique_tokens)}
    return token_to_idx, idx_to_token


def prepare_sequences(
    tokens: List[str],
    token_to_idx: Dict[str, int],
    sequence_length: int = None,
    step: int = None
) -> Tuple[List[List[int]], List[int]]:
    """
    Create sliding-window input sequences X and next-token targets y.
    """
    seq_len = sequence_length or config.SEQUENCE_LENGTH
    stride = step or config.STEP_SIZE
    
    network_input = []
    network_output = []
    
    if len(tokens) <= seq_len:
        # If dataset is small, pad tokens or adjust sequence length
        seq_len = max(4, len(tokens) // 2)
        print(f"Adjusted sequence_length to {seq_len} to fit dataset of {len(tokens)} tokens.")
        
    for i in range(0, len(tokens) - seq_len, stride):
        sequence_in = tokens[i:i + seq_len]
        target = tokens[i + seq_len]
        network_input.append([token_to_idx[char] for char in sequence_in])
        network_output.append(token_to_idx[target])
        
    print(f"Generated {len(network_input)} training sequences of length {seq_len}.")
    return network_input, network_output


def save_preprocessed_data(
    tokens: List[str],
    token_to_idx: Dict[str, int],
    idx_to_token: Dict[int, str],
    network_input: List[List[int]],
    network_output: List[int],
    processed_dir: Path = None
):
    """Save all processed tokens, vocabulary, and sequences to disk."""
    out_dir = Path(processed_dir or config.PROCESSED_DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save vocabulary mapping as JSON (human-readable)
    mapping_data = {
        "vocab_size": len(token_to_idx),
        "token_to_idx": token_to_idx,
        "idx_to_token": {str(k): v for k, v in idx_to_token.items()}
    }
    with open(out_dir / "vocab_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)
        
    # Save full sequences and tokens as pickle
    data_bundle = {
        "tokens": tokens,
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "network_input": network_input,
        "network_output": network_output,
        "vocab_size": len(token_to_idx)
    }
    with open(out_dir / "notes_data.pkl", "wb") as f:
        pickle.dump(data_bundle, f)
        
    print(f"Saved preprocessed data to {out_dir}:")
    print(f"  - vocab_mapping.json (Vocab size: {len(token_to_idx)})")
    print(f"  - notes_data.pkl (Sequences: {len(network_input)})")


def load_preprocessed_data(processed_dir: Path = None) -> Dict[str, Any]:
    """Load previously saved preprocessed data bundle."""
    target_dir = Path(processed_dir or config.PROCESSED_DATA_DIR)
    pkl_path = target_dir / "notes_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Processed data not found at {pkl_path}. Run preprocessing first.")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def run_preprocessing_pipeline() -> Dict[str, Any]:
    """Full end-to-end preprocessing pipeline execution."""
    tokens = load_dataset_tokens()
    token_to_idx, idx_to_token = build_vocabulary(tokens)
    network_input, network_output = prepare_sequences(tokens, token_to_idx)
    save_preprocessed_data(tokens, token_to_idx, idx_to_token, network_input, network_output)
    return {
        "tokens_count": len(tokens),
        "vocab_size": len(token_to_idx),
        "sequences_count": len(network_input)
    }


if __name__ == "__main__":
    stats = run_preprocessing_pipeline()
    print("Preprocessing completed successfully:", stats)
