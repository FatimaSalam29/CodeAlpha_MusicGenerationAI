"""
CodeAlpha Internship Project 3: Music Generation with AI
Postprocessing Module
Converts predicted note, chord, and duration token sequences back into
playable Standard MIDI (.mid) files and synthesizes audio (.wav).
"""

import os
import math
import struct
import wave
from pathlib import Path
from typing import List, Tuple, Union

try:
    from . import config
except ImportError:
    import config

try:
    import music21
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False


def note_name_to_midi_number(name: str) -> int:
    """Convert pitch name like 'C4', 'F#4', 'E-4', or 'A4' to MIDI integer (60, 66, etc.)."""
    pitch_map = {
        'C': 0, 'C#': 1, 'D-': 1,
        'D': 2, 'D#': 3, 'E-': 3,
        'E': 4, 'F': 5, 'F#': 6, 'G-': 6,
        'G': 7, 'G#': 8, 'A-': 8,
        'A': 9, 'A#': 10, 'B-': 10,
        'B': 11
    }
    name = name.strip()
    if not name:
        return 60
    # Split letter (+accidental) and octave digit
    if len(name) >= 2 and name[-1].isdigit():
        octave = int(name[-1])
        note_letter = name[:-1]
    elif len(name) >= 3 and name[-2] == '-' and name[-1].isdigit():
        octave = -int(name[-1])
        note_letter = name[:-2]
    else:
        octave = 4
        note_letter = name

    note_letter = note_letter.capitalize()
    semitone = pitch_map.get(note_letter, 0)
    midi_num = (octave + 1) * 12 + semitone
    return max(0, min(127, midi_num))


def parse_token(token: str) -> Tuple[Union[int, List[int], None], float]:
    """
    Parse a token string like 'C4_0.5', 'C4.E4.G4_1.0', 'rest_0.5', or '60_0.5'.
    Returns (pitches, duration_beats).
    """
    if "_" in token:
        parts = token.rsplit("_", 1)
        pitch_part, dur_str = parts[0], parts[1]
        try:
            dur = float(dur_str)
        except ValueError:
            dur = 0.5
    else:
        pitch_part = token
        dur = 0.5

    if pitch_part.lower() == "rest":
        return None, dur

    if "." in pitch_part:
        # Chord
        sub_pitches = pitch_part.split(".")
        chord_nums = []
        for p in sub_pitches:
            if p.isdigit():
                chord_nums.append(int(p))
            else:
                chord_nums.append(note_name_to_midi_number(p))
        return sorted(chord_nums), dur
    else:
        # Single note
        if pitch_part.isdigit():
            return int(pitch_part), dur
        else:
            return note_name_to_midi_number(pitch_part), dur


def tokens_to_midi_music21(tokens: List[str], output_path: str, tempo_bpm: int = 100):
    """Convert sequence of tokens to MIDI using music21."""
    if not HAS_MUSIC21:
        raise ImportError("music21 is not installed.")

    stream_part = music21.stream.Part()
    stream_part.append(music21.tempo.MetronomeMark(number=tempo_bpm))
    stream_part.append(music21.instrument.Piano())

    offset = 0.0
    for token in tokens:
        pitches, dur = parse_token(token)
        if pitches is None:
            # Rest
            new_rest = music21.note.Rest()
            new_rest.quarterLength = dur
            new_rest.offset = offset
            stream_part.append(new_rest)
        elif isinstance(pitches, list):
            # Chord
            chord_notes = [music21.note.Note(p) for p in pitches]
            new_chord = music21.chord.Chord(chord_notes)
            new_chord.quarterLength = dur
            new_chord.offset = offset
            stream_part.append(new_chord)
        else:
            # Single note
            new_note = music21.note.Note(pitches)
            new_note.quarterLength = dur
            new_note.offset = offset
            stream_part.append(new_note)
        offset += dur

    score = music21.stream.Score([stream_part])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    score.write("midi", fp=output_path)
    return output_path


def tokens_to_midi_builtin(tokens: List[str], output_path: str, tempo_bpm: int = 100, ticks_per_beat: int = 480):
    """
    Built-in Standard MIDI File writer with zero external dependencies.
    Encodes tokens to format 0 SMF file.
    """
    def encode_vlq(value):
        buf = [value & 0x7F]
        value >>= 7
        while value > 0:
            buf.append((value & 0x7F) | 0x80)
            value >>= 7
        buf.reverse()
        return bytes(buf)

    track_data = bytearray()
    
    # Tempo: 60,000,000 / BPM
    tempo_us = int(60000000 / tempo_bpm)
    track_data.extend(encode_vlq(0))
    track_data.extend(bytes([0xFF, 0x51, 0x03, (tempo_us >> 16) & 0xFF, (tempo_us >> 8) & 0xFF, tempo_us & 0xFF]))
    
    # Time Signature 4/4
    track_data.extend(encode_vlq(0))
    track_data.extend(bytes([0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]))

    for token in tokens:
        pitches, dur = parse_token(token)
        ticks = int(dur * ticks_per_beat)
        velocity = 90

        if pitches is None:
            # Rest: wait delta ticks on next event
            track_data.extend(encode_vlq(ticks))
            track_data.extend(bytes([0x80, 0, 0]))  # Silent note-off to advance delta
        elif isinstance(pitches, list):
            # Chord
            for i, p in enumerate(pitches):
                track_data.extend(encode_vlq(0))
                track_data.extend(bytes([0x90, p, velocity]))
            for i, p in enumerate(pitches):
                delta = ticks if i == 0 else 0
                track_data.extend(encode_vlq(delta))
                track_data.extend(bytes([0x80, p, 0]))
        else:
            # Note on
            track_data.extend(encode_vlq(0))
            track_data.extend(bytes([0x90, pitches, velocity]))
            # Note off after ticks
            track_data.extend(encode_vlq(ticks))
            track_data.extend(bytes([0x80, pitches, 0]))

    # End of Track
    track_data.extend(encode_vlq(0))
    track_data.extend(bytes([0xFF, 0x2F, 0x00]))

    header = struct.pack(">4sIHHH", b"MThd", 6, 0, 1, ticks_per_beat)
    track_chunk = struct.pack(">4sI", b"MTrk", len(track_data)) + track_data

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header + track_chunk)
    return output_path


def create_midi_from_tokens(tokens: List[str], output_path: str = None, tempo_bpm: int = 100) -> str:
    """Universal MIDI generator: uses music21 if available, falls back to built-in writer."""
    dest = str(output_path or config.OUTPUT_MIDI_PATH)
    if HAS_MUSIC21:
        try:
            return tokens_to_midi_music21(tokens, dest, tempo_bpm)
        except Exception as e:
            print(f"music21 write notice ({e}), falling back to standard MIDI writer.")
    return tokens_to_midi_builtin(tokens, dest, tempo_bpm)


def synthesize_tokens_to_wav(
    tokens: List[str],
    output_wav_path: str = None,
    tempo_bpm: int = 100,
    sample_rate: int = 44100
) -> str:
    """
    Synthesize music tokens to a 16-bit PCM WAV audio file using an authentic
    additive acoustic piano synthesis model with ADSR envelope.
    Ensures free, zero-dependency audio generation that works on any laptop and Google Colab!
    """
    dest = str(output_wav_path or config.OUTPUT_WAV_PATH)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    sec_per_beat = 60.0 / tempo_bpm
    audio_samples = []

    def midi_to_freq(midi_num: int) -> float:
        return 440.0 * (2.0 ** ((midi_num - 69.0) / 12.0))

    for token in tokens:
        pitches, dur_beats = parse_token(token)
        dur_sec = dur_beats * sec_per_beat
        num_samples = int(dur_sec * sample_rate)
        if num_samples <= 0:
            continue

        if pitches is None:
            # Silence for rest
            audio_samples.extend([0.0] * num_samples)
            continue

        pitch_list = pitches if isinstance(pitches, list) else [pitches]
        
        # Synthesize notes for this duration window
        for i in range(num_samples):
            t = i / sample_rate
            # ADSR Envelope: Fast attack (0.01s), exponential decay
            attack = min(1.0, t / 0.01) if t < 0.01 else 1.0
            decay = math.exp(-3.0 * t / max(0.1, dur_sec))
            envelope = attack * decay

            sample_val = 0.0
            for p in pitch_list:
                f = midi_to_freq(p)
                # Multi-harmonic acoustic piano tone: fundamental + 2nd & 3rd harmonics
                tone = (
                    0.60 * math.sin(2.0 * math.pi * f * t) +
                    0.25 * math.sin(4.0 * math.pi * f * t) +
                    0.15 * math.sin(6.0 * math.pi * f * t)
                )
                sample_val += tone
                
            sample_val = (sample_val / len(pitch_list)) * envelope * 0.7
            audio_samples.append(sample_val)

    # Convert to 16-bit PCM signed integers (-32768 to 32767)
    with wave.open(dest, "wb") as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)        # 16-bit (2 bytes per sample)
        wf.setframerate(sample_rate)
        
        raw_bytes = bytearray()
        for sample in audio_samples:
            clamped = max(-1.0, min(1.0, sample))
            int_val = int(clamped * 32767)
            raw_bytes.extend(struct.pack("<h", int_val))
        wf.writeframes(raw_bytes)

    print(f"Synthesized WAV: {dest} ({len(audio_samples) / sample_rate:.2f} seconds)")
    return dest


if __name__ == "__main__":
    sample_tokens = ["C4_0.5", "E4_0.5", "G4_0.5", "C4.E4.G4_1.0", "rest_0.5", "C5_1.5"]
    midi_res = create_midi_from_tokens(sample_tokens, "test_out.mid")
    wav_res = synthesize_tokens_to_wav(sample_tokens, "test_out.wav")
    print("Postprocessing test completed:", midi_res, wav_res)
