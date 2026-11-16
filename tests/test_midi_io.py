"""
Unit Tests for MIDI File I/O and Synthesis
"""

import os
import unittest
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, midi_dataset, postprocess, preprocess


class TestMidiIO(unittest.TestCase):

    def setUp(self):
        self.raw_files = midi_dataset.list_raw_midi_files()

    def test_raw_midi_files_exist(self):
        """Ensure curated MIDI files exist in data/raw_midi."""
        self.assertGreaterEqual(len(self.raw_files), 5, "Expected at least 5 curated raw MIDI files.")

    def test_midi_header_validity(self):
        """Ensure all raw MIDI files have valid SMF MThd headers."""
        for fpath in self.raw_files:
            with open(fpath, "rb") as f:
                header = f.read(4)
            self.assertEqual(header, b"MThd", f"Invalid SMF header in {fpath}")

    def test_midi_parsing_and_token_extraction(self):
        """Ensure parser extracts non-empty token lists from raw files."""
        for fpath in self.raw_files:
            tokens = preprocess.parse_midi_file(fpath)
            self.assertGreater(len(tokens), 10, f"Too few tokens extracted from {fpath}")
            # Verify token structure (e.g. C4_0.5)
            self.assertIn("_", tokens[0], f"Token format invalid: {tokens[0]}")

    def test_token_to_midi_creation(self):
        """Ensure tokens can be written to a valid MIDI file."""
        sample_tokens = ["C4_0.5", "E4_0.5", "G4_0.5", "C5_1.0"]
        test_out = str(config.OUTPUT_DIR / "unit_test_sample.mid")
        created_path = postprocess.create_midi_from_tokens(sample_tokens, test_out)
        self.assertTrue(os.path.exists(created_path))
        with open(created_path, "rb") as f:
            header = f.read(4)
        self.assertEqual(header, b"MThd")
        if os.path.exists(created_path):
            os.remove(created_path)

    def test_wav_synthesis(self):
        """Ensure tokens can be synthesized into a valid WAV audio file."""
        sample_tokens = ["C4_0.5", "E4_0.5", "G4_0.5"]
        test_out = str(config.OUTPUT_DIR / "unit_test_sample.wav")
        created_path = postprocess.synthesize_tokens_to_wav(sample_tokens, test_out)
        self.assertTrue(os.path.exists(created_path))
        self.assertGreater(os.path.getsize(created_path), 1000)
        if os.path.exists(created_path):
            os.remove(created_path)


if __name__ == "__main__":
    unittest.main()
