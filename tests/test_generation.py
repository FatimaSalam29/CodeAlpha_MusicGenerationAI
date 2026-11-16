"""
Unit Tests for Music Generation and Temperature Sampling
"""

import unittest
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import generate, model as music_model


class TestGeneration(unittest.TestCase):

    def test_temperature_sampling_bounds(self):
        """Ensure sampled index is strictly within vocabulary range."""
        probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        for temp in [0.2, 0.5, 0.8, 1.2, 2.0]:
            idx = generate.sample_with_temperature(probs, temperature=temp)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(probs))

    def test_low_temperature_peaks_highest_probability(self):
        """Low temperature (T -> 0) should overwhelmingly pick the highest probability token."""
        probs = [0.05, 0.05, 0.80, 0.05, 0.05]
        picks = [generate.sample_with_temperature(probs, temperature=0.01) for _ in range(20)]
        self.assertTrue(all(p == 2 for p in picks), "Expected argmax (index 2) under very low temperature")

    def test_end_to_end_generation_flow(self):
        """Test full generation call producing MIDI and WAV files."""
        result = generate.generate(temperature=0.8, length=30)
        self.assertEqual(result["tokens_count"], 30)
        self.assertTrue(Path(result["midi_path"]).exists())
        self.assertTrue(Path(result["wav_path"]).exists())
        self.assertGreater(Path(result["midi_path"]).stat().st_size, 50)
        self.assertGreater(Path(result["wav_path"]).stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
