"""
Unit Tests for Preprocessing and Token Sequence Preparation
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import preprocess


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_tokens = [
            "C4_0.5", "D4_0.5", "E4_0.5", "F4_0.5",
            "G4_1.0", "A4_1.0", "B4_1.0", "C5_2.0",
            "C4.E4.G4_1.0", "rest_0.5"
        ] * 5

    def test_vocabulary_mapping_invertibility(self):
        """Ensure token_to_idx and idx_to_token are true inverses."""
        token_to_idx, idx_to_token = preprocess.build_vocabulary(self.sample_tokens)
        self.assertEqual(len(token_to_idx), len(idx_to_token))
        for token, idx in token_to_idx.items():
            self.assertEqual(idx_to_token[idx], token)

    def test_quantize_duration(self):
        """Ensure duration quantizer snaps to musically correct steps."""
        self.assertEqual(preprocess.quantize_duration(0.24), 0.25)
        self.assertEqual(preprocess.quantize_duration(0.51), 0.5)
        self.assertEqual(preprocess.quantize_duration(0.99), 1.0)
        self.assertEqual(preprocess.quantize_duration(0.01), 0.25)  # Enforce minimum step

    def test_sequence_preparation_dimensions(self):
        """Ensure sliding window sequences match sequence length and target alignment."""
        token_to_idx, _ = preprocess.build_vocabulary(self.sample_tokens)
        seq_len = 8
        stride = 1
        X, y = preprocess.prepare_sequences(self.sample_tokens, token_to_idx, sequence_length=seq_len, step=stride)

        expected_count = len(self.sample_tokens) - seq_len
        self.assertEqual(len(X), expected_count)
        self.assertEqual(len(y), expected_count)
        self.assertEqual(len(X[0]), seq_len)
        # Verify first target corresponds to element at index seq_len
        first_target_token = self.sample_tokens[seq_len]
        self.assertEqual(y[0], token_to_idx[first_target_token])


if __name__ == "__main__":
    unittest.main()
