"""
Test Runner for CodeAlpha Music Generation AI Project
Discovers and executes all unit tests in the tests/ directory.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_all():
    print("=" * 65)
    print("Running Test Suite: CodeAlpha MusicGenerationAI")
    print("=" * 65)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 65)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures:  {len(result.failures)}")
    print(f"Errors:    {len(result.errors)}")
    print("=" * 65)

    if result.wasSuccessful():
        print(">> ALL UNIT TESTS PASSED SUCCESSFULLY! <<")
        return 0
    else:
        print(">> SOME TESTS FAILED. PLEASE REVIEW LOGS. <<")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
