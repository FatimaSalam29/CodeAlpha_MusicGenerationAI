"""
CodeAlpha Internship Project 3: Music Generation with AI
Utility Functions Module
Provides audio display helpers, system checks, and diagnostics.
"""

import os
import sys
from pathlib import Path

try:
    from . import config
except ImportError:
    import config


def check_environment() -> dict:
    """Inspect current runtime environment for GPU, Python version, and libraries."""
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "gpu_available": False,
        "gpu_name": None,
        "has_tensorflow": False,
        "has_music21": False,
    }

    try:
        import tensorflow as tf
        info["has_tensorflow"] = True
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["gpu_available"] = True
            info["gpu_name"] = [g.name for g in gpus]
    except ImportError:
        pass

    try:
        import music21
        info["has_music21"] = True
    except ImportError:
        pass

    return info


def print_system_summary():
    """Print clean formatted report of current environment."""
    env = check_environment()
    print("=" * 60)
    print("Environment & Hardware Diagnostics")
    print("=" * 60)
    print(f"Python:       {env['python_version'].split()[0]}")
    print(f"Platform:     {env['platform']}")
    print(f"TensorFlow:   {'Available' if env['has_tensorflow'] else 'Not Installed'}")
    print(f"Music21:      {'Available' if env['has_music21'] else 'Not Installed'}")
    if env["gpu_available"]:
        print(f"GPU Support:  ACTIVE ({env['gpu_name']})")
    else:
        print(f"GPU Support:  None detected (Standard CPU mode)")
    print("=" * 60)


def display_audio_colab(wav_path: str):
    """Render interactive audio player in Google Colab / Jupyter."""
    try:
        from IPython.display import Audio, display
        if os.path.exists(wav_path):
            display(Audio(wav_path, rate=44100, autoplay=False))
        else:
            print(f"Audio file not found: {wav_path}")
    except ImportError:
        print(f"Audio display is available inside Jupyter / Google Colab for {wav_path}")
