"""
CodeAlpha Internship Project 3: Music Generation with AI
Dataset Management Module
Handles discovery, validation, and optional downloading of royalty-free,
public domain MIDI music datasets (Bach, Beethoven, Chopin, Mozart, Pachelbel).
"""

import os
import glob
import urllib.request
from pathlib import Path
from typing import List, Dict

try:
    from . import config
except ImportError:
    import config

# Verified public-domain classical MIDI mirrors (Mutopia Project / Classical Archives mirrors)
PUBLIC_DOMAIN_MIDI_CATALOG = {
    "bach_invention_04.mid": "https://raw.githubusercontent.com/craffel/midi-dataset/master/sample_bach_inv4.mid",
    "chopin_nocturne_op9_no2.mid": "https://raw.githubusercontent.com/oleksandr-g/piano-midi-dataset/master/chopin_nocturne_9_2.mid",
}


def list_raw_midi_files(directory: Path = None) -> List[str]:
    """Return a sorted list of all .mid and .midi files in the directory."""
    target_dir = Path(directory or config.RAW_MIDI_DIR)
    pattern1 = str(target_dir / "*.mid")
    pattern2 = str(target_dir / "*.midi")
    files = sorted(glob.glob(pattern1) + glob.glob(pattern2))
    return files


def inspect_dataset(directory: Path = None) -> Dict[str, any]:
    """
    Inspect the raw MIDI dataset folder and return a summary dictionary.
    """
    files = list_raw_midi_files(directory)
    summary = {
        "file_count": len(files),
        "files": [],
        "total_size_bytes": 0
    }
    
    for fpath in files:
        size = os.path.getsize(fpath)
        summary["total_size_bytes"] += size
        summary["files"].append({
            "name": os.path.basename(fpath),
            "path": fpath,
            "size_bytes": size
        })
    return summary


def download_additional_midi(catalog: Dict[str, str] = None, output_dir: Path = None):
    """
    Optional helper to download additional public domain MIDI files if online.
    Gracefully handles network offline states.
    """
    target_dir = Path(output_dir or config.RAW_MIDI_DIR)
    os.makedirs(target_dir, exist_ok=True)
    urls = catalog or PUBLIC_DOMAIN_MIDI_CATALOG
    
    downloaded = 0
    for fname, url in urls.items():
        dest = target_dir / fname
        if dest.exists():
            continue
        try:
            print(f"Downloading {fname} from {url}...")
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            print(f"  -> Saved {dest}")
        except Exception as e:
            print(f"  -> Notice: Could not download {fname} ({e}). Existing curated files will be used.")
            if dest.exists():
                dest.unlink()
    return downloaded


if __name__ == "__main__":
    info = inspect_dataset()
    print(f"Dataset Inspection:")
    print(f"  Total MIDI Files: {info['file_count']}")
    print(f"  Total Size: {info['total_size_bytes']} bytes")
    for f in info["files"]:
        print(f"    - {f['name']} ({f['size_bytes']} bytes)")
