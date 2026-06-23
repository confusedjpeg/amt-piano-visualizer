"""Download the TimGM6mb.sf2 SoundFont for FluidSynth piano synthesis.

Run from the project root:
    python scripts/download_sf2.py
"""

import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "assets" / "TimGM6mb.sf2"
URL = (
    "https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/"
    "mscore/share/sound/TimGM6mb.sf2?format=raw"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading SoundFont to {OUTPUT_PATH}...")
urllib.request.urlretrieve(URL, str(OUTPUT_PATH))
print("Download complete.")
