# AI Piano Arranger — Audio to Synthesia Pipeline

> Transform any song into a playable two-handed piano arrangement with a Synthesia-style falling-notes video.

## Features

- **Automatic Stem Separation** — Isolates vocals, bass, drums, and other instruments using Meta's Demucs
- **Vocal-to-Piano Transcription** — Converts vocal melodies to right-hand piano parts using Spotify's BasicPitch
- **Piano Transcription** — Extracts existing piano from audio using ByteDance's transcription model
- **Algorithmic Arrangement** — Generates accompaniment from detected chords when no piano is present
- **Playability Filter** — Enforces human-playable constraints (hand span, polyphony, quantization)
- **Synthesia Video Rendering** — Produces falling-notes videos synced to the original audio

## Quick Start

### Prerequisites

| Requirement | Minimum Version | Purpose |
|:------------|:----------------|:--------|
| Python | 3.10+ | Runtime |
| FFmpeg | 4.4+ | Audio processing |
| CUDA Toolkit *(optional)* | 11.7+ | GPU acceleration |
| MIDIVisualizer | Latest | Video rendering |

### Installation

```bash
# 1. Clone and enter the project
cd Piano

# 2. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install PyTorch (choose one)
# GPU (CUDA 11.8):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify
python -c "import demucs; import basic_pitch; import librosa; import pretty_midi; print('All OK')"
```

### MIDIVisualizer Setup

Download from [GitHub Releases](https://github.com/kosua20/MIDIVisualizer/releases) and add to PATH, or set the path in `config.yaml`:

```yaml
video:
  midi_visualizer_path: "/path/to/MIDIVisualizer"
```

## Usage

```bash
# Full pipeline: vocals + existing piano
python scripts/run_pipeline.py "data/input/song.mp3" --include-vocals --has-piano

# No vocals, generate accompaniment algorithmically
python scripts/run_pipeline.py "data/input/edm_track.mp3" --no-vocals --no-piano

# Custom config file
python scripts/run_pipeline.py "data/input/rock.mp3" --config custom.yaml

# Verbose logging
python scripts/run_pipeline.py "data/input/song.mp3" --log-level DEBUG
```

### Pipeline Modes

| `--include-vocals` | `--has-piano` | Right Hand | Left Hand |
|:---:|:---:|:---|:---|
| ✅ | ✅ | Vocal melody (BasicPitch) | Piano transcription |
| ✅ | ❌ | Vocal melody (BasicPitch) | Algorithmic arrangement |
| ❌ | ✅ | *(unused)* | Piano transcription |
| ❌ | ❌ | *(unused)* | Algorithmic arrangement |

## Output

Each run produces:
- `data/output/<run_id>_final.mid` — Playable two-handed piano MIDI
- `data/output/<run_id>_synthesia.mp4` — Falling-notes video with audio
- `data/intermediate/<run_id>/` — All intermediate artifacts (stems, raw MIDIs, etc.)

## Configuration

All tunable parameters are in [`config.yaml`](config.yaml). Key settings:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `separator.model` | `htdemucs` | Demucs model variant |
| `playability.max_hand_span_semitones` | `15` | Max interval per hand |
| `playability.max_polyphony_per_hand` | `4` | Max simultaneous notes |
| `playability.quantization_grid` | `16` | 16th note quantization |
| `arranger.default_pattern` | `pop_ballad` | Accompaniment style |
| `video.fps` | `60` | Video frame rate |

## Testing

```bash
# Run unit tests (fast, no GPU needed)
pytest tests/ -v -k "not e2e"

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing -k "not e2e"

# Run all tests including integration
pytest tests/ -v
```

## Project Structure

```
Piano/
├── src/
│   ├── audio/          # Stem separation & mixing
│   ├── transcription/  # Vocal & piano transcription
│   ├── arranger/       # Chord detection & pattern-based arrangement
│   ├── midi/           # Cleaning, quantization, playability, merging
│   ├── video/          # MIDIVisualizer rendering
│   ├── pipeline/       # Orchestration, config, models, errors
│   └── utils/          # Logging, file I/O, validation
├── tests/              # pytest test suite
├── assets/patterns/    # JSON arrangement patterns
├── scripts/            # CLI entry point
├── config.yaml         # Runtime configuration
└── requirements.txt    # Python dependencies
```

## License

This project is for personal/educational use.
