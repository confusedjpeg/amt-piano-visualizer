# MASTER DEVELOPMENT PLAN
## AI-Powered Audio-to-Piano-Synthesia Pipeline

> **Version:** 1.0 — MVP Blueprint  
> **Last Updated:** 2026-05-25  
> **Status:** Architectural Plan (No source code — implementation follows approval)

---

## Table of Contents

1. [System Architecture Diagram](#1-system-architecture-diagram)
2. [Directory Structure](#2-directory-structure)
3. [Phase 1: Environment Setup & Dependencies](#3-phase-1-environment-setup--dependencies)
4. [Phase 2: Core Module Implementation](#4-phase-2-core-module-implementation)
   - [Step 1 — Stem Separation](#step-1-stem-separation-the-demucs-module)
   - [Step 2 — Vocal Transcription](#step-2-vocal-transcription-the-basic-pitch-module)
   - [Step 3 — Accompaniment Generation](#step-3-accompaniment-generation-the-fork)
   - [Step 4 — Playability Filter & MIDI Merging](#step-4-the-playability-filter--midi-merging)
   - [Step 5 — Visual Rendering](#step-5-visual-rendering)
5. [Phase 3: Integration & Testing](#5-phase-3-integration--testing)
6. [Appendices](#6-appendices)

---

## 1. System Architecture Diagram

### High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT                                      │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│  │ original_audio   │   │ include_vocals   │   │ has_piano            │  │
│  │ (.mp3 / .wav)    │   │ (True / False)   │   │ (True / False)       │  │
│  └────────┬─────────┘   └────────┬─────────┘   └──────────┬───────────┘  │
└───────────┼──────────────────────┼─────────────────────────┼─────────────┘
            │                      │                         │
            ▼                      │                         │
  ┌─────────────────────┐         │                         │
  │   STEP 1: Demucs    │         │                         │
  │  Stem Separation    │         │                         │
  └──────┬──────────────┘         │                         │
         │                        │                         │
         ├── vocals.wav ──────────┼─────────┐               │
         ├── bass.wav             │         │               │
         ├── other.wav            │         │               │
         └── drums.wav (discard)  │         │               │
              │                   │         │               │
              ▼                   │         │               │
   ┌─────────────────┐           │         │               │
   │ Mix bass + other │           │         │               │
   │ → instrumental   │           │         │               │
   │     .wav         │           │         │               │
   └────────┬─────────┘           │         │               │
            │                     │         │               │
            │                     ▼         ▼               ▼
            │          ┌───────────────────────┐  ┌─────────────────────┐
            │          │ include_vocals==True?  │  │   has_piano toggle  │
            │          └─────┬───────────┬──────┘  └──────┬──────┬──────┘
            │                │ YES       │ NO             │      │
            │                ▼           ▼                ▼      ▼
            │     ┌──────────────┐   (skip)      ┌───────┐  ┌────────┐
            │     │  STEP 2:     │               │Path A │  │Path B  │
            │     │  BasicPitch  │               │Piano  │  │Algo    │
            │     │  Vocal→MIDI  │               │Transc.│  │Arranger│
            │     └──────┬───────┘               └───┬───┘  └───┬────┘
            │            │                           │           │
            │            ▼                           ▼           ▼
            │     ┌──────────────┐           ┌──────────────────────┐
            │     │ raw_vocals   │           │ accompaniment.mid    │
            │     │ .mid         │           │                      │
            │     │ (cleaned,    │           │ (Path A: from piano  │
            │     │  quantized,  │           │  transcription)      │
            │     │  Ch.1 / RH)  │           │ (Path B: from algo   │
            │     └──────┬───────┘           │  chord→pattern map)  │
            │            │                   └──────────┬───────────┘
            │            │                              │
            │            └─────────────┬────────────────┘
            │                          │
            │                          ▼
            │              ┌───────────────────────┐
            │              │   STEP 4: Playability  │
            │              │   Filter & Merge       │
            │              │                        │
            │              │  • Hand-span ≤ 15 ST   │
            │              │  • Polyphony ≤ 4/hand  │
            │              │  • 16th-note quantize  │
            │              └───────────┬────────────┘
            │                          │
            │                          ▼
            │              ┌───────────────────────┐
            │              │ final_playable.mid     │
            │              └───────────┬────────────┘
            │                          │
            ▼                          ▼
  ┌─────────────────────────────────────────────┐
  │          STEP 5: MIDIVisualizer             │
  │  Inputs: final_playable.mid + original_audio│
  │  Output: final_synthesia_video.mp4          │
  └─────────────────────────────────────────────┘
```

### Control Flow Matrix

| `include_vocals` | `has_piano` | Steps Executed                        | Right Hand Source     | Left Hand Source           |
|:-----------------:|:-----------:|:--------------------------------------|:----------------------|:---------------------------|
| `True`            | `True`      | 1 → 2 → 3A → 4 → 5                  | BasicPitch (vocals)   | Piano Transcription        |
| `True`            | `False`     | 1 → 2 → 3B → 4 → 5                  | BasicPitch (vocals)   | Algorithmic Arranger       |
| `False`           | `True`      | 1 → 3A → 4 → 5                      | *(none — RH unused)*  | Piano Transcription        |
| `False`           | `False`     | 1 → 3B → 4 → 5                      | *(none — RH unused)*  | Algorithmic Arranger       |

> **Note:** When `include_vocals=False`, the final MIDI contains only the accompaniment (left-hand). Step 2 is skipped entirely. The playability filter still runs on the single-channel output.

---

## 2. Directory Structure

```
Piano/
├── MASTER_DEVELOPMENT_PLAN.md        # This document
├── README.md                         # User-facing quickstart guide
├── requirements.txt                  # Pinned Python dependencies
├── setup.py                          # Optional — for editable installs
├── .env.example                      # Template for environment variables
├── config.yaml                       # Runtime configuration defaults
│
├── src/
│   ├── __init__.py
│   │
│   ├── audio/                        # Step 1 — Audio I/O & Stem Separation
│   │   ├── __init__.py
│   │   ├── separator.py              # Demucs wrapper
│   │   └── mixer.py                  # Stem combination utilities
│   │
│   ├── transcription/                # Steps 2 & 3A — Audio-to-MIDI
│   │   ├── __init__.py
│   │   ├── vocal_transcriber.py      # BasicPitch wrapper (Step 2)
│   │   └── piano_transcriber.py      # ByteDance wrapper (Step 3A)
│   │
│   ├── arranger/                     # Step 3B — Algorithmic Arrangement
│   │   ├── __init__.py
│   │   ├── chord_detector.py         # Chord extraction from audio
│   │   ├── beat_analyzer.py          # Tempo & beat grid via librosa
│   │   ├── pattern_library.py        # Pre-defined MIDI chord voicings
│   │   └── arranger.py              # Orchestrates chord→MIDI generation
│   │
│   ├── midi/                         # Step 4 — MIDI Processing
│   │   ├── __init__.py
│   │   ├── cleaner.py                # Pitch-bend stripping, channel assignment
│   │   ├── quantizer.py              # Grid snapping (16th / 32nd)
│   │   ├── playability.py            # Hand-span & polyphony constraints
│   │   └── merger.py                 # Multi-track MIDI merge
│   │
│   ├── video/                        # Step 5 — Visual Rendering
│   │   ├── __init__.py
│   │   └── renderer.py              # MIDIVisualizer subprocess wrapper
│   │
│   ├── pipeline/                     # Orchestration
│   │   ├── __init__.py
│   │   ├── config.py                 # Pydantic settings / config loader
│   │   └── orchestrator.py           # Main pipeline controller
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── logger.py                 # Structured logging (loguru)
│       ├── file_io.py                # Path helpers, temp dir management
│       └── validators.py            # Input validation helpers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared pytest fixtures
│   ├── test_separator.py
│   ├── test_vocal_transcriber.py
│   ├── test_piano_transcriber.py
│   ├── test_arranger.py
│   ├── test_playability.py
│   ├── test_quantizer.py
│   ├── test_merger.py
│   ├── test_renderer.py
│   └── test_pipeline_e2e.py          # End-to-end integration tests
│
├── data/
│   ├── input/                        # User drops audio files here
│   ├── output/                       # Final MIDI + MP4 artifacts
│   └── intermediate/                 # Stems, raw MIDIs (per-run subfolders)
│
├── assets/
│   └── patterns/                     # JSON/YAML chord voicing patterns
│       ├── pop_ballad.json
│       ├── driving_eighths.json
│       └── arpeggiated.json
│
└── scripts/
    ├── run_pipeline.py               # CLI entry point
    └── install_midi_visualizer.sh    # Helper to install MIDIVisualizer binary
```

### Key Design Decisions

| Decision | Rationale |
|:---------|:----------|
| One module per pipeline step | Each step can be developed, tested, and debugged in isolation. |
| `data/intermediate/` with per-run subfolders | Prevents file collisions on concurrent runs; makes debugging trivial (inspect any intermediate artifact). |
| Pattern library as JSON files | Non-developer musicians can contribute new patterns without touching Python. |
| `config.yaml` + Pydantic | Single source of truth for all tunable parameters (quantization grid size, polyphony limit, hand span, etc.) with runtime type validation. |

---

## 3. Phase 1: Environment Setup & Dependencies

### 3.1 System-Level Prerequisites

| Requirement | Minimum Version | Purpose |
|:------------|:----------------|:--------|
| Python | 3.10+ | Type hints, `match` statements, library compat |
| FFmpeg | 4.4+ | Audio decoding/encoding for Demucs and librosa |
| CUDA Toolkit *(optional)* | 11.7+ | GPU acceleration for Demucs and piano transcription |
| MIDIVisualizer | Latest release | CLI binary for Synthesia-style rendering |

### 3.2 MIDIVisualizer Installation

MIDIVisualizer is a standalone C++ binary, not a Python package. It must be installed separately.

```bash
# Linux
wget https://github.com/kosua20/MIDIVisualizer/releases/latest/download/MIDIVisualizer-linux.tar.gz
tar -xzf MIDIVisualizer-linux.tar.gz -C /usr/local/bin/

# macOS (Homebrew)
brew install --cask midivisualizer

# Windows
# Download the .exe from GitHub Releases and add to PATH
# https://github.com/kosua20/MIDIVisualizer/releases
```

Store the binary path in `config.yaml` → `video.midi_visualizer_path`.

### 3.3 Python Dependencies

#### `requirements.txt`

```
# ── Audio Source Separation ──
demucs>=4.0.0

# ── Vocal Transcription ──
basic-pitch>=0.3.0

# ── Piano Transcription ──
piano-transcription-inference>=0.0.5

# ── Audio Analysis ──
librosa>=0.10.0
soundfile>=0.12.0       # Backend for librosa / soundfile I/O

# ── MIDI Manipulation ──
pretty_midi>=0.2.10

# ── Chord Detection (Path B) ──
# Option A: Use a lightweight chroma-to-chord mapper (custom)
# Option B: Use a pre-trained model — see Section 4.3.1
numpy>=1.24.0
scipy>=1.10.0

# ── Configuration & Validation ──
pydantic>=2.0
pyyaml>=6.0

# ── Logging ──
loguru>=0.7.0

# ── CLI ──
click>=8.1.0

# ── Testing ──
pytest>=7.4.0
pytest-cov>=4.1.0

# ── Utilities ──
tqdm>=4.65.0             # Progress bars for long-running steps
```

#### Installation Commands

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch (select CUDA or CPU variant)
# GPU (CUDA 11.8):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Verify critical imports
python -c "import demucs; import basic_pitch; import piano_transcription_inference; import librosa; import pretty_midi; print('All dependencies OK')"
```

### 3.4 Configuration File (`config.yaml`)

```yaml
# ── Pipeline Defaults ──
pipeline:
  include_vocals: true
  has_piano: true
  output_dir: "data/output"
  intermediate_dir: "data/intermediate"

# ── Step 1: Demucs ──
separator:
  model: "htdemucs"         # htdemucs | htdemucs_ft | mdx_extra
  device: "auto"            # auto | cpu | cuda
  shifts: 1                 # Higher = better quality, slower
  overlap: 0.25             # Overlap ratio between chunks
  mp3_bitrate: 320          # Bitrate if saving stems as mp3
  output_format: "wav"      # wav | mp3

# ── Step 2: BasicPitch ──
vocal_transcription:
  onset_threshold: 0.5      # Note onset confidence threshold
  frame_threshold: 0.3      # Frame-level activation threshold
  minimum_note_length_ms: 58  # ~32nd note at 120 BPM
  minimum_frequency_hz: 80.0  # Roughly E2 — floor for vocal range
  maximum_frequency_hz: 1100.0  # Roughly C#6 — ceiling for vocal range

# ── Step 3A: Piano Transcription ──
piano_transcription:
  device: "auto"
  checkpoint: null           # null = use bundled default

# ── Step 3B: Algorithmic Arranger ──
arranger:
  default_pattern: "pop_ballad"
  pattern_dir: "assets/patterns"
  chord_detection_method: "chroma"  # chroma | crema (if available)
  beats_per_bar: 4

# ── Step 4: Playability Filter ──
playability:
  max_hand_span_semitones: 15
  max_polyphony_per_hand: 4
  quantization_grid: 16      # 16 = 16th note, 32 = 32nd note
  pruning_priority:           # Which intervals to drop first (descending priority)
    - "P5"                    # Perfect 5th
    - "M3"                    # Major 3rd (only if still over limit)
    - "m3"                    # Minor 3rd

# ── Step 5: MIDIVisualizer ──
video:
  midi_visualizer_path: "MIDIVisualizer"  # Must be on PATH or absolute
  resolution: "1920x1080"
  fps: 60
  background_color: "0.1 0.1 0.12"       # Dark charcoal
  note_speed: 1.0
  additional_args: []         # Extra CLI flags passed verbatim
```

---

## 4. Phase 2: Core Module Implementation

Each subsection below corresponds to one pipeline step. For every module, we define:
- **Purpose** — what the module does and why.
- **Class / Function Signatures** — the public API surface.
- **Internal Logic** — pseudocode-level description of the algorithm.
- **Error Handling** — expected failure modes and recovery strategies.
- **I/O Contract** — exact input/output types and file paths.

---

### Step 1: Stem Separation (The `demucs` Module)

**Modules:** `src/audio/separator.py`, `src/audio/mixer.py`

#### 4.1.1 `separator.py` — `StemSeparator`

**Purpose:** Wrap the Demucs Python API to decompose an input audio file into four stems: `vocals`, `bass`, `drums`, `other`.

```python
class StemSeparator:
    """Wraps Meta's Demucs htdemucs model for 4-stem source separation."""

    def __init__(self, config: SeparatorConfig) -> None:
        """
        Args:
            config: Pydantic model containing:
                - model (str): Demucs model name, e.g. "htdemucs"
                - device (str): "auto" | "cpu" | "cuda"
                - shifts (int): Number of random shifts for prediction averaging
                - overlap (float): Overlap ratio between audio chunks
        """

    def separate(self, audio_path: Path) -> StemResult:
        """
        Run source separation on the given audio file.

        Args:
            audio_path: Absolute path to the input audio file (.mp3, .wav, .flac).

        Returns:
            StemResult: A dataclass containing paths to the four output stems:
                - vocals_path: Path    (data/intermediate/<run_id>/vocals.wav)
                - bass_path: Path      (data/intermediate/<run_id>/bass.wav)
                - drums_path: Path     (data/intermediate/<run_id>/drums.wav)
                - other_path: Path     (data/intermediate/<run_id>/other.wav)

        Raises:
            FileNotFoundError: If audio_path does not exist.
            SeparationError: If Demucs fails (model load error, OOM, corrupt audio).
        """
```

**Internal Logic:**

1. Validate that `audio_path` exists and is a supported format.
2. Load the Demucs model using `demucs.pretrained.get_model(model_name)`.
3. Load the audio waveform using `demucs.audio.AudioFile(audio_path).read(...)`.
4. Apply the separation model: `demucs.apply.apply_model(model, mix, ...)`.
5. Write each stem tensor to WAV using `torchaudio.save()` or `soundfile.write()`.
6. Return a `StemResult` dataclass with the four file paths.

**Key Implementation Notes:**
- Demucs returns a tensor of shape `(sources, channels, samples)` where `sources` order is `['drums', 'bass', 'other', 'vocals']` for htdemucs — **verify stem ordering at runtime** by checking `model.sources`.
- For GPU acceleration: move the model and input tensors to the configured device. Wrap in `torch.no_grad()` for inference.
- For large files (>10 min), Demucs chunks internally using the `overlap` parameter. Ensure sufficient RAM/VRAM — log a warning if file duration exceeds 15 minutes.

#### 4.1.2 `mixer.py` — `StemMixer`

**Purpose:** Combine selected stems into a single mixed audio file.

```python
class StemMixer:
    """Utility for mixing audio stems via waveform summation."""

    @staticmethod
    def mix_stems(
        stem_paths: list[Path],
        output_path: Path,
        normalize: bool = True,
    ) -> Path:
        """
        Sum multiple audio stems into a single WAV file.

        Args:
            stem_paths: List of WAV file paths to mix.
            output_path: Where to write the mixed result.
            normalize: If True, normalize the mixed waveform to prevent clipping.

        Returns:
            Path to the output mixed file.

        Raises:
            ValueError: If stems have mismatched sample rates or channel counts.
        """
```

**Internal Logic:**

1. Load each stem via `librosa.load(path, sr=None)` to preserve native sample rate.
2. Assert all sample rates are identical (they will be, since Demucs outputs them uniformly — but validate defensively).
3. Pad shorter stems with silence to match the longest stem's length.
4. Sum the waveforms element-wise: `mixed = stem_1 + stem_2 + ... + stem_n`.
5. If `normalize=True`, scale the result to `[-1.0, 1.0]` by dividing by `max(abs(mixed))`.
6. Write to `output_path` using `soundfile.write()`.

**Usage in Pipeline:**

```python
# After separation, create instrumental.wav = bass + other
mixer = StemMixer()
instrumental_path = mixer.mix_stems(
    stem_paths=[stem_result.bass_path, stem_result.other_path],
    output_path=run_dir / "instrumental.wav",
)
```

---

### Step 2: Vocal Transcription (The `basic-pitch` Module)

**Module:** `src/transcription/vocal_transcriber.py`

**Condition:** This step is **only executed** when `include_vocals == True`.

#### 4.2.1 `vocal_transcriber.py` — `VocalTranscriber`

**Purpose:** Convert the isolated vocal stem (`vocals.wav`) to a MIDI file using Spotify's BasicPitch, then apply cleanup to produce a musically clean right-hand part.

```python
class VocalTranscriber:
    """Transcribe monophonic/polyphonic vocals to MIDI using BasicPitch."""

    def __init__(self, config: VocalTranscriptionConfig) -> None:
        """
        Args:
            config: Pydantic model containing:
                - onset_threshold (float)
                - frame_threshold (float)
                - minimum_note_length_ms (float)
                - minimum_frequency_hz (float)
                - maximum_frequency_hz (float)
        """

    def transcribe(self, vocals_path: Path, output_path: Path) -> Path:
        """
        Transcribe vocals to MIDI and apply cleanup.

        Args:
            vocals_path: Path to vocals.wav (from Step 1).
            output_path: Path for the cleaned output MIDI.

        Returns:
            Path to the cleaned MIDI file (all notes on Channel 0 = Right Hand).

        Pipeline:
            1. Run BasicPitch inference → raw PrettyMIDI object.
            2. Strip all pitch bend events.
            3. Clamp notes to vocal range [minimum_frequency_hz, maximum_frequency_hz].
            4. Remove notes shorter than minimum_note_length_ms.
            5. Quantize note start/end times to nearest 16th note grid.
            6. Assign all notes to MIDI Channel 0 (displayed as "Right Hand").
            7. Write cleaned MIDI to output_path.
        """
```

**Internal Logic — Cleanup Pipeline:**

```
raw_midi = basic_pitch.inference.predict(vocals_path, ...)
    │
    ▼
┌──────────────────────────────────────┐
│  1. STRIP PITCH BENDS               │
│  For each instrument in raw_midi:    │
│    instrument.pitch_bends = []       │
│  Rationale: Vocal vibrato creates    │
│  erratic pitch bends that make the   │
│  piano part sound unnatural.         │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  2. CLAMP TO VOCAL RANGE            │
│  Convert min/max Hz to MIDI notes:   │
│    min_note = hz_to_midi(80)  ≈ 43  │
│    max_note = hz_to_midi(1100) ≈ 84 │
│  Remove any note outside this range. │
│  Rationale: BasicPitch sometimes     │
│  hallucinates sub-bass or artifacts  │
│  above the vocal register.           │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  3. REMOVE SHORT NOTES              │
│  If note.duration < min_note_length: │
│    discard note                      │
│  Rationale: Transient artifacts from │
│  consonants, breaths, etc.           │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  4. QUANTIZE TO GRID                 │
│  (Delegated to src/midi/quantizer)   │
│  Snap note.start and note.end to     │
│  nearest 16th-note grid position.    │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  5. ASSIGN CHANNEL                   │
│  Move all notes into a single        │
│  Instrument on Channel 0 (RH).      │
│  Set program = 0 (Acoustic Grand).   │
└──────────────────────────────────────┘
```

**BasicPitch API Integration:**

```python
# BasicPitch returns three numpy arrays + a PrettyMIDI object:
# model_output, midi_data, note_events = basic_pitch.inference.predict(
#     audio_path,
#     onset_threshold=...,
#     frame_threshold=...,
#     minimum_note_length=...,   # in seconds
#     minimum_frequency=...,     # in Hz
#     maximum_frequency=...,     # in Hz
# )
# We primarily use the midi_data (PrettyMIDI) object.
```

**Error Handling:**
- `FileNotFoundError` — vocals.wav missing (separator failure).
- `TranscriptionError` — BasicPitch model failure (wrap with informative message).
- Edge case: vocals.wav contains silence or very low energy → BasicPitch returns an empty MIDI → log a warning, return an empty but valid MIDI file.

---

### Step 3: Accompaniment Generation (The Fork)

This step branches into two mutually exclusive paths based on `has_piano`.

---

#### Step 3A: Piano Transcription (`has_piano == True`)

**Module:** `src/transcription/piano_transcriber.py`

**Purpose:** When the source audio already contains piano, use the ByteDance `piano_transcription_inference` model to extract a high-fidelity piano transcription from the instrumental mix.

```python
class PianoTranscriber:
    """Transcribe piano audio to MIDI using ByteDance's piano transcription model."""

    def __init__(self, config: PianoTranscriptionConfig) -> None:
        """
        Args:
            config: Pydantic model containing:
                - device (str): "auto" | "cpu" | "cuda"
                - checkpoint (Optional[str]): Path to model checkpoint, or None for default.
        """

    def transcribe(self, audio_path: Path, output_path: Path) -> Path:
        """
        Transcribe piano content from an audio file to MIDI.

        Args:
            audio_path: Path to instrumental.wav (bass + other from Step 1).
            output_path: Path for the output MIDI file.

        Returns:
            Path to the transcribed MIDI file.
            Preserves: note polyphony, velocities, sustain pedal (CC64) events.

        Raises:
            TranscriptionError: If the model fails or produces empty output.
        """
```

**Internal Logic:**

1. Load audio at 16kHz (the model's expected sample rate) using `librosa.load()`.
2. Instantiate the transcriber: `PianoTranscription(device=device, checkpoint_path=checkpoint)`.
3. Call `transcriber.transcribe(audio, output_path)`.
4. Load the output MIDI via `pretty_midi.PrettyMIDI(output_path)` for validation.
5. Verify that at least one instrument with notes exists; raise `TranscriptionError` if empty.
6. The output preserves velocity dynamics and sustain pedal (CC64) control changes — **do not strip these** as they are critical for a natural-sounding piano arrangement.

**Key Notes:**
- The ByteDance model transcribes **all piano-like content** — if the source has both piano and synth pads, both may appear. This is acceptable for MVP; future versions could add a secondary filtering step.
- The model outputs to a single MIDI track. Post-processing in Step 4 will split into hands if needed.

---

#### Step 3B: Algorithmic Arranger (`has_piano == False`)

**Modules:** `src/arranger/chord_detector.py`, `src/arranger/beat_analyzer.py`, `src/arranger/pattern_library.py`, `src/arranger/arranger.py`

**Purpose:** When the source audio has no piano, programmatically generate a piano accompaniment by detecting chords and beats, then mapping them onto musical patterns.

##### 4.3.1 `beat_analyzer.py` — `BeatAnalyzer`

```python
class BeatAnalyzer:
    """Extract tempo, beat positions, and downbeat locations from audio."""

    def analyze(self, audio_path: Path) -> BeatGrid:
        """
        Args:
            audio_path: Path to instrumental.wav.

        Returns:
            BeatGrid dataclass:
                - tempo (float): Estimated BPM.
                - beat_times (np.ndarray): Timestamps of every beat in seconds.
                - downbeat_times (np.ndarray): Timestamps of bar-starting beats.
                - time_signature (tuple): e.g., (4, 4) — estimated or default.
        """
```

**Internal Logic:**

1. Load audio via `librosa.load(audio_path, sr=22050)`.
2. Extract tempo and beat frames: `tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)`.
3. Convert frames to time: `beat_times = librosa.frames_to_time(beat_frames, sr=sr)`.
4. Estimate downbeats: assume 4/4 time, take every 4th beat as a downbeat (for MVP).
5. Return `BeatGrid`.

##### 4.3.2 `chord_detector.py` — `ChordDetector`

```python
class ChordDetector:
    """Detect chord progressions from audio using chroma feature analysis."""

    def detect(self, audio_path: Path, beat_grid: BeatGrid) -> list[ChordEvent]:
        """
        Args:
            audio_path: Path to instrumental.wav.
            beat_grid: Beat timing information from BeatAnalyzer.

        Returns:
            List of ChordEvent, each containing:
                - start_time (float): Start timestamp in seconds.
                - end_time (float): End timestamp in seconds.
                - chord_label (str): e.g., "Cmaj", "Am", "F#dim"
                - root_note (int): MIDI note number of the chord root.
                - chord_type (str): "major" | "minor" | "diminished" | "augmented" | "dominant7"
                - intervals (list[int]): Semitone intervals from root, e.g., [0, 4, 7] for major.
        """
```

**Internal Logic:**

1. Compute chroma features: `chroma = librosa.feature.chroma_cqt(y=y, sr=sr)`.
2. Segment chroma by beat boundaries — average the chroma vectors within each beat window.
3. For each beat-level chroma vector, compare against a **chord template dictionary**:
   ```
   CHORD_TEMPLATES = {
       "major":      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
       "minor":      [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
       "diminished": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
       "dominant7":  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
       ...
   }
   ```
4. For each of the 12 pitch classes as potential root, circularly shift the chroma and compute cosine similarity against each template.
5. The highest-scoring (root, chord_type) pair wins.
6. Consolidate consecutive identical chords into single `ChordEvent` spans.

**Accuracy Caveat:** Chroma-based chord detection is approximate. For MVP, this is acceptable. Future improvements could integrate a pre-trained chord recognition model (e.g., `madmom` or a fine-tuned neural network).

##### 4.3.3 `pattern_library.py` — `PatternLibrary`

```python
class PatternLibrary:
    """Load and manage MIDI arrangement patterns from JSON definitions."""

    def __init__(self, pattern_dir: Path) -> None:
        """Load all .json pattern files from the given directory."""

    def get_pattern(self, pattern_name: str) -> ArrangementPattern:
        """
        Returns:
            ArrangementPattern dataclass:
                - name (str): e.g., "pop_ballad"
                - beats_per_bar (int): e.g., 4
                - events (list[PatternEvent]): Rhythmic blueprint, each containing:
                    - beat_position (float): e.g., 1.0, 2.0, 2.5 (for "and of 2")
                    - note_type (str): "root" | "triad" | "octave" | "fifth"
                    - octave_offset (int): e.g., -1 for bass register
                    - velocity (int): 0-127
                    - duration_beats (float): How long the note sustains
        """
```

**Example Pattern JSON (`pop_ballad.json`):**

```json
{
  "name": "pop_ballad",
  "beats_per_bar": 4,
  "events": [
    {"beat_position": 1.0, "note_type": "root",  "octave_offset": -2, "velocity": 80, "duration_beats": 1.0},
    {"beat_position": 2.0, "note_type": "triad", "octave_offset": -1, "velocity": 60, "duration_beats": 0.5},
    {"beat_position": 3.0, "note_type": "triad", "octave_offset": -1, "velocity": 65, "duration_beats": 0.5},
    {"beat_position": 4.0, "note_type": "triad", "octave_offset": -1, "velocity": 60, "duration_beats": 0.5}
  ]
}
```

**Note Type Resolution:**

| `note_type` | Produced Notes (relative to chord root) |
|:------------|:----------------------------------------|
| `"root"` | Root note only (e.g., C2) |
| `"fifth"` | Root + P5 (e.g., C2 + G2) |
| `"triad"` | Root + 3rd + 5th (e.g., C3 + E3 + G3, using chord's intervals) |
| `"octave"` | Root at specified octave (e.g., C4) |

##### 4.3.4 `arranger.py` — `AlgorithmicArranger`

```python
class AlgorithmicArranger:
    """
    Generate a piano accompaniment MIDI by mapping detected chords
    onto rhythmic patterns aligned with beat positions.
    """

    def __init__(
        self,
        chord_detector: ChordDetector,
        beat_analyzer: BeatAnalyzer,
        pattern_library: PatternLibrary,
        config: ArrangerConfig,
    ) -> None: ...

    def arrange(
        self,
        audio_path: Path,
        output_path: Path,
        pattern_name: str = "pop_ballad",
    ) -> Path:
        """
        Generate accompaniment MIDI from audio analysis.

        Args:
            audio_path: Path to instrumental.wav.
            output_path: Path for the output MIDI file.
            pattern_name: Which rhythmic pattern to apply.

        Returns:
            Path to the generated accompaniment MIDI.
            All notes are assigned to MIDI Channel 1 (Left Hand).

        Algorithm:
            1. Analyze beats → BeatGrid
            2. Detect chords → list[ChordEvent]
            3. Load pattern → ArrangementPattern
            4. For each bar (defined by consecutive downbeats):
                a. Look up the active chord at this bar's start time.
                b. For each PatternEvent in the pattern:
                    - Convert beat_position to absolute time using beat_grid.
                    - Resolve note_type against the active chord's intervals.
                    - Apply octave_offset.
                    - Create pretty_midi.Note with computed pitch, start, end, velocity.
                c. Add all notes to a PrettyMIDI Instrument on Channel 1.
            5. Write MIDI to output_path.
        """
```

**Detailed Algorithm Walkthrough:**

```
Given:
  beat_grid.downbeat_times = [0.0, 2.0, 4.0, 6.0, ...]  (bars at 120 BPM)
  chords = [ChordEvent("Cmaj", 0.0, 4.0), ChordEvent("Am", 4.0, 8.0), ...]
  pattern = pop_ballad (as defined above)

For bar starting at t=0.0 (Cmaj, root=C3=60):
  Event 1: beat 1.0 → t=0.0, root at octave_offset=-2 → C2 (36), vel=80, dur=0.5s
  Event 2: beat 2.0 → t=0.5, triad at -1 → [C3(48), E3(52), G3(55)], vel=60, dur=0.25s
  Event 3: beat 3.0 → t=1.0, triad at -1 → [C3(48), E3(52), G3(55)], vel=65, dur=0.25s
  Event 4: beat 4.0 → t=1.5, triad at -1 → [C3(48), E3(52), G3(55)], vel=60, dur=0.25s

For bar starting at t=2.0 (still Cmaj): ... repeat ...
For bar starting at t=4.0 (Am, root=A2=57):
  Event 1: beat 1.0 → t=4.0, root at -2 → A1 (33), vel=80, dur=0.5s
  Event 2: beat 2.0 → t=4.5, triad at -1 → [A2(45), C3(48), E3(52)], vel=60, dur=0.25s
  ...
```

---

### Step 4: The Playability Filter & MIDI Merging

**Modules:** `src/midi/cleaner.py`, `src/midi/quantizer.py`, `src/midi/playability.py`, `src/midi/merger.py`

This is the **most algorithmically dense step** and is critical for output quality.

#### 4.4.1 `quantizer.py` — `Quantizer`

```python
class Quantizer:
    """Snap MIDI note start and end times to a musical grid."""

    def __init__(self, grid_resolution: int = 16) -> None:
        """
        Args:
            grid_resolution: 16 for 16th notes, 32 for 32nd notes, etc.
        """

    def quantize(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Quantize all note on/off times to the nearest grid position.

        Args:
            midi: Input PrettyMIDI object.

        Returns:
            New PrettyMIDI object with quantized note timings.

        Algorithm:
            1. Extract tempo from the MIDI (or use a fixed tempo if not embedded).
            2. Compute grid spacing: grid_sec = (60.0 / tempo) / (grid_resolution / 4)
               Example: 120 BPM, 16th notes → 0.125 seconds per grid unit.
            3. For each note:
                note.start = round(note.start / grid_sec) * grid_sec
                note.end   = round(note.end / grid_sec) * grid_sec
                Ensure note.end > note.start (minimum duration = 1 grid unit).
            4. Merge/deduplicate notes that collapse onto identical (start, end, pitch).
        """
```

**Edge Cases:**
- Two notes quantized to the same start but with different pitches → keep both (this is a chord).
- A note quantized to zero duration → set `end = start + grid_sec` (minimum 1 grid unit).
- Overlapping notes on the same pitch after quantization → merge into a single longer note.

#### 4.4.2 `cleaner.py` — `MidiCleaner`

```python
class MidiCleaner:
    """Remove artifacts and normalize MIDI data for piano playback."""

    @staticmethod
    def strip_pitch_bends(midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """Remove all pitch bend events from all instruments."""

    @staticmethod
    def clamp_note_range(
        midi: pretty_midi.PrettyMIDI,
        min_note: int = 21,   # A0 — lowest piano key
        max_note: int = 108,  # C8 — highest piano key
    ) -> pretty_midi.PrettyMIDI:
        """Remove notes outside the standard 88-key piano range."""

    @staticmethod
    def assign_channel(
        midi: pretty_midi.PrettyMIDI,
        channel: int,
    ) -> pretty_midi.PrettyMIDI:
        """Move all notes in all instruments to a specific MIDI channel.
        Channel 0 = Right Hand, Channel 1 = Left Hand."""

    @staticmethod
    def set_instrument(
        midi: pretty_midi.PrettyMIDI,
        program: int = 0,   # 0 = Acoustic Grand Piano
    ) -> pretty_midi.PrettyMIDI:
        """Set all instruments to the specified MIDI program."""
```

#### 4.4.3 `playability.py` — `PlayabilityFilter`

This is the most complex module. It enforces human-playability constraints on the merged MIDI.

```python
class PlayabilityFilter:
    """
    Enforce human-playability constraints on piano MIDI:
    - Maximum hand span (interval between lowest and highest simultaneous note)
    - Maximum polyphony per hand
    - Final quantization pass
    """

    def __init__(self, config: PlayabilityConfig) -> None:
        """
        Args:
            config: Contains:
                - max_hand_span_semitones (int): default 15
                - max_polyphony_per_hand (int): default 4
                - quantization_grid (int): default 16
                - pruning_priority (list[str]): interval names to prune first
        """

    def apply(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Apply all playability constraints, in order:
            1. Hand span enforcement
            2. Polyphony reduction
            3. Final quantization

        Processes each MIDI channel (hand) independently.

        Returns:
            A new PrettyMIDI object satisfying all constraints.
        """

    def _enforce_hand_span(
        self, notes: list[pretty_midi.Note], max_span: int
    ) -> list[pretty_midi.Note]:
        """
        At each time slice, if the interval between the lowest and highest
        note exceeds max_span semitones, prune inner notes.

        Strategy:
            1. Build a timeline of note events (on/off) sorted by time.
            2. At each "note on" event, compute the set of currently sounding notes.
            3. If span(current_notes) > max_span:
                a. Always preserve the lowest note (bass/root) and highest note (melody).
                b. Remove inner notes starting from those closest to the middle,
                   until span <= max_span.
            4. Return the pruned note list.
        """

    def _enforce_polyphony(
        self, notes: list[pretty_midi.Note], max_voices: int
    ) -> list[pretty_midi.Note]:
        """
        At each time slice, if more than max_voices notes sound simultaneously,
        prune excess notes.

        Strategy:
            1. Build a timeline of note events sorted by time.
            2. At each "note on" event, if len(sounding_notes) > max_voices:
                a. Classify each sounding note by its interval relative to the
                   lowest sounding note (the root).
                b. Remove notes matching the pruning_priority list, in order:
                   - First remove P5 (7 semitones from root)
                   - Then M3 (4 semitones from root)
                   - Then m3 (3 semitones from root)
                   - As a last resort, remove the note closest to the median pitch
                c. Repeat until len(sounding_notes) <= max_voices.
            3. Return the pruned note list.
        """
```

**Time-Slice Analysis Algorithm:**

The core challenge is efficiently computing "which notes are sounding at time T." The approach:

```
1. Collect all (time, event_type, note) tuples:
   For each note n:
     events.append( (n.start, "ON",  n) )
     events.append( (n.end,   "OFF", n) )

2. Sort events by time (break ties: "OFF" before "ON" at same timestamp).

3. Maintain a set `active_notes`.

4. Walk through events:
   If event is "ON":
     active_notes.add(note)
     if constraint_violated(active_notes):
       prune(active_notes)  # modifies note list in-place by truncating/removing
   If event is "OFF":
     active_notes.discard(note)
```

**Why preserve lowest and highest?**
- The **lowest note** is typically the harmonic root — removing it destroys the chord identity.
- The **highest note** is typically the melody — removing it destroys the musical intent.
- **Inner voices** (3rds, 5ths) are perceptually less critical and can be dropped with minimal musical damage.

#### 4.4.4 `merger.py` — `MidiMerger`

```python
class MidiMerger:
    """Merge multiple MIDI files into a single multi-channel MIDI."""

    @staticmethod
    def merge(
        midi_paths: list[Path],
        output_path: Path,
        tempo: float | None = None,
    ) -> Path:
        """
        Merge multiple MIDI files into a single file.

        Args:
            midi_paths: Ordered list of MIDI file paths.
                - midi_paths[0] is expected to be Right Hand (Channel 0)
                - midi_paths[1] is expected to be Left Hand (Channel 1)
                Channel assignments should already be set by upstream modules.
            output_path: Where to write the merged MIDI.
            tempo: If provided, override the tempo in the output MIDI. If None,
                   use the tempo from the first MIDI file.

        Returns:
            Path to the merged output MIDI file.

        Logic:
            1. Create a new PrettyMIDI object with the target tempo.
            2. For each input MIDI:
                a. Load via pretty_midi.PrettyMIDI(path).
                b. Copy all Instrument objects into the output MIDI.
                c. Preserve control changes (sustain pedal, etc.).
            3. Write to output_path.
        """
```

**Step 4 Integration Flow:**

```
vocals.mid (Ch.0)  ──┐
                      ├──→ MidiMerger.merge() ──→ combined.mid
accompaniment.mid     │                              │
(Ch.1) ──────────────┘                              │
                                                     ▼
                                            PlayabilityFilter.apply()
                                                     │
                                                     ▼
                                            Quantizer.quantize()
                                                     │
                                                     ▼
                                            final_playable.mid
```

---

### Step 5: Visual Rendering

**Module:** `src/video/renderer.py`

#### 4.5.1 `renderer.py` — `VideoRenderer`

**Purpose:** Invoke MIDIVisualizer as a subprocess to render the final Synthesia-style video.

```python
class VideoRenderer:
    """Render a falling-notes video from MIDI + audio using MIDIVisualizer CLI."""

    def __init__(self, config: VideoConfig) -> None:
        """
        Args:
            config: Pydantic model containing:
                - midi_visualizer_path (str): Path to MIDIVisualizer binary.
                - resolution (str): e.g., "1920x1080"
                - fps (int): e.g., 60
                - background_color (str): e.g., "0.1 0.1 0.12"
                - note_speed (float): Falling speed multiplier.
                - additional_args (list[str]): Extra CLI flags.
        """

    def render(
        self,
        midi_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> Path:
        """
        Render the Synthesia-style video.

        Args:
            midi_path: Path to final_playable.mid.
            audio_path: Path to original_audio.mp3 (for audio track sync).
            output_path: Path for the output MP4 file.

        Returns:
            Path to the rendered MP4 video.

        Raises:
            FileNotFoundError: If MIDIVisualizer binary is not found.
            RenderingError: If the subprocess exits with a non-zero code.
            TimeoutError: If rendering exceeds the timeout (configurable).
        """

    def _build_command(
        self,
        midi_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> list[str]:
        """
        Construct the MIDIVisualizer CLI command.

        Example output:
            [
                "/usr/local/bin/MIDIVisualizer",
                "--midi", "/path/to/final_playable.mid",
                "--export", "/path/to/output.mp4",
                "--format", "MPEG4",
                "--size", "1920", "1080",
                "--framerate", "60",
                "--bg-color", "0.1", "0.1", "0.12",
                "--speed", "1.0",
            ]
        """

    def _validate_binary(self) -> None:
        """Check that MIDIVisualizer is accessible and executable."""
```

**Internal Logic:**

1. Validate that the MIDIVisualizer binary exists and is executable.
2. Build the CLI command list from config parameters.
3. Execute via `subprocess.run(cmd, capture_output=True, timeout=600)`.
4. Check return code. On failure, log stderr and raise `RenderingError`.
5. Verify the output MP4 file exists and has a non-zero size.
6. **Audio sync:** MIDIVisualizer natively supports audio overlay via `--audio` flag (verify availability in the installed version). If the version doesn't support `--audio`, use a post-processing step with FFmpeg:
   ```bash
   ffmpeg -i video_no_audio.mp4 -i original_audio.mp3 \
          -c:v copy -c:a aac -shortest final_synthesia_video.mp4
   ```

**Audio Synchronization Note:**

MIDIVisualizer renders visuals from MIDI timestamps, not from the audio. Since the MIDI was derived from the audio (via transcription), they are inherently time-aligned **if we do not modify the global timing**. Steps 1-4 preserve absolute timestamps. The only risk is if quantization shifts notes significantly — but with a 16th-note grid at typical tempos (100-140 BPM), the maximum shift is ~37-53 ms, which is perceptually acceptable.

---

## 5. Phase 3: Integration & Testing

### 5.1 The Orchestrator (`src/pipeline/orchestrator.py`)

```python
class PipelineOrchestrator:
    """
    Main entry point that wires together all pipeline steps
    based on user-defined toggles.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize all sub-modules from config.

        Instantiates:
            - StemSeparator
            - StemMixer
            - VocalTranscriber
            - PianoTranscriber
            - AlgorithmicArranger (with ChordDetector, BeatAnalyzer, PatternLibrary)
            - MidiCleaner
            - Quantizer
            - PlayabilityFilter
            - MidiMerger
            - VideoRenderer
            - Logger
        """

    def run(
        self,
        audio_path: Path,
        include_vocals: bool,
        has_piano: bool,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline.

        Args:
            audio_path: Path to the input audio file.
            include_vocals: Whether to transcribe vocals into the right hand.
            has_piano: Whether the source audio contains piano.
            output_dir: Override for the output directory.

        Returns:
            PipelineResult dataclass:
                - midi_path (Path): Path to final_playable.mid
                - video_path (Path): Path to final_synthesia_video.mp4
                - run_id (str): Unique identifier for this pipeline run
                - duration_seconds (float): Total wall-clock time
                - steps_completed (list[str]): Names of steps that ran
                - warnings (list[str]): Non-fatal issues encountered

        Raises:
            PipelineError: If any critical step fails unrecoverably.
        """
```

**Orchestrator Control Flow (Pseudocode):**

```python
def run(self, audio_path, include_vocals, has_piano, output_dir):
    run_id = generate_run_id()  # e.g., "run_20260525_150204"
    run_dir = self._create_run_directory(run_id)
    log = self.logger.bind(run_id=run_id)

    try:
        # ── STEP 1: Stem Separation ──
        log.info("Step 1: Separating stems...")
        stems = self.separator.separate(audio_path)
        instrumental_path = self.mixer.mix_stems(
            [stems.bass_path, stems.other_path],
            run_dir / "instrumental.wav",
        )

        # ── STEP 2: Vocal Transcription (conditional) ──
        vocals_midi_path = None
        if include_vocals:
            log.info("Step 2: Transcribing vocals...")
            raw_vocals_midi = self.vocal_transcriber.transcribe(
                stems.vocals_path,
                run_dir / "raw_vocals.mid",
            )
            # Clean: strip pitch bends, clamp range, assign channel 0 (RH)
            vocals_midi = load_midi(raw_vocals_midi)
            vocals_midi = self.cleaner.strip_pitch_bends(vocals_midi)
            vocals_midi = self.cleaner.clamp_note_range(vocals_midi, 43, 84)
            vocals_midi = self.cleaner.assign_channel(vocals_midi, channel=0)
            vocals_midi = self.cleaner.set_instrument(vocals_midi, program=0)
            vocals_midi = self.quantizer.quantize(vocals_midi)
            vocals_midi_path = run_dir / "vocals_cleaned.mid"
            vocals_midi.write(str(vocals_midi_path))
        else:
            log.info("Step 2: Skipped (include_vocals=False)")

        # ── STEP 3: Accompaniment Generation (fork) ──
        if has_piano:
            log.info("Step 3A: Transcribing existing piano...")
            accompaniment_path = self.piano_transcriber.transcribe(
                instrumental_path,
                run_dir / "accompaniment.mid",
            )
        else:
            log.info("Step 3B: Generating algorithmic arrangement...")
            accompaniment_path = self.arranger.arrange(
                instrumental_path,
                run_dir / "accompaniment.mid",
            )

        # Assign accompaniment to Channel 1 (LH)
        acc_midi = load_midi(accompaniment_path)
        acc_midi = self.cleaner.assign_channel(acc_midi, channel=1)
        acc_midi = self.cleaner.set_instrument(acc_midi, program=0)
        acc_midi.write(str(accompaniment_path))

        # ── STEP 4: Merge & Apply Playability Filter ──
        log.info("Step 4: Merging and filtering for playability...")
        midi_inputs = []
        if vocals_midi_path:
            midi_inputs.append(vocals_midi_path)
        midi_inputs.append(accompaniment_path)

        combined_path = self.merger.merge(
            midi_inputs,
            run_dir / "combined.mid",
        )

        combined_midi = load_midi(combined_path)
        filtered_midi = self.playability_filter.apply(combined_midi)
        final_midi = self.quantizer.quantize(filtered_midi)  # Final pass

        final_midi_path = run_dir / "final_playable.mid"
        final_midi.write(str(final_midi_path))

        # Copy to output directory
        output_midi = output_dir / f"{run_id}_final.mid"
        shutil.copy2(final_midi_path, output_midi)

        # ── STEP 5: Render Video ──
        log.info("Step 5: Rendering Synthesia video...")
        output_video = output_dir / f"{run_id}_synthesia.mp4"
        self.renderer.render(
            midi_path=final_midi_path,
            audio_path=audio_path,
            output_path=output_video,
        )

        log.info(f"Pipeline complete. Output: {output_video}")

        return PipelineResult(
            midi_path=output_midi,
            video_path=output_video,
            run_id=run_id,
            ...
        )

    except Exception as e:
        log.error(f"Pipeline failed at step: {e}")
        raise PipelineError(run_id=run_id, cause=e)
```

### 5.2 CLI Entry Point (`scripts/run_pipeline.py`)

```python
@click.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--include-vocals / --no-vocals", default=True,
              help="Include vocal melody in the right hand.")
@click.option("--has-piano / --no-piano", default=True,
              help="Source audio contains piano (True) or needs arrangement (False).")
@click.option("--config", type=click.Path(), default="config.yaml",
              help="Path to configuration file.")
@click.option("--output-dir", type=click.Path(), default="data/output",
              help="Output directory for final artifacts.")
def main(audio_path, include_vocals, has_piano, config, output_dir):
    """AI Piano Arranger — Audio to Synthesia Pipeline."""
    # 1. Load config
    # 2. Initialize PipelineOrchestrator
    # 3. Call orchestrator.run(...)
    # 4. Print results summary
```

**Example Invocations:**

```bash
# Full pipeline: vocals + existing piano
python scripts/run_pipeline.py "data/input/song.mp3" --include-vocals --has-piano

# No vocals, generate piano arrangement from scratch
python scripts/run_pipeline.py "data/input/edm_track.mp3" --no-vocals --no-piano

# Custom config
python scripts/run_pipeline.py "data/input/rock.mp3" --no-vocals --has-piano --config custom.yaml
```

### 5.3 Configuration System (`src/pipeline/config.py`)

```python
from pydantic import BaseModel, Field
from pathlib import Path

class SeparatorConfig(BaseModel):
    model: str = "htdemucs"
    device: str = "auto"
    shifts: int = 1
    overlap: float = 0.25
    output_format: str = "wav"

class VocalTranscriptionConfig(BaseModel):
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 58.0
    minimum_frequency_hz: float = 80.0
    maximum_frequency_hz: float = 1100.0

class PianoTranscriptionConfig(BaseModel):
    device: str = "auto"
    checkpoint: str | None = None

class ArrangerConfig(BaseModel):
    default_pattern: str = "pop_ballad"
    pattern_dir: Path = Path("assets/patterns")
    chord_detection_method: str = "chroma"
    beats_per_bar: int = 4

class PlayabilityConfig(BaseModel):
    max_hand_span_semitones: int = 15
    max_polyphony_per_hand: int = 4
    quantization_grid: int = 16
    pruning_priority: list[str] = Field(default=["P5", "M3", "m3"])

class VideoConfig(BaseModel):
    midi_visualizer_path: str = "MIDIVisualizer"
    resolution: str = "1920x1080"
    fps: int = 60
    background_color: str = "0.1 0.1 0.12"
    note_speed: float = 1.0
    additional_args: list[str] = Field(default_factory=list)

class PipelineConfig(BaseModel):
    """Root configuration aggregating all sub-configs."""
    include_vocals: bool = True
    has_piano: bool = True
    output_dir: Path = Path("data/output")
    intermediate_dir: Path = Path("data/intermediate")

    separator: SeparatorConfig = SeparatorConfig()
    vocal_transcription: VocalTranscriptionConfig = VocalTranscriptionConfig()
    piano_transcription: PianoTranscriptionConfig = PianoTranscriptionConfig()
    arranger: ArrangerConfig = ArrangerConfig()
    playability: PlayabilityConfig = PlayabilityConfig()
    video: VideoConfig = VideoConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file, merging with defaults."""
```

### 5.4 Data Models (`src/pipeline/config.py` or separate `models.py`)

```python
@dataclass
class StemResult:
    vocals_path: Path
    bass_path: Path
    drums_path: Path
    other_path: Path

@dataclass
class BeatGrid:
    tempo: float
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    time_signature: tuple[int, int] = (4, 4)

@dataclass
class ChordEvent:
    start_time: float
    end_time: float
    chord_label: str        # e.g., "Cmaj"
    root_note: int          # MIDI note number
    chord_type: str         # "major", "minor", etc.
    intervals: list[int]    # semitone intervals, e.g., [0, 4, 7]

@dataclass
class PipelineResult:
    midi_path: Path
    video_path: Path
    run_id: str
    duration_seconds: float
    steps_completed: list[str]
    warnings: list[str]
```

### 5.5 Testing Strategy

#### 5.5.1 Unit Tests

Each module gets its own test file. Tests are organized by **function** and **edge case**.

| Test File | Module Under Test | Key Test Cases |
|:----------|:------------------|:---------------|
| `test_separator.py` | `StemSeparator` | • Valid MP3 → 4 stems output<br>• Invalid file path → `FileNotFoundError`<br>• Corrupt audio → `SeparationError`<br>• Verify stem count and file existence |
| `test_vocal_transcriber.py` | `VocalTranscriber` | • Clean vocal → MIDI with notes<br>• Silent audio → empty MIDI (no crash)<br>• Pitch bends stripped<br>• Notes clamped to vocal range<br>• Channel assignment = 0 |
| `test_piano_transcriber.py` | `PianoTranscriber` | • Piano audio → MIDI with polyphony<br>• Sustain pedal events preserved<br>• Velocity dynamics present<br>• Non-piano audio → graceful degradation |
| `test_arranger.py` | `AlgorithmicArranger` | • Known chord sequence → expected note pattern<br>• Pattern loading from JSON<br>• Beat grid alignment<br>• Channel assignment = 1 |
| `test_playability.py` | `PlayabilityFilter` | • **Hand span:** 2-octave chord → pruned to ≤15 ST<br>• **Polyphony:** 6-note chord → pruned to ≤4<br>• **Preservation:** Lowest + highest always kept<br>• **Pruning order:** P5 removed before M3<br>• **Edge:** Single note → no change<br>• **Edge:** Exactly at limit → no change |
| `test_quantizer.py` | `Quantizer` | • Off-grid note → snapped to nearest 16th<br>• Already on-grid → unchanged<br>• Zero-duration after quantization → minimum 1 grid unit<br>• Duplicate notes merged |
| `test_merger.py` | `MidiMerger` | • Two MIDIs → single file with both channels<br>• Tempo preserved<br>• Control changes preserved<br>• Empty MIDI input handled |
| `test_renderer.py` | `VideoRenderer` | • Valid MIDI + audio → MP4 created<br>• Missing binary → `FileNotFoundError`<br>• Binary failure → `RenderingError` with stderr<br>• Command construction matches expected format |

#### 5.5.2 Test Fixtures (`conftest.py`)

```python
# Shared fixtures for all tests

@pytest.fixture
def sample_midi() -> pretty_midi.PrettyMIDI:
    """Create a minimal PrettyMIDI with a C major chord."""

@pytest.fixture
def sample_vocals_midi() -> pretty_midi.PrettyMIDI:
    """Create a PrettyMIDI simulating vocal transcription output
    (with pitch bends, out-of-range notes, short notes)."""

@pytest.fixture
def wide_span_midi() -> pretty_midi.PrettyMIDI:
    """Create a PrettyMIDI with notes spanning > 15 semitones."""

@pytest.fixture
def dense_polyphony_midi() -> pretty_midi.PrettyMIDI:
    """Create a PrettyMIDI with > 4 simultaneous notes."""

@pytest.fixture
def sample_audio_path(tmp_path) -> Path:
    """Generate a short sine wave WAV file for testing."""

@pytest.fixture
def beat_grid() -> BeatGrid:
    """Create a BeatGrid at 120 BPM with 16 beats."""

@pytest.fixture
def chord_events() -> list[ChordEvent]:
    """Create a simple I-V-vi-IV progression in C major."""
```

#### 5.5.3 Integration / End-to-End Tests (`test_pipeline_e2e.py`)

```python
class TestPipelineE2E:
    """End-to-end tests using real (short) audio fixtures."""

    def test_full_pipeline_vocals_and_piano(self):
        """Run with include_vocals=True, has_piano=True.
        Assert: final MIDI has 2 channels, video exists."""

    def test_full_pipeline_no_vocals_with_piano(self):
        """Run with include_vocals=False, has_piano=True.
        Assert: final MIDI has 1 channel (LH only)."""

    def test_full_pipeline_vocals_no_piano(self):
        """Run with include_vocals=True, has_piano=False.
        Assert: final MIDI has 2 channels, LH is algorithmically generated."""

    def test_full_pipeline_no_vocals_no_piano(self):
        """Run with include_vocals=False, has_piano=False.
        Assert: final MIDI has 1 channel (LH only, algorithmically generated)."""

    def test_playability_constraints_met(self):
        """After full pipeline, verify:
        - No hand exceeds 15 semitone span at any time slice
        - No hand exceeds 4 simultaneous notes at any time slice
        - All note starts/ends align to 16th note grid"""

    def test_idempotency(self):
        """Running the pipeline twice on the same input produces
        identical MIDI output (deterministic)."""
```

#### 5.5.4 Running Tests

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run only unit tests (fast, no GPU/audio required)
pytest tests/ -v -k "not e2e"

# Run only the playability tests
pytest tests/test_playability.py -v
```

---

## 6. Appendices

### Appendix A: MIDI Channel Conventions

| Channel | Hand | Usage |
|:--------|:-----|:------|
| 0 | Right Hand (RH) | Vocal melody transcription |
| 1 | Left Hand (LH) | Piano accompaniment (transcribed or generated) |
| 9 | *(reserved)* | General MIDI drum channel — never used |

> **Note:** MIDI channels are 0-indexed in `pretty_midi`. Channel 0 = the first non-drum channel, conventionally used for melody. Channel 1 = accompaniment. MIDIVisualizer will render Channel 0 and Channel 1 in different colors by default, providing visual hand distinction.

### Appendix B: Chord Template Dictionary

Used by `ChordDetector` for chroma-to-chord matching.

```python
CHORD_TEMPLATES: dict[str, list[int]] = {
    # Each list is a 12-element binary chroma vector
    # Index 0 = root, index 1 = minor 2nd, ..., index 11 = major 7th
    "major":       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # R, M3, P5
    "minor":       [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # R, m3, P5
    "diminished":  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # R, m3, TT
    "augmented":   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # R, M3, m6
    "dominant7":   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # R, M3, P5, m7
    "major7":      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # R, M3, P5, M7
    "minor7":      [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # R, m3, P5, m7
    "sus2":        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # R, M2, P5
    "sus4":        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # R, P4, P5
}

# Interval-to-semitone mapping (used by PlayabilityFilter pruning)
INTERVAL_SEMITONES: dict[str, int] = {
    "P1": 0,  "m2": 1,  "M2": 2,  "m3": 3,  "M3": 4,
    "P4": 5,  "TT": 6,  "P5": 7,  "m6": 8,  "M6": 9,
    "m7": 10, "M7": 11, "P8": 12,
}
```

### Appendix C: Error Hierarchy

```python
class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    def __init__(self, run_id: str, cause: Exception | None = None):
        self.run_id = run_id
        self.cause = cause

class SeparationError(PipelineError):
    """Raised when Demucs stem separation fails."""

class TranscriptionError(PipelineError):
    """Raised when vocal or piano transcription fails."""

class ArrangementError(PipelineError):
    """Raised when the algorithmic arranger fails."""

class PlayabilityError(PipelineError):
    """Raised when playability filtering encounters an unrecoverable state."""

class RenderingError(PipelineError):
    """Raised when MIDIVisualizer fails to produce output."""
```

### Appendix D: Performance Estimates

Rough wall-clock time estimates per step for a **4-minute pop song** on a system with an NVIDIA RTX 3060 GPU:

| Step | Operation | Estimated Time |
|:-----|:----------|:---------------|
| 1 | Demucs stem separation | 30-60 seconds |
| 2 | BasicPitch vocal transcription | 5-10 seconds |
| 3A | ByteDance piano transcription | 10-20 seconds |
| 3B | Algorithmic arrangement | 2-5 seconds |
| 4 | Playability filter + merge | < 1 second |
| 5 | MIDIVisualizer rendering | 60-120 seconds |
| | **Total (worst case)** | **~3.5 minutes** |

CPU-only execution will be significantly slower for Steps 1, 2, and 3A (potentially 3-5x).

### Appendix E: Future Enhancements (Post-MVP)

These are explicitly **out of scope** for the MVP but documented for roadmap planning:

1. **Hand Splitting for Piano Transcription (Step 3A):** Use a pitch-based heuristic (e.g., notes below C4 → LH, above → RH) to split the single-track piano transcription into two hands.
2. **Difficulty Levels:** Add a `difficulty` parameter ("easy", "medium", "hard") that adjusts polyphony limits, pattern complexity, and quantization resolution.
3. **Chord Recognition Model:** Replace the chroma-template matching in Step 3B with a pre-trained deep learning model (e.g., using `madmom` or a custom CNN) for more accurate chord detection.
4. **Web API / GUI:** Wrap the pipeline in a FastAPI server or Streamlit app for browser-based interaction.
5. **Real-time Preview:** Generate a short (30-second) preview video before rendering the full song.
6. **Multiple Arrangement Patterns:** Let the user choose or cycle through different accompaniment styles (arpeggiated, broken chords, waltz pattern, etc.) within a single song based on sections.
7. **Key Detection:** Automatically detect the song key and transpose the arrangement to be more piano-friendly (e.g., avoiding lots of black keys).
8. **Dynamic Velocity Curves:** Shape the generated accompaniment's velocity to follow the energy contour of the original audio (louder in choruses, softer in verses).

---

*End of Master Development Plan.*
