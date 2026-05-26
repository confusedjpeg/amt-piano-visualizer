# PIPELINE WORKFLOW & IMPLEMENTATION STATUS
## AI-Powered Audio-to-Piano-Synthesia Pipeline

This document provides a comprehensive, non-code technical breakdown of the **Project Piano** audio-to-piano pipeline. It details the step-by-step operational flow, the underlying algorithms, the specific technology stack utilized for each module, and the musical reasoning behind the processing steps.

---

## 1. High-Level Pipeline Architecture

The pipeline is designed as a highly modular, configurable system. It processes a raw audio file (such as a `.wav` or `.mp3` recording of a song) through advanced neural audio source separation, vocal and piano transcription, rhythmic beat/tempo tracking, chord detection, algorithmic keyboard arrangement, ergonomic playability filtering, and dynamic video synthesis.

### Modular System Flow

```mermaid
graph TD
    A[Input Audio File: .mp3 / .wav] --> B[Validation & Directory Setup]
    B --> C[Step 1: Meta Demucs Stem Separation]
    
    %% Stems
    C -->|vocals.wav| D[Vocal Stream]
    C -->|bass.wav & other.wav| E[Accompaniment Stream]
    C -->|drums.wav| F[Discarded Stems]
    
    %% Step 2: Vocal
    D --> G{include_vocals?}
    G -->|Yes| H[Step 2: Spotify BasicPitch Vocal Transcription]
    G -->|No| I[Skip Vocal Transcription]
    
    %% Step 3: Accompaniment
    E --> J[Waveform Mixer: Summation & Normalization]
    J -->|instrumental.wav| K{has_piano?}
    
    %% Step 3A/3B
    K -->|Yes: Path A| L[Step 3A: ByteDance Neural Piano Transcription]
    K -->|No: Path B| M[Step 3B: Algorithmic Keyboard Arranger]
    
    %% Path B Details
    M --> M1[Librosa Beat & Tempo Tracking]
    M --> M2[Librosa CQT Chroma Chord Extraction]
    M --> M3[JSON Voicing Pattern Mapping]
    
    %% Step 4: MIDI Processing
    H --> N[Vocal MIDI Cleaning & Post-Processing]
    N -->|Channel 0: Right Hand| O[Step 4: MIDI Merger]
    L -->|Channel 1: Left Hand| O
    M3 -->|Channel 1: Left Hand| O
    
    O --> P[Quantizer: Grid Snapping & Deduplication]
    P --> Q[Playability Filter: Ergonomic Constraint Solver]
    Q -->|final_playable.mid| R[Step 5: Synthesia-Style Video Rendering]
    
    %% Step 5 Details
    R --> S[MIDIVisualizer Subprocess Execution]
    S --> T{Direct Audio Support?}
    T -->|No| U[FFmpeg Fallback Muxer]
    T -->|Yes| V[Rendered Output Video]
    U --> V
    V --> W[data/output/<run_id>_synthesia.mp4]
    Q -->|final_playable.mid| X[data/output/<run_id>_final.mid]
```

---

## 2. Core Technology Stack Matrix

The following table summarizes the technology stack used across the pipeline and the specific responsibility of each dependency:

| Pipeline Step | Library / Technology | Version (Min) | Operational Role & Purpose |
| :--- | :--- | :--- | :--- |
| **Separation** | `demucs` | `4.0.0` | Meta's Hybrid Transformer Demucs model (`htdemucs`) for high-fidelity audio source separation. |
| **Separation** | `torchcodec` | `0.13.0` | High-efficiency audio decoding and tensor saving for virtual environment execution. |
| **Vocal MIDI** | `onnxruntime` | `1.16.0` | Executes Spotify's `basic-pitch` neural networks via ONNX for Windows compatibility. |
| **Piano MIDI** | `basic-pitch` (via `onnxruntime`) | `0.3.0` | Spotify's modern polyphonic piano transcription model, configured for the full piano range (A0–C8). |
| **Audio Analysis**| `librosa` | `0.10.0` | Audio loading, sample rate management, Constant-Q Transform (CQT) chroma computation, and tempo tracking. |
| **Audio I/O** | `soundfile` | `0.12.0` | Reads and writes wave files, summing and normalizing audio waveforms. |
| **MIDI Math** | `pretty_midi` | `0.2.10` | Object-oriented parsing, manipulation, editing, and writing of MIDI tracks, notes, and control changes. |
| **Math Engine** | `numpy` & `scipy` | `1.24.0` | Linear algebra, cosine similarity computations, grid rounding, and signal pad operations. |
| **Orchestration**| `pydantic` & `pyyaml` | `2.0` / `6.0` | Strictly typed runtime configuration management and YAML file parsing. |
| **Diagnostics** | `loguru` | `0.7.0` | Comprehensive logging, console tracing, and run-specific error reporting. |
| **Rendering** | `MIDIVisualizer` | *Latest* | Standalone compiled C++ command-line tool for ultra-smooth falling-notes rendering. |
| **Rendering** | `FFmpeg` | `4.4` | System-level binary used to mux high-quality audio tracks back onto rendered video streams. |

---

## 3. Dynamic Control Flow Matrix

The pipeline dynamically shifts its execution graph based on two boolean parameters: `include_vocals` (whether to extract and transcribe the singer's melody) and `has_piano` (whether to transcribe an existing piano or synthesize an accompaniment from scratch).

| `include_vocals` | `has_piano` | Right Hand (Channel 0) | Left Hand (Channel 1) | Core Operations Executed |
| :---: | :---: | :--- | :--- | :--- |
| **`True`** | **`True`** | Neural Vocal Transcription | Neural Piano Transcription | Separation → Vocal Transc. → Piano Transc. → Merging → Filtering → Rendering |
| **`True`** | **`False`** | Neural Vocal Transcription | Algorithmic Chord Arranger | Separation → Vocal Transc. → Algo Arrangement → Merging → Filtering → Rendering |
| **`False`** | **`True`** | *Unused* | Neural Piano Transcription | Separation → Piano Transc. → Channel Assignment → Filtering → Rendering |
| **`False`** | **`False`** | *Unused* | Algorithmic Chord Arranger | Separation → Algo Arrangement → Channel Assignment → Filtering → Rendering |

---

## 4. Step-by-Step Pipeline Workflow

### Step 1: Audio Pre-processing & Stem Separation
* **Primary Technology**: Meta's `demucs` (HTDemucs architecture), `torch`, `torchaudio`, and `soundfile`
* **Workflow & Execution**:
  1. **Validation**: The orchestrator receives the target audio file, validates its existence and integrity, and assigns a unique `run_id`.
  2. **Stem Decomposition**: The system loads Meta’s four-stem Hybrid Transformer Demucs model (`htdemucs`). This neural network analyzes the stereo frequency spectra and separates the mixed master audio track into four discrete, high-quality `.wav` audio files: `vocals.wav`, `bass.wav`, `drums.wav`, and `other.wav`.
  3. **Waveform Mixing**: The drums are discarded since drum beats are not mapped to harmonic piano voices. The `bass` and `other` (guitars, synths, strings) stems are fed into a waveform mixer.
  4. **Summation & Normalization**: The mixer reads both waveforms using `soundfile`, pads the shorter track with digital silence, performs an element-wise numerical summation of the wave arrays, and divides the resulting waveform by its peak value. This creates a balanced, clipping-free `instrumental.wav` file that serves as the basis for accompaniment.

> [!NOTE]
> Separating the vocals from the instrumentation first prevents the vocal frequencies from interfering with the chord detection and piano transcription algorithms downstream.

---

### Step 2: Vocal Transcription (Right Hand Melody)
* **Primary Technology**: Spotify's `basic-pitch` (leveraging `onnxruntime`), `pretty_midi`
* **Condition**: Triggered only if `include_vocals` is enabled.
* **Workflow & Execution**:
  1. **ONNX Inference**: The pipeline passes the isolated `vocals.wav` stem into the BasicPitch model. For Windows compatibility on Python 3.12, this runs inside the high-performance `onnxruntime` engine, generating raw MIDI events.
  2. **Pitch Bend Stripping**: Singers naturally employ continuous pitch slides and vibrato. On a mechanical piano, these represent hundreds of microtonal pitch wheel shifts. The transcriber strips all pitch bends in-place, yielding clear, solid notes.
  3. **Range Clamping**: The transcription is clamped to standard vocal frequencies (MIDI pitch 43 to 84, roughly E2 to C#6). This filters out deep sub-bass hums or high-frequency breaths that the model occasionally mistakes for notes.
  4. **Duration Filtering**: Brief notes shorter than a configured duration (e.g., 58ms) are discarded to clean up transients from consonants (like "t" or "k") and breathing patterns.
  5. **Melody Allocation**: The cleaned notes are assigned to MIDI **Channel 0** (internally tagged as the **Right Hand** melody) using the Acoustic Grand Piano instrument configuration.

---

### Step 3: Accompaniment Generation (Left Hand Harmony)
This phase forks into two distinct processing paths based on the `has_piano` config parameter.

#### Path 3A: Neural Piano Transcription (`has_piano == True`)
* **Primary Technology**: Spotify's `basic-pitch` (via `onnxruntime`)
* **Workflow & Execution**:
  1. **ONNX Inference**: The pipeline passes the isolated `instrumental.wav` stem into the BasicPitch model — the same modern neural network used for vocal transcription — but configured for the full piano frequency range (A0 at 27.5 Hz to C8 at 4186 Hz). This produces significantly cleaner onset timings and more accurate polyphonic pitch detection than the previous ByteDance model.
  2. **Polyphonic Transcription**: BasicPitch extracts polyphonic piano notes with precise onset/offset timings and velocity dynamics. Notes shorter than 50ms are filtered out during inference to eliminate micro-transients from harmonics.
  3. **Output Generation**: The raw MIDI output is validated to ensure notes exist and is assigned to MIDI **Channel 1** (representing the **Left Hand** accompaniment).

#### Path 3B: Algorithmic Keyboard Arrangement (`has_piano == False`)
* **Primary Technology**: `librosa`, `numpy`, `scipy`
* **Workflow & Execution**:
  1. **Tempo and Beat Grid Tracking**: Librosa's onset envelope spectral flux analyzer calculates the global tempo in Beats Per Minute (BPM) and maps every beat's exact timestamp in seconds. The arranger divides the timeline into bars using an estimated downbeat grid (e.g., every 4 beats for 4/4 time).
  2. **Constant-Q Transform (CQT) Chroma Extraction**: The `instrumental.wav` file is analyzed via Constant-Q Transform. This folds the audio spectrum into a 12-semitone chroma vector representing the intensity of the 12 chromatic pitches (C, C#, D... B) present in the music at any moment.
  3. **Beat-Synchronous Smoothing**: Chroma vectors are averaged within each detected beat boundary to smooth out transients and establish steady, beat-aligned harmonic states.
  4. **Cosine Similarity Template Matching**: Each beat's smoothed chroma vector is compared against a comprehensive template dictionary of chord qualities:
     * *Supported chords*: Major, Minor, Diminished, Augmented, Dominant 7th, Major 7th, Minor 7th, sus2, and sus4.
     * The algorithm circularly rolls the chroma vector across all 12 potential roots. It computes the cosine similarity between the vector and the chord templates. The highest similarity score determines the chord (e.g., "A minor").
  5. **Chord Consolidation**: Adjacent beats with identical chord classifications are merged to form musical chord spans.
  6. **Voicing Pattern Synthesis**: The arranger reads rhythmic blueprints from a JSON Pattern Library (e.g., `pop_ballad.json` or `arpeggiated.json`). For each bar, it retrieves the active chord and maps it onto the pattern:
     * *`root` note events* are placed in the deep bass register (transposed down).
     * *`triad` events* lay down full harmonic chord stack structures.
     * *`fifth` / `octave` events* fill in syncopated rhythmic pulses.
  7. **Accompaniment Allocation**: The resulting notes are written to MIDI **Channel 1** (the **Left Hand** track).

---

### Step 4: Rhythmic Quantization, Merging, & Playability Filtering
* **Primary Technology**: `pretty_midi`, `numpy`
* **Workflow & Execution**:
  1. **MIDI Merging**: If vocals are present, the vocal MIDI (Channel 0) and the accompaniment MIDI (Channel 1) are merged into a single multi-channel file.
  2. **Rhythmic Quantization**: Note start and end times are snapped to a strict musical grid (configured as 16th notes or 32nd notes relative to the tracked tempo). Snapping ensures visual alignments in the final video.
  3. **Timing Cleanup & Deduplication**: If multiple notes of the same pitch collapse to the same start/end time after quantization, they are merged. Notes that overlap are stitched together, and notes with zero duration are expanded to a minimum of one grid unit.
  4. **Ergonomic Playability Constraints**: Real humans have mechanical limits (e.g., ten fingers, maximum stretch). To make the MIDI physically playable on a real keyboard, the pipeline applies a **Time-Slice Event Sweep**:
     * The notes are converted into chronological ON/OFF events. The solver sweeps from the beginning to the end of the song, examining all sounding pitches at every microsecond.
     * **Hand Span Filtering**: If the distance between the lowest and highest simultaneous note on a single hand exceeds the human hand span limit (e.g., 15 semitones / 1.2 octaves), the filter prunes the inner notes. It calculates the median pitch of the chord and deletes notes closest to that median, leaving the critical low bass note and high melody note intact.
     * **Polyphony Limit Filtering**: If the number of sounding notes on one hand exceeds the physical finger count limit (e.g., 4 simultaneous notes), the filter drops the excess. It determines the interval of each note relative to the bass root and prunes notes based on a harmonic priority queue: Perfect 5ths are dropped first (as they add thick, redundant weight), followed by Major 3rds, then Minor 3rds.
  5. **Output Delivery**: The normalized, human-playable, and quantized file is saved as the master MIDI artifact in `data/output`.

---

### Step 5: Synthesia-Style Video Rendering
* **Primary Technology**: `MIDIVisualizer` CLI binary, `ffmpeg`
* **Workflow & Execution**:
  1. **Binary Verification**: The pipeline validates the presence of the standalone compiled `MIDIVisualizer` C++ application, verifying its path.
  2. **Synthesia Visual Rendering**: The final playable MIDI file is passed to the visualizer as a background subprocess. The tool renders a stunning, high-definition (1920x1080), 60 FPS video of colored blocks falling onto an interactive piano keyboard, customized with a harmonious dark charcoal theme.
  3. **Audio-Video Synchronization (Muxing)**:
     * *Primary Method*: The system attempts to feed the original audio directly into the visualizer using the `--audio` flag so it encodes the audio track in a single pass.
     * *Fallback Muxing*: If the local version of `MIDIVisualizer` does not support direct audio embedding, the pipeline renders a silent MP4. It then invokes `FFmpeg` in a secondary subprocess to copy the visual stream while encoding and muxing the original high-quality audio file back onto the video track.
  4. **Final Export**: The output is validated to ensure non-zero file sizes and saved directly to `data/output` as a shareable Synthesia video file, synced perfectly to the milliseconds of the music.

---

## 5. Summary of Key Algorithmic and Musical Decisions

* **Vocal Vibrato Removal**: Stripping pitch bends prevents synthesized pianos from sounding out-of-tune or "wobbly" when mimicking vocal contours.
* **Ergonomic Pruning (Bass & Melody Protection)**: When pruning notes for hand span or polyphony constraints, the algorithm *never* deletes the highest note (retains melody definition) or the lowest note (retains the harmonic foundation).
* **Harmonic Pruning Priority**: Perfect 5ths are pruned before 3rds because the 3rd defines whether a chord is major or minor (crucial for emotional flavor), whereas the 5th is harmonically redundant and can be omitted without changing the chord quality.
* **ONNX Execution on Windows**: Utilizing ONNX runtime for Spotify's BasicPitch allows the pipeline to bypass complex Tensorflow setup issues and run extremely fast on standard consumer laptops.
* **FFmpeg Audio Copier**: Muxing the original audio back onto the video ensures the user hears the rich production value of the original song, while viewing their simplified, playable piano arrangements.
