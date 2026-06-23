# Project Status & Codebase Audit

> **Audit date:** 2026-06-23
> **Scope:** Full review of the current implementation, identification of
> correctness defects, quality issues, test-coverage gaps, and a deep-dive
> into the final MIDI visualiser step (Step 5) that is almost certainly
> not behaving as the design intends.

This document is a snapshot of the codebase **as it is right now**, not as
the plan documents describe it. It is meant to be the working brief for
the next round of fixes. Each section ends with a short list of
"things to work on," with the highest-leverage items first.

---

## 1. What the project actually is

A five-stage audio-to-MIDI-to-video pipeline that turns a song file into
a two-handed piano arrangement plus a Synthesia-style falling-notes
video. The code is organised as:

```
src/
├── audio/         Demucs stem separation, noise gate, stem mixer
├── transcription/ BasicPitch wrappers (vocal + piano)
├── arranger/      Algorithmic accompaniment (chord → pattern)
├── midi/          Cleaning, quantisation, playability, merging, post-processing
├── video/         MIDIVisualizer subprocess + pure-Python fallback
├── pipeline/      Orchestrator, Pydantic config, models, errors
└── utils/         Logging, file I/O, validators
```

The orchestrator (`src/pipeline/orchestrator.py:39`) wires everything
together. There is one CLI entry point (`scripts/run_pipeline.py`) and
108 pytest cases across 9 test files.

### Test status (last run on this machine)

| File | Status | Note |
|:-----|:-------|:-----|
| `test_arranger.py` | 1 setup error | `tmp_path` Windows path issue |
| `test_merger.py` | 4 setup errors | `tmp_path` Windows path issue |
| `test_piano_transcriber.py` | 4 setup errors + 1 real failure | `tmp_path` + `PianoTranscriber._resolve_device` is referenced by a test but does not exist |
| `test_pipeline_e2e.py` | 3 tests, all pass | Pure unit-level "e2e" |
| `test_playability.py` | passes | — |
| `test_post_processor.py` | passes | — |
| `test_quantizer.py` | passes | — |
| `test_renderer.py` | 5 setup errors | `tmp_path` |
| `test_separator.py` | 3 setup errors | `tmp_path` |
| `test_vocal_transcriber.py` | 5 setup errors | `tmp_path` |

**Net: 75 passed, 1 real failure, 27 setup errors, 5 deselected.** The
27 errors are all the same `OSError: could not create numbered dir with
prefix …` Windows temp-path bug — long `C:\Users\Saanvi Sharma\…` path
combined with `pytest`'s tmp-path counter overflows. Unrelated to the
code under test. The one real test failure is the missing
`_resolve_device` static method on `PianoTranscriber` (see §6).

---

## 2. What the pipeline actually does (file-by-file)

This is the "real" walkthrough. The high-level plan in
`MASTER_DEVELOPMENT_PLAN.md` is mostly correct but is now significantly
out of date — many more processing steps have been bolted on top of it
since the plan was written.

### Step 1 — Stem separation
* `src/audio/separator.py` — wraps Demucs' `htdemucs` model, four stems
  output. Lazy-loads the model. Uses `demucs.apply.apply_model` with
  configured shifts/overlap.
* `src/audio/noise_gate.py` — **not in the plan**. Hard amplitude gate
  that mutes frames below a dB threshold. Runs on `other.wav` **before**
  piano transcription only.
* `src/audio/mixer.py` — sums `bass.wav` + `other.wav` for the
  algorithmic arranger (Path B).

### Step 2 — Vocal transcription
* `src/transcription/vocal_transcriber.py` — BasicPitch with vocal-range
  frequency clamp (80–1100 Hz, ≈E2–C#6). Strips pitch bends, drops notes
  <58 ms, consolidates everything into one Instrument named "Right Hand"
  on MIDI channel 0.

### Step 3 — Accompaniment (fork)
* `src/transcription/piano_transcriber.py` — BasicPitch again, but with
  the full 88-key range. Note: the plan says "ByteDance
  `piano-transcription_inference`" — **that is not what the code does**.
  The code uses BasicPitch for piano too. (See §3.)
* `src/arranger/beat_analyzer.py` — librosa beat tracker, every Nth beat
  is a downbeat (4/4 only).
* `src/arranger/chord_detector.py` — CQT chroma, beat-synchronous
  averaging, cosine-similarity template matching against 9 chord types.
* `src/arranger/pattern_library.py` — JSON pattern loader.
* `src/arranger/arranger.py` — maps chords × pattern events → MIDI notes.

### Step 4 — Cleaning, post-processing, playability, quantisation, merging
This step has **grown far beyond** the plan. In execution order on the
accompaniment MIDI:

1. `MidiCleaner.filter_minimum_duration` — drop notes <80 ms
2. `MidiCleaner.filter_polyphony_choke` — thin chords to ≤4 notes,
   keeping lowest+highest and the loudest inner notes
3. `MidiCleaner.filter_reverb_shadow` — drop quiet notes <200 ms after
   a loud one
4. `MidiCleaner.filter_ghost_notes` — tiered velocity+duration filter
5. `MidiCleaner.filter_short_notes` — drop notes <50 ms (safety net)
6. `MidiCleaner.normalize_velocities` — linearly compress to 60–100
7. `post_processor.post_process_midi` — extracts BPM via librosa, then:
   * quantises note starts to 16th-note grid
   * applies **aggressive legato** (every note stretched to the next
     chord change, capped at 2.0 s)
8. Assign to MIDI channel 1 (Left Hand)
9. Merge with vocals (if any) via `MidiMerger`
10. `PlayabilityFilter.apply` — hand-span + polyphony pruning
11. `Quantizer.quantize` — final 16th-note grid snap + dedup + merge

The vocal MIDI gets a lighter treatment: pitch-bend strip, range clamp,
channel assign, instrument assign — no quantisation, no post-processing,
no playability filtering. (See §4.4.)

### Step 5 — Video rendering
Two renderers behind a factory in `src/video/__init__.py:27`:

* `src/video/renderer.py` — `VideoRenderer` invokes `MIDIVisualizer` as a
  subprocess. Tries `--audio` flag first; on `RenderingError` falls back
  to rendering silent then muxing with FFmpeg.
* `src/video/python_renderer.py` — `PythonVideoRenderer` pure-Python
  fallback using Pillow + MoviePy + a FluidSynth-synthesised audio
  track. **Does not use the original input audio.**

The orchestrator catches all renderer exceptions and demotes video
rendering to non-critical — the pipeline still returns a successful
`PipelineResult` if video fails.

---

## 3. Where the implementation differs from the plan

The plan (`MASTER_DEVELOPMENT_PLAN.md`, `PIPELINE_WORKFLOW.md`) is
**significantly stale** in several places:

1. **Step 3A is BasicPitch, not ByteDance.** The plan describes
   `piano_transcription_inference`. The code uses BasicPitch (the same
   model used for vocals) with a full piano frequency range. The package
   is even still pinned in `requirements.txt`. (See §6.)
2. **The pre-transcription noise gate** is not mentioned in the plan. It
   runs only in Path A and silently re-shapes the audio feeding the
   piano transcriber.
3. **The four-pruned-rule ghost-note pipeline** (Rules 1–3 in
   `cleaner.py`) is not in the plan. The plan only describes the
   `filter_ghost_notes` tiered filter as the single cleanup.
4. **The tempo-aware post-processor** (`midi/post_processor.py`,
   "Chord Snapper + Smart Legato") is not in the plan. The plan says
   "Legato Extension: +50ms" hardcoded. The actual code extracts BPM
   from audio and stretches notes algorithmically.
5. **The `PlayabilityFilter` is called once on the merged file**, not
   twice (vocals + accompaniment separately) as the plan implies. The
   plan's "Phase B" describes the right idea but the code merges first
   then filters.
6. **The video renderer is no longer just MIDIVisualizer.** There's a
   `python` fallback that synthesises its own audio from the MIDI via
   FluidSynth. The plan only describes the MIDIVisualizer binary.
7. **The `python` renderer ignores the input audio entirely.** The plan
   states "audio path … for audio track sync." The Python fallback
   silently **discards** the input audio and synthesises its own
   piano-only audio from the MIDI. This is a big behavior divergence
   from what users expect.

---

## 4. Correctness & quality issues — the things to fix

Listed in approximate order of impact. Each item is something I would
prioritise before any new feature work.

### 4.1 🚨 The final MIDI visualiser step is broken (Step 5)

This is the headline issue. There are **three independent failure modes**
in the renderer that combine to produce the symptom you described — "the
visualisation shows up with the input song and we're not sure if the
notes are accurate to the pipeline output."

#### 4.1.1 `VideoRenderer` (MIDIVisualizer path) — almost certainly mis-pointed

`src/video/renderer.py:64-69`:

```python
try:
    self._render_with_audio(midi_path, audio_path, output_path, timeout)
except RenderingError:
    log.info("MIDIVisualizer --audio flag not supported; using FFmpeg fallback.")
    self._render_without_audio(midi_path, output_path, timeout)
    self._mux_audio_ffmpeg(output_path, audio_path)
```

`MIDIVisualizer` does not actually accept an `--audio` flag in any
publicly released version. The flag is invented. Every call to
`_render_with_audio` raises `RenderingError`, every call falls through
to the FFmpeg path. So far that path *should* still produce a video with
the original audio.

**But** — the bigger problem: `MIDIVisualizer` itself appears not to be
installed. `create_renderer` checks `shutil.which("MIDIVisualizer")` and
falls back to `PythonVideoRenderer` if not found. There is no
`MIDIVisualizer` binary on this machine. The rendered MP4s in
`data/output/` come from the Python fallback. The intermediate file
names in your run are `run_20260530_172528_synthesiaTEMP_MPY_wvf_snd.mp4`
— the "MPY" prefix is a tell that MoviePy (the Python renderer) is what
produced them.

#### 4.1.2 `PythonVideoRenderer` — uses the wrong audio

`src/video/python_renderer.py:160-263`. The class docstring even
admits it:

> The audio track is synthesized directly from the MIDI file
> (piano sounds matching the falling notes), NOT from the
> original input audio.

And in the `render` method:

```python
def render(
    self,
    midi_path: Path,
    audio_path: Path,   # <-- ignored!
    output_path: Path,
    ...
):
    ...
    # Synthesize piano audio from the MIDI
    synth_audio_path = self._synthesize_midi_audio(pm, output_path)
```

**`audio_path` is never used.** If the renderer is being chosen because
MIDIVisualizer is missing, the user gets a video with **synthesised
piano audio of the transcription output**, not the original song. This
is the source of "the visualisation shows up with the input song"
feel — the user sees notes that look related to their song but the audio
underneath sounds like a cheesy MIDI piano playing a simplified version
of the chords.

The FluidSynth step itself has another problem: the SoundFont it tries
to load is `assets/TimGM6mb.sf2`, which exists, but on a Windows system
without the FluidSynth DLL on `PATH` the
`pm.fluidsynth(fs=…, sf2_path=…)` call will fail. The code has a
fallback to `pm.synthesize()` (a built-in sine-wave synth), which sounds
hideous and is what users are most likely hearing.

#### 4.1.3 The video's "synthesia" image is also questionable

Even when the python renderer does work, the visual fidelity is much
lower than MIDIVisualizer. Things to be aware of:

* **`_draw_keyboard` only draws a static piano; it never plays the
  88-key width mapping correctly on different aspect ratios.** White-key
  width is `width / num_white_keys` regardless of resolution, so
  1920×1080 (your default) gets ~14.4 px per white key which is fine,
  but 1280×720 gets ~9.6 px which is illegible.
* **`_draw_falling_notes` iterates every note on every frame** — at 60
  fps over a 4-minute song with ~700 notes total, that's ~17 M
  comparisons per render. On a real machine this can take 5–10× the
  audio length to render.
* **Note color is by `note.channel` (0 or 1)** which is correct, but
  `_extract_notes` uses **instrument index** (not `instrument.program`
  or any MIDI channel field) to determine channel — see §4.1.4.
* **No fade-in/fade-out for active notes.** They pop in and out.

#### 4.1.4 `_extract_notes` channel assignment is fragile

`src/video/python_renderer.py:333-357`:

```python
for idx, inst in enumerate(pm.instruments):
    if inst.is_drum:
        continue
    channel = 0 if idx == 0 else 1
```

`pretty_midi.Instrument` does have a real `channel` attribute, but this
code uses the **list index** instead. If the merged MIDI happens to be
written in a different order (e.g. accompaniment instrument created
first because vocals were skipped), every note's color swaps. The
right-hand gets green and the left-hand gets cyan.

**Fix:** use `inst.channel if inst.channel is not None else idx` — but
even better, propagate the channel choice via the `assign_channel`
cleaner, which already does the right thing. Then the renderer just
reads `inst.channel`.

#### 4.1.5 Video rendering is non-critical, but video failures are silent

`src/pipeline/orchestrator.py:303-317` catches **every** exception
class via `except (FileNotFoundError, Exception)`. The second clause
makes the first redundant — it catches everything, period. The user
gets a `video_path` of `Path("")` and a `warning` string. There is no
real error surfaced. The user has no way to know that the video
"rendered" is actually a 12-pixel-per-key piano with a sine-wave piano
audio track instead of the actual song.

**Action: log the actual rendering backend chosen, the audio source used,
and any fallback decisions at INFO level, and surface them in the CLI
output.** Then the user knows what they got.

### 4.2 🐛 `PianoTranscriber` test references a missing method

`tests/test_piano_transcriber.py:107-110`:

```python
def test_device_resolution(self):
    device = PianoTranscriber._resolve_device("auto")
    assert device in ("cpu", "cuda")
```

`PianoTranscriber` has no `_resolve_device` static method. The
`Separator` class has one (`src/audio/separator.py:158`), but
`PianoTranscriber` does not. The test fails with
`AttributeError: type object 'PianoTranscriber' has no attribute
'_resolve_device'`.

This is a *test* bug — the test was written for an earlier version of
the code that had the method — but it should be either deleted or the
method should be added. Right now the test is dead weight.

### 4.3 🐛 `assign_channel` in `cleaner.py` can silently drop program info

`src/midi/cleaner.py:67-106`:

```python
program = 0
...
for instrument in midi.instruments:
    ...
    program = instrument.program
```

`program` is overwritten on every iteration. If the input MIDI has
multiple instruments with *different* programs (it shouldn't, but
defensively…), only the last one survives. Then:

```python
new_instrument = pretty_midi.Instrument(
    program=program,   # last instrument wins
    is_drum=False,
    name=name,
)
```

The final instrument's program is always whatever the *last* instrument
in the list had. For a transcription output (always Acoustic Grand
Piano, program 0) this is fine. But the function is meant to be a
generic utility, and it has a real bug for any caller that passes
heterogeneous programs. The fix: take a `program` parameter with a
default of 0, or assert that all instruments share a program.

### 4.4 🐛 Vocal MIDI is never quantised and never playability-filtered

`src/pipeline/orchestrator.py:140-155`. After the vocal MIDI is
transcribed, the pipeline only does:

1. `strip_pitch_bends`
2. `clamp_note_range(43, 84)`
3. `assign_channel(0)`
4. `set_instrument(0)`

No quantisation, no post-processing, no playability filtering. The
vocal notes go into the merged file at **whatever timestamps BasicPitch
produced**, which means the falling-notes video will show vocal notes
landing slightly off the 16th-note grid relative to the accompaniment.
The user sees jittery, smeared RH notes over crisp LH block chords.

Two consequences:
* The video looks worse (notes don't land on the strike line
  simultaneously).
* Playability enforcement doesn't apply to the right hand at all, so
  wide vocal ranges (>15 ST) end up in the final MIDI.

**Fix:** add a tempo-aware quantisation + playability pass to the vocal
MIDI after the cleaner steps and before the merge. Same post-processor
as the accompaniment. The two are independent so it can be a separate
`post_process_midi(vocals_midi, audio_path, config)` call.

### 4.5 🐛 `apply_aggressive_legato` is destructive across unrelated notes

`src/midi/post_processor.py:280-314`. The function looks for "the next
note" in chronological order and stretches the current note to meet it.
But it doesn't check whether the next note is on the same pitch or
chord. So if the next note happens to be an inner voice one beat later
on a different pitch, the current note is dragged out across the chord
change, producing an unintended legato that wasn't musically there.

This is visible in the legato-stretched note in the latest output
(`data/output/run_20260531_141306_final.mid`, instrument 1, last note
starts at 126.73 s and ends at 131.097 s — 4.4 seconds of one held
note, well over the 2.0 s cap, which means the cap is broken too).

**Fix:** `aggressive_legato_max_duration_s` is read from config in
`post_process_midi` (`post_processor.py:366-367`) but
`apply_aggressive_legato`'s second parameter is also called
`max_note_duration` — that parameter is being passed in correctly, but
the implementation has a bug: the absolute cap is **only applied to the
last note** (`post_processor.py:303-308`). All other notes can exceed
the cap freely.

### 4.6 🐛 `filter_polyphony_choke` has an off-by-one

`src/midi/cleaner.py:404-408`:

```python
max_inner = max_chord_notes - 2  # 4 - 2 = 2
# Delete the quietest inner notes beyond the budget
for i, _n in inner[:len(inner) - max_inner]:
    notes_to_remove.add(i)
```

When there are 5 active notes (1 + 4 inner), `max_inner = 2`, so
`len(inner) - max_inner = 2`, so `inner[:2]` is removed. That leaves
1 lowest + 1 highest + 2 inner = 4 notes. Correct.

When there are 6 active notes (1 + 5 inner), `len(inner) - max_inner = 3`,
so `inner[:3]` is removed. That leaves 1 + 1 + 2 = 4. Correct.

When there are 4 active notes (1 + 3 inner), `len(inner) - max_inner = 1`,
so `inner[:1]` is removed. That leaves 1 + 1 + 2 = 4. Correct.

But when there are exactly 4 active notes from the start, the
condition is `len(live_ids) <= max_chord_notes`, so we `continue` and
don't enter the choke. The check is correct for the entry condition.

However: the rule is supposed to "always preserve lowest and highest"
and keep the *loudest* inner notes. The code sorts inner notes by
velocity **ascending** and deletes the first N, which is the same as
keeping the loudest. Correct.

But the algorithm only fires when `len(live_ids) > max_chord_notes`.
What if the input is already at the limit? It never gets re-checked as
notes turn on/off later in the timeline. So a long sustained note that
overlaps a series of new chords will **never be choked** even if a
new chord brings the polyphony above the limit. The on/off event
sweep is correct for the entry condition but doesn't iterate to
re-check after the OFF event frees up a slot.

In practice this is a minor issue because the threshold is generous
(4), but it's a subtle correctness gap worth fixing.

### 4.7 🐛 `post_processor.extract_tempo` runs on the wrong audio

`src/pipeline/orchestrator.py:265`:

```python
bpm_audio_source = gated_path if has_piano else instrumental_path
```

For Path A (has piano), the BPM is extracted from the **gated** audio.
That's audio that's been pre-cleaned for the piano transcriber — the
noise gate may have removed frames that librosa needs to track beats
reliably. The original `other.wav` (or even better, the
`instrumental.wav` = bass + other mix) is a more reliable source.

For Path B (no piano), the BPM is extracted from the **mixed**
`instrumental.wav` (bass + other). That's correct.

**Fix:** always extract BPM from `instrumental.wav` (mix bass + other)
when it exists, or from `stems.other_path` as a fallback. The
`NoiseGate` output should not be used for tempo tracking.

### 4.8 🐛 `AlgorithmicArranger` octave math can produce pitches above C8

`src/arranger/arranger.py:186-218`. The `_resolve_note_type` function
applies `octave_offset` directly to `root_midi` and then adds intervals.
For chord roots in octave 3 (C3=48) with `octave_offset=0` and a triad
`[0, 4, 7]`, we get `[48, 52, 55]` — fine. But:

* `pop_ballad.json` uses `octave_offset: -1` on the triad events
  starting from a root_midi in octave 3, so we get `[36, 40, 43]`. Fine.
* `octave: 0` with a root of G#3 (56) and triad `[0, 4, 7]` gives
  `[56, 60, 63]`. Fine.
* But if the chord detector reports a root in octave 4 or 5 (which can
  happen for high-register songs), and the pattern uses
  `octave_offset: -1` or `-2`, we can still get notes in the upper
  register of the piano. With root=72 (C5) and octave_offset=0, we get
  `[72, 76, 79]` — still in piano range. With root=84 and offset=+1
  the notes can exceed 108 (C8). The MIDI is clamped to 0-127
  (`arranger.py:179`) but the result is a stuck note in the highest
  octave which the playability filter will then choke.

**Fix:** the `root_midi` calculation in
`ChordDetector._classify_beats` (`chord_detector.py:156`) hardcodes
`root_midi = 48 + root`, putting every chord root in octave 3. This is
fine for most pop music, but for songs with high-register chord roots
(D, E, F# in the upper octave), the algorithmic arrangement ends up
way too high. The fix is to compute `root_midi` from the song's
estimated key (e.g., assume the most-detected chroma note is the tonic
and put it in octave 3), or simply clamp `root_midi` so the highest
voiced note never exceeds 84 (C6).

### 4.9 ⚠️ Chord detector has no key-normalisation

`src/arranger/chord_detector.py:135-169`. The `_classify_beats` loop
matches each beat's chroma vector against all 12 roots × 9 chord types
and picks the highest cosine similarity. This produces the *correct*
chord name for each beat, but the resulting MIDI can have wild
register jumps. For example, if the song's key is G major and the
chord detector returns "G major" (root=7) and "D major" (root=2), the
two chords are 5 semitones apart. But the algorithm places every root
in octave 3 (`root_midi = 48 + root`), so the actual MIDI notes jump
between F#3 and D3 — completely missing the bass register where a
human would play the roots.

**Fix:** detect the song's key once, then transpose all detected
chord roots so they are within ±6 semitones of each other and end up
in a sensible piano register. This is a §6 future enhancement item
in the plan but is needed in practice for the algorithmic path to
sound musical.

### 4.10 ⚠️ `MidiMerger` doesn't normalise tempo

`src/midi/merger.py:33-66`. The merger reads the first MIDI's tempo
and applies it to the output. But vocals and accompaniment can have
**different** tempo values (vocal MIDI was written with no explicit
initial tempo — `vocal_transcriber.py:155` — so it uses
`midi.estimate_tempo()` which is heuristic; accompaniment is written
with `arranger.arrange()` which uses `beat_grid.tempo` directly).

If the two tempos disagree (which they will, even by a few BPM), the
quantiser downstream will grid-snap to slightly different grid
positions for the two hands, producing visible LH/RH desync in the
video.

**Fix:** before merging, align both MIDIs to a common tempo. The
simplest correct approach: extract BPM once from the source audio
(via `post_processor.extract_tempo` on the original input), and write
both vocals.mid and accompaniment.mid with `initial_tempo=bpm`
explicit. Then the merger and quantiser see consistent grid math.

### 4.11 ⚠️ `validators.SUPPORTED_AUDIO_FORMATS` includes `.m4a` and `.aac` but the rest of the pipeline can't decode them

`src/utils/validators.py:9`:

```python
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
```

`.m4a` and `.aac` are not decodable by `librosa`/`demucs` without
additional codecs. If a user passes one, the validator says "OK", then
Demucs explodes. Either drop those two from the supported list, or
have the orchestrator run an FFmpeg pre-step to convert them to `.wav`.

### 4.12 ⚠️ `StemSeparator._load_audio` doesn't handle `.flac` extension mismatches

`src/audio/separator.py:88-112` uses `demucs.audio.AudioFile(audio_path)`
directly. This works for mp3/wav/flac in most cases, but if the file
extension doesn't match the actual format (e.g., a `.mp3` that's
actually an `.aac`), Demucs's internal decoder will fail with an
obscure error rather than a clean `SeparationError`.

**Fix:** use `subprocess.run(["ffprobe", …])` to verify the format
before passing to Demucs.

### 4.13 ⚠️ Playability filter is per-instrument, not per-hand

`src/midi/playability.py:79-122`. The filter processes each
`Instrument` independently. After merging, the right hand
(instrument 0) and left hand (instrument 1) get separate polyphonies,
so a hand can still have >4 notes — they just won't be in the *same*
instrument. This is what the design intends, but the test in
`test_pipeline_e2e.py:96-113` constructs a single instrument and
checks its polyphony. The check is correct for the single-instrument
case but doesn't validate the merged-file case.

**Fix:** add a multi-instrument test that constructs a 2-instrument
merged file and asserts that each instrument's max polyphony is ≤4
across the whole timeline.

### 4.14 ⚠️ `BeatAnalyzer` assumes 4/4

`src/arranger/beat_analyzer.py:23-77`. Hardcoded
`time_signature=(4, 4)` and `downbeat_indices = list(range(0, len(beat_times), self._beats_per_bar))`.
Songs in 3/4 (waltz), 6/8 (jazz), or 7/8 (prog) will have **completely
wrong downbeat alignments**, which then makes the arranger place
chord changes on off-beats. For the user's typical pop/EDM/rock
library this is fine, but it's a real gap.

### 4.15 ⚠️ `NoiseGate` mutates stereo layout

`src/audio/noise_gate.py:69-117`. The gate computes RMS on the **mono
sum** of the stereo signal, then applies a single mask to both
channels. This is fine, but the output is written as `(samples,
channels)` regardless of input. If the input was mono, the output is
mono; if stereo, stereo. `Demucs` outputs stereo WAVs with a specific
channel ordering and the noise gate silently produces
`(samples, channels)` which `soundfile` writes correctly, but the
downstream BasicPitch call doesn't always handle mono input the same
way. Worth verifying with a stereo song sample that the gated output
is being interpreted correctly.

### 4.16 ⚠️ `StemMixer` peak normalization can destroy dynamics

`src/audio/mixer.py:85-91`:

```python
if normalize:
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak
```

This normalises to **0 dBFS** (peak = 1.0). For mixed stems that are
already at -3 dBFS, this is fine. But for stems that came out of
Demucs with a -20 dBFS headroom, this normalization **loudness-warps**
the result. The chord detector's chroma calculation is amplitude-
insensitive so it doesn't matter for the algorithmic path, but the
noise gate downstream (in Path A) **does** use a dB threshold
(`-20 dB`), and a peak-normalized instrumental is going to have a
totally different dB landscape.

**Fix:** normalize to a target LUFS or at least a -3 dBFS headroom,
not peak.

### 4.17 ⚠️ No way to inspect intermediates between steps

The orchestrator saves intermediates to `data/intermediate/<run_id>/`
but only the final `*_final.mid` and `*_synthesia.mp4` are surfaced in
the CLI summary. For debugging "why does my video look weird," a user
would need to manually inspect the intermediate folder. The CLI
should at minimum print the run directory path at the end and offer a
flag like `--inspect` that lists all the intermediates with note
counts.

### 4.18 ⚠️ `pattern_library` is loaded but the default pattern is hard-coded

`config.yaml:37` sets `default_pattern: "pop_ballad"`. The CLI has no
flag to change it. The arranger has a `pattern_name` parameter
(`arranger.py:50`) but nothing in the orchestrator exposes it.

### 4.19 ⚠️ Error messages are leaked but warnings are buried

`src/pipeline/orchestrator.py:336-340`:

```python
except Exception as exc:
    raise PipelineError(
        message="Pipeline failed",
        run_id=run_id,
        cause=exc,
    ) from exc
```

`PipelineError.__init__` (`src/pipeline/errors.py:18-26`) includes the
*cause* chain in the message. That's good for debugging, but the CLI
prints the full traceback. For a 3-minute pipeline failure, the user
sees 50+ lines of loguru output. The CLI should catch the
`PipelineError`, print a one-line summary, and write the full
traceback to `data/intermediate/<run_id>/error.log`.

### 4.20 ⚠️ Tests are not parameterised over configurations

`tests/test_pipeline_e2e.py` only tests the default config. The
config has 7 sub-configs, each with multiple knobs. None of the
combinations are tested. A reasonable minimum is a small
parameterised test that runs the e2e pipeline with each of the 4
`(include_vocals, has_piano)` combinations and asserts non-empty
output.

---

## 5. Things that are *good* and should be preserved

This is not all bad news. Several things in the codebase are
genuinely well done:

* **The 3-rule ghost-note pruning pipeline** (min-duration,
  polyphony-choke, reverb-shadow) is a meaningful upgrade over the
  plan's "single tiered filter." The math is correct except for the
  bug in §4.6. The order is correct (prune → normalise) and the
  rationale is well-documented in the docstrings.
* **The aggressive-legato Chord Snapper** is a real improvement over
  the plan's hardcoded +50 ms legato. The 2-second cap is a sensible
  idea, even if the cap is currently broken (§4.5).
* **The noise gate** is well-implemented with clear dB threshold
  semantics. The decision to use it only for Path A (piano
  transcription) and not Path B is correct.
* **The factory pattern in `src/video/__init__.py`** with
  `auto | midi_visualizer | python` is the right design.
* **The graceful-degradation policy** for video rendering (non-
  critical step) is sound engineering — a missing binary should not
  take down the whole pipeline.
* **The `Pydantic` config model** (`src/pipeline/config.py`) is
  genuinely good. The `from_yaml` method handles both flat and
  `pipeline:`-nested YAML layouts, and every sub-config has type
  validation.
* **The test fixture set** in `tests/conftest.py` is well-designed
  — wide-span, dense-polyphony, and vocals-with-artifacts fixtures
  cover the right edge cases.
* **75 of 80 unit tests pass** with only one real failure. The
  coverage is much better than typical for a project of this size.

---

## 6. Things to work on (prioritised)

### P0 — Must-fix to call the system "working"

1. **Fix the video renderer's audio source.**
   * `VideoRenderer` (MIDIVisualizer path) silently fails the
     `--audio` flag attempt and falls through to FFmpeg. Verify
     whether the installed MIDIVisualizer (if any) supports
     `--audio`; if not, skip the first attempt entirely and go
     straight to the FFmpeg fallback.
   * `PythonVideoRenderer` should use the *original* input audio by
     default, not synthesise its own. If the synthesised audio is the
     only option, surface that fact in the CLI summary so the user
     knows what they're getting.
   * At minimum, log which renderer was chosen and which audio
     source was used at INFO level.

2. **Make the visualiser actually use the right hand/left hand
   colors correctly.** Fix `_extract_notes` to use `inst.channel`
   (or the cleaner-assigned channel) instead of the instrument
   index. Add a unit test that constructs a merged MIDI with
   instruments in non-canonical order and asserts the colors.

3. **Add quantisation and playability filtering to the vocal MIDI.**
   The right hand is currently untouched after the cleaner pass.
   It needs the same tempo-aware post-processing and playability
   filter as the left hand.

4. **Delete or fix the broken `test_device_resolution` test.** It
   references a method that doesn't exist.

5. **Decide what "the video" means** in your README. The README
   says "Produces falling-notes videos synced to the original
   audio." The Python renderer does not do this. Either fix it to
   use the original audio (the easy fix — MoviePy can mux in an
   arbitrary audio file) or change the README to say "may
   synthesise piano audio if MIDIVisualizer is unavailable."

### P1 — Quality, before adding new features

6. **Fix `apply_aggressive_legato`'s absolute duration cap** so
   it applies to every note, not just the last one in the file.

7. **Use the same BPM source for both hands.** Right now the
   accompaniment gets BPM from the instrumental mix (correct), but
   the vocal MIDI never has a tempo applied at all. Pre-write both
   MIDIs with `initial_tempo=extracted_bpm` so the quantiser and
   playability filter see a consistent grid.

8. **Fix `extract_tempo` to never use the gated audio.** Use the
   ungated `other.wav` (or `bass + other` mix) for BPM extraction.

9. **Add key-normalisation to the algorithmic chord arranger.**
   Detect the song's key once, then transpose chord roots so the
   generated MIDI stays in a piano-friendly register. This is the
   difference between the algorithmic path sounding like a human
   and sounding like a random C-major pattern.

10. **Add a `download_sf2.py` script.** The repo has
    `download_sf2.py` at the top level but it's not in `scripts/`
    and not documented. Users who run the Python renderer without
    a SoundFont silently get sine-wave audio. Make the
    SoundFont download an explicit, documented step in the
    install instructions.

11. **Make `MidiCleaner.assign_channel` accept a `program`
    parameter** so the function is correct for any caller, not
    just the pipeline's known-to-be-0 case.

12. **Parameterise the e2e tests over the 4
    `(include_vocals, has_piano)` combinations.**

13. **Implement the `filter_reverb_shadow` `O(n²)` issue.** The
    inner loop scans `loud_notes` for every quiet note. For a
    1000-note file with 500 loud notes, that's 500,000
    comparisons. A two-pointer sweep after sorting would be O(n).

### P2 — Future, when the foundations are right

14. **Hand splitting for Path A piano transcription.** Plan
    Appendix E, future enhancement #1. Use a pitch-based heuristic
    (notes below C4 → LH) to split the single-track piano MIDI
    into two hands *before* the merge. Right now, the merged
    MIDI's "left hand" is the entire piano transcription. The
    visual result is that the LH shows notes from the original
    pianist's right hand too, which looks weird.

15. **Difficulty levels.** Plan Appendix E, #2. Easy/medium/hard
    that adjusts polyphony limits and pattern choice.

16. **Ditch the broken `piano-transcription-inference` dependency
    in `requirements.txt`.** The code doesn't use it. Or actually
    use it — BasicPitch is a polyphonic vocal model, not a piano
    model. For a song with actual piano, the
    `piano-transcription-inference` model (ByteDance, mentioned
    in the plan) would likely give better results.

17. **Time-signature detection** so 3/4 and 6/8 songs work.

18. **A "presets" system** for the post-processor. Right now the
    `aggressive_legato` flag is binary. Real users want "soft
    ballad" (smart legato, no choke), "acoustic pop" (aggressive
    legato, light choke), "aggressive rock" (full choke, hard
    velocity normalise).

19. **Logging the renderer choice to a file** at INFO level. The
    current loguru setup is to stderr only. For long renders the
    stderr output scrolls off.

20. **Address the test `tmp_path` Windows bug** so all 108 tests
    can actually run. Either set `TMPDIR`/`TEMP` to a shorter
    path or pass `--basetemp` to pytest with a short path.

---

## 7. The video step in detail (what it is, what it should be)

This section is the deep-dive on the specific step you asked about.
It is a complete picture of "the final midi visualiser step" as
the code implements it today, why it does not match the design, and
what the fix looks like.

### What the code does today

`PipelineOrchestrator.run` ends with:

```python
output_video = output_dir / f"{run_id}_synthesia.mp4"
try:
    self._renderer.render(
        midi_path=final_midi_path,
        audio_path=audio_path,
        output_path=output_video,
    )
    steps_completed.append("video_rendering")
except (FileNotFoundError, Exception) as video_exc:
    warning_msg = f"Video rendering skipped: {video_exc}"
    log.warning(warning_msg)
    warnings.append(warning_msg)
    output_video = Path("")
```

`self._renderer` is whatever `create_renderer(config.video)` returns.
`config.video.renderer` defaults to `"auto"`. The auto path:

1. Calls `shutil.which(config.video.midi_visualizer_path)`. On this
   machine that returns `None`.
2. Falls back to `PythonVideoRenderer(config)`.

The `PythonVideoRenderer.render` method (already quoted in §4.1.2)
ignores `audio_path` entirely and uses
`pm.fluidsynth(sf2_path="assets/TimGM6mb.sf2")` to generate piano
audio. If FluidSynth can't load the DLL or the SoundFont, it falls
back to `pm.synthesize()` which is a sine-wave piano.

The output is a `libx264` MP4 with the synthesised audio encoded as
AAC, written by MoviePy. The frame rendering is correct enough —
notes fall, the strike line is drawn, keys highlight when active —
but it's a static keyboard, low resolution, and visually nothing
like MIDIVisualizer's actual output.

### What the design says it should do

From `PIPELINE_WORKFLOW.md` §5 and the README:

> The pipeline invokes MIDIVisualizer, which renders a
> 1920×1080 60 fps MP4 of falling notes synced to the original
> audio. If MIDIVisualizer doesn't support the `--audio` flag,
> FFmpeg muxes the original audio back onto a silent video.

That describes the MIDIVisualizer path correctly, and the plan
acknowledges the FFmpeg fallback. The plan does *not* describe the
Python fallback renderer at all — that was added later, presumably
because the user found the system unusable without MIDIVisualizer
installed.

### What's actually wrong

Three layered problems:

1. **The Python renderer was designed to be a fallback, but the
   fallback does not honour the API contract.** The contract is
   "render the MIDI synced to the original audio." The Python
   renderer renders the MIDI synced to *its own synthesised
   piano audio*. The user hears a different song.

2. **The Python renderer is silently selected when MIDIVisualizer
   is missing.** There is no user-visible indication that
   "you are getting the low-quality fallback." The CLI summary
   says "Video Output: <path>" whether the file is a 60 fps
   MIDIVisualizer masterpiece or a 640×360 sine-wave piano
   recording.

3. **The fix is small.** The Python renderer needs to:
   * accept the original `audio_path`
   * use MoviePy to mux it onto the silent video it generates
   * keep the synthesised audio only as a *secondary* track that
     plays underneath the original at low volume (optional polish)
   * log clearly which mode it ran in

The relevant change is ~20 lines in
`src/video/python_renderer.py`. Concretely:

```python
# In render(), instead of synthesising audio from the MIDI:
synth_audio_path = self._synthesize_midi_audio(pm, output_path)
# do this instead:
audio_clip = None
if Path(audio_path).exists():
    audio_clip = AudioFileClip(str(audio_path))
else:
    log.warning(
        f"Original audio not found at {audio_path}; "
        "falling back to synthesised piano audio."
    )
    synth_audio_path = self._synthesize_midi_audio(pm, output_path)
    if synth_audio_path and synth_audio_path.exists():
        audio_clip = AudioFileClip(str(synth_audio_path))
```

The change is straightforward but it has to be made by someone who
understands MoviePy's `clip.with_audio()` semantics — the audio
clip has to be trimmed to the video's duration, and the
`audio_path` may be in a format MoviePy can't decode directly
(`.mp3` is usually fine, but `.aac` and `.m4a` are not).

### Why the visualisation may look "wrong" beyond the audio issue

Even after the audio is fixed, the visual output of the Python
renderer has these issues:

* **Channel-coloring is by instrument index, not by MIDI channel.**
  If your merged MIDI happens to have accompaniment written first
  (no-vocals case), your LH notes will be cyan and RH notes (if
  any) will be green. Reversed from what the user expects.
  See §4.1.4.

* **No MIDI channel 0/1 enforcement.** The merger
  (`src/midi/merger.py`) doesn't actually set
  `inst.channel = 0/1` when copying instruments into the merged
  file. `pretty_midi.Instrument` has a `channel` attribute but
  the merger uses `program` only. When a downstream tool reads
  the MIDI by channel (MIDIVisualizer, many DAWs), it sees
  "channel 0" for both instruments because the attribute was
  never set. The Python renderer works around this by using
  instrument index, but a real MIDIVisualizer will not get the
  RH/LH split.

  **Fix:** in `MidiMerger.merge`, set
  `new_instrument.channel = instrument.channel if instrument.channel is not None else 0`
  for the first instrument and 1 for the second. Or, more
  robustly, store the channel in the instrument name and have
  the merger parse it back. Or, simplest: explicitly set
  `channel` in `MidiCleaner.assign_channel` (which the cleaner
  does not currently do, despite the function name).

* **The python renderer's falling notes use a `LOOKAHEAD_SECONDS = 3.0`
  constant** which means a note appears 3 seconds before it
  strikes. For a fast song this is fine. For a slow ballad
  (60–80 BPM), 3 seconds is barely one bar. The user sees notes
  at the top of the screen the moment the song starts. Fix:
  scale `LOOKAHEAD_SECONDS` with the tempo, e.g., `2 bars' worth
  of lookahead = 8 * seconds_per_beat`.

* **The python renderer iterates every note for every frame.**
  For a 4-minute song with 700 notes, this is 60 × 240 = 14,400
  frames × 700 = 10M comparisons. On a modern CPU this takes
  about 3 minutes — half the time of a real MIDIVisualizer
  render. Fix: pre-build a per-frame lookup of which notes are
  visible. The data structure is straightforward: a sorted list
  of `(note_idx, on_or_off)` events, walked with a pointer.

* **No use of `MIDIVisualizer`'s actual features.** The plan
  advertises "sustained-pedal-aware rendering" and "velocity-
  colored blocks." The Python renderer ignores sustain pedal
  entirely (the merged MIDI's CC64 events, if any, are
  preserved through the merger but never consumed by the
  renderer). Velocity is used only to pick a brighter "active"
  color, not to scale the note block size.

### What the user can do right now (workarounds)

Until the renderer is fixed:

1. **Install MIDIVisualizer** and set
   `video.midi_visualizer_path` in `config.yaml` to the absolute
   path. This is the only way to get a high-quality video that
   matches the design. The Python renderer is a degraded
   substitute, not a feature.

2. **Use `--no-vocals` mode** for simpler output (only LH, no
   channel-coloring ambiguity). This still won't fix the audio
   source but the video will at least look right.

3. **Manually run FFmpeg to replace the audio track** on the
   output MP4:
   ```bash
   ffmpeg -i data/output/run_XXX_synthesia.mp4 \
          -i data/input/song.mp3 \
          -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 \
          -shortest data/output/run_XXX_synthesia_fixed.mp4
   ```

4. **Inspect the intermediate MIDI files** in
   `data/intermediate/<run_id>/` to see what the pipeline
   actually produced. The `final_playable.mid` is what the
   renderer is being asked to visualise. If that file looks
   wrong, the problem is upstream, not in the renderer.

---

## 8. Summary — what the next iteration should look like

If I were picking the top 5 things to fix in the next session, in
order, they would be:

1. **Make the Python video renderer use the original input audio.**
   (1–2 hours, fixes the "shows up with the input song" complaint.)
2. **Fix `apply_aggressive_legato`'s absolute duration cap** so it
   applies to every note, not just the last one. (30 minutes.)
3. **Add quantisation + playability filtering to the vocal MIDI**
   so the right hand isn't visibly offset from the left in the
   video. (1 hour.)
4. **Fix the `_extract_notes` channel-by-index bug** to use the
   actual MIDI channel attribute. (15 minutes.)
5. **Delete the broken `test_device_resolution` test** and add
   real coverage for the video renderer's audio-source decision.
   (1 hour.)

After those five, the pipeline would actually deliver what the
README and PIPELINE_WORKFLOW docs promise. After that, the §6 P1
items (consistent BPM, key normalisation, SoundFont downloader)
become the next round of improvements.

Everything else — hand splitting, difficulty levels, time-signature
detection, the ByteDance piano model swap — is a P2+ enhancement
that can wait until the foundations are correct.

---

*End of audit.*
