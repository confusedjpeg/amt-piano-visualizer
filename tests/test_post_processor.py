"""
Tests for src.midi.post_processor — Tempo-Aware MIDI Post-Processing.

Covers:
  - Dynamic tempo extraction (with mocked librosa)
  - Grid framework computation
  - Frame-based chord quantization
  - Smart legato stretcher (gap filling, rest preservation, duration cap)
  - The combined post_process_instrument pipeline
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pretty_midi
import pytest

from src.midi.post_processor import (
    apply_aggressive_legato,
    apply_smart_legato,
    compute_grid_framework,
    extract_tempo,
    post_process_instrument,
    post_process_midi,
    quantize_note_starts,
)
from src.pipeline.config import PostProcessingConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_instrument(
    notes: list[tuple[float, float, int, int]] | None = None,
    name: str = "Test",
) -> pretty_midi.Instrument:
    """Create a pretty_midi.Instrument with the given notes.

    Each note is (start, end, pitch, velocity).
    """
    inst = pretty_midi.Instrument(program=0, name=name)
    if notes:
        for start, end, pitch, vel in notes:
            inst.notes.append(
                pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
            )
    return inst


# ── Feature 1: Dynamic Tempo Extraction ──────────────────────────────────────

class TestExtractTempo:
    """Tests for the extract_tempo function."""

    @patch("src.midi.post_processor.librosa")
    def test_returns_scalar_bpm(self, mock_librosa: MagicMock) -> None:
        """Scalar return from beat_track is handled correctly."""
        mock_librosa.load.return_value = (np.zeros(22050), 22050)
        mock_librosa.beat.beat_track.return_value = (130.0, np.array([]))

        bpm = extract_tempo("fake.wav")
        assert bpm == pytest.approx(130.0)

    @patch("src.midi.post_processor.librosa")
    def test_returns_ndarray_bpm(self, mock_librosa: MagicMock) -> None:
        """ndarray return from beat_track (older librosa) is unpacked."""
        mock_librosa.load.return_value = (np.zeros(22050), 22050)
        mock_librosa.beat.beat_track.return_value = (np.array([95.5]), np.array([]))

        bpm = extract_tempo("fake.wav")
        assert bpm == pytest.approx(95.5)

    @patch("src.midi.post_processor.librosa")
    def test_fallback_on_zero_bpm(self, mock_librosa: MagicMock) -> None:
        """Falls back to 120 BPM when beat_track returns 0."""
        mock_librosa.load.return_value = (np.zeros(22050), 22050)
        mock_librosa.beat.beat_track.return_value = (0.0, np.array([]))

        bpm = extract_tempo("fake.wav")
        assert bpm == pytest.approx(120.0)

    @patch("src.midi.post_processor.librosa")
    def test_fallback_on_exception(self, mock_librosa: MagicMock) -> None:
        """Falls back to 120 BPM on librosa exceptions."""
        mock_librosa.load.side_effect = RuntimeError("corrupt file")

        bpm = extract_tempo("fake.wav")
        assert bpm == pytest.approx(120.0)


# ── Grid Framework Computation ───────────────────────────────────────────────

class TestComputeGridFramework:
    """Tests for the compute_grid_framework function."""

    def test_120_bpm(self) -> None:
        """120 BPM produces canonical values."""
        grid = compute_grid_framework(120.0)

        assert grid["bpm"] == pytest.approx(120.0)
        assert grid["seconds_per_beat"] == pytest.approx(0.5)
        assert grid["grid_size"] == pytest.approx(0.125)            # 16th note
        assert grid["max_stretch_gap"] == pytest.approx(0.25)       # 8th note
        assert grid["max_note_duration"] == pytest.approx(2.0)      # whole note

    def test_60_bpm(self) -> None:
        """60 BPM — slow ballad."""
        grid = compute_grid_framework(60.0)

        assert grid["seconds_per_beat"] == pytest.approx(1.0)
        assert grid["grid_size"] == pytest.approx(0.25)
        assert grid["max_stretch_gap"] == pytest.approx(0.5)
        assert grid["max_note_duration"] == pytest.approx(4.0)

    def test_180_bpm(self) -> None:
        """180 BPM — fast rock/punk."""
        grid = compute_grid_framework(180.0)

        spb = 60.0 / 180.0  # 0.333...
        assert grid["seconds_per_beat"] == pytest.approx(spb)
        assert grid["grid_size"] == pytest.approx(spb / 4.0)
        assert grid["max_stretch_gap"] == pytest.approx(spb / 2.0)
        assert grid["max_note_duration"] == pytest.approx(spb * 4.0)

    def test_all_values_scale_with_bpm(self) -> None:
        """Grid values at double BPM are exactly half those at the base."""
        slow = compute_grid_framework(80.0)
        fast = compute_grid_framework(160.0)

        assert fast["grid_size"] == pytest.approx(slow["grid_size"] / 2.0)
        assert fast["max_stretch_gap"] == pytest.approx(slow["max_stretch_gap"] / 2.0)
        assert fast["max_note_duration"] == pytest.approx(slow["max_note_duration"] / 2.0)


# ── Feature 2: Frame-Based Chord Quantization ───────────────────────────────

class TestQuantizeNoteStarts:
    """Tests for the quantize_note_starts function."""

    def test_snaps_to_nearest_grid(self) -> None:
        """A note at 0.13s with grid=0.125 snaps to 0.125."""
        inst = _make_instrument([(0.13, 0.30, 60, 80)])
        quantize_note_starts(inst, grid_size=0.125)

        assert inst.notes[0].start == pytest.approx(0.125)
        # Duration should be preserved
        assert inst.notes[0].end - inst.notes[0].start == pytest.approx(0.17, abs=1e-6)

    def test_near_simultaneous_notes_become_chord(self) -> None:
        """Three notes within a few ms of each other all snap to the same grid point."""
        inst = _make_instrument([
            (0.124, 0.50, 60, 80),
            (0.126, 0.50, 64, 80),
            (0.130, 0.50, 67, 80),
        ])
        quantize_note_starts(inst, grid_size=0.125)

        starts = [n.start for n in inst.notes]
        # All should snap to 0.125
        assert all(s == pytest.approx(0.125) for s in starts)

    def test_notes_at_exact_grid_position_unchanged(self) -> None:
        """A note already on the grid is not modified."""
        inst = _make_instrument([(0.250, 0.50, 60, 80)])
        quantize_note_starts(inst, grid_size=0.125)

        assert inst.notes[0].start == pytest.approx(0.250)

    def test_notes_are_resorted_after_quantization(self) -> None:
        """After quantization, notes are sorted by start time."""
        # Note A starts at 0.24 (will snap to 0.25), Note B starts at 0.26 (also 0.25)
        # Note C starts at 0.10 (will snap to 0.125)
        inst = _make_instrument([
            (0.24, 0.50, 60, 80),
            (0.26, 0.50, 64, 80),
            (0.10, 0.50, 67, 80),
        ])
        quantize_note_starts(inst, grid_size=0.125)

        starts = [n.start for n in inst.notes]
        assert starts == sorted(starts)

    def test_duration_preserved(self) -> None:
        """The original duration of each note is preserved after snapping."""
        inst = _make_instrument([(0.13, 0.63, 60, 80)])  # duration = 0.5s
        quantize_note_starts(inst, grid_size=0.125)

        duration = inst.notes[0].end - inst.notes[0].start
        assert duration == pytest.approx(0.5)

    def test_zero_grid_size_skips(self) -> None:
        """Grid size <= 0 is a no-op (safety guard)."""
        inst = _make_instrument([(0.13, 0.30, 60, 80)])
        quantize_note_starts(inst, grid_size=0.0)

        assert inst.notes[0].start == pytest.approx(0.13)

    def test_empty_instrument(self) -> None:
        """No crash on empty instrument."""
        inst = _make_instrument([])
        quantize_note_starts(inst, grid_size=0.125)
        assert len(inst.notes) == 0


# ── Feature 3: Smart Legato Stretcher ────────────────────────────────────────

class TestSmartLegato:
    """Tests for the apply_smart_legato function."""

    def test_fills_small_gap(self) -> None:
        """A gap smaller than max_stretch_gap is filled."""
        # Note ends at 0.5, next starts at 0.6 → gap = 0.1
        inst = _make_instrument([
            (0.0, 0.5, 60, 80),
            (0.6, 1.0, 64, 80),
        ])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        # First note should now end at 0.6 (meeting the next note)
        assert inst.notes[0].end == pytest.approx(0.6)
        # Second note is untouched (it's the last note)
        assert inst.notes[1].end == pytest.approx(1.0)

    def test_preserves_intentional_rest(self) -> None:
        """A gap larger than max_stretch_gap is an intentional rest — leave it."""
        # Note ends at 0.5, next starts at 1.5 → gap = 1.0 (bigger than 0.25)
        inst = _make_instrument([
            (0.0, 0.5, 60, 80),
            (1.5, 2.0, 64, 80),
        ])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        # First note's end is unchanged — the rest is intentional
        assert inst.notes[0].end == pytest.approx(0.5)

    def test_does_not_stretch_overlapping_notes(self) -> None:
        """If notes already overlap (gap < 0), no stretching occurs."""
        # Note A: 0.0–0.7, Note B: 0.5–1.0 → gap = 0.5 - 0.7 = -0.2
        inst = _make_instrument([
            (0.0, 0.7, 60, 80),
            (0.5, 1.0, 64, 80),
        ])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(0.7)  # unchanged

    def test_absolute_duration_cap(self) -> None:
        """Notes stretched beyond max_note_duration are capped."""
        # A single very long note (10 seconds)
        inst = _make_instrument([(0.0, 10.0, 60, 80)])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(2.0)  # capped to 2s

    def test_cap_applied_after_stretch(self) -> None:
        """If stretching makes a note exceed the cap, the cap wins."""
        # Note 1: 0.0–0.1 (duration 0.1), stretches to meet Note 2 at 0.2
        # Then duration = 0.2 — within cap. OK.
        # But let's test where stretch + original start makes it too long:
        # Note at 0.0–1.9, next at 1.95. Gap = 0.05 → stretch to 1.95.
        # Duration becomes 1.95 — within 2.0 cap.
        inst = _make_instrument([
            (0.0, 1.9, 60, 80),
            (1.95, 2.5, 64, 80),
        ])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(1.95)  # stretched, still within cap

    def test_cap_applied_when_stretch_exceeds_limit(self) -> None:
        """Stretch that would exceed the cap is truncated."""
        # Note at 0.0–1.85, next at 1.9. Gap=0.05 → stretch to 1.9.
        # Duration = 1.9 → within 2.0. OK.
        # Now: note at 0.0–1.92, next at 1.95. Gap=0.03 → stretch to 1.95.
        # Duration = 1.95 → within 2.0. OK.
        # Force exceed: note at 0.0–5.0 (already 5s, no gap to stretch).
        inst = _make_instrument([(0.0, 5.0, 60, 80)])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(2.0)

    def test_zero_gap_no_stretch(self) -> None:
        """A gap of exactly 0 (notes perfectly abutting) → no stretch needed."""
        inst = _make_instrument([
            (0.0, 0.5, 60, 80),
            (0.5, 1.0, 64, 80),
        ])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        # gap = 0.5 - 0.5 = 0, but condition is gap > 0, so no stretch
        assert inst.notes[0].end == pytest.approx(0.5)

    def test_gap_exactly_at_threshold(self) -> None:
        """A gap equal to max_stretch_gap is NOT stretched (must be strictly less)."""
        inst = _make_instrument([
            (0.0, 0.5, 60, 80),
            (0.75, 1.0, 64, 80),  # gap = 0.25 == max_stretch_gap
        ])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        # gap == max_stretch_gap → not an artifact (boundary case → leave alone)
        assert inst.notes[0].end == pytest.approx(0.5)

    def test_empty_instrument(self) -> None:
        """No crash on empty instrument."""
        inst = _make_instrument([])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)
        assert len(inst.notes) == 0

    def test_single_note(self) -> None:
        """Single note — only the cap is applied if needed."""
        inst = _make_instrument([(0.0, 0.3, 60, 80)])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        # duration 0.3 < 2.0, nothing changes
        assert inst.notes[0].end == pytest.approx(0.3)

    def test_multiple_consecutive_gaps_filled(self) -> None:
        """Chain of small gaps all get filled."""
        inst = _make_instrument([
            (0.0, 0.20, 60, 80),   # gap to next: 0.05
            (0.25, 0.45, 64, 80),  # gap to next: 0.05
            (0.50, 0.70, 67, 80),  # last note
        ])
        apply_smart_legato(inst, max_stretch_gap=0.25, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(0.25)  # stretched
        assert inst.notes[1].end == pytest.approx(0.50)  # stretched
        assert inst.notes[2].end == pytest.approx(0.70)  # untouched (last note)


# ── Feature 3b: Aggressive Legato / Chord Snapper ───────────────────────────

class TestAggressiveLegato:
    """Tests for the apply_aggressive_legato function."""

    def test_stretches_note_to_next_chord(self) -> None:
        """A note is dragged to the start of the next chord."""
        # Note 1: 0.0–0.3, Note 2 starts at 1.5
        # Gap = 1.2 — smart legato would NOT fill this; aggressive does.
        inst = _make_instrument([
            (0.0, 0.3, 60, 80),
            (1.5, 2.0, 64, 80),
        ])
        apply_aggressive_legato(inst, max_note_duration=2.0)

        # First note should now end at 1.5 (meeting the next chord)
        assert inst.notes[0].end == pytest.approx(1.5)

    def test_respects_duration_cap(self) -> None:
        """Stretch is capped at max_note_duration."""
        # Note 1: 0.0–0.3, Note 2 starts at 5.0
        # Without cap: would stretch to 5.0 (duration=5.0)
        # With cap of 2.0: clamps to 2.0
        inst = _make_instrument([
            (0.0, 0.3, 60, 80),
            (5.0, 6.0, 64, 80),
        ])
        apply_aggressive_legato(inst, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(2.0)

    def test_chord_notes_same_start_all_stretch_to_next(self) -> None:
        """Notes in the same chord (same start) all stretch to the next chord."""
        # Chord 1: C-E-G at 0.0, all short
        # Chord 2: starts at 2.0
        inst = _make_instrument([
            (0.0, 0.2, 60, 80),  # C
            (0.0, 0.2, 64, 80),  # E
            (0.0, 0.2, 67, 80),  # G
            (2.0, 2.5, 72, 80),  # next chord
        ])
        apply_aggressive_legato(inst, max_note_duration=3.0)

        # All three chord notes should stretch to 2.0
        assert inst.notes[0].end == pytest.approx(2.0)
        assert inst.notes[1].end == pytest.approx(2.0)
        assert inst.notes[2].end == pytest.approx(2.0)

    def test_multiple_chord_chain(self) -> None:
        """Three chords in sequence: each stretches to the next."""
        inst = _make_instrument([
            (0.0, 0.2, 60, 80),   # Chord 1
            (1.0, 1.2, 64, 80),   # Chord 2
            (2.0, 2.2, 67, 80),   # Chord 3 (last)
        ])
        apply_aggressive_legato(inst, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(1.0)  # stretched to chord 2
        assert inst.notes[1].end == pytest.approx(2.0)  # stretched to chord 3
        assert inst.notes[2].end == pytest.approx(2.2)   # last note, untouched (within cap)

    def test_last_note_capped_if_too_long(self) -> None:
        """The last note has its duration capped if it exceeds the max."""
        inst = _make_instrument([(0.0, 5.0, 60, 80)])
        apply_aggressive_legato(inst, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(2.0)

    def test_last_note_left_alone_if_within_cap(self) -> None:
        """The last note is not modified if its duration is within the cap."""
        inst = _make_instrument([(0.0, 1.5, 60, 80)])
        apply_aggressive_legato(inst, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(1.5)

    def test_empty_instrument(self) -> None:
        """No crash on empty instrument."""
        inst = _make_instrument([])
        apply_aggressive_legato(inst, max_note_duration=2.0)
        assert len(inst.notes) == 0

    def test_single_note_short(self) -> None:
        """Single short note — no stretch target, duration within cap."""
        inst = _make_instrument([(0.0, 0.3, 60, 80)])
        apply_aggressive_legato(inst, max_note_duration=2.0)

        assert inst.notes[0].end == pytest.approx(0.3)


# ── Combined Pipeline ────────────────────────────────────────────────────────

class TestPostProcessInstrument:
    """Tests for the full post_process_instrument pipeline."""

    @patch("src.midi.post_processor.extract_tempo", return_value=120.0)
    def test_full_pipeline_smart_legato(self, _mock_tempo: MagicMock) -> None:
        """End-to-end with smart legato: quantize → conservative gap fill."""
        # At 120 BPM: grid=0.125, max_stretch=0.25, max_dur=2.0
        smart_config = PostProcessingConfig(aggressive_legato=False)
        inst = _make_instrument([
            (0.124, 0.40, 60, 80),   # snaps to 0.125, duration=0.276
            (0.126, 0.40, 64, 80),   # snaps to 0.125, duration=0.274
            (0.55, 0.90, 67, 80),    # snaps to 0.5, duration=0.35
        ])

        result = post_process_instrument(inst, "fake.wav", config=smart_config)

        # After quantization: Note 1 & 2 both start at 0.125
        # Note 3 snaps to 0.5
        assert result.notes[0].start == pytest.approx(0.125)
        assert result.notes[1].start == pytest.approx(0.125)
        assert result.notes[2].start == pytest.approx(0.5)

    @patch("src.midi.post_processor.extract_tempo", return_value=120.0)
    def test_full_pipeline_aggressive_legato(self, _mock_tempo: MagicMock) -> None:
        """End-to-end with aggressive legato: quantize → chord snapper."""
        aggressive_config = PostProcessingConfig(
            aggressive_legato=True,
            aggressive_legato_max_duration_s=2.0,
        )
        # Two notes far apart — smart legato would leave the gap,
        # aggressive should fill it.
        inst = _make_instrument([
            (0.124, 0.40, 60, 80),   # snaps to 0.125
            (1.55, 1.90, 67, 80),    # snaps to 1.5
        ])

        result = post_process_instrument(inst, "fake.wav", config=aggressive_config)

        # After quantization + aggressive stretch:
        # Note 1 starts at 0.125, stretched to Note 2's start at 1.5
        assert result.notes[0].start == pytest.approx(0.125)
        assert result.notes[0].end == pytest.approx(1.5)

    @patch("src.midi.post_processor.extract_tempo", return_value=120.0)
    def test_default_config_uses_aggressive(self, _mock_tempo: MagicMock) -> None:
        """Default PostProcessingConfig has aggressive_legato=True."""
        inst = _make_instrument([
            (0.124, 0.40, 60, 80),
            (1.55, 1.90, 67, 80),
        ])

        result = post_process_instrument(inst, "fake.wav")

        # Default config is aggressive, so the note should stretch across the big gap
        assert result.notes[0].end == pytest.approx(1.5)


class TestPostProcessMidi:
    """Tests for the post_process_midi convenience wrapper."""

    @patch("src.midi.post_processor.extract_tempo", return_value=120.0)
    def test_processes_all_instruments(self, _mock_tempo: MagicMock) -> None:
        """All instruments in the MIDI object are post-processed."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)

        inst1 = _make_instrument(
            [(0.13, 0.40, 60, 80)], name="Right Hand"
        )
        inst2 = _make_instrument(
            [(0.13, 0.40, 48, 80)], name="Left Hand"
        )
        midi.instruments.extend([inst1, inst2])

        result = post_process_midi(midi, "fake.wav")

        # Both instruments' notes should be quantized
        for inst in result.instruments:
            assert inst.notes[0].start == pytest.approx(0.125)

    @patch("src.midi.post_processor.extract_tempo", return_value=120.0)
    def test_respects_config_toggle(self, _mock_tempo: MagicMock) -> None:
        """post_process_midi passes config through to each instrument."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        # Note at 0.0–0.3, next at 2.0 — big gap
        inst = _make_instrument([
            (0.0, 0.3, 60, 80),
            (2.0, 2.5, 64, 80),
        ], name="Left Hand")
        midi.instruments.append(inst)

        # Smart legato: gap of 1.7 is way bigger than 8th note (~0.25), so NOT filled
        smart_config = PostProcessingConfig(aggressive_legato=False)
        result_smart = post_process_midi(midi, "fake.wav", config=smart_config)
        assert result_smart.instruments[0].notes[0].end == pytest.approx(0.3)

        # Reset the note for the aggressive test
        midi2 = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst2 = _make_instrument([
            (0.0, 0.3, 60, 80),
            (2.0, 2.5, 64, 80),
        ], name="Left Hand")
        midi2.instruments.append(inst2)

        # Aggressive legato: fills the gap
        aggressive_config = PostProcessingConfig(aggressive_legato=True)
        result_agg = post_process_midi(midi2, "fake.wav", config=aggressive_config)
        assert result_agg.instruments[0].notes[0].end == pytest.approx(2.0)
