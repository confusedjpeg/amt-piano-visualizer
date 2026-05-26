"""
End-to-end integration tests for the full pipeline.

These tests require real audio processing and may be slow.
Mark with 'e2e' for selective execution:
    pytest tests/test_pipeline_e2e.py -v
    pytest tests/ -k "not e2e" -v   # skip these
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from src.midi.playability import PlayabilityFilter
from src.midi.quantizer import Quantizer
from src.pipeline.config import PipelineConfig, PlayabilityConfig


@pytest.fixture
def pipeline_config(tmp_path) -> PipelineConfig:
    """Create a PipelineConfig pointing to temp directories."""
    pattern_dir = tmp_path / "patterns"
    pattern_dir.mkdir()

    # Create a minimal pattern
    import json
    (pattern_dir / "pop_ballad.json").write_text(json.dumps({
        "name": "pop_ballad",
        "beats_per_bar": 4,
        "events": [
            {"beat_position": 1.0, "note_type": "root", "octave_offset": -2,
             "velocity": 80, "duration_beats": 1.0},
        ],
    }))

    return PipelineConfig(
        output_dir=tmp_path / "output",
        intermediate_dir=tmp_path / "intermediate",
        arranger={"pattern_dir": str(pattern_dir)},
    )


class TestPlayabilityConstraints:
    """Verify playability constraints on synthetic MIDI data."""

    def test_hand_span_constraint(self):
        """After playability filtering, no hand should exceed 15 ST span."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")

        # Create notes with various wide spans
        note_groups = [
            # Group 1: span = 24 ST (C3 to C5)
            [(48, 0.0, 1.0), (55, 0.0, 1.0), (60, 0.0, 1.0), (72, 0.0, 1.0)],
            # Group 2: span = 19 ST
            [(50, 2.0, 3.0), (57, 2.0, 3.0), (64, 2.0, 3.0), (69, 2.0, 3.0)],
        ]

        for group in note_groups:
            for pitch, start, end in group:
                inst.notes.append(
                    pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
                )

        midi.instruments.append(inst)

        config = PlayabilityConfig(max_hand_span_semitones=15, max_polyphony_per_hand=4)
        pf = PlayabilityFilter(config)
        result = pf.apply(midi)

        # Check span at every time slice
        for inst in result.instruments:
            # Check at each note start/end
            time_points = set()
            for note in inst.notes:
                time_points.add(note.start)
                time_points.add(note.end)

            for t in sorted(time_points):
                active = [n for n in inst.notes if n.start <= t < n.end]
                if len(active) >= 2:
                    pitches = sorted(n.pitch for n in active)
                    span = pitches[-1] - pitches[0]
                    assert span <= 15, (
                        f"Span {span} ST at t={t:.3f} exceeds limit"
                    )

    def test_polyphony_constraint(self):
        """After filtering, no more than 4 simultaneous notes per hand."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")

        # 8 simultaneous notes (within span)
        for pitch in [60, 61, 62, 63, 64, 65, 66, 67]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
            )

        midi.instruments.append(inst)

        config = PlayabilityConfig(max_polyphony_per_hand=4)
        pf = PlayabilityFilter(config)
        result = pf.apply(midi)

        for inst in result.instruments:
            active = [n for n in inst.notes if n.start <= 0.5 < n.end]
            assert len(active) <= 4

    def test_quantization_alignment(self):
        """After quantization, all notes should be on the 16th-note grid."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")

        # Off-grid notes (need at least 2 notes for tempo estimation)
        for pitch, start, end in [(60, 0.03, 0.28), (64, 0.17, 0.45), (67, 0.31, 0.6)]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            )

        midi.instruments.append(inst)

        q = Quantizer(grid_resolution=16)
        result = q.quantize(midi)

        # Get the actual tempo the quantizer used (from get_tempo_changes)
        _, tempos = midi.get_tempo_changes()
        actual_tempo = float(tempos[0]) if len(tempos) > 0 else 120.0
        grid_sec = (60.0 / actual_tempo) / 4  # 16th note grid spacing

        for inst in result.instruments:
            for note in inst.notes:
                start_remainder = round(note.start % grid_sec, 10)
                end_remainder = round(note.end % grid_sec, 10)
                assert start_remainder < 1e-4 or abs(start_remainder - grid_sec) < 1e-4, (
                    f"Note start {note.start} not on grid (remainder={start_remainder})"
                )
                assert end_remainder < 1e-4 or abs(end_remainder - grid_sec) < 1e-4, (
                    f"Note end {note.end} not on grid (remainder={end_remainder})"
                )


class TestIdempotency:
    """Verify that processing the same MIDI twice produces identical output."""

    def test_playability_idempotent(self):
        """Running the playability filter twice should not change the output."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        for pitch in [48, 55, 60, 67, 72]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
            )
        midi.instruments.append(inst)

        config = PlayabilityConfig()
        pf = PlayabilityFilter(config)

        first_pass = pf.apply(midi)
        second_pass = pf.apply(first_pass)

        first_notes = sorted(
            [(n.pitch, n.start, n.end) for i in first_pass.instruments for n in i.notes]
        )
        second_notes = sorted(
            [(n.pitch, n.start, n.end) for i in second_pass.instruments for n in i.notes]
        )

        assert first_notes == second_notes

    def test_quantization_idempotent(self):
        """Running quantization twice should not change the output."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.03, end=0.28)
        )
        midi.instruments.append(inst)

        q = Quantizer(grid_resolution=16)

        first_pass = q.quantize(midi)
        second_pass = q.quantize(first_pass)

        first_notes = [
            (n.pitch, round(n.start, 6), round(n.end, 6))
            for i in first_pass.instruments for n in i.notes
        ]
        second_notes = [
            (n.pitch, round(n.start, 6), round(n.end, 6))
            for i in second_pass.instruments for n in i.notes
        ]

        assert first_notes == second_notes
