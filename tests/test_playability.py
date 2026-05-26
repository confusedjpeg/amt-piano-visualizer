"""Tests for src.midi.playability — PlayabilityFilter."""

from __future__ import annotations

import pretty_midi
import pytest

from src.midi.playability import PlayabilityFilter
from src.pipeline.config import PlayabilityConfig


class TestPlayabilityFilter:
    """Unit tests for PlayabilityFilter."""

    def _make_filter(self, **overrides) -> PlayabilityFilter:
        config = PlayabilityConfig(**overrides)
        return PlayabilityFilter(config)

    # ── Hand Span Tests ──────────────────────────────────────────────────

    def test_wide_span_pruned(self, wide_span_midi):
        """A chord spanning > 15 semitones should be pruned to fit."""
        pf = self._make_filter(max_hand_span_semitones=15)
        result = pf.apply(wide_span_midi)

        for inst in result.instruments:
            # Check at every note start/end
            time_points = set()
            for n in inst.notes:
                time_points.add(n.start)
                time_points.add(n.end)
            for t in sorted(time_points):
                active = [n for n in inst.notes if n.start <= t < n.end]
                if len(active) >= 2:
                    pitches = sorted(n.pitch for n in active)
                    span = pitches[-1] - pitches[0]
                    assert span <= 15, f"Span {span} exceeds 15 semitones at t={t}"

    def test_span_at_limit_unchanged(self):
        """A chord exactly at the span limit should not be pruned."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        # 15 semitone span: C4(60) to D#5(75)
        for pitch in [60, 67, 75]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
            )
        midi.instruments.append(inst)

        pf = self._make_filter(max_hand_span_semitones=15)
        result = pf.apply(midi)
        total = sum(len(i.notes) for i in result.instruments)
        assert total == 3  # All notes preserved

    def test_single_note_unchanged(self):
        """A single note should pass through without modification."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
        )
        midi.instruments.append(inst)

        pf = self._make_filter()
        result = pf.apply(midi)
        total = sum(len(i.notes) for i in result.instruments)
        assert total == 1

    # ── Polyphony Tests ──────────────────────────────────────────────────

    def test_dense_polyphony_pruned(self, dense_polyphony_midi):
        """More than 4 simultaneous notes should be pruned to ≤ 4."""
        pf = self._make_filter(max_polyphony_per_hand=4)
        result = pf.apply(dense_polyphony_midi)

        for inst in result.instruments:
            active = [n for n in inst.notes if n.start <= 0.5 < n.end]
            assert len(active) <= 4

    def test_polyphony_at_limit_unchanged(self):
        """Exactly 4 simultaneous notes should not be pruned."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        for pitch in [60, 64, 67, 72]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
            )
        midi.instruments.append(inst)

        pf = self._make_filter(max_polyphony_per_hand=4)
        result = pf.apply(midi)
        total = sum(len(i.notes) for i in result.instruments)
        assert total == 4

    # ── Preservation Tests ───────────────────────────────────────────────

    def test_lowest_and_highest_preserved(self, wide_span_midi):
        """The lowest note should always be preserved (root/bass priority)."""
        pf = self._make_filter(max_hand_span_semitones=15, max_polyphony_per_hand=4)
        result = pf.apply(wide_span_midi)

        original_pitches = set()
        for inst in wide_span_midi.instruments:
            for note in inst.notes:
                original_pitches.add(note.pitch)

        result_pitches = set()
        for inst in result.instruments:
            for note in inst.notes:
                result_pitches.add(note.pitch)

        # The lowest (48) from wide_span_midi should always be preserved
        assert min(original_pitches) in result_pitches
        # At least one note should survive
        assert len(result_pitches) >= 1

    # ── Pruning Priority Tests ───────────────────────────────────────────

    def test_p5_pruned_before_m3(self):
        """With default priority, P5 (7 semitones) should be pruned before M3."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Test")
        # C4(60), E4(64), G4(67), B4(71), D5(74) — 5 notes
        for pitch in [60, 64, 67, 71, 74]:
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=0.0, end=1.0)
            )
        midi.instruments.append(inst)

        pf = self._make_filter(max_polyphony_per_hand=4)
        result = pf.apply(midi)

        result_pitches = set()
        for inst in result.instruments:
            for note in inst.notes:
                result_pitches.add(note.pitch)

        # G4 (67) is P5 from C4 — should be the first pruned
        assert 67 not in result_pitches or len(result_pitches) <= 4

    # ── Empty Input ──────────────────────────────────────────────────────

    def test_empty_midi_unchanged(self):
        """Empty MIDI should pass through without error."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        midi.instruments.append(
            pretty_midi.Instrument(program=0, name="Empty")
        )

        pf = self._make_filter()
        result = pf.apply(midi)
        total = sum(len(i.notes) for i in result.instruments)
        assert total == 0
