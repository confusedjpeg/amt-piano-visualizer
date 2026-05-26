"""
MIDI merger — combines multiple MIDI files into a single multi-channel MIDI.

Preserves channel assignments, control changes, and tempo.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pretty_midi

from src.utils.logger import get_logger

log = get_logger(__name__)


def _safe_estimate_tempo(midi: pretty_midi.PrettyMIDI, default: float = 120.0) -> float:
    """Safely extract tempo, preferring explicit tempo from the MIDI file."""
    tempo_times, tempos = midi.get_tempo_changes()
    if len(tempos) > 0:
        return float(tempos[0])
    try:
        return midi.estimate_tempo()
    except ValueError:
        return default


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
            tempo: If provided, override the tempo in the output MIDI.
                   If None, use the tempo from the first MIDI file.

        Returns:
            Path to the merged output MIDI file.

        Raises:
            FileNotFoundError: If any input path does not exist.
            ValueError: If no input paths are provided.
        """
        if not midi_paths:
            raise ValueError("At least one MIDI file path is required.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine tempo from first file if not overridden
        first_midi = pretty_midi.PrettyMIDI(str(midi_paths[0]))
        effective_tempo = tempo if tempo is not None else _safe_estimate_tempo(first_midi)

        # Create the output MIDI
        merged = pretty_midi.PrettyMIDI(initial_tempo=effective_tempo)

        total_notes = 0
        for midi_path in midi_paths:
            midi_path = Path(midi_path)
            if not midi_path.exists():
                raise FileNotFoundError(f"MIDI file not found: {midi_path}")

            source_midi = pretty_midi.PrettyMIDI(str(midi_path))

            for instrument in source_midi.instruments:
                new_instrument = pretty_midi.Instrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name,
                )
                new_instrument.notes = copy.deepcopy(instrument.notes)
                new_instrument.control_changes = copy.deepcopy(
                    instrument.control_changes
                )
                new_instrument.pitch_bends = copy.deepcopy(
                    instrument.pitch_bends
                )

                total_notes += len(instrument.notes)
                merged.instruments.append(new_instrument)

        merged.write(str(output_path))
        log.info(
            f"Merged {len(midi_paths)} MIDI files "
            f"({total_notes} total notes) → {output_path}"
        )

        return output_path
