"""
Pattern library for arrangement patterns.

Loads JSON-defined rhythmic blueprints that describe how to voice
chords across a bar of music.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.models import ArrangementPattern, PatternEvent
from src.utils.logger import get_logger

log = get_logger(__name__)


class PatternLibrary:
    """Load and manage MIDI arrangement patterns from JSON definitions."""

    def __init__(self, pattern_dir: Path) -> None:
        """Load all .json pattern files from the given directory.

        Args:
            pattern_dir: Directory containing pattern JSON files.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        self._pattern_dir = Path(pattern_dir)
        self._patterns: dict[str, ArrangementPattern] = {}
        self._load_all()

    def get_pattern(self, pattern_name: str) -> ArrangementPattern:
        """Retrieve a pattern by name.

        Args:
            pattern_name: The name of the pattern (e.g., "pop_ballad").

        Returns:
            The matching ArrangementPattern.

        Raises:
            KeyError: If no pattern with the given name exists.
        """
        if pattern_name not in self._patterns:
            available = ", ".join(sorted(self._patterns.keys()))
            raise KeyError(
                f"Pattern '{pattern_name}' not found. "
                f"Available patterns: {available}"
            )
        return self._patterns[pattern_name]

    def list_patterns(self) -> list[str]:
        """Return a sorted list of available pattern names."""
        return sorted(self._patterns.keys())

    # ── Private Helpers ──────────────────────────────────────────────────

    def _load_all(self) -> None:
        """Load all JSON pattern files from the pattern directory."""
        if not self._pattern_dir.exists():
            log.warning(f"Pattern directory not found: {self._pattern_dir}")
            return

        json_files = sorted(self._pattern_dir.glob("*.json"))
        if not json_files:
            log.warning(f"No pattern files found in {self._pattern_dir}")
            return

        for json_path in json_files:
            try:
                pattern = self._load_pattern(json_path)
                self._patterns[pattern.name] = pattern
                log.info(f"  Loaded pattern: {pattern.name} ({len(pattern.events)} events)")
            except Exception as exc:
                log.warning(f"  Failed to load pattern {json_path}: {exc}")

        log.info(f"Pattern library: {len(self._patterns)} patterns loaded")

    @staticmethod
    def _load_pattern(json_path: Path) -> ArrangementPattern:
        """Parse a single pattern JSON file."""
        with open(json_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        events = [
            PatternEvent(
                beat_position=ev["beat_position"],
                note_type=ev["note_type"],
                octave_offset=ev["octave_offset"],
                velocity=ev["velocity"],
                duration_beats=ev["duration_beats"],
            )
            for ev in raw.get("events", [])
        ]

        return ArrangementPattern(
            name=raw["name"],
            beats_per_bar=raw.get("beats_per_bar", 4),
            events=events,
        )
