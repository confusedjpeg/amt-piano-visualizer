"""
Pure-Python Synthesia-style falling-notes video renderer.

Uses Pillow for frame drawing and MoviePy for video assembly.
This is a fallback renderer for when the MIDIVisualizer binary
is not installed.  It produces a visually appealing falling-notes
MP4 with piano audio synthesized directly from the MIDI file.

Requires: moviepy >= 2.0, Pillow >= 10.0, pretty_midi, numpy, soundfile
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
from PIL import Image, ImageDraw, ImageFont

from src.pipeline.config import VideoConfig
from src.pipeline.errors import RenderingError
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Visual Constants ─────────────────────────────────────────────────────────

# Piano keyboard layout (MIDI note 21 = A0 through 108 = C8)
PIANO_MIN_NOTE = 21
PIANO_MAX_NOTE = 108
TOTAL_KEYS = PIANO_MAX_NOTE - PIANO_MIN_NOTE + 1

# Which offsets within an octave are black keys (C#, D#, F#, G#, A#)
BLACK_KEY_OFFSETS = {1, 3, 6, 8, 10}

# Keyboard dimensions (fraction of total height)
KEYBOARD_HEIGHT_RATIO = 0.10

# How many seconds of "look-ahead" are visible above the strike line
LOOKAHEAD_SECONDS = 3.0

# ── Color Palette ────────────────────────────────────────────────────────────

# Right-hand (channel 0) — vibrant cyan-blue
RH_NOTE_COLOR = (80, 180, 255)
RH_NOTE_GLOW = (50, 140, 220)
RH_NOTE_ACTIVE = (130, 210, 255)

# Left-hand (channel 1) — vibrant green-teal
LH_NOTE_COLOR = (80, 220, 160)
LH_NOTE_GLOW = (50, 180, 120)
LH_NOTE_ACTIVE = (130, 255, 190)

# Fallback for other channels
OTHER_NOTE_COLOR = (200, 160, 80)
OTHER_NOTE_ACTIVE = (240, 200, 120)

# Keyboard colors
WHITE_KEY_COLOR = (240, 240, 240)
BLACK_KEY_COLOR = (30, 30, 35)
KEY_BORDER_COLOR = (180, 180, 185)
ACTIVE_WHITE_KEY = (180, 220, 255)
ACTIVE_BLACK_KEY = (80, 130, 200)

# Strike line
STRIKE_LINE_COLOR = (255, 255, 255, 80)


def _is_black_key(midi_note: int) -> bool:
    """Return True if the MIDI note corresponds to a black piano key."""
    return (midi_note % 12) in BLACK_KEY_OFFSETS


def _parse_background_color(color_str: str) -> tuple[int, int, int]:
    """Parse the config background_color (space-separated 0.0–1.0 floats)."""
    parts = color_str.strip().split()
    if len(parts) == 3:
        try:
            return tuple(int(float(p) * 255) for p in parts)  # type: ignore[return-value]
        except ValueError:
            pass
    return (25, 25, 30)  # Default dark charcoal


class _NoteEvent:
    """Lightweight container for a single MIDI note event."""

    __slots__ = ("pitch", "start", "end", "velocity", "channel")

    def __init__(
        self,
        pitch: int,
        start: float,
        end: float,
        velocity: int,
        channel: int,
    ) -> None:
        self.pitch = pitch
        self.start = start
        self.end = end
        self.velocity = velocity
        self.channel = channel


class PythonVideoRenderer:
    """Pure-Python Synthesia-style falling-notes video renderer."""

    def __init__(self, config: VideoConfig) -> None:
        self._config = config

        # Parse resolution
        parts = config.resolution.split("x")
        self._width = int(parts[0])
        self._height = int(parts[1])
        self._fps = config.fps
        self._bg_color = _parse_background_color(config.background_color)

        # Keyboard geometry
        self._kb_height = int(self._height * KEYBOARD_HEIGHT_RATIO)
        self._fall_height = self._height - self._kb_height  # pixels for falling notes

        # Key width calculation — count white keys
        self._white_key_indices: list[int] = [
            n for n in range(PIANO_MIN_NOTE, PIANO_MAX_NOTE + 1)
            if not _is_black_key(n)
        ]
        self._num_white_keys = len(self._white_key_indices)
        self._white_key_width = self._width / self._num_white_keys
        self._black_key_width = self._white_key_width * 0.6
        self._black_key_height = int(self._kb_height * 0.62)

        # Pre-compute x-position for every MIDI note
        self._note_x: dict[int, float] = {}
        self._note_w: dict[int, float] = {}
        self._compute_key_positions()

    def _compute_key_positions(self) -> None:
        """Pre-compute the x-position and width for every piano key."""
        # Build a mapping from white-key index to x position
        white_x: dict[int, float] = {}
        for i, note in enumerate(self._white_key_indices):
            x = i * self._white_key_width
            white_x[note] = x
            self._note_x[note] = x
            self._note_w[note] = self._white_key_width

        # For black keys, position them between their neighboring white keys
        for note in range(PIANO_MIN_NOTE, PIANO_MAX_NOTE + 1):
            if _is_black_key(note):
                # The black key sits to the right of the white key below it
                lower_white = note - 1
                while lower_white >= PIANO_MIN_NOTE and _is_black_key(lower_white):
                    lower_white -= 1

                if lower_white in white_x:
                    bk_x = white_x[lower_white] + self._white_key_width - self._black_key_width / 2
                    self._note_x[note] = bk_x
                    self._note_w[note] = self._black_key_width

    def render(
        self,
        midi_path: Path,
        audio_path: Path,
        output_path: Path,
        timeout: int = 600,
    ) -> Path:
        """
        Render the Synthesia-style video using pure Python.

        The audio track is synthesized from the MIDI file (piano playing
        the same notes that fall on screen).  The original input song is
        NOT used — a Synthesia video must sound like piano, not like the
        original artist.

        Args:
            midi_path: Path to final_playable.mid.
            audio_path: Path to original audio file (unused — kept for
                API compatibility with VideoRendererProtocol).
            output_path: Path for the output MP4 file.
            timeout: Maximum time in seconds (unused, kept for API compat).

        Returns:
            Path to the rendered MP4 video.

        Raises:
            FileNotFoundError: If input files are not found.
            RenderingError: If rendering fails.
        """
        midi_path = Path(midi_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        # Parse MIDI
        log.info(f"Parsing MIDI: {midi_path}")
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as exc:
            raise RenderingError(
                message=f"Failed to parse MIDI file: {exc}"
            ) from exc

        notes = self._extract_notes(pm)
        if not notes:
            raise RenderingError(message="MIDI file contains no notes to render.")

        duration = pm.get_end_time() + 1.0  # +1s for trailing notes to fade out
        log.info(
            f"Rendering {len(notes)} notes over {duration:.1f}s "
            f"at {self._width}x{self._height} @ {self._fps}fps"
        )

        # Sort notes by start time for efficient look-up
        notes.sort(key=lambda n: n.start)

        # Build the video
        synth_audio_path: Path | None = None
        try:
            from moviepy import VideoClip, AudioFileClip

            def make_frame(t: float) -> np.ndarray:
                return self._draw_frame(t, notes)

            clip = VideoClip(make_frame, duration=duration)

            # ── Audio source ─────────────────────────────────────────────
            # For a Synthesia-style video, the audio MUST be the piano
            # playing the same notes that are falling on screen.  We
            # synthesize it from the final MIDI.  The original input song
            # is deliberately NOT used — it would not match the visuals.
            audio_attached = False
            synth_audio_path = self._synthesize_midi_audio(pm, midi_path, output_path)
            if synth_audio_path and synth_audio_path.exists():
                try:
                    audio_clip = AudioFileClip(str(synth_audio_path))
                    clip = clip.with_audio(audio_clip)
                    audio_attached = True
                    log.info("Attached synthesized piano audio (matches MIDI notes)")
                except Exception as synth_exc:
                    log.warning(
                        f"Could not attach synthesized audio: {synth_exc}"
                    )

            if not audio_attached:
                log.warning(
                    "No piano audio available — rendering silent video. "
                    "(The original input song is intentionally NOT used; "
                    "it would not match the falling notes.)"
                )

            log.info(f"Writing video to {output_path}...")
            clip.write_videofile(
                str(output_path),
                fps=self._fps,
                codec="libx264",
                audio_codec="aac",
                logger=None,  # Suppress moviepy's progress bar
            )

        except ImportError as exc:
            raise RenderingError(
                message=(
                    "moviepy is required for the Python video renderer. "
                    "Install it with: pip install moviepy>=2.0"
                )
            ) from exc
        except Exception as exc:
            raise RenderingError(
                message=f"Video rendering failed: {exc}"
            ) from exc
        finally:
            # Clean up temp synthesized audio (if one was created)
            if synth_audio_path and synth_audio_path.exists():
                try:
                    synth_audio_path.unlink()
                except OSError:
                    pass

        # Validate output
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RenderingError(
                message=f"Rendering produced no output at {output_path}"
            )

        log.info(f"Video rendered: {output_path}")
        return output_path

    # ── MIDI Audio Synthesis ─────────────────────────────────────────────

    def _synthesize_midi_audio(
        self, pm: pretty_midi.PrettyMIDI, midi_path: Path, output_path: Path
    ) -> Path | None:
        """Synthesize piano audio from the MIDI file.

        Tries in order:
          1. FluidSynth CLI (bundled binary in assets/fluidsynth/bin/)
          2. pyfluidsynth Python binding (if installed)
          3. Built-in sine-wave synthesizer (ugly last resort)

        Returns:
            Path to the temporary WAV file, or None if synthesis fails.
        """
        import soundfile as sf
        import subprocess
        import os

        synth_path = output_path.with_suffix(".synth.wav")
        sample_rate = 44100
        sf2_path = Path("assets/TimGM6mb.sf2").resolve()

        # ── Approach 1: FluidSynth CLI (bundled binary) ─────────────────
        fluidsynth_bin = Path("assets/fluidsynth/bin/fluidsynth.exe").resolve()
        if fluidsynth_bin.is_file() and sf2_path.exists():
            try:
                log.info("Synthesizing piano audio via FluidSynth CLI ...")
                result = subprocess.run(
                    [
                        str(fluidsynth_bin),
                        "-ni",                     # no interactive mode
                        "-r", str(sample_rate),
                        "-T", "wav",
                        "-F", str(synth_path),
                        str(sf2_path),
                        str(midi_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0 and synth_path.exists():
                    audio_data, _sr = sf.read(str(synth_path))
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)  # stereo → mono
                    # Normalize to prevent clipping
                    peak = np.max(np.abs(audio_data))
                    if peak > 0:
                        audio_data = audio_data / peak * 0.9
                    sf.write(str(synth_path), audio_data, sample_rate)
                    log.info("Piano audio synthesized via FluidSynth CLI")
                    return synth_path
                log.warning(f"FluidSynth CLI failed (rc={result.returncode})")
            except (subprocess.TimeoutExpired, OSError) as e:
                log.warning(f"FluidSynth CLI failed: {e}")

        # ── Approach 2: pyfluidsynth (Python binding, if installed) ─────
        try:
            if sf2_path.exists():
                audio_data = pm.fluidsynth(fs=sample_rate, sf2_path=str(sf2_path))
                log.info("Synthesized via pyfluidsynth")
            else:
                raise FileNotFoundError("SoundFont not found")
        except Exception:
            # ── Approach 3: Built-in sine-wave fallback ────────────────
            log.warning(
                "No FluidSynth available — using built-in sine-wave synth. "
                "Install pyfluidsynth or ensure assets/fluidsynth/bin/ "
                "contains fluidsynth.exe for realistic piano sound."
            )
            try:
                audio_data = pm.synthesize(fs=sample_rate)
            except Exception as exc:
                log.warning(f"MIDI audio synthesis failed: {exc}")
                return None

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            audio_data = audio_data / peak * 0.9

        sf.write(str(synth_path), audio_data, sample_rate)
        return synth_path

    # ── MIDI Parsing ─────────────────────────────────────────────────────

    def _extract_notes(self, pm: pretty_midi.PrettyMIDI) -> list[_NoteEvent]:
        """Extract all notes from the PrettyMIDI object.

        Channel assignment: instruments named 'Right Hand' → channel 0,
        'Left Hand' → channel 1. Falls back to instrument list index
        for any instrument with an unrecognized name.
        """
        events: list[_NoteEvent] = []
        for idx, inst in enumerate(pm.instruments):
            if inst.is_drum:
                continue
            if "Right Hand" in inst.name:
                channel = 0
            elif "Left Hand" in inst.name:
                channel = 1
            else:
                channel = 0 if idx == 0 else 1
            for note in inst.notes:
                # Clamp to piano range
                if note.pitch < PIANO_MIN_NOTE or note.pitch > PIANO_MAX_NOTE:
                    continue
                events.append(
                    _NoteEvent(
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                        velocity=note.velocity,
                        channel=channel,
                    )
                )
        return events

    # ── Frame Drawing ────────────────────────────────────────────────────

    def _draw_frame(self, t: float, notes: list[_NoteEvent]) -> np.ndarray:
        """Draw a single frame at time t."""
        img = Image.new("RGB", (self._width, self._height), self._bg_color)
        draw = ImageDraw.Draw(img, "RGBA")

        # Determine which notes are active (currently sounding) at time t
        active_pitches: set[int] = set()
        for note in notes:
            if note.start <= t < note.end:
                active_pitches.add(note.pitch)

        # Draw falling notes
        self._draw_falling_notes(draw, t, notes)

        # Draw strike line (translucent white line where notes "land")
        strike_y = self._fall_height
        draw.rectangle(
            [0, strike_y - 1, self._width, strike_y + 1],
            fill=STRIKE_LINE_COLOR,
        )

        # Draw keyboard
        self._draw_keyboard(draw, active_pitches)

        return np.array(img)

    def _draw_falling_notes(
        self,
        draw: ImageDraw.ImageDraw,
        t: float,
        notes: list[_NoteEvent],
    ) -> None:
        """Draw all visible falling note rectangles."""
        # Visible time window: notes that will be struck between t and t + LOOKAHEAD
        # Notes scroll down from top; their bottom edge hits strike_y at note.start
        pixels_per_second = self._fall_height / LOOKAHEAD_SECONDS

        for note in notes:
            # Skip notes that are completely out of view
            if note.end < t - 0.5:
                continue
            if note.start > t + LOOKAHEAD_SECONDS + 0.5:
                continue

            # Calculate vertical position
            # When t == note.start, the bottom of the note is at strike_y
            note_bottom_y = self._fall_height - (note.start - t) * pixels_per_second
            note_duration_px = (note.end - note.start) * pixels_per_second
            note_top_y = note_bottom_y - note_duration_px

            # Clamp to visible area
            if note_bottom_y < 0 or note_top_y > self._fall_height:
                continue

            # Get horizontal position
            if note.pitch not in self._note_x:
                continue
            x = self._note_x[note.pitch]
            w = self._note_w[note.pitch]

            # Pick color based on channel
            is_active = note.start <= t < note.end
            if note.channel == 0:
                color = RH_NOTE_ACTIVE if is_active else RH_NOTE_COLOR
                border_color = RH_NOTE_GLOW
            elif note.channel == 1:
                color = LH_NOTE_ACTIVE if is_active else LH_NOTE_COLOR
                border_color = LH_NOTE_GLOW
            else:
                color = OTHER_NOTE_ACTIVE if is_active else OTHER_NOTE_COLOR
                border_color = OTHER_NOTE_COLOR

            # Clamp drawing coordinates
            draw_top = max(0, int(note_top_y))
            draw_bottom = min(self._fall_height, int(note_bottom_y))

            if draw_bottom - draw_top < 1:
                continue

            # Margin between adjacent notes for visual clarity
            margin = 1
            rect_x0 = int(x) + margin
            rect_x1 = int(x + w) - margin

            # Draw note rectangle with slight rounding effect
            # Main body
            draw.rectangle(
                [rect_x0, draw_top, rect_x1, draw_bottom],
                fill=color,
            )

            # Subtle border for depth
            draw.rectangle(
                [rect_x0, draw_top, rect_x1, draw_bottom],
                outline=border_color,
                width=1,
            )

            # Active glow: draw a brighter strip at the bottom when note is playing
            if is_active:
                glow_h = min(4, draw_bottom - draw_top)
                draw.rectangle(
                    [rect_x0, draw_bottom - glow_h, rect_x1, draw_bottom],
                    fill=(255, 255, 255, 120),
                )

    def _draw_keyboard(
        self,
        draw: ImageDraw.ImageDraw,
        active_pitches: set[int],
    ) -> None:
        """Draw the piano keyboard at the bottom of the frame."""
        kb_top = self._fall_height
        kb_bottom = self._height

        # Draw white keys first
        for i, note in enumerate(self._white_key_indices):
            x = int(i * self._white_key_width)
            x_end = int((i + 1) * self._white_key_width)

            if note in active_pitches:
                fill = ACTIVE_WHITE_KEY
            else:
                fill = WHITE_KEY_COLOR

            draw.rectangle([x, kb_top, x_end, kb_bottom], fill=fill)
            # Key separator line
            draw.line(
                [x, kb_top, x, kb_bottom],
                fill=KEY_BORDER_COLOR,
                width=1,
            )

        # Draw black keys on top
        for note in range(PIANO_MIN_NOTE, PIANO_MAX_NOTE + 1):
            if not _is_black_key(note):
                continue
            if note not in self._note_x:
                continue

            x = int(self._note_x[note])
            w = int(self._black_key_width)

            if note in active_pitches:
                fill = ACTIVE_BLACK_KEY
            else:
                fill = BLACK_KEY_COLOR

            draw.rectangle(
                [x, kb_top, x + w, kb_top + self._black_key_height],
                fill=fill,
            )
