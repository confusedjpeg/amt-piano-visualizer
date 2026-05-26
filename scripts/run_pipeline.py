#!/usr/bin/env python3
"""
CLI entry point for the AI Piano Arranger pipeline.

Usage:
    python scripts/run_pipeline.py "data/input/song.mp3" --include-vocals --has-piano
    python scripts/run_pipeline.py "data/input/edm.mp3" --no-vocals --no-piano
    python scripts/run_pipeline.py "data/input/rock.mp3" --config custom.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Ensure the project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.logger import setup_logging


@click.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option(
    "--include-vocals/--no-vocals",
    default=None,
    help="Include vocal melody in the right hand. Default: use config.",
)
@click.option(
    "--has-piano/--no-piano",
    default=None,
    help="Source audio contains piano (True) or needs arrangement (False). Default: use config.",
)
@click.option(
    "--config",
    type=click.Path(),
    default="config.yaml",
    help="Path to configuration file.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for final artifacts. Default: use config.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging verbosity.",
)
def main(
    audio_path: str,
    include_vocals: bool | None,
    has_piano: bool | None,
    config: str,
    output_dir: str | None,
    log_level: str,
) -> None:
    """AI Piano Arranger — Audio to Synthesia Pipeline.

    Transforms a raw audio file into a playable two-handed piano MIDI
    and a synced Synthesia-style falling-notes video.
    """
    setup_logging(log_level.upper())

    # Load configuration
    config_path = Path(config)
    if config_path.exists():
        click.echo(f"Loading config from {config_path}")
        pipeline_config = PipelineConfig.from_yaml(config_path)
    else:
        click.echo(f"Config file not found ({config_path}), using defaults.")
        pipeline_config = PipelineConfig()

    # Initialize and run
    orchestrator = PipelineOrchestrator(pipeline_config)

    click.echo("=" * 60)
    click.echo("  AI Piano Arranger — Audio to Synthesia Pipeline")
    click.echo("=" * 60)
    click.echo(f"  Input:          {audio_path}")
    click.echo(f"  Include Vocals: {include_vocals if include_vocals is not None else 'config default'}")
    click.echo(f"  Has Piano:      {has_piano if has_piano is not None else 'config default'}")
    click.echo("=" * 60)

    try:
        result = orchestrator.run(
            audio_path=Path(audio_path),
            include_vocals=include_vocals,
            has_piano=has_piano,
            output_dir=Path(output_dir) if output_dir else None,
        )

        click.echo("")
        click.echo("=" * 60)
        click.echo("  ✅ Pipeline Complete!")
        click.echo("=" * 60)
        click.echo(f"  Run ID:         {result.run_id}")
        click.echo(f"  Duration:       {result.duration_seconds:.1f}s")
        click.echo(f"  MIDI Output:    {result.midi_path}")
        if result.video_path and str(result.video_path):
            click.echo(f"  Video Output:   {result.video_path}")
        click.echo(f"  Steps:          {', '.join(result.steps_completed)}")
        if result.warnings:
            click.echo(f"  Warnings:       {len(result.warnings)}")
            for w in result.warnings:
                click.echo(f"    ⚠ {w}")
        click.echo("=" * 60)

    except Exception as exc:
        click.echo(f"\n❌ Pipeline failed: {exc}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
