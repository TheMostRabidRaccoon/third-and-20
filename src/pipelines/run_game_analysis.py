#!/usr/bin/env python3
"""
Third & 20 - Game Analysis Runner v1.0

Runs the SDI pipeline on multiple clips and aggregates results into
game-level defensive tendencies and positional metrics.

Usage:
    python run_game_analysis.py *.mp4 --home-team BRUSH --away-team BEDFORD \\
        --home-color brown --away-color white

Outputs:
    - {output}_offensive_sdi.csv         (per-player offensive metrics)
    - {output}_defensive_analysis.csv    (per-play scheme detection)
    - {output}_defensive_players.csv     (per-player defensive metrics)
    - {output}_game_summary.txt          (aggregated tendencies + tells)
    - {output}_game_summary.json         (machine-readable aggregated data)
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from typing import List, Optional
import argparse
import json

# Import pipeline components
from sdi_pipeline_v5_with_defense import SDIPipelineV5, CombinedPlayAnalysis
from game_aggregator import (
    GameAggregator,
    PlayResult,
    GeometricPosition,
    create_play_result_from_pipeline
)
from defensive_inference import DefensiveInference


def convert_to_play_result(
    play: CombinedPlayAnalysis,
    play_number: int = 0
) -> Optional[PlayResult]:
    """
    Convert CombinedPlayAnalysis to PlayResult for aggregation.

    This bridges the per-play pipeline output to the aggregator.
    """
    if play.defensive_analysis is None:
        return None

    da = play.defensive_analysis
    play_state = play.play_state  # Now exposed directly in CombinedPlayAnalysis

    # Determine validity from both PlayState and DefensiveAnalysis
    is_valid = (da.confidence > 0.3)
    if play_state:
        is_valid = is_valid and play_state.is_valid

    result = PlayResult(
        clip_name=play.clip_file,
        play_number=play_number,
        is_valid=is_valid,
        los_x=play_state.los_x if play_state else None,
        drive_dir=play_state.drive_dir if play_state else None,
        coverage_shell=da.coverage_shell.value if hasattr(da.coverage_shell, 'value') else str(da.coverage_shell),
        front=da.front.value if hasattr(da.front, 'value') else str(da.front),
        box_count=da.box_count,
        safety_alignment=da.safety_alignment.value if hasattr(da.safety_alignment, 'value') else str(da.safety_alignment)
    )

    # Build geometric positions from defensive_analysis player positions
    inference = DefensiveInference()
    if play_state:
        inference.los_x = play_state.los_x
        inference.drive_dir = play_state.drive_dir

    for player in da.player_positions:
        if player.team == "defense":
            position, alignment = inference.classify_player_position(player, da)

            # Compute signed depth if we have PlayState
            depth = 0.0
            if play_state and play_state.los_x is not None and play_state.drive_dir is not None:
                depth = (player.x - play_state.los_x) * play_state.drive_dir

            geo_pos = GeometricPosition(
                track_id=player.track_id,
                role="defense",
                position=position,
                alignment=alignment,
                depth_from_los=depth,
                width_from_center=abs(player.y - 0.5)
            )

            # Try to get movement metrics from defensive_metrics
            for dm in play.defensive_metrics:
                if dm.track_id == player.track_id:
                    geo_pos.first_step_sec = dm.first_step_sec
                    geo_pos.penetration = dm.penetration_depth
                    geo_pos.lateral_movement = dm.lateral_movement
                    break

            result.defensive_positions.append(geo_pos)

    return result


class GameAnalysisRunner:
    """
    Runs the SDI pipeline and aggregates results.

    Wraps SDIPipelineV5 and feeds results to GameAggregator.
    """

    def __init__(self, home_team: str, away_team: str,
                 home_color: str, away_color: str,
                 roster_path: str = None):
        self.pipeline = SDIPipelineV5(
            roster_path=roster_path,
            home_team=home_team,
            away_team=away_team,
            home_color=home_color,
            away_color=away_color
        )

        self.aggregator = GameAggregator(
            home_team=home_team,
            away_team=away_team
        )

    def process_clips(self, clip_paths: List[str], output_prefix: str):
        """Process all clips and generate aggregated outputs"""

        print(f"\n{'='*60}")
        print(f"PROCESSING {len(clip_paths)} CLIPS")
        print(f"{'='*60}")

        # Process each clip
        for i, clip_path in enumerate(clip_paths, 1):
            clip_name = Path(clip_path).name
            print(f"\n[{i}/{len(clip_paths)}] {clip_name}")

            try:
                # Run pipeline
                play = self.pipeline.process_clip(clip_path)

                if play:
                    # Convert to PlayResult (PlayState is now in CombinedPlayAnalysis)
                    result = convert_to_play_result(play, play_number=i)

                    if result and result.is_valid:
                        self.aggregator.add_play(result)
                        print(f"  Added to aggregation: {result.coverage_shell}, {result.front}")
                    elif result:
                        print(f"  Skipped (invalid): confidence too low or PlayState invalid")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Write standard pipeline outputs
        self.pipeline._write_results(output_prefix)

        # Write aggregated outputs
        self._write_aggregated_outputs(output_prefix)

    def _write_aggregated_outputs(self, output_prefix: str):
        """Write aggregated game-level outputs"""

        print(f"\n{'='*60}")
        print("WRITING AGGREGATED OUTPUTS")
        print(f"{'='*60}")

        # 1. Human-readable summary
        summary_txt = f"{output_prefix}_game_summary.txt"
        summary = self.aggregator.get_summary()
        with open(summary_txt, 'w') as f:
            f.write(summary)
        print(f"\nGame summary saved to: {summary_txt}")

        # Print to console too
        print(f"\n{summary}")

        # 2. Machine-readable JSON
        summary_json = f"{output_prefix}_game_summary.json"
        data = self.aggregator.to_dict()
        with open(summary_json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nJSON data saved to: {summary_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Third & 20 Game Analysis - Aggregate defensive tendencies and positional metrics"
    )
    parser.add_argument("clips", nargs="+", help="Video clip files")
    parser.add_argument("-o", "--output", default="game_analysis",
                        help="Output file prefix (creates multiple files)")
    parser.add_argument("--roster", help="Roster CSV file")
    parser.add_argument("--home-team", default="HOME", help="Home team name")
    parser.add_argument("--away-team", default="AWAY", help="Away team name")
    parser.add_argument("--home-color", default="white",
                        help="Home jersey color")
    parser.add_argument("--away-color", default="red",
                        help="Away jersey color")

    args = parser.parse_args()

    # Expand glob patterns
    clips = []
    for pattern in args.clips:
        path = Path(pattern)
        if path.exists():
            clips.append(str(path))
        else:
            # Try glob
            clips.extend(str(p) for p in Path('.').glob(pattern))

    clips = sorted(set(clips))

    if not clips:
        print("No clips found!")
        return

    print(f"Third & 20 Game Analysis v1.0")
    print(f"=" * 60)
    print(f"Clips: {len(clips)}")
    print(f"Home: {args.home_team} ({args.home_color})")
    print(f"Away: {args.away_team} ({args.away_color})")
    print(f"Output: {args.output}")

    # Run analysis
    runner = GameAnalysisRunner(
        home_team=args.home_team,
        away_team=args.away_team,
        home_color=args.home_color,
        away_color=args.away_color,
        roster_path=args.roster
    )

    runner.process_clips(clips, args.output)


if __name__ == "__main__":
    main()
