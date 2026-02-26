#!/usr/bin/env python3
"""
Quick test of the game aggregator on available clips
"""

import sys
from pathlib import Path

# Test paths
SHAKER_BASE = Path("/Users/rabidraccoon/Library/CloudStorage/GoogleDrive-kad@rabidraccoonintelligence.org/My Drive/THIRD_AND_20/data/games/Brush 2025 games/Shaker ✅")
BEDFORD_BASE = Path("/Users/rabidraccoon/Library/CloudStorage/GoogleDrive-kad@rabidraccoonintelligence.org/My Drive/THIRD_AND_20/data/games/Brush 2025 games/Bedford✅")

def find_clips(base_path, max_clips=5):
    """Find up to max_clips mp4 files"""
    if not base_path.exists():
        print(f"Path does not exist: {base_path}")
        return []
    clips = list(base_path.glob("*.mp4"))[:max_clips]
    return [str(c) for c in clips]


def main():
    from run_game_analysis import GameAnalysisRunner

    # Try Shaker (more clips, better coverage)
    clips = find_clips(SHAKER_BASE, max_clips=5)

    if not clips:
        # Fallback to Bedford
        clips = find_clips(BEDFORD_BASE, max_clips=2)

    if not clips:
        print("No clips found!")
        return

    print(f"Found {len(clips)} clips:")
    for c in clips:
        print(f"  {Path(c).name}")

    # Run analysis
    runner = GameAnalysisRunner(
        home_team="BRUSH",
        away_team="BEDFORD" if "Bedford" in clips[0] else "SHAKER",
        home_color="brown",
        away_color="white",
        roster_path=None
    )

    runner.process_clips(clips, "/tmp/test_aggregation")


if __name__ == "__main__":
    main()
