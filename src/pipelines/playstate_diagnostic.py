#!/usr/bin/env python3
"""
PlayState Diagnostic - Verify the sensor before trusting metrics.

This runs the minimum pipeline needed to log PlayState inference:
1. Snap detection
2. Player tracking
3. Jersey OCR + identity assignment
4. PlayState inference (THE SENSOR)

We log ONLY:
- play_state.is_valid
- los_x, los_confidence
- drive_dir, drive_dir_confidence
- possession_team_id, possession_confidence
- offense_team_id, defense_team_id

If everything comes back valid with high confidence, that's a RED FLAG.
Real HS film should have some invalid states and low confidence plays.
"""

import cv2
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from third_and_20_cv_v2_fixed import SnapDetectorV2
from player_tracker import PlayerTracker, PlayTracking
from play_state import PlayState, infer_play_state


def run_diagnostic(clip_path: str, home_team: str = "BRUSH", away_team: str = "EUCLID",
                   home_color: str = "brown", away_color: str = "white") -> Optional[Dict]:
    """
    Run minimal pipeline to extract and log PlayState.

    Returns dict with PlayState fields for logging, or None if snap not detected.
    """
    clip_name = Path(clip_path).name
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: {clip_name}")
    print(f"{'='*60}")

    # Step 1: Detect snap
    print("  [1/3] Detecting snap...")
    snap_detector = SnapDetectorV2()
    snap_result = snap_detector.analyze_clip(clip_path)

    if snap_result.snap_frame is None:
        print(f"  SKIP: No snap detected")
        return None

    print(f"  Snap at frame {snap_result.snap_frame}")

    # Step 2: Track players
    print("  [2/3] Tracking players...")
    tracker = PlayerTracker()
    tracking = tracker.track_clip(clip_path, snap_result.snap_frame)
    print(f"  Tracked {len(tracking.players)} players")

    # Step 3: Get positions at snap and assign mock identities
    # (In full pipeline, jersey OCR does this. Here we skip OCR and
    # assign identities based on position heuristic for diagnostic.)
    print("  [3/3] Inferring PlayState...")

    positions = tracking.get_all_positions_at_frame(snap_result.snap_frame)

    # For diagnostic: assign team_id based on Y position (top/bottom half)
    # This is a TEMPORARY heuristic - real pipeline uses jersey OCR
    team_ids: Dict[int, Optional[str]] = {}
    for track_id, (x, y) in positions.items():
        if y < 0.4:
            team_ids[track_id] = home_team  # Top of frame
        elif y > 0.6:
            team_ids[track_id] = away_team  # Bottom of frame
        else:
            team_ids[track_id] = None  # Unknown (middle)

    # Count known identities
    known_home = sum(1 for t in team_ids.values() if t == home_team)
    known_away = sum(1 for t in team_ids.values() if t == away_team)
    known_total = known_home + known_away
    print(f"  Identity heuristic: {known_home} {home_team}, {known_away} {away_team}, {len(positions) - known_total} unknown")

    # Get post-snap positions (30 frames after snap, ~1 second)
    post_snap_frame = snap_result.snap_frame + 30
    positions_post = tracking.get_all_positions_at_frame(post_snap_frame)
    print(f"  Post-snap positions: {len(positions_post)} tracks at frame {post_snap_frame}")

    # THE MOMENT: Infer PlayState
    play_state = infer_play_state(positions, positions_post, team_ids)

    # Log the sensor output
    print(f"\n  {'─'*40}")
    print(f"  PLAYSTATE SENSOR OUTPUT")
    print(f"  {'─'*40}")
    print(f"  is_valid:              {play_state.is_valid}")
    print(f"  los_x:                 {play_state.los_x}")
    print(f"  los_confidence:        {play_state.los_confidence:.3f}")
    print(f"  drive_dir:             {play_state.drive_dir}")
    print(f"  drive_dir_confidence:  {play_state.drive_dir_confidence:.3f}")
    print(f"  possession_team_id:    {play_state.possession_team_id}")
    print(f"  possession_confidence: {play_state.possession_confidence:.3f}")
    print(f"  offense_team_id:       {play_state.offense_team_id}")
    print(f"  defense_team_id:       {play_state.defense_team_id}")
    print(f"  {'─'*40}")

    # Return as dict for aggregation
    return {
        'clip': clip_name,
        'snap_frame': snap_result.snap_frame,
        'is_valid': play_state.is_valid,
        'los_x': play_state.los_x,
        'los_confidence': play_state.los_confidence,
        'drive_dir': play_state.drive_dir,
        'drive_dir_confidence': play_state.drive_dir_confidence,
        'possession_team_id': play_state.possession_team_id,
        'possession_confidence': play_state.possession_confidence,
        'offense_team_id': play_state.offense_team_id,
        'defense_team_id': play_state.defense_team_id,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PlayState Diagnostic")
    parser.add_argument("clips", nargs="+", help="Video clips to diagnose")
    parser.add_argument("--home-team", default="BRUSH")
    parser.add_argument("--away-team", default="EUCLID")
    parser.add_argument("--home-color", default="brown")
    parser.add_argument("--away-color", default="white")

    args = parser.parse_args()

    results = []
    for clip in args.clips:
        try:
            result = run_diagnostic(
                clip,
                home_team=args.home_team,
                away_team=args.away_team,
                home_color=args.home_color,
                away_color=args.away_color
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")

    if not results:
        print("No plays analyzed")
        return

    valid_count = sum(1 for r in results if r['is_valid'])
    invalid_count = len(results) - valid_count

    print(f"Total plays:    {len(results)}")
    print(f"Valid states:   {valid_count} ({100*valid_count/len(results):.1f}%)")
    print(f"Invalid states: {invalid_count} ({100*invalid_count/len(results):.1f}%)")

    # Check for red flags
    if valid_count == len(results) and len(results) > 3:
        print("\n⚠️  RED FLAG: 100% valid - likely too permissive")

    # Confidence distribution
    los_confs = [r['los_confidence'] for r in results if r['los_confidence'] is not None]
    drive_confs = [r['drive_dir_confidence'] for r in results if r['drive_dir_confidence'] is not None]
    poss_confs = [r['possession_confidence'] for r in results if r['possession_confidence'] is not None]

    if los_confs:
        print(f"\nLOS confidence:        avg={sum(los_confs)/len(los_confs):.3f}, min={min(los_confs):.3f}, max={max(los_confs):.3f}")
    if drive_confs:
        print(f"Drive dir confidence:  avg={sum(drive_confs)/len(drive_confs):.3f}, min={min(drive_confs):.3f}, max={max(drive_confs):.3f}")
    if poss_confs:
        print(f"Possession confidence: avg={sum(poss_confs)/len(poss_confs):.3f}, min={min(poss_confs):.3f}, max={max(poss_confs):.3f}")

    # Drive direction distribution
    drive_dirs = [r['drive_dir'] for r in results if r['drive_dir'] is not None]
    if drive_dirs:
        left = drive_dirs.count(-1)
        right = drive_dirs.count(1)
        print(f"\nDrive direction: {left} left (-1), {right} right (+1)")
        if left == 0 or right == 0:
            print("⚠️  WARNING: All plays same direction - check for quarter switch handling")


if __name__ == "__main__":
    main()
