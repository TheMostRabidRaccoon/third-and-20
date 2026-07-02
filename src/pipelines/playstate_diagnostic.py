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
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from third_and_20_cv_v2_fixed import SnapDetectorV2
from player_tracker import PlayerTracker, PlayTracking
from play_state import PlayState, infer_play_state
from camera_motion import run_motion_diagnostic
from validation_overlay import draw_play_overlay


def _read_frame(clip_path: str, frame_num: int):
    """Read one frame from a clip, or None"""
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _player_boxes_at_frame(tracking: PlayTracking, frame_num: int):
    """Normalized (cx, cy, w, h) boxes for every track at a frame"""
    boxes = []
    for track in tracking.players.values():
        for f in track.frames:
            if f.frame_num == frame_num:
                boxes.append((f.x, f.y, f.width, f.height))
                break
    return boxes


def run_diagnostic(clip_path: str, home_team: str = "BRUSH", away_team: str = "EUCLID",
                   home_color: str = "brown", away_color: str = "white",
                   axis: str = "x", overlay_dir: Optional[str] = None) -> Optional[Dict]:
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
    post_snap_overlap = len(set(positions.keys()) & set(positions_post.keys()))
    print(f"  Post-snap positions: {len(positions_post)} tracks at frame {post_snap_frame} "
          f"(overlap with snap: {post_snap_overlap})")

    # THE MOMENT: Infer PlayState
    play_state = infer_play_state(positions, positions_post, team_ids, axis=axis)

    # Camera-motion diagnostic: is the pan/zoom corrupting drive direction?
    # Diagnostic only - production drive_dir is unchanged.
    motion_record = None
    frame_pre = _read_frame(clip_path, snap_result.snap_frame)
    frame_post = _read_frame(clip_path, post_snap_frame)
    if frame_pre is not None and frame_post is not None:
        motion = run_motion_diagnostic(
            frame_pre, frame_post, positions, positions_post,
            axis=play_state.axis,
            player_boxes=_player_boxes_at_frame(tracking, snap_result.snap_frame)
        )
        motion_record = motion.to_dict()
        print(f"  Camera motion: ok={motion.estimation_ok} "
              f"conf={motion.confidence:.2f} zoom={motion.camera_scale:.3f} "
              f"drive raw={motion.drive_dir_raw} vs compensated={motion.drive_dir_compensated}")
        if motion.drive_dir_disagrees:
            print(f"  ⚠️  CAMERA MOTION CHANGES DRIVE DIRECTION on this play")

    # Overlay snapshot: see what the sensor saw
    overlay_path = None
    if overlay_dir and frame_pre is not None:
        Path(overlay_dir).mkdir(parents=True, exist_ok=True)
        overlay_path = str(Path(overlay_dir) / f"{Path(clip_path).stem}_snap_overlay.jpg")
        draw_play_overlay(frame_pre, positions, team_ids, play_state,
                          overlay_path, motion_summary=motion_record, title=clip_name)
        print(f"  Overlay written: {overlay_path}")

    # Log the sensor output
    print(f"\n  {'─'*40}")
    print(f"  PLAYSTATE SENSOR OUTPUT")
    print(f"  {'─'*40}")
    print(f"  geometry_valid:        {play_state.geometry_valid}")
    print(f"  identity_valid:        {play_state.identity_valid}")
    print(f"  failure_reasons:       {', '.join(play_state.failure_reasons) or 'none'}")
    print(f"  axis:                  {play_state.axis}")
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
        'geometry_valid': play_state.geometry_valid,
        'identity_valid': play_state.identity_valid,
        'is_valid': play_state.is_valid,
        'failure_reasons': play_state.failure_reasons,
        'axis': play_state.axis,
        'los_x': play_state.los_x,
        'los_confidence': play_state.los_confidence,
        'drive_dir': play_state.drive_dir,
        'drive_dir_confidence': play_state.drive_dir_confidence,
        'possession_team_id': play_state.possession_team_id,
        'possession_confidence': play_state.possession_confidence,
        'offense_team_id': play_state.offense_team_id,
        'defense_team_id': play_state.defense_team_id,
        'snap_tracks': len(positions),
        'post_snap_overlap': post_snap_overlap,
        'known_team_ids': known_total,
        'team_counts': {home_team: known_home, away_team: known_away},
        # CAVEAT: team_ids here come from a top/bottom-of-frame position
        # heuristic, NOT the OCR/color identity pipeline. identity_valid in
        # this record tests the possession gate mechanics, not real identity.
        'identity_source': 'position_heuristic',
        'camera_motion': motion_record,
        'overlay': overlay_path,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PlayState Diagnostic")
    parser.add_argument("clips", nargs="+", help="Video clips to diagnose")
    parser.add_argument("--home-team", default="BRUSH")
    parser.add_argument("--away-team", default="EUCLID")
    parser.add_argument("--home-color", default="brown")
    parser.add_argument("--away-color", default="white")
    parser.add_argument("--axis", choices=["x", "y"], default="x",
                        help="Depth axis: 'x' for endzone camera, 'y' for sideline camera")
    parser.add_argument("--json-out", help="Write per-play diagnostic records to this JSON file")
    parser.add_argument("--overlay-dir",
                        help="Write annotated snap-frame overlays into this directory")

    args = parser.parse_args()

    results = []
    for clip in args.clips:
        try:
            result = run_diagnostic(
                clip,
                home_team=args.home_team,
                away_team=args.away_team,
                home_color=args.home_color,
                away_color=args.away_color,
                axis=args.axis,
                overlay_dir=args.overlay_dir
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

    geometry_count = sum(1 for r in results if r['geometry_valid'])
    identity_count = sum(1 for r in results if r['identity_valid'])

    print(f"Total plays:     {len(results)}")
    print(f"Geometry valid:  {geometry_count} ({100*geometry_count/len(results):.1f}%)  "
          f"<- opponent structure computable")
    print(f"Identity valid:  {identity_count} ({100*identity_count/len(results):.1f}%)  "
          f"<- team/player metrics computable")

    # Failure reason histogram - this is what tells you WHERE the sensor breaks
    reason_counts: Dict[str, int] = {}
    for r in results:
        for reason in r['failure_reasons']:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        print("\nFailure reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")

    # Camera-motion verdict: does pan/zoom change drive direction on real film?
    motion_records = [r['camera_motion'] for r in results if r.get('camera_motion')]
    estimated = [m for m in motion_records if m['estimation_ok']]
    disagreements = [m for m in estimated if m['drive_dir_disagrees']]
    if motion_records:
        print(f"\nCamera motion: estimated on {len(estimated)}/{len(motion_records)} plays")
        print(f"Drive-dir changed by compensation: {len(disagreements)} plays")
        if disagreements:
            print("⚠️  Camera pan/zoom is affecting drive direction - "
                  "compensation is worth promoting into the pipeline")
        elif estimated:
            print("Camera motion is NOT changing drive direction - "
                  "keep raw drive_dir for now")

    # Check for red flags
    if identity_count == len(results) and len(results) > 3:
        print("\n⚠️  RED FLAG: 100% valid - likely too permissive")

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nPer-play records written to: {args.json_out}")

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
