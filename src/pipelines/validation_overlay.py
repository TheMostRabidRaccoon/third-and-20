#!/usr/bin/env python3
"""
Third & 20 - Validation Overlay v1.0

Draws what the sensor saw onto the snap frame: tracked players colored by
derived role, the LOS on the correct axis, the drive-direction arrow, and
a validity text block. One glance answers "did it read this play right?"

This is the seed of the coach-facing audit appendix: for every report,
show the plays we trusted and why.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

from play_state import PlayState


ROLE_COLORS = {
    'offense': (255, 160, 0),    # BGR: blue
    'defense': (0, 0, 255),      # BGR: red
    'unknown': (160, 160, 160),  # gray
}
LOS_COLOR = (0, 255, 255)        # yellow
ARROW_COLOR = (0, 255, 0)        # green
TEXT_COLOR = (255, 255, 255)


def derive_roles(positions: Dict[int, Tuple[float, float]],
                 team_ids: Dict[int, Optional[str]],
                 play_state: PlayState) -> Dict[int, str]:
    """Role per track, mirroring the pipeline's identity/geometry logic"""
    roles = {}
    for track_id, (x, y) in positions.items():
        if play_state.identity_valid:
            team = team_ids.get(track_id)
            if team == play_state.offense_team_id:
                roles[track_id] = 'offense'
            elif team == play_state.defense_team_id:
                roles[track_id] = 'defense'
            else:
                roles[track_id] = 'unknown'
        elif play_state.geometry_valid:
            depth = play_state.depth_at(x, y)
            if depth is None:
                roles[track_id] = 'unknown'
            else:
                roles[track_id] = 'defense' if depth > 0 else 'offense'
        else:
            roles[track_id] = 'unknown'
    return roles


def draw_play_overlay(
    frame: np.ndarray,
    positions: Dict[int, Tuple[float, float]],
    team_ids: Dict[int, Optional[str]],
    play_state: PlayState,
    out_path: str,
    motion_summary: Optional[dict] = None,
    title: str = ""
) -> str:
    """
    Render the diagnostic overlay for one play and write it to out_path.

    Args:
        frame: snap frame (BGR)
        positions: track_id -> normalized (x, y) at snap
        team_ids: track_id -> team identity (may be empty)
        play_state: inferred PlayState
        out_path: image file to write
        motion_summary: optional MotionCompensationResult.to_dict()
        title: optional label (clip name)

    Returns:
        out_path
    """
    img = frame.copy()
    h, w = img.shape[:2]
    roles = derive_roles(positions, team_ids, play_state)

    # LOS line, perpendicular to the depth axis
    if play_state.los_x is not None:
        if play_state.axis == 'x':
            px = int(play_state.los_x * w)
            cv2.line(img, (px, 0), (px, h), LOS_COLOR, 2)
        else:
            py = int(play_state.los_x * h)
            cv2.line(img, (0, py), (w, py), LOS_COLOR, 2)

    # Drive-direction arrow along the depth axis, anchored near frame center
    if play_state.drive_dir is not None:
        length = int(0.08 * (w if play_state.axis == 'x' else h))
        cx, cy = w // 2, int(0.08 * h)
        if play_state.axis == 'x':
            start, end = (cx, cy), (cx + play_state.drive_dir * length, cy)
        else:
            cx, cy = int(0.06 * w), h // 2
            start, end = (cx, cy), (cx, cy + play_state.drive_dir * length)
        cv2.arrowedLine(img, start, end, ARROW_COLOR, 3, tipLength=0.3)

    # Players: circle colored by role, track id label
    for track_id, (x, y) in positions.items():
        px, py = int(x * w), int(y * h)
        color = ROLE_COLORS[roles[track_id]]
        cv2.circle(img, (px, py), 10, color, 2)
        cv2.putText(img, str(track_id), (px + 12, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Validity text block
    lines = []
    if title:
        lines.append(title)
    lines.append(f"axis={play_state.axis}  los={play_state.los_x:.3f}" if play_state.los_x is not None
                 else f"axis={play_state.axis}  los=None")
    lines.append(f"geometry_valid={play_state.geometry_valid}  "
                 f"identity_valid={play_state.identity_valid}  "
                 f"roles={'identity' if play_state.identity_valid else 'geometry' if play_state.geometry_valid else 'n/a'}")
    reasons = ', '.join(play_state.failure_reasons) or 'none'
    lines.append(f"failure_reasons: {reasons}")
    if motion_summary:
        lines.append(f"drive raw={motion_summary.get('drive_dir_raw')}  "
                     f"compensated={motion_summary.get('drive_dir_compensated')}  "
                     f"cam_conf={motion_summary.get('confidence')}  "
                     f"zoom={motion_summary.get('camera_scale')}")
        if motion_summary.get('drive_dir_disagrees'):
            lines.append("!! CAMERA MOTION CHANGES DRIVE DIRECTION !!")

    y_cursor = h - 14 * len(lines) - 10
    for line in lines:
        cv2.putText(img, line, (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
        y_cursor += 14

    cv2.imwrite(out_path, img)
    return out_path
