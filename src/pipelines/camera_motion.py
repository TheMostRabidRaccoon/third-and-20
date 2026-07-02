#!/usr/bin/env python3
"""
Third & 20 - Camera Motion Diagnostic v1.0

HS film is shot by a human panning (and zooming) to follow the ball, so
player displacement in FRAME coordinates is not player displacement on the
FIELD. If the camera pans with the offense, the offense can appear nearly
stationary while the defense appears to move backward - which can flip the
inferred drive direction on exactly the plays with the most movement.

This module is DIAGNOSTIC-FIRST: it estimates the camera transform between
two frames from background features (player boxes masked out), compensates
tracked displacements by warping pre-snap positions through the FULL
transform, and reports drive direction raw vs compensated side by side.

It does NOT change production drive_dir. Promote compensated motion into
the pipeline only after the diagnostic shows it helps on real film.

Why warp, not translation subtraction: a single global dx/dy only models
pure pan. Under zoom, apparent motion depends on where a player sits
relative to the zoom center, so compensation must be
    field_disp = post_pos - warp(pre_pos, transform)
never
    field_disp = post_pos - pre_pos - camera_translation
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from play_state import infer_drive_direction


# Estimation knobs
MIN_FEATURES = 20          # below this, refuse to estimate
MIN_INLIERS = 12           # below this, treat the transform as unreliable
MAX_FEATURES = 400
RANSAC_REPROJ_THRESHOLD = 3.0  # pixels


@dataclass
class MotionCompensationResult:
    """Side-by-side record of raw vs camera-compensated bulk motion.

    All displacement values are normalized (fraction of frame size).
    `transform` is the 2x3 similarity matrix (pixel coords) as nested lists,
    or None when estimation failed - in which case only raw values are set
    and confidence is 0.
    """
    transform: Optional[List[List[float]]]
    camera_dx: float               # frame-center translation summary (diagnostic only)
    camera_dy: float
    camera_scale: float            # 1.0 = no zoom
    confidence: float              # inlier ratio, 0 when estimation failed
    inlier_count: int
    feature_count: int
    raw_median_dx: float
    raw_median_dy: float
    compensated_median_dx: float
    compensated_median_dy: float
    drive_dir_raw: Optional[int]
    drive_dir_compensated: Optional[int]
    axis: str = 'x'

    @property
    def estimation_ok(self) -> bool:
        return self.transform is not None and self.inlier_count >= MIN_INLIERS

    @property
    def drive_dir_disagrees(self) -> bool:
        """True when compensation changes the drive-direction answer"""
        return self.estimation_ok and self.drive_dir_raw != self.drive_dir_compensated

    def to_dict(self) -> dict:
        return {
            'estimation_ok': self.estimation_ok,
            'camera_dx': round(self.camera_dx, 5),
            'camera_dy': round(self.camera_dy, 5),
            'camera_scale': round(self.camera_scale, 4),
            'confidence': round(self.confidence, 3),
            'inlier_count': self.inlier_count,
            'feature_count': self.feature_count,
            'raw_median_dx': round(self.raw_median_dx, 5),
            'raw_median_dy': round(self.raw_median_dy, 5),
            'compensated_median_dx': round(self.compensated_median_dx, 5),
            'compensated_median_dy': round(self.compensated_median_dy, 5),
            'drive_dir_raw': self.drive_dir_raw,
            'drive_dir_compensated': self.drive_dir_compensated,
            'drive_dir_disagrees': self.drive_dir_disagrees,
            'axis': self.axis,
        }


def _build_background_mask(shape: Tuple[int, int],
                           player_boxes: Optional[List[Tuple[float, float, float, float]]],
                           pad: float = 0.15) -> np.ndarray:
    """Mask that excludes (padded) player boxes so features come from the
    field, not from the moving bodies we are trying to de-camera.

    Args:
        shape: (height, width) of the frame
        player_boxes: normalized (cx, cy, w, h) boxes, or None
        pad: fractional padding added around each box
    """
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    if not player_boxes:
        return mask

    for cx, cy, bw, bh in player_boxes:
        half_w = bw * (1 + pad) / 2
        half_h = bh * (1 + pad) / 2
        x1 = int(max(0, (cx - half_w) * w))
        y1 = int(max(0, (cy - half_h) * h))
        x2 = int(min(w, (cx + half_w) * w))
        y2 = int(min(h, (cy + half_h) * h))
        mask[y1:y2, x1:x2] = 0

    return mask


def estimate_camera_transform(
    frame_pre: np.ndarray,
    frame_post: np.ndarray,
    player_boxes: Optional[List[Tuple[float, float, float, float]]] = None
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Estimate the camera motion between two frames as a 2x3 similarity
    transform (rotation + uniform scale + translation, pixel coordinates).

    Similarity (4 DOF) rather than full affine/homography: it captures the
    pan + zoom that HS tripod film actually exhibits while staying robust
    on the sparse, low-texture features a grass field offers.

    Args:
        frame_pre / frame_post: BGR or grayscale frames
        player_boxes: normalized (cx, cy, w, h) boxes to exclude from
                      feature detection (players are not background)

    Returns:
        (transform, inlier_count, feature_count); transform is None when
        there is not enough trackable background to estimate reliably.
    """
    gray_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY) if frame_pre.ndim == 3 else frame_pre
    gray_post = cv2.cvtColor(frame_post, cv2.COLOR_BGR2GRAY) if frame_post.ndim == 3 else frame_post

    mask = _build_background_mask(gray_pre.shape[:2], player_boxes)

    pts = cv2.goodFeaturesToTrack(
        gray_pre, maxCorners=MAX_FEATURES, qualityLevel=0.01,
        minDistance=10, mask=mask
    )
    if pts is None or len(pts) < MIN_FEATURES:
        return None, 0, 0 if pts is None else len(pts)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_pre, gray_post, pts, None)
    if next_pts is None:
        return None, 0, len(pts)

    ok = status.ravel() == 1
    src = pts[ok].reshape(-1, 2)
    dst = next_pts[ok].reshape(-1, 2)
    if len(src) < MIN_FEATURES:
        return None, 0, len(pts)

    transform, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
    )
    if transform is None:
        return None, 0, len(src)

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return transform, inlier_count, len(src)


def warp_point(transform: np.ndarray, x_px: float, y_px: float) -> Tuple[float, float]:
    """Apply a 2x3 transform to one pixel-space point"""
    wx = transform[0, 0] * x_px + transform[0, 1] * y_px + transform[0, 2]
    wy = transform[1, 0] * x_px + transform[1, 1] * y_px + transform[1, 2]
    return wx, wy


def compensate_positions(
    positions_pre: Dict[int, Tuple[float, float]],
    transform: np.ndarray,
    width: int,
    height: int
) -> Dict[int, Tuple[float, float]]:
    """
    Warp normalized pre positions through the camera transform, i.e.
    'where would this player appear in the post frame if only the camera
    had moved'. Differencing post positions against these yields
    field-relative displacement.
    """
    warped = {}
    for track_id, (x, y) in positions_pre.items():
        wx, wy = warp_point(transform, x * width, y * height)
        warped[track_id] = (wx / width, wy / height)
    return warped


def _median_displacement(pre: Dict[int, Tuple[float, float]],
                         post: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
    dxs, dys = [], []
    for tid, (x0, y0) in pre.items():
        if tid in post:
            x1, y1 = post[tid]
            dxs.append(x1 - x0)
            dys.append(y1 - y0)
    if not dxs:
        return 0.0, 0.0
    return float(np.median(dxs)), float(np.median(dys))


def run_motion_diagnostic(
    frame_pre: np.ndarray,
    frame_post: np.ndarray,
    positions_pre: Dict[int, Tuple[float, float]],
    positions_post: Dict[int, Tuple[float, float]],
    axis: str = 'x',
    player_boxes: Optional[List[Tuple[float, float, float, float]]] = None
) -> MotionCompensationResult:
    """
    Full diagnostic for one play: estimate camera motion between the two
    frames and report drive direction raw vs compensated.

    Args:
        frame_pre / frame_post: the frames the two position snapshots came from
        positions_pre / positions_post: track_id -> normalized (x, y)
        axis: depth axis ('x' endzone, 'y' sideline)
        player_boxes: normalized (cx, cy, w, h) boxes at the pre frame
    """
    height, width = frame_pre.shape[:2]

    raw_dx, raw_dy = _median_displacement(positions_pre, positions_post)
    # Same thresholds the pipeline uses for production drive_dir
    drive_raw, _ = infer_drive_direction(
        positions_pre, positions_post, axis=axis, min_tracks=2, threshold=0.002
    )

    transform, inlier_count, feature_count = estimate_camera_transform(
        frame_pre, frame_post, player_boxes
    )

    if transform is None or inlier_count < MIN_INLIERS:
        return MotionCompensationResult(
            transform=None,
            camera_dx=0.0, camera_dy=0.0, camera_scale=1.0,
            confidence=0.0,
            inlier_count=inlier_count, feature_count=feature_count,
            raw_median_dx=raw_dx, raw_median_dy=raw_dy,
            compensated_median_dx=raw_dx, compensated_median_dy=raw_dy,
            drive_dir_raw=drive_raw, drive_dir_compensated=drive_raw,
            axis=axis,
        )

    # Translation summary at frame center (for logging only - compensation
    # below uses the full warp, which also accounts for zoom)
    cx, cy = width / 2.0, height / 2.0
    wcx, wcy = warp_point(transform, cx, cy)
    camera_dx = (wcx - cx) / width
    camera_dy = (wcy - cy) / height
    camera_scale = float(np.hypot(transform[0, 0], transform[1, 0]))

    warped_pre = compensate_positions(positions_pre, transform, width, height)
    comp_dx, comp_dy = _median_displacement(warped_pre, positions_post)
    drive_comp, _ = infer_drive_direction(
        warped_pre, positions_post, axis=axis, min_tracks=2, threshold=0.002
    )

    return MotionCompensationResult(
        transform=transform.tolist(),
        camera_dx=camera_dx, camera_dy=camera_dy, camera_scale=camera_scale,
        confidence=inlier_count / feature_count if feature_count else 0.0,
        inlier_count=inlier_count, feature_count=feature_count,
        raw_median_dx=raw_dx, raw_median_dy=raw_dy,
        compensated_median_dx=comp_dx, compensated_median_dy=comp_dy,
        drive_dir_raw=drive_raw, drive_dir_compensated=drive_comp,
        axis=axis,
    )
