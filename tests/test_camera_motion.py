"""Camera-motion diagnostic tests.

Synthetic frames with known camera transforms, players laid out like the
play_state tests: offense behind the LOS, defense downfield.
"""

import cv2
import numpy as np
import pytest

from camera_motion import (
    MotionCompensationResult,
    estimate_camera_transform,
    run_motion_diagnostic,
)
from validation_overlay import draw_play_overlay, derive_roles
from play_state import PlayState

W, H = 640, 480


def make_background(seed=7):
    """Trackable synthetic background: blurred noise + a few field lines"""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, (H, W), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # yard-line-ish verticals give strong features
    for x in range(40, W, 80):
        cv2.line(img, (x, 0), (x, H), 255, 2)
    return img


def shift_frame(frame, dx_px, dy_px=0.0):
    """Simulate a camera pan of (+dx, +dy): content shifts by (-dx, -dy)"""
    M = np.float32([[1, 0, -dx_px], [0, 1, -dy_px]])
    return cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def zoom_frame(frame, scale):
    """Simulate a zoom-in around the frame center"""
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
    return cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def formation():
    """track_id -> normalized (x, y); ids 1-7 offense, 8-14 defense"""
    return {
        1: (0.35, 0.42), 2: (0.35, 0.46), 3: (0.35, 0.50),
        4: (0.35, 0.54), 5: (0.35, 0.58),
        6: (0.32, 0.50), 7: (0.30, 0.50),
        8: (0.45, 0.42), 9: (0.45, 0.47), 10: (0.45, 0.53), 11: (0.45, 0.58),
        12: (0.50, 0.45), 13: (0.50, 0.55),
        14: (0.62, 0.50),
    }


def test_recovers_pure_pan():
    pre = make_background()
    pan_px = 25.0
    post = shift_frame(pre, pan_px)

    transform, inliers, features = estimate_camera_transform(pre, post)

    assert transform is not None
    assert inliers >= 12
    # Content shifted -pan_px, so the transform's x-translation is ~ -pan_px
    assert transform[0, 2] == pytest.approx(-pan_px, abs=2.0)
    assert abs(transform[1, 2]) < 2.0


def test_pan_following_offense_flips_raw_drive_dir():
    """The motivating failure: camera pans with the drive, offense looks
    almost stationary in-frame and the defense appears to move backward,
    so the RAW median flips the drive direction. Compensation fixes it."""
    pre = make_background()
    pan_px = 30.0          # camera pans +x, tracking the play
    field_px = 35.0        # offense actually advances +x on the field
    post = shift_frame(pre, pan_px)

    snap = formation()
    post_positions = {}
    for tid, (x, y) in snap.items():
        field_dx_px = field_px if tid <= 7 else 0.0   # only offense moves
        post_positions[tid] = ((x * W + field_dx_px - pan_px) / W, y)

    result = run_motion_diagnostic(pre, post, snap, post_positions, axis='x')

    assert result.estimation_ok
    assert result.drive_dir_raw == -1            # frame coords lie
    assert result.drive_dir_compensated == 1     # field coords tell the truth
    assert result.drive_dir_disagrees
    assert result.camera_dx == pytest.approx(-pan_px / W, abs=0.005)


def test_zoom_produces_no_false_drive():
    """Zoom-in spreads static players outward in frame; warp-based
    compensation must cancel it (translation subtraction would not)."""
    pre = make_background()
    scale = 1.08
    post = zoom_frame(pre, scale)

    snap = formation()
    post_positions = {}
    for tid, (x, y) in snap.items():
        # Static on the field: frame position follows the zoom exactly
        px = (x * W - W / 2) * scale + W / 2
        py = (y * H - H / 2) * scale + H / 2
        post_positions[tid] = (px / W, py / H)

    result = run_motion_diagnostic(pre, post, snap, post_positions, axis='x')

    assert result.estimation_ok
    assert result.camera_scale == pytest.approx(scale, abs=0.02)
    assert result.compensated_median_dx == pytest.approx(0.0, abs=0.003)
    assert result.drive_dir_compensated is None  # nobody actually moved


def test_featureless_frame_fails_safely():
    flat = np.full((H, W), 128, dtype=np.uint8)
    snap = formation()
    post_positions = {tid: (x + 0.02, y) for tid, (x, y) in snap.items()}

    result = run_motion_diagnostic(flat, flat, snap, post_positions, axis='x')

    assert not result.estimation_ok
    assert result.confidence == 0.0
    # Falls back to raw values rather than inventing a compensation
    assert result.compensated_median_dx == result.raw_median_dx
    assert result.drive_dir_compensated == result.drive_dir_raw


def test_player_mask_does_not_break_estimation():
    pre = make_background()
    post = shift_frame(pre, 15.0)
    boxes = [(x, y, 0.04, 0.10) for (x, y) in formation().values()]

    transform, inliers, _ = estimate_camera_transform(pre, post, player_boxes=boxes)

    assert transform is not None
    assert inliers >= 12
    assert transform[0, 2] == pytest.approx(-15.0, abs=2.0)


def test_overlay_writes_annotated_frame(tmp_path):
    frame = cv2.cvtColor(make_background(), cv2.COLOR_GRAY2BGR)
    state = PlayState(los_x=0.40, possession_team_id=None, drive_dir=1,
                      los_confidence=0.8, drive_dir_confidence=0.8, axis='x')
    out = str(tmp_path / "overlay.jpg")

    path = draw_play_overlay(frame, formation(), {}, state, out,
                             motion_summary={'drive_dir_raw': -1,
                                             'drive_dir_compensated': 1,
                                             'drive_dir_disagrees': True,
                                             'confidence': 0.9,
                                             'camera_scale': 1.0},
                             title="synthetic.mp4")

    assert path == out
    assert (tmp_path / "overlay.jpg").stat().st_size > 10_000


def test_derive_roles_geometry_mode():
    state = PlayState(los_x=0.40, possession_team_id=None, drive_dir=1,
                      los_confidence=0.8, drive_dir_confidence=0.8, axis='x')
    roles = derive_roles(formation(), {}, state)

    assert all(roles[tid] == 'offense' for tid in range(1, 8))
    assert all(roles[tid] == 'defense' for tid in range(8, 15))
