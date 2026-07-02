"""Defensive inference axis-consistency and geometry-role tests."""

import pytest

from play_state import PlayState
from defensive_inference import DefensiveInference, build_defensive_metrics


LOS = 0.365

SNAP_POSITIONS = {
    # Offense: line at 0.35 depth, QB/RB behind
    1: (0.35, 0.42), 2: (0.35, 0.46), 3: (0.35, 0.50),
    4: (0.35, 0.54), 5: (0.35, 0.58),
    6: (0.32, 0.50), 7: (0.30, 0.50),
    # Defense: 4 down linemen, 2 LBs, 1 deep safety
    8: (0.38, 0.42), 9: (0.38, 0.47), 10: (0.38, 0.53), 11: (0.38, 0.58),
    12: (0.43, 0.45), 13: (0.43, 0.55),
    14: (0.55, 0.50),
}

TEAM_IDS = {tid: ('TEAM_A' if tid <= 7 else 'TEAM_B') for tid in SNAP_POSITIONS}


def make_state(axis='x', possession='TEAM_A'):
    state = PlayState(
        los_x=LOS,
        possession_team_id=possession,
        drive_dir=1,
        los_confidence=0.8,
        possession_confidence=0.8 if possession else 0.0,
        drive_dir_confidence=0.8,
        axis=axis,
    )
    if possession:
        state.offense_team_id = 'TEAM_A'
        state.defense_team_id = 'TEAM_B'
    return state


def transpose(positions):
    return {tid: (y, x) for tid, (x, y) in positions.items()}


def analyze(positions, team_ids, state):
    return DefensiveInference().analyze_positions_with_state(
        positions, team_ids, state, clip_file="synthetic.mp4", frame_num=100
    )


def test_identity_roles_endzone_structure():
    analysis = analyze(SNAP_POSITIONS, TEAM_IDS, make_state())

    assert analysis.roles_source == "identity"
    assert analysis.total_defensive_players == 7
    assert analysis.defenders_on_los == 4
    assert analysis.deep_defenders == 1
    assert analysis.safety_alignment.value == "1-high"
    assert analysis.confidence > 0


def test_sideline_matches_endzone_structure():
    """The same play filmed from the sideline must yield identical structure."""
    endzone = analyze(SNAP_POSITIONS, TEAM_IDS, make_state(axis='x'))
    sideline = analyze(transpose(SNAP_POSITIONS), TEAM_IDS, make_state(axis='y'))

    assert sideline.total_defensive_players == endzone.total_defensive_players
    assert sideline.defenders_on_los == endzone.defenders_on_los
    assert sideline.box_count == endzone.box_count
    assert sideline.deep_defenders == endzone.deep_defenders
    assert sideline.safety_alignment == endzone.safety_alignment
    assert sideline.front == endzone.front


def test_geometry_roles_when_identity_unknown():
    """No possession: roles come from side-of-LOS, structure still computed."""
    state = make_state(possession=None)
    identity = analyze(SNAP_POSITIONS, TEAM_IDS, make_state())
    geometry = analyze(SNAP_POSITIONS, {}, state)

    assert geometry.roles_source == "geometry"
    assert geometry.total_defensive_players == identity.total_defensive_players
    assert geometry.defenders_on_los == identity.defenders_on_los
    assert geometry.box_count == identity.box_count
    assert geometry.safety_alignment == identity.safety_alignment
    # Geometry-derived roles carry less confidence than identity-derived ones
    assert 0 < geometry.confidence < identity.confidence


def test_invalid_geometry_aborts():
    state = PlayState(los_x=None, possession_team_id=None, drive_dir=None)
    analysis = analyze(SNAP_POSITIONS, TEAM_IDS, state)

    assert analysis.confidence == 0.0
    assert analysis.total_defensive_players == 0


class MockTrack:
    """Minimal stand-in for PlayerTrack in build_defensive_metrics."""

    def __init__(self, positions_by_frame):
        self.positions_by_frame = dict(positions_by_frame)
        self.frames = []

    def calculate_first_step_time(self, snap_frame, fps):
        return 0.25

    def get_position_at_frame(self, frame_num):
        return self.positions_by_frame.get(frame_num)


class MockTracking:
    def __init__(self, players):
        self.players = players


def test_penetration_uses_playstate_geometry():
    """A nose tackle crossing the LOS into the backfield gets positive
    penetration measured along the PlayState axis, not frame-center y."""
    state = make_state()
    analysis = analyze(SNAP_POSITIONS, TEAM_IDS, state)

    fps = 30.0
    snap_frame = 100
    tracks = {}
    for tid, (x, y) in SNAP_POSITIONS.items():
        end = (x, y)
        if tid == 10:  # DL at (0.38, 0.53) penetrates to 0.30 depth
            end = (0.30, 0.53)
        tracks[tid] = MockTrack({snap_frame: (x, y), snap_frame + int(fps): end})

    metrics = build_defensive_metrics(
        analysis, MockTracking(tracks), snap_frame, fps, play_state=state
    )

    by_track = {m.track_id: m for m in metrics}
    assert set(by_track) == {8, 9, 10, 11, 12, 13, 14}

    penetrator = by_track[10]
    assert penetrator.inferred_position in ("DL", "EDGE")
    assert penetrator.penetration_depth == pytest.approx(LOS - 0.30, abs=1e-6)

    # Non-penetrating linemen ended the play at the LOS, not in the backfield
    assert by_track[8].penetration_depth == pytest.approx(0.0, abs=1e-6)

    # Depth is LOS-relative, not the old |y - 0.5| frame-center guess
    safety = by_track[14]
    assert safety.depth_from_los == pytest.approx(0.55 - LOS, abs=1e-6)
    assert not safety.in_box
