"""PlayState axis-safety and validity-tier tests.

The synthetic play used throughout: offense behind the LOS at ~0.365 along
the depth axis, defense downfield, offense flowing +depth after the snap.
"""

import pytest

from play_state import PlayState, infer_play_state


def endzone_play():
    """Synthetic endzone play (depth along x). Returns (snap, post, team_ids)."""
    snap = {
        # Offense (TEAM_A): line at x=0.35, QB/RB in backfield
        1: (0.35, 0.42), 2: (0.35, 0.46), 3: (0.35, 0.50),
        4: (0.35, 0.54), 5: (0.35, 0.58),
        6: (0.32, 0.50), 7: (0.30, 0.50),
        # Defense (TEAM_B): 4 on the line, 2 LBs, 1 deep safety
        8: (0.38, 0.42), 9: (0.38, 0.47), 10: (0.38, 0.53), 11: (0.38, 0.58),
        12: (0.43, 0.45), 13: (0.43, 0.55),
        14: (0.55, 0.50),
    }
    # Offense flows +x, defense essentially static
    post = {tid: (x + 0.03, y) if tid <= 7 else (x - 0.001, y)
            for tid, (x, y) in snap.items()}
    team_ids = {tid: ('TEAM_A' if tid <= 7 else 'TEAM_B') for tid in snap}
    return snap, post, team_ids


def transpose(positions):
    """Swap x/y: turns an endzone-framed play into a sideline-framed one."""
    return {tid: (y, x) for tid, (x, y) in positions.items()}


def test_endzone_x_axis_play():
    snap, post, team_ids = endzone_play()
    state = infer_play_state(snap, post, team_ids, axis='x')

    assert state.axis == 'x'
    assert state.geometry_valid
    assert state.identity_valid
    assert state.drive_dir == 1
    assert 0.35 < state.los_x < 0.38
    assert state.offense_team_id == 'TEAM_A'
    assert state.defense_team_id == 'TEAM_B'
    assert state.failure_reasons == []


def test_sideline_y_axis_play():
    snap, post, team_ids = endzone_play()
    state = infer_play_state(transpose(snap), transpose(post), team_ids, axis='y')

    assert state.axis == 'y'
    assert state.geometry_valid
    assert state.identity_valid
    assert state.drive_dir == 1
    assert 0.35 < state.los_x < 0.38
    assert state.offense_team_id == 'TEAM_A'


def test_fallback_rejected_when_no_los_on_other_axis():
    """Lateral-only flow must not produce a mixed-axis 'valid' state.

    LOS is separable on x, but the only motion is along y where players are
    tightly packed (no LOS). The old code would keep the x-axis LOS and take
    the y-axis drive sign, silently corrupting every downstream depth.
    """
    snap = {}
    for i in range(6):
        snap[i] = (0.32, 0.495 + i * 0.002)          # cluster A on x
    for i in range(6, 12):
        snap[i] = (0.42, 0.495 + (i - 6) * 0.002)    # cluster B on x
    # Everyone drifts +y (sweep-like), nobody moves along x
    post = {tid: (x, y + 0.05) for tid, (x, y) in snap.items()}

    state = infer_play_state(snap, post, {}, axis='x')

    assert state.axis == 'x'          # no incoherent axis switch
    assert state.drive_dir is None    # lateral flow is not a drive direction
    assert not state.geometry_valid
    assert 'missing_drive_dir' in state.failure_reasons


def test_fallback_switches_axes_coherently():
    """If the camera guess is wrong, the fallback may switch axes - but only
    with LOS and drive_dir recomputed together on the new axis."""
    snap, post, team_ids = endzone_play()
    # Sideline-framed play, but caller wrongly claims endzone (axis='x')
    state = infer_play_state(transpose(snap), transpose(post), team_ids, axis='x')

    # Fallback fired: the whole state must live on y, including the LOS
    assert state.drive_dir == 1
    assert state.axis == 'y'
    assert 0.35 < state.los_x < 0.38
    assert state.geometry_valid


def test_geometry_valid_without_identity():
    """Unknown team IDs: structure is computable, team metrics are not."""
    snap, post, _ = endzone_play()
    state = infer_play_state(snap, post, {}, axis='x')

    assert state.geometry_valid
    assert not state.identity_valid
    assert not state.is_valid
    assert state.possession_team_id is None
    assert state.failure_reasons == ['missing_possession']


def test_depth_at_uses_declared_axis():
    endzone = PlayState(los_x=0.4, possession_team_id=None, drive_dir=1, axis='x')
    sideline = PlayState(los_x=0.4, possession_team_id=None, drive_dir=1, axis='y')

    assert endzone.depth_at(0.5, 0.9) == pytest.approx(0.1)
    assert sideline.depth_at(0.9, 0.5) == pytest.approx(0.1)


def test_invalid_geometry_refuses_depth():
    state = PlayState(los_x=None, possession_team_id=None, drive_dir=1)
    assert state.depth_at(0.5, 0.5) is None
