#!/usr/bin/env python3
"""
Third & 20 - PlayState Keystone Module v1.0

This module implements the canonical state inference that MUST run before
any offense/defense labeling or metric computation.

Architecture (correct ordering):
    Phase 0: Tracks in field (x, y) coordinates (from homography or normalized)
    Phase 1: Infer PlayState (los_x, possession_team_id, drive_dir) <- THIS MODULE
    Phase 2: Derive offense/defense from PlayState (deterministic)
    Phase 3: Compute metrics relative to PlayState
    Phase 4: Render perspective (UI layer only)

Key invariants:
    - LOS is computed from geometry, NOT from offense/defense labels
    - drive_dir is inferred from post-snap bulk flow, NOT from camera orientation
    - possession_team_id is inferred from pre-snap clustering + post-snap flow
    - No circular dependencies

Definition-of-done tests:
    1. Horizontal flip invariance: mirror video -> same PlayState
    2. Quarter switch invariance: drive_dir changes, depth_along_drive stays consistent
    3. No circular deps: LOS computed without offense/defense labels
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class PlayState:
    """
    The keystone object for neutral state inference.

    All downstream computation (offense/defense labels, metrics, etc.)
    MUST consume this object. Nothing runs without it.

    UNITS: All coordinates are NORMALIZED 0-1 (not yards).
    Convert at the homography boundary if needed.

    Coordinates:
        - x: field length axis (normalized 0-1)
        - y: field width axis (normalized 0-1)
        - los_x: line of scrimmage position on x-axis (normalized)
        - drive_dir: +1 if offense moving toward +x, -1 if toward -x, None if unknown
    """
    los_x: Optional[float]  # None if LOS could not be determined
    possession_team_id: Optional[str]  # Team IDENTITY (BRUSH, SHAKER), not role
    drive_dir: Optional[int]  # +1, -1, or None (unknown) — NEVER 0

    # Confidence scores (0-1)
    los_confidence: float = 0.0
    possession_confidence: float = 0.0
    drive_dir_confidence: float = 0.0

    # Derived (set after inference, only if possession known)
    offense_team_id: Optional[str] = None
    defense_team_id: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if PlayState has minimum required data for decision-grade analysis"""
        return (
            self.los_x is not None and
            self.drive_dir is not None and
            self.possession_team_id is not None
        )

    @property
    def overall_confidence(self) -> float:
        """Combined confidence score"""
        if not self.is_valid:
            return 0.0
        return (self.los_confidence + self.drive_dir_confidence + self.possession_confidence) / 3

    def depth_along_drive(self, x: float) -> Optional[float]:
        """
        Compute signed depth relative to LOS, normalized by drive direction.

        Returns:
            Positive = downfield (toward opponent end zone)
            Negative = behind LOS (toward own end zone)
            None = cannot compute (drive_dir or los_x unknown)
        """
        if self.drive_dir is None or self.los_x is None:
            return None  # Refuse to compute garbage
        return (x - self.los_x) * self.drive_dir


@dataclass
class TrackSnapshot:
    """Position and velocity of one track at a specific time"""
    track_id: int
    x: float
    y: float
    team_id: Optional[str] = None  # From jersey classifier, NOT "offense"/"defense"
    vx: float = 0.0  # Velocity in x
    vy: float = 0.0  # Velocity in y


def compute_los_from_geometry(
    positions: Dict[int, Tuple[float, float]],
    axis: str = 'x',
    min_per_side: int = 5
) -> Tuple[Optional[float], float]:
    """
    Compute line of scrimmage from player positions using two-line clustering.

    This is the ONLY correct way to compute LOS:
    - No offense/defense labels required
    - Uses the gap between two dense clusters (lines facing each other)

    Args:
        positions: Dict of track_id -> (x, y) positions at snap
        axis: 'x' for endzone camera, 'y' for sideline camera
        min_per_side: Minimum players required on each side of LOS (default 5)

    Returns:
        (los_value, confidence) where los_value is None if LOS cannot be determined
    """
    if not positions:
        return (None, 0.0)

    # Extract positions along the depth axis
    if axis == 'x':
        values = [pos[0] for pos in positions.values()]
    else:
        values = [pos[1] for pos in positions.values()]

    # Need at least min_per_side * 2 players to have a valid LOS
    if len(values) < min_per_side * 2:
        return (None, 0.0)  # Not enough players, refuse to guess

    values = sorted(values)

    # Check for single-cluster failure (all players on one side - sideline camera issue)
    spread = values[-1] - values[0]
    if spread < 0.1:  # All players within 10% of field = no clear LOS
        return (None, 0.0)

    # Find the densest gap between two clusters
    # The LOS is where two lines of players face each other
    best_los = None
    best_score = 0.0
    best_confidence = 0.0

    for i in range(len(values) - 1):
        gap_start = values[i]
        gap_end = values[i + 1]
        gap_size = gap_end - gap_start

        if gap_size < 0.015:  # Too small to be the LOS gap (~1.8 yards)
            continue

        # Count players on each side of this gap
        left_count = sum(1 for v in values if v <= gap_start)
        right_count = sum(1 for v in values if v >= gap_end)

        # HARD REQUIREMENT: minimum players per side
        if left_count < min_per_side or right_count < min_per_side:
            continue

        # Good LOS gap has roughly equal players on each side
        balance = min(left_count, right_count) / max(left_count, right_count)

        # Score: balance * gap_size * min_count (want balanced split with clear gap)
        score = balance * gap_size * min(left_count, right_count)

        if score > best_score:
            best_score = score
            best_los = (gap_start + gap_end) / 2
            # Confidence: based on balance and gap clarity
            best_confidence = min(1.0, balance * 0.5 + (gap_size / 0.1) * 0.5)

    if best_los is None:
        # No valid LOS found - don't pretend
        return (None, 0.0)

    return (best_los, best_confidence)


def infer_drive_direction(
    tracks_pre: Dict[int, Tuple[float, float]],
    tracks_post: Dict[int, Tuple[float, float]],
    axis: str = 'x',
    min_tracks: int = 5,
    threshold: float = 0.008  # ~1 yard in normalized coords
) -> Tuple[Optional[int], float]:
    """
    Infer drive direction from bulk player flow immediately after snap.

    Uses median displacement to be robust to outliers (WR running backward,
    DB bailing deep, etc.)

    Args:
        tracks_pre: Positions at snap (or snap - epsilon)
        tracks_post: Positions at snap + 0.5-1.0 seconds
        axis: 'x' for endzone camera, 'y' for sideline camera
        min_tracks: Minimum tracks required to infer direction (default 5)
        threshold: Minimum median displacement to declare direction (default 0.008)

    Returns:
        (drive_dir, confidence) where drive_dir is +1, -1, or None (unknown)
        NEVER returns 0 - use None for unknown
    """
    if not tracks_pre or not tracks_post:
        return (None, 0.0)

    # Compute displacement for each track that exists in both snapshots
    displacements = []

    for track_id, (x_pre, y_pre) in tracks_pre.items():
        if track_id not in tracks_post:
            continue

        x_post, y_post = tracks_post[track_id]

        if axis == 'x':
            dx = x_post - x_pre
        else:
            dx = y_post - y_pre

        displacements.append(dx)

    if len(displacements) < min_tracks:
        return (None, 0.0)  # Not enough data, refuse to guess

    # Use median for robustness
    median_dx = np.median(displacements)
    mad = np.median(np.abs(np.array(displacements) - median_dx))  # Median absolute deviation


    # Threshold: need clear directional movement
    if abs(median_dx) < threshold:
        return (None, 0.0)  # Movement too small to determine direction

    drive_dir = 1 if median_dx > 0 else -1

    # Confidence: how consistent is the flow?
    # High confidence = large median displacement relative to MAD
    if mad < 1e-6:
        confidence = 1.0
    else:
        # Scale confidence by consistency (low MAD = consistent flow)
        consistency = abs(median_dx) / (mad + 1e-6)
        # Also factor in magnitude (larger movement = more confident)
        magnitude_factor = min(1.0, abs(median_dx) / 0.05)  # Max out at 5% field movement
        confidence = min(1.0, consistency * 0.4 + magnitude_factor * 0.6)

    return (drive_dir, confidence)


def infer_possession(
    positions_at_snap: Dict[int, Tuple[float, float]],
    team_ids: Dict[int, str],
    los_x: Optional[float],
    drive_dir: Optional[int],
    tracks_post: Dict[int, Tuple[float, float]] = None,
    axis: str = 'x',
    min_known_total: int = 8,
    min_known_per_team: int = 2  # Lowered from 3 for HS film with challenging jersey detection
) -> Tuple[Optional[str], float]:
    """
    Infer which team has possession (is on offense).

    Uses two signals:
    1. Pre-snap: which team has more players behind LOS?
    2. Post-snap: which team is moving in drive_dir? (only if drive_dir known)

    Args:
        positions_at_snap: Dict of track_id -> (x, y)
        team_ids: Dict of track_id -> team_id (from jersey classifier)
        los_x: Line of scrimmage (from compute_los_from_geometry), or None
        drive_dir: Drive direction (+1, -1, or None)
        tracks_post: Optional post-snap positions for flow analysis
        axis: 'x' for endzone, 'y' for sideline
        min_known_total: Minimum total tracks with known team_id (default 8)
        min_known_per_team: Minimum tracks per team (default 3)

    Returns:
        (possession_team_id, confidence) - returns (None, 0.0) if cannot determine
    """
    if not positions_at_snap or not team_ids:
        return (None, 0.0)

    # Cannot infer possession without LOS
    if los_x is None:
        return (None, 0.0)

    # Count known vs unknown identities
    known_team_ids = {tid: team for tid, team in team_ids.items()
                      if team not in ('unknown', None, '')}
    unknown_count = len(team_ids) - len(known_team_ids)

    # Check minimum known identity threshold
    if len(known_team_ids) < min_known_total:
        return (None, 0.0)  # Too many unknowns, refuse to guess

    # Get unique teams (excluding unknown/None)
    teams = set(known_team_ids.values())

    if len(teams) < 2:
        return (None, 0.0)  # Need at least 2 distinct teams

    teams = list(teams)

    # Count tracks per team
    team_counts = {team: sum(1 for t in known_team_ids.values() if t == team)
                   for team in teams}

    # Check minimum per team
    for team, count in team_counts.items():
        if count < min_known_per_team:
            return (None, 0.0)  # Not enough known players for one team

    # Score each team on "offense likelihood"
    scores = {team: 0.0 for team in teams}

    # Signal 1: Pre-snap side of LOS
    # If drive_dir is known: use signed depth
    # If drive_dir is None: use unsigned distance (less accurate but safe)
    for track_id, (x, y) in positions_at_snap.items():
        team = known_team_ids.get(track_id)
        if team not in scores:
            continue

        pos_along_axis = x if axis == 'x' else y

        if drive_dir is not None:
            # Signed depth: negative = behind LOS (offense territory)
            depth = (pos_along_axis - los_x) * drive_dir

            if depth < -0.02:  # Behind LOS
                scores[team] += 1.5
            elif abs(depth) < 0.02:  # On LOS (deadband)
                scores[team] += 0.5
            # In front of LOS: no points (defense)
        else:
            # Unsigned fallback: just use distance from LOS
            # Assume players farther from LOS on one side are backfield (offense)
            distance = pos_along_axis - los_x
            # This is less reliable, so lower weights
            if distance < -0.03:  # One side of LOS
                scores[team] += 0.8
            elif distance > 0.03:  # Other side of LOS
                scores[team] += 0.8  # Equal weight both sides without direction

    # Signal 2: Post-snap flow direction (ONLY if drive_dir is known)
    if tracks_post and drive_dir is not None:
        for track_id, (x_pre, y_pre) in positions_at_snap.items():
            if track_id not in tracks_post:
                continue
            team = known_team_ids.get(track_id)
            if team not in scores:
                continue

            x_post, y_post = tracks_post[track_id]

            if axis == 'x':
                dx = x_post - x_pre
            else:
                dx = y_post - y_pre

            # Moving in drive direction = more likely offense
            if dx * drive_dir > 0.01:
                scores[team] += 1.0

    # Pick team with highest score
    if not scores or all(s == 0 for s in scores.values()):
        return (None, 0.0)

    best_team = max(scores, key=scores.get)
    best_score = scores[best_team]

    # Confidence based on margin between top two teams
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
        margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        confidence = min(1.0, margin * 0.7 + 0.3)
    else:
        confidence = 0.3

    # Penalize confidence if many unknowns
    unknown_ratio = unknown_count / len(team_ids) if team_ids else 1.0
    if unknown_ratio > 0.3:
        confidence *= (1.0 - unknown_ratio * 0.5)

    # Penalize confidence if drive_dir was unknown (used fallback logic)
    if drive_dir is None:
        confidence *= 0.6

    return (best_team, confidence)


def infer_play_state(
    positions_at_snap: Dict[int, Tuple[float, float]],
    positions_post_snap: Dict[int, Tuple[float, float]],
    team_ids: Dict[int, str],
    axis: str = 'x'
) -> PlayState:
    """
    Main entry point: infer complete PlayState from track data.

    This is the keystone function. Call this BEFORE any offense/defense
    labeling or metric computation.

    Args:
        positions_at_snap: Dict of track_id -> (x, y) at snap frame
        positions_post_snap: Dict of track_id -> (x, y) at snap + ~0.5-1.0s
        team_ids: Dict of track_id -> team_id (from jersey color classifier)
        axis: 'x' for endzone camera, 'y' for sideline camera

    Returns:
        PlayState with all fields populated.
        Check state.is_valid before using for decision-grade analysis.
        If is_valid is False, do NOT derive offense/defense roles.
    """
    # Step 1: Compute LOS from geometry (no labels needed)
    # Note: min_per_side lowered to 3 for HS film where tracker fragmentation
    # often causes fewer than 11 players to be detected at snap
    los_x, los_conf = compute_los_from_geometry(positions_at_snap, axis=axis, min_per_side=3)

    # Step 2: Infer drive direction from bulk flow
    # Try primary axis first, then fallback to other axis if no direction detected
    # This handles cases where play develops unexpectedly (e.g., sweep on sideline film)
    drive_dir, drive_conf = infer_drive_direction(
        positions_at_snap, positions_post_snap, axis=axis, min_tracks=2, threshold=0.002
    )

    # If primary axis didn't detect direction, try the other axis
    if drive_dir is None:
        other_axis = 'y' if axis == 'x' else 'x'
        drive_dir_alt, drive_conf_alt = infer_drive_direction(
            positions_at_snap, positions_post_snap, axis=other_axis, min_tracks=2, threshold=0.002
        )
        if drive_dir_alt is not None:
            drive_dir = drive_dir_alt
            drive_conf = drive_conf_alt * 0.8  # Slightly lower confidence for fallback axis
            print(f"  drive_dir fallback: used {other_axis}-axis")

    # Step 3: Infer possession team (may return None if insufficient data)
    possession_team, poss_conf = infer_possession(
        positions_at_snap, team_ids, los_x, drive_dir,
        tracks_post=positions_post_snap, axis=axis
    )

    # Build PlayState
    state = PlayState(
        los_x=los_x,
        possession_team_id=possession_team,
        drive_dir=drive_dir,
        los_confidence=los_conf,
        possession_confidence=poss_conf,
        drive_dir_confidence=drive_conf
    )

    # ONLY derive offense/defense team IDs if we have valid possession
    # Do NOT set these if state is invalid - that propagates bad data
    if state.is_valid and possession_team:
        state.offense_team_id = possession_team
        # Find the other team (excluding unknown/None/empty)
        other_teams = set(team_ids.values()) - {possession_team, 'unknown', None, ''}
        if len(other_teams) == 1:
            state.defense_team_id = list(other_teams)[0]
        elif len(other_teams) > 1:
            # Multiple teams? Pick the one with most players (shouldn't happen normally)
            team_counts = {t: sum(1 for v in team_ids.values() if v == t)
                          for t in other_teams}
            state.defense_team_id = max(team_counts, key=team_counts.get)
        # If no other teams found, defense_team_id stays None

    return state


def assign_roles_from_play_state(
    track_ids: List[int],
    team_ids: Dict[int, str],
    play_state: PlayState
) -> Dict[int, str]:
    """
    Assign "offense" or "defense" role to each track based on PlayState.

    This is Phase 2: deterministic role assignment.
    No heuristics. No camera-relative logic.

    Args:
        track_ids: List of track IDs to assign
        team_ids: Dict of track_id -> team_id (from jersey classifier)
        play_state: The inferred PlayState

    Returns:
        Dict of track_id -> "offense" | "defense" | "unknown"
    """
    roles = {}

    for track_id in track_ids:
        team = team_ids.get(track_id, 'unknown')

        if team == play_state.offense_team_id:
            roles[track_id] = "offense"
        elif team == play_state.defense_team_id:
            roles[track_id] = "defense"
        else:
            roles[track_id] = "unknown"

    return roles


# =============================================================================
# TESTS (run with: python play_state.py)
# =============================================================================

def _test_horizontal_flip_invariance():
    """Test that mirroring positions produces same PlayState"""
    # Original positions (offense on left, defense on right)
    positions_snap = {
        1: (0.3, 0.5), 2: (0.32, 0.4), 3: (0.32, 0.6),  # Offense behind LOS
        4: (0.35, 0.5), 5: (0.35, 0.45), 6: (0.35, 0.55),  # O-line on LOS
        7: (0.38, 0.5), 8: (0.38, 0.45), 9: (0.38, 0.55),  # D-line on LOS
        10: (0.45, 0.5), 11: (0.5, 0.4), 12: (0.5, 0.6),  # Defense deep
    }

    positions_post = {
        1: (0.35, 0.5), 2: (0.37, 0.4), 3: (0.37, 0.6),  # Offense moved forward
        4: (0.38, 0.5), 5: (0.38, 0.45), 6: (0.38, 0.55),
        7: (0.40, 0.5), 8: (0.40, 0.45), 9: (0.40, 0.55),
        10: (0.47, 0.5), 11: (0.52, 0.4), 12: (0.52, 0.6),
    }

    team_ids = {
        1: 'TEAM_A', 2: 'TEAM_A', 3: 'TEAM_A',
        4: 'TEAM_A', 5: 'TEAM_A', 6: 'TEAM_A',
        7: 'TEAM_B', 8: 'TEAM_B', 9: 'TEAM_B',
        10: 'TEAM_B', 11: 'TEAM_B', 12: 'TEAM_B',
    }

    # Original
    state1 = infer_play_state(positions_snap, positions_post, team_ids)

    # Mirrored (flip x around 0.5)
    positions_snap_mirror = {k: (1.0 - x, y) for k, (x, y) in positions_snap.items()}
    positions_post_mirror = {k: (1.0 - x, y) for k, (x, y) in positions_post.items()}

    state2 = infer_play_state(positions_snap_mirror, positions_post_mirror, team_ids)

    # Check invariants
    assert state1.possession_team_id == state2.possession_team_id, \
        f"Possession changed: {state1.possession_team_id} vs {state2.possession_team_id}"
    assert state1.offense_team_id == state2.offense_team_id, \
        f"Offense changed: {state1.offense_team_id} vs {state2.offense_team_id}"

    # drive_dir should flip (because we flipped the field)
    # but depth_along_drive for any player should be consistent

    print("PASS: Horizontal flip invariance")


def _test_no_circular_dependency():
    """Test that LOS is computed without offense/defense labels"""
    # Need 10+ players (5 per side minimum)
    positions = {
        # Offense side (5 players)
        1: (0.30, 0.50), 2: (0.32, 0.45), 3: (0.32, 0.55),
        4: (0.34, 0.48), 5: (0.34, 0.52),
        # Defense side (5 players)
        6: (0.40, 0.50), 7: (0.42, 0.45), 8: (0.42, 0.55),
        9: (0.44, 0.48), 10: (0.44, 0.52),
    }

    # compute_los should work with NO team labels at all
    los, conf = compute_los_from_geometry(positions)

    assert los is not None, "LOS should be computed"
    assert 0.34 < los < 0.40, f"LOS {los} should be in the gap between clusters"

    print(f"PASS: No circular dependency (LOS={los:.3f}, conf={conf:.3f})")


def _test_drive_dir_returns_none():
    """Test that drive_dir returns None (not 0) when uncertain"""
    # Stationary players - no clear movement
    positions_snap = {1: (0.3, 0.5), 2: (0.4, 0.5), 3: (0.5, 0.5),
                      4: (0.6, 0.5), 5: (0.7, 0.5)}
    positions_post = {1: (0.3, 0.5), 2: (0.4, 0.5), 3: (0.5, 0.5),
                      4: (0.6, 0.5), 5: (0.7, 0.5)}  # No movement

    drive_dir, conf = infer_drive_direction(positions_snap, positions_post)

    assert drive_dir is None, f"drive_dir should be None, got {drive_dir}"
    assert conf == 0.0, f"confidence should be 0.0, got {conf}"

    print("PASS: drive_dir returns None when uncertain")


def _test_possession_requires_minimum_known():
    """Test that possession returns None if too few known identities"""
    positions = {1: (0.3, 0.5), 2: (0.4, 0.5), 3: (0.5, 0.5)}
    # Only 3 known - below threshold of 8
    team_ids = {1: 'TEAM_A', 2: 'TEAM_A', 3: 'TEAM_B'}

    possession, conf = infer_possession(positions, team_ids, los_x=0.4, drive_dir=1)

    assert possession is None, f"possession should be None with few knowns, got {possession}"

    print("PASS: possession returns None when insufficient known identities")


if __name__ == "__main__":
    print("PlayState Module Tests")
    print("=" * 40)

    _test_no_circular_dependency()
    _test_drive_dir_returns_none()
    _test_possession_requires_minimum_known()
    _test_horizontal_flip_invariance()

    print("\nAll tests passed.")
