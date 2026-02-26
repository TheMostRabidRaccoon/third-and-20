#!/usr/bin/env python3
"""
Third & 20 - Defensive Front Inference Module v1.0
Infers defensive alignments from pre-snap CV player detection

This module analyzes pre-snap player positions to infer:
1. Box count (6-man vs 7-man vs 8-man box)
2. Safety alignment (1-high vs 2-high vs Cover-0)
3. Defensive front (4-down vs 3-down vs 5-down)
4. Edge alignment and coverage shell

Since teams haven't uploaded defensive data to Hudl, this uses CV inference
from player detection to estimate defensive schemes.

Usage:
    from defensive_inference import DefensiveInference, DefensiveAnalysis

    inference = DefensiveInference()
    analysis = inference.analyze_snap_frame(frame, player_positions, ball_position)

Dependencies:
    - numpy
    - opencv-python (cv2)
    - Player detection from player_tracker.py
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum
import json

# Import PlayState for type hints
try:
    from play_state import PlayState
except ImportError:
    PlayState = None  # Type hint only, not required at runtime


# =============================================================================
# CONSTANTS - Football field geometry (normalized 0-1 coordinates)
# =============================================================================

# Box boundaries (relative to ball/line of scrimmage)
BOX_DEPTH_SHALLOW = 0.08   # Depth from LOS for box count (8% of frame = ~5 yards)
BOX_DEPTH_DEEP = 0.15      # Extended box for 2nd level (15% = ~8-9 yards)
BOX_WIDTH = 0.25           # Width from center for box (tackle box)

# Safety depth thresholds
SAFETY_SHALLOW = 0.12      # Safeties within 12% of LOS = in the box
SAFETY_MEDIUM = 0.20       # 12-20% = robber/overhang
SAFETY_DEEP = 0.30         # 20-30% = deep half
SAFETY_SINGLE_HIGH = 0.35  # >30% = single high centerfield

# Alignment thresholds
CENTER_THRESHOLD = 0.05    # How close to center to be considered "middle"
WIDE_THRESHOLD = 0.35      # How far from center to be "wide" (outside numbers)


class DefensiveFront(Enum):
    """Defensive front classification"""
    FOUR_DOWN = "4-down"       # 4-3, 4-2-5
    THREE_DOWN = "3-down"      # 3-4, 3-3-5
    FIVE_DOWN = "5-down"       # Goal line, 5-2
    EVEN = "even"              # 4-man with even alignment
    ODD = "odd"                # 3-man with nose
    UNKNOWN = "unknown"


class SafetyAlignment(Enum):
    """Safety shell classification"""
    SINGLE_HIGH = "1-high"     # Cover 1, Cover 3
    TWO_HIGH = "2-high"        # Cover 2, Cover 4, Quarters
    COVER_0 = "0-high"         # No deep safety, all man/blitz
    ROBBER = "robber"          # One safety shallow, one deep
    UNKNOWN = "unknown"


class CoverageShell(Enum):
    """Coverage shell classification"""
    COVER_0 = "Cover 0"        # No deep help, pure man
    COVER_1 = "Cover 1"        # 1 deep, man under
    COVER_2 = "Cover 2"        # 2 deep halves
    COVER_3 = "Cover 3"        # 3 deep, 4 under
    COVER_4 = "Cover 4"        # Quarters
    COVER_6 = "Cover 6"        # Quarter-quarter-half
    TWO_MAN = "2-Man"          # 2 deep, man under
    UNKNOWN = "Unknown"


@dataclass
class PlayerPosition:
    """Single player position at snap"""
    track_id: int
    x: float  # Normalized 0-1 (left-right)
    y: float  # Normalized 0-1 (top-bottom, top = deep)
    team: str  # "offense", "defense", "unknown"
    bbox_width: float = 0.0
    bbox_height: float = 0.0

    @property
    def is_in_box(self) -> bool:
        """Check if player is in the tackle box area"""
        # Box is defined relative to center of field
        return abs(self.x - 0.5) < BOX_WIDTH

    @property
    def depth_from_los(self) -> float:
        """Estimated depth from line of scrimmage (assuming LOS at y=0.5)"""
        return abs(self.y - 0.5)

    @property
    def is_on_line(self) -> bool:
        """Check if player is on the line of scrimmage"""
        return self.depth_from_los < 0.03  # Within 3% of LOS


@dataclass
class DefensiveAnalysis:
    """Complete defensive analysis for one snap"""
    clip_file: str
    frame_num: int

    # Player counts
    total_defensive_players: int = 0
    box_count: int = 0
    defenders_on_los: int = 0
    second_level_count: int = 0  # LBs
    deep_defenders: int = 0

    # Classifications
    front: DefensiveFront = DefensiveFront.UNKNOWN
    front_description: str = ""
    safety_alignment: SafetyAlignment = SafetyAlignment.UNKNOWN
    coverage_shell: CoverageShell = CoverageShell.UNKNOWN

    # Specific alignments
    nose_tackle: bool = False          # NT over center
    two_gap_front: bool = False        # Likely 2-gap technique
    edge_count: int = 0                # Players on edge (DE/OLB)

    # Blitz indicators
    potential_blitz: bool = False
    blitz_indicators: List[str] = field(default_factory=list)

    # Raw data
    player_positions: List[PlayerPosition] = field(default_factory=list)

    # Confidence
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Export to dictionary for CSV/JSON"""
        return {
            'clip_file': self.clip_file,
            'frame_num': self.frame_num,
            'total_defensive_players': self.total_defensive_players,
            'box_count': self.box_count,
            'defenders_on_los': self.defenders_on_los,
            'second_level_count': self.second_level_count,
            'deep_defenders': self.deep_defenders,
            'front': self.front.value,
            'front_description': self.front_description,
            'safety_alignment': self.safety_alignment.value,
            'coverage_shell': self.coverage_shell.value,
            'nose_tackle': self.nose_tackle,
            'edge_count': self.edge_count,
            'potential_blitz': self.potential_blitz,
            'blitz_indicators': ','.join(self.blitz_indicators),
            'confidence': f"{self.confidence:.2f}"
        }

    def summary(self) -> str:
        """Human-readable summary"""
        return (f"{self.front.value} front, {self.safety_alignment.value} safety, "
                f"{self.box_count}-man box ({self.coverage_shell.value})")


@dataclass
class DefensiveMetrics:
    """Full defensive metrics for one player on one play"""
    clip_file: str
    track_id: int
    jersey_number: Optional[int] = None
    player_name: str = ""
    team_name: str = ""

    # Position classification
    inferred_position: str = ""  # DL, EDGE, LB, CB, S
    alignment: str = ""          # Technique (0, 1, 3, 5, 7, 9) or coverage
    depth_from_los: float = 0.0

    # Pre-snap context
    in_box: bool = False
    on_line: bool = False

    # Movement metrics (post-snap)
    first_step_sec: Optional[float] = None
    first_step_grade: str = ""
    penetration_depth: Optional[float] = None
    lateral_movement: Optional[float] = None

    # Composite grade
    defensive_grade: str = ""

    def to_dict(self) -> dict:
        return {
            'clip_file': self.clip_file,
            'track_id': self.track_id,
            'jersey_number': self.jersey_number if self.jersey_number else "",
            'player_name': self.player_name,
            'team_name': self.team_name,
            'inferred_position': self.inferred_position,
            'alignment': self.alignment,
            'depth_from_los': f"{self.depth_from_los:.3f}",
            'in_box': self.in_box,
            'on_line': self.on_line,
            'first_step_sec': f"{self.first_step_sec:.3f}" if self.first_step_sec else "",
            'first_step_grade': self.first_step_grade,
            'penetration_depth': f"{self.penetration_depth:.3f}" if self.penetration_depth else "",
            'lateral_movement': f"{self.lateral_movement:.3f}" if self.lateral_movement else "",
            'defensive_grade': self.defensive_grade
        }


# =============================================================================
# DEFENSIVE INFERENCE ENGINE
# =============================================================================

class DefensiveInference:
    """
    Infers defensive alignment from pre-snap player positions.

    Uses geometric analysis of player positions to classify:
    - Defensive front (3-down vs 4-down)
    - Box count (6/7/8 man box)
    - Safety alignment (1-high vs 2-high)
    - Coverage shell estimation

    IMPORTANT: Use analyze_positions_with_state() with a PlayState object.
    The old analyze_positions() method is deprecated and will raise an error.
    """

    def __init__(self, camera_angle: str = "endzone"):
        """
        Initialize inference engine.

        Args:
            camera_angle: "endzone" (behind offense) or "sideline" (perpendicular)

        NOTE: LOS and drive_dir now come from PlayState, not estimated here.
        """
        self.camera_angle = camera_angle
        # These are set per-analysis by analyze_positions_with_state()
        self.los_x: Optional[float] = None
        self.drive_dir: Optional[int] = None
        # Geometry axes for sideline/endzone
        # Endzone: depth=x (along field), width=y (across field)
        # Sideline: depth=y (along field), width=x (across field)
        self.depth_axis = 'x' if camera_angle == 'endzone' else 'y'
        self.width_axis = 'y' if camera_angle == 'endzone' else 'x'

    def analyze_positions_with_state(self,
                                      positions: Dict[int, Tuple[float, float]],
                                      team_ids: Dict[int, str],
                                      play_state: 'PlayState',
                                      clip_file: str = "",
                                      frame_num: int = 0) -> DefensiveAnalysis:
        """
        Analyze player positions using PlayState for LOS and role derivation.

        This is the ONLY correct entry point. It:
        - Uses LOS from PlayState (no circular estimation)
        - Derives offense/defense from PlayState.possession_team_id
        - Uses drive_dir for signed depth calculations

        Args:
            positions: Dict of track_id -> (x, y) normalized position
            team_ids: Dict of track_id -> team IDENTITY ("BRUSH", not "offense")
            play_state: PlayState with los_x, drive_dir, possession_team_id
            clip_file: Source clip filename
            frame_num: Frame number being analyzed

        Returns:
            DefensiveAnalysis with inferred scheme
        """
        # Store PlayState values for depth calculations
        self.los_x = play_state.los_x
        self.drive_dir = play_state.drive_dir

        analysis = DefensiveAnalysis(clip_file=clip_file, frame_num=frame_num)

        # If PlayState is invalid, abort early
        if not play_state.is_valid:
            analysis.confidence = 0.0
            return analysis

        # Derive offense/defense from PlayState (not from stored roles)
        defense_players = []
        offense_players = []

        for track_id, (x, y) in positions.items():
            team_id = team_ids.get(track_id, "unknown")

            # Derive role from PlayState
            if team_id == play_state.offense_team_id:
                role = "offense"
            elif team_id == play_state.defense_team_id:
                role = "defense"
            else:
                role = "unknown"

            player = PlayerPosition(track_id=track_id, x=x, y=y, team=role)
            analysis.player_positions.append(player)

            if role == "defense":
                defense_players.append(player)
            elif role == "offense":
                offense_players.append(player)

        if not defense_players:
            analysis.confidence = 0.0
            return analysis

        analysis.total_defensive_players = len(defense_players)

        # NOTE: We do NOT call _estimate_los() - we use play_state.los_x

        # Classify each defender using signed depth
        on_line = []
        in_box = []
        second_level = []
        deep = []

        for player in defense_players:
            depth = self._get_depth_signed(player)
            if depth is None:
                continue  # Skip if depth cannot be computed

            width_from_center = abs(player.x - 0.5)

            if depth < 0.03:  # On the line
                on_line.append(player)
                if width_from_center < BOX_WIDTH:
                    in_box.append(player)
            elif depth < BOX_DEPTH_SHALLOW:  # In the box (LB depth)
                if width_from_center < BOX_WIDTH:
                    in_box.append(player)
                second_level.append(player)
            elif depth < BOX_DEPTH_DEEP:  # Second level / shallow safety
                second_level.append(player)
                if width_from_center < BOX_WIDTH:
                    in_box.append(player)
            else:  # Deep
                deep.append(player)

        analysis.defenders_on_los = len(on_line)
        analysis.box_count = len(in_box)
        analysis.second_level_count = len(second_level)
        analysis.deep_defenders = len(deep)

        # Classify front
        analysis.front = self._classify_front(on_line, second_level)
        analysis.front_description = self._get_front_description(
            analysis.front, len(on_line), len(second_level)
        )

        # Check for nose tackle
        analysis.nose_tackle = self._has_nose_tackle(on_line)

        # Count edge players
        analysis.edge_count = self._count_edges(on_line, second_level)

        # Classify safety alignment
        analysis.safety_alignment = self._classify_safeties(deep, second_level)

        # Estimate coverage shell
        analysis.coverage_shell = self._estimate_coverage(
            analysis.safety_alignment,
            analysis.box_count,
            deep
        )

        # Check for blitz indicators
        analysis.potential_blitz, analysis.blitz_indicators = self._check_blitz_indicators(
            defense_players, analysis
        )

        # Calculate confidence
        analysis.confidence = self._calculate_confidence(analysis)

        return analysis

    def analyze_positions(self,
                          positions: Dict[int, Tuple[float, float]],
                          teams: Dict[int, str],
                          clip_file: str = "",
                          frame_num: int = 0) -> DefensiveAnalysis:
        """
        DEPRECATED: Use analyze_positions_with_state() instead.

        This method has a circular dependency: it uses offense/defense labels
        to estimate LOS, but those labels were computed using LOS.
        """
        raise RuntimeError(
            "analyze_positions() is deprecated and has circular dependencies. "
            "Use analyze_positions_with_state(positions, team_ids, play_state, ...) "
            "where team_ids contains team IDENTITIES (e.g., 'BRUSH', 'SHAKER'), "
            "not roles ('offense', 'defense')."
        )

    def _get_depth_signed(self, player: PlayerPosition) -> Optional[float]:
        """
        Get SIGNED depth from LOS along drive direction.

        Returns:
            Positive = downfield (toward opponent end zone)
            Negative = behind LOS (toward own end zone)
            None = cannot compute (drive_dir or los_x unknown)
        """
        if self.los_x is None or self.drive_dir is None:
            return None
        return (player.x - self.los_x) * self.drive_dir

    def _analyze_positions_legacy(self,
                          positions: Dict[int, Tuple[float, float]],
                          teams: Dict[int, str],
                          clip_file: str = "",
                          frame_num: int = 0) -> DefensiveAnalysis:
        """
        LEGACY METHOD - kept for reference only.
        DO NOT USE - has circular LOS estimation bug.
        """
        analysis = DefensiveAnalysis(clip_file=clip_file, frame_num=frame_num)

        # Convert to PlayerPosition objects
        defense_players = []
        offense_players = []

        for track_id, (x, y) in positions.items():
            team = teams.get(track_id, "unknown")
            player = PlayerPosition(track_id=track_id, x=x, y=y, team=team)
            analysis.player_positions.append(player)

            if team == "defense":
                defense_players.append(player)
            elif team == "offense":
                offense_players.append(player)

        if not defense_players:
            analysis.confidence = 0.0
            return analysis

        analysis.total_defensive_players = len(defense_players)

        # Find line of scrimmage from offensive line
        self._estimate_los(offense_players, defense_players)

        # Classify each defender
        on_line = []
        in_box = []
        second_level = []
        deep = []

        for player in defense_players:
            depth = self._get_depth(player)
            width_from_center = abs(self._get_width(player) - 0.5)

            if depth < 0.03:  # On the line
                on_line.append(player)
                if width_from_center < BOX_WIDTH:
                    in_box.append(player)
            elif depth < BOX_DEPTH_SHALLOW:  # In the box (LB depth)
                if width_from_center < BOX_WIDTH:
                    in_box.append(player)
                second_level.append(player)
            elif depth < BOX_DEPTH_DEEP:  # Second level / shallow safety
                second_level.append(player)
                if width_from_center < BOX_WIDTH:
                    in_box.append(player)
            else:  # Deep
                deep.append(player)

        analysis.defenders_on_los = len(on_line)
        analysis.box_count = len(in_box)
        analysis.second_level_count = len(second_level)
        analysis.deep_defenders = len(deep)

        # Classify front
        analysis.front = self._classify_front(on_line, second_level)
        analysis.front_description = self._get_front_description(
            analysis.front, len(on_line), len(second_level)
        )

        # Check for nose tackle
        analysis.nose_tackle = self._has_nose_tackle(on_line)

        # Count edge players
        analysis.edge_count = self._count_edges(on_line, second_level)

        # Classify safety alignment
        analysis.safety_alignment = self._classify_safeties(deep, second_level)

        # Estimate coverage shell
        analysis.coverage_shell = self._estimate_coverage(
            analysis.safety_alignment,
            analysis.box_count,
            deep
        )

        # Check for blitz indicators
        analysis.potential_blitz, analysis.blitz_indicators = self._check_blitz_indicators(
            defense_players, analysis
        )

        # Calculate confidence based on player count and consistency
        analysis.confidence = self._calculate_confidence(analysis)

        return analysis

    def _get_depth(self, player: PlayerPosition) -> float:
        """Get UNSIGNED player depth from LOS based on camera angle"""
        los = self.los_x if self.los_x is not None else 0.5
        if self.depth_axis == 'y':
            # Endzone view: higher y = deeper (defense side)
            return player.y - los if player.y > los else 0
        else:
            # Sideline view: x is depth
            return abs(player.x - los)

    def _get_width(self, player: PlayerPosition) -> float:
        """Get player width position based on camera angle"""
        if self.width_axis == 'x':
            return player.x
        else:
            return player.y

    def _estimate_los(self, offense: List[PlayerPosition], defense: List[PlayerPosition]):
        """Estimate line of scrimmage from player positions"""
        if not offense and not defense:
            return

        # Find cluster of players near LOS (offensive line + defensive line)
        all_players = offense + defense

        if self.depth_axis == 'y':
            # Find the y-coordinate where offense and defense meet
            offense_y = [p.y for p in offense] if offense else [0.5]
            defense_y = [p.y for p in defense] if defense else [0.5]

            # LOS is approximately where offense max meets defense min
            if offense and defense:
                self.los_estimate = (max(offense_y) + min(defense_y)) / 2
        else:
            offense_x = [p.x for p in offense] if offense else [0.5]
            defense_x = [p.x for p in defense] if defense else [0.5]

            if offense and defense:
                self.los_estimate = (max(offense_x) + min(defense_x)) / 2

    def _classify_front(self, on_line: List[PlayerPosition],
                        second_level: List[PlayerPosition]) -> DefensiveFront:
        """Classify defensive front based on players on LOS"""
        num_on_line = len(on_line)

        if num_on_line >= 5:
            return DefensiveFront.FIVE_DOWN
        elif num_on_line == 4:
            return DefensiveFront.FOUR_DOWN
        elif num_on_line == 3:
            return DefensiveFront.THREE_DOWN
        elif num_on_line < 3:
            # Likely a sub package or misdetection
            return DefensiveFront.THREE_DOWN  # Assume 3-down as default

        return DefensiveFront.UNKNOWN

    def _get_front_description(self, front: DefensiveFront,
                                on_line: int, second_level: int) -> str:
        """Get descriptive name for front"""
        total_box = on_line + second_level

        if front == DefensiveFront.FOUR_DOWN:
            if second_level >= 3:
                return "4-3"
            elif second_level == 2:
                return "4-2-5 (Nickel)"
            else:
                return "4-1-6 (Dime)"
        elif front == DefensiveFront.THREE_DOWN:
            if second_level >= 4:
                return "3-4"
            elif second_level == 3:
                return "3-3-5"
            else:
                return "3-2-6"
        elif front == DefensiveFront.FIVE_DOWN:
            return "5-2 / Goal Line"

        return f"{on_line}-{second_level}"

    def _has_nose_tackle(self, on_line: List[PlayerPosition]) -> bool:
        """Check if there's a nose tackle (0-technique over center)"""
        for player in on_line:
            width = self._get_width(player)
            if abs(width - 0.5) < CENTER_THRESHOLD:
                return True
        return False

    def _count_edges(self, on_line: List[PlayerPosition],
                     second_level: List[PlayerPosition]) -> int:
        """Count edge players (DEs or OLBs in 3-4)"""
        edges = 0

        for player in on_line:
            width = self._get_width(player)
            # Edge if outside the tackles (beyond ~0.35 from center)
            if abs(width - 0.5) > 0.15:
                edges += 1

        # Also check for stand-up edge rushers in second level
        for player in second_level:
            depth = self._get_depth(player)
            width = self._get_width(player)
            # On edge at LB depth
            if depth < 0.05 and abs(width - 0.5) > 0.18:
                edges += 1

        return edges

    def _classify_safeties(self, deep: List[PlayerPosition],
                           second_level: List[PlayerPosition]) -> SafetyAlignment:
        """Classify safety alignment from deep defenders"""
        if not deep:
            # No deep defenders = Cover 0 look
            return SafetyAlignment.COVER_0

        num_deep = len(deep)

        if num_deep == 1:
            # Single high safety
            return SafetyAlignment.SINGLE_HIGH
        elif num_deep >= 2:
            # Check if both are deep or one is a robber
            deep_depths = [self._get_depth(p) for p in deep]

            # If one is significantly shallower, it's a robber look
            if max(deep_depths) - min(deep_depths) > 0.08:
                return SafetyAlignment.ROBBER

            return SafetyAlignment.TWO_HIGH

        return SafetyAlignment.UNKNOWN

    def _estimate_coverage(self, safety: SafetyAlignment,
                           box_count: int,
                           deep: List[PlayerPosition]) -> CoverageShell:
        """Estimate coverage shell from safety alignment and box count"""
        if safety == SafetyAlignment.COVER_0:
            return CoverageShell.COVER_0

        elif safety == SafetyAlignment.SINGLE_HIGH:
            if box_count >= 8:
                return CoverageShell.COVER_1  # Likely Cover 1 with loaded box
            else:
                return CoverageShell.COVER_3  # Could be Cover 3

        elif safety == SafetyAlignment.TWO_HIGH:
            if box_count <= 6:
                return CoverageShell.COVER_2  # Light box = Cover 2
            else:
                return CoverageShell.COVER_4  # Loaded box = likely quarters

        elif safety == SafetyAlignment.ROBBER:
            return CoverageShell.COVER_3  # Robber often with Cover 3

        return CoverageShell.UNKNOWN

    def _check_blitz_indicators(self, defense: List[PlayerPosition],
                                 analysis: DefensiveAnalysis) -> Tuple[bool, List[str]]:
        """Check for pre-snap blitz indicators"""
        indicators = []

        # High box count often indicates blitz
        if analysis.box_count >= 8:
            indicators.append("8+ in box")

        # Cover 0 look
        if analysis.safety_alignment == SafetyAlignment.COVER_0:
            indicators.append("No deep safety (Cover 0)")

        # LB walked up to line
        for player in defense:
            depth = self._get_depth(player)
            if 0.02 < depth < 0.05:  # Between line and LB depth
                width = self._get_width(player)
                if abs(width - 0.5) < 0.2:  # Inside the box
                    indicators.append("LB showing blitz")
                    break

        # DB in box
        for player in defense:
            depth = self._get_depth(player)
            width = self._get_width(player)
            # Deep player (normally) now in box
            if abs(width - 0.5) > WIDE_THRESHOLD and depth < BOX_DEPTH_SHALLOW:
                indicators.append("DB walked into box")
                break

        return len(indicators) >= 2, indicators

    def _calculate_confidence(self, analysis: DefensiveAnalysis) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence

        # More players = more confident
        if analysis.total_defensive_players >= 10:
            confidence += 0.2
        elif analysis.total_defensive_players >= 8:
            confidence += 0.1

        # Reasonable box count = more confident
        if 5 <= analysis.box_count <= 9:
            confidence += 0.1

        # Reasonable LOS count = more confident
        if 3 <= analysis.defenders_on_los <= 5:
            confidence += 0.1

        # Has deep defenders = more confident (full picture)
        if analysis.deep_defenders >= 1:
            confidence += 0.1

        return min(confidence, 1.0)

    def classify_player_position(self, player: PlayerPosition,
                                  analysis: DefensiveAnalysis) -> Tuple[str, str]:
        """
        Classify individual defensive player's position and alignment.

        Returns:
            Tuple of (position, alignment)
            position: DL, EDGE, LB, CB, S
            alignment: technique number or coverage position
        """
        depth = self._get_depth(player)
        width = self._get_width(player)
        width_from_center = abs(width - 0.5)

        # Defensive Line (on LOS, inside)
        if depth < 0.03 and width_from_center < 0.18:
            # Determine technique
            if width_from_center < CENTER_THRESHOLD:
                return "DL", "0-tech (Nose)"
            elif width_from_center < 0.08:
                return "DL", "1-tech"
            elif width_from_center < 0.12:
                return "DL", "3-tech"
            else:
                return "DL", "5-tech"

        # Edge (on LOS or close, outside)
        if depth < 0.05 and width_from_center >= 0.15:
            if depth < 0.03:
                return "EDGE", "DE (down)"
            else:
                return "EDGE", "OLB (up)"

        # Linebacker (behind LOS, inside box)
        if 0.03 <= depth < BOX_DEPTH_DEEP and width_from_center < BOX_WIDTH:
            if width_from_center < CENTER_THRESHOLD:
                return "LB", "MIKE (middle)"
            elif width < 0.5:
                return "LB", "WILL (weak)"
            else:
                return "LB", "SAM (strong)"

        # Cornerback (wide, any depth)
        if width_from_center > WIDE_THRESHOLD:
            if depth < 0.08:
                return "CB", "Press"
            elif depth < 0.15:
                return "CB", "Off (5-7 yards)"
            else:
                return "CB", "Deep"

        # Safety (deep, inside)
        if depth >= SAFETY_MEDIUM:
            if width_from_center < CENTER_THRESHOLD:
                return "S", "Free Safety (centerfield)"
            elif width_from_center < 0.2:
                return "S", "Strong Safety"
            else:
                return "S", "Deep half"

        # Nickel/Slot corner (middle depth, slot area)
        if 0.05 <= depth < 0.15 and 0.2 < width_from_center < 0.35:
            return "SLOT", "Nickel/Slot"

        return "UNKNOWN", ""


# =============================================================================
# GRADING FUNCTIONS
# =============================================================================

def grade_defensive_first_step(first_step_sec: float, position: str) -> str:
    """Grade defensive player first step quickness by position"""
    if first_step_sec is None:
        return ""

    # DL and EDGE have faster thresholds than LBs/DBs
    if position in ["DL", "EDGE"]:
        if first_step_sec < 0.20:
            return "Elite"
        elif first_step_sec < 0.30:
            return "Above Average"
        elif first_step_sec < 0.40:
            return "Average"
        elif first_step_sec < 0.55:
            return "Below Average"
        else:
            return "Developmental"
    else:
        # LB, CB, S
        if first_step_sec < 0.25:
            return "Elite"
        elif first_step_sec < 0.35:
            return "Above Average"
        elif first_step_sec < 0.50:
            return "Average"
        elif first_step_sec < 0.65:
            return "Below Average"
        else:
            return "Developmental"


def grade_defensive_player(metrics: DefensiveMetrics) -> str:
    """Calculate overall defensive grade"""
    score = 50  # Base

    # First step component
    if metrics.first_step_grade == "Elite":
        score += 25
    elif metrics.first_step_grade == "Above Average":
        score += 15
    elif metrics.first_step_grade == "Average":
        score += 5
    elif metrics.first_step_grade == "Below Average":
        score -= 5
    elif metrics.first_step_grade == "Developmental":
        score -= 15

    # Penetration for DL/EDGE
    if metrics.inferred_position in ["DL", "EDGE"]:
        if metrics.penetration_depth:
            if metrics.penetration_depth > 0.15:
                score += 15
            elif metrics.penetration_depth > 0.08:
                score += 8

    # Lateral movement for DBs
    if metrics.inferred_position in ["CB", "S", "SLOT"]:
        if metrics.lateral_movement:
            if metrics.lateral_movement > 0.10:
                score += 10

    # Convert to grade
    if score >= 75:
        return "Elite"
    elif score >= 65:
        return "Above Average"
    elif score >= 50:
        return "Average"
    elif score >= 35:
        return "Below Average"
    else:
        return "Developmental"


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def analyze_defensive_from_tracking(tracking_data, snap_frame: int,
                                    clip_file: str = "") -> DefensiveAnalysis:
    """
    Convenience function to analyze defense from PlayTracking object.

    Args:
        tracking_data: PlayTracking from player_tracker.py
        snap_frame: Frame number of snap
        clip_file: Source clip filename

    Returns:
        DefensiveAnalysis
    """
    # Get positions at snap
    positions = tracking_data.get_all_positions_at_frame(snap_frame)

    # Get team assignments
    teams = {track_id: track.team
             for track_id, track in tracking_data.players.items()}

    # Run analysis
    inference = DefensiveInference()
    return inference.analyze_positions(positions, teams, clip_file, snap_frame)


def build_defensive_metrics(analysis: DefensiveAnalysis,
                            tracking_data,
                            snap_frame: int,
                            fps: float,
                            jersey_map: Dict[int, int] = None,
                            roster: Dict[int, Dict] = None) -> List[DefensiveMetrics]:
    """
    Build per-player defensive metrics from analysis and tracking.

    Args:
        analysis: DefensiveAnalysis from analyze_positions
        tracking_data: PlayTracking from player_tracker.py
        snap_frame: Frame number of snap
        fps: Video frame rate
        jersey_map: Dict of track_id -> jersey_number
        roster: Dict of jersey_number -> player info

    Returns:
        List of DefensiveMetrics for each defensive player
    """
    inference = DefensiveInference()
    metrics_list = []

    for player in analysis.player_positions:
        if player.team != "defense":
            continue

        # Get track for movement metrics
        track = tracking_data.players.get(player.track_id)
        if not track:
            continue

        # Classify position
        position, alignment = inference.classify_player_position(player, analysis)

        # Calculate movement metrics
        first_step = track.calculate_first_step_time(snap_frame, fps)

        # Calculate penetration (for DL)
        penetration = None
        if position in ["DL", "EDGE"]:
            # How far did they get past LOS?
            end_pos = track.get_position_at_frame(snap_frame + int(fps * 1.0))  # 1 sec later
            if end_pos:
                penetration = max(0, 0.5 - end_pos[1])  # How much into backfield

        # Calculate lateral movement (for DBs)
        lateral = None
        if position in ["CB", "S", "SLOT"]:
            start_pos = player.x
            positions_after = [f.x for f in track.frames if f.frame_num > snap_frame]
            if positions_after:
                lateral = max(abs(p - start_pos) for p in positions_after[:int(fps)])

        # Get jersey info
        jersey_num = jersey_map.get(player.track_id) if jersey_map else None
        player_name = ""
        if jersey_num and roster and jersey_num in roster:
            player_name = roster[jersey_num].get('name', '')

        # Build metrics object
        m = DefensiveMetrics(
            clip_file=analysis.clip_file,
            track_id=player.track_id,
            jersey_number=jersey_num,
            player_name=player_name,
            team_name="DEFENSE",
            inferred_position=position,
            alignment=alignment,
            depth_from_los=player.depth_from_los,
            in_box=player.is_in_box,
            on_line=player.is_on_line,
            first_step_sec=first_step,
            penetration_depth=penetration,
            lateral_movement=lateral
        )

        # Add grades
        m.first_step_grade = grade_defensive_first_step(first_step, position)
        m.defensive_grade = grade_defensive_player(m)

        metrics_list.append(m)

    return metrics_list


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Defensive Inference Module v1.0")
    print("This module is designed to be imported by sdi_pipeline_v4.py")
    print("\nUsage in code:")
    print("  from defensive_inference import DefensiveInference, analyze_defensive_from_tracking")
    print("  ")
    print("  # From tracking data:")
    print("  analysis = analyze_defensive_from_tracking(tracking, snap_frame, clip_file)")
    print("  print(analysis.summary())")
    print("  ")
    print("  # Get per-player metrics:")
    print("  metrics = build_defensive_metrics(analysis, tracking, snap_frame, fps)")
