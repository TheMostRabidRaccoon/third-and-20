"""
Third & 20 Recommendation Engine - Data Models
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class PlayType(Enum):
    RUN = "run"
    PASS = "pass"
    SCREEN = "screen"
    PLAY_ACTION = "play_action"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"


class PlayResult(Enum):
    RUSH = "rush"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INTERCEPTION = "interception"
    FUMBLE = "fumble"
    SACK = "sack"
    PENALTY = "penalty"
    TOUCHDOWN = "touchdown"
    DOWNED = "downed"  # For punts


class HashMark(Enum):
    LEFT = "L"
    MIDDLE = "M"
    RIGHT = "R"


class DefensiveFront(Enum):
    THREE_DOWN = "3-down"
    FOUR_DOWN = "4-down"
    FIVE_DOWN = "5-down"
    UNKNOWN = "unknown"


class CoverageShell(Enum):
    COVER_0 = "Cover 0"
    COVER_1 = "Cover 1"
    COVER_2 = "Cover 2"
    COVER_3 = "Cover 3"
    COVER_4 = "Cover 4"
    COVER_6 = "Cover 6"
    MAN = "Man"
    UNKNOWN = "unknown"


# ============================================================================
# INPUT: Situation from sideline
# ============================================================================

@dataclass
class SituationInput:
    """What the coach enters on the tablet"""
    # Required
    down: int                      # 1, 2, 3, 4
    distance: int                  # Yards to first down
    yard_line: int                 # -50 to +50 (negative = own territory)
    hash_mark: HashMark            # L, M, R

    # Context
    quarter: int = 1               # 1, 2, 3, 4
    game_clock_seconds: int = 900  # Seconds remaining in quarter
    score_differential: int = 0   # Our score - their score
    timeouts_remaining: int = 3   # 0, 1, 2, 3

    # Optional
    personnel_on_field: str = "11"  # "11", "12", "21", "22"
    fatigue_flags: List[str] = field(default_factory=list)
    injured_out: List[str] = field(default_factory=list)

    @property
    def situation_bucket(self) -> str:
        """Canonical situation key for tendency lookup"""
        # Down
        down_str = f"{self.down}"

        # Distance bucket
        if self.distance <= 2:
            dist_str = "short"
        elif self.distance <= 5:
            dist_str = "medium"
        elif self.distance <= 10:
            dist_str = "long"
        else:
            dist_str = "very_long"

        # Field position
        if self.yard_line <= -20:
            field_str = "backed_up"
        elif self.yard_line <= 0:
            field_str = "own_territory"
        elif self.yard_line <= 35:
            field_str = "midfield"
        elif self.yard_line <= 45:
            field_str = "plus_territory"
        else:
            field_str = "redzone"

        return f"{down_str}_{dist_str}_{field_str}"


# ============================================================================
# OUTPUT: Play recommendations
# ============================================================================

@dataclass
class PlayRecommendation:
    """One recommended play"""
    play_name: str                 # "WISCONSIN RED"
    formation: str                 # "Shotgun Trips"
    play_type: PlayType

    success_probability: float     # 0.0 - 1.0
    expected_yards: float          # Average yards in similar situations
    confidence: str                # "HIGH", "MEDIUM", "LOW"

    reasoning: List[str]           # Why this play
    risk_flags: List[str]          # What could go wrong

    # Historical stats for this play
    times_called: int = 0
    success_rate: float = 0.0
    avg_yards: float = 0.0


@dataclass
class RecommendationResponse:
    """Full response to coach"""
    situation: SituationInput
    response_time_ms: int

    recommendations: List[PlayRecommendation]  # Top 3-5, ranked

    # Predicted defense
    predicted_front: DefensiveFront
    predicted_coverage: CoverageShell
    defense_confidence: float


# ============================================================================
# TRAINING DATA: Historical play outcomes
# ============================================================================

@dataclass
class PlayOutcome:
    """
    One play from game film - merged Hudl + CV data.
    This is the training data for the recommendation engine.
    """
    # Identifiers
    game_id: str
    play_number: int
    clip_file: str

    # Situation (from Hudl)
    down: int
    distance: int
    yard_line: int                 # Converted from YARD LN
    hash_mark: HashMark

    # What was called (from Hudl)
    play_type: PlayType            # Run, Pass
    formation: str                 # "IOWA BLACK", "WISCONSIN RED"
    play_direction: Optional[str]  # "L", "R"

    # Defensive alignment (from CV pipeline)
    def_front: Optional[DefensiveFront]
    def_coverage: Optional[CoverageShell]
    def_box_count: Optional[int]
    def_safety_look: Optional[str]

    # Result (from Hudl)
    result: PlayResult
    yards_gained: Optional[int]
    first_down: bool = False
    turnover: bool = False

    # CV confidence
    cv_valid: bool = False
    cv_confidence: float = 0.0

    @property
    def is_success(self) -> bool:
        """Was this play successful? (gained expected yards)"""
        if self.turnover:
            return False
        if self.yards_gained is None:
            return False

        # Success = gained at least 50% of needed yards on early downs
        # Success = converted on 3rd/4th down
        if self.down in (3, 4):
            return self.first_down
        elif self.down == 1:
            return self.yards_gained >= 4  # 40% of 10
        elif self.down == 2:
            return self.yards_gained >= self.distance * 0.5
        return self.yards_gained > 0


@dataclass
class OpponentTendencies:
    """
    Aggregated defensive tendencies for one opponent.
    Built from multiple games of CV analysis.
    """
    opponent_id: str
    games_analyzed: int
    plays_analyzed: int

    # Tendencies by situation bucket
    # situation_bucket -> {coverage: probability}
    coverage_tendencies: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Front tendencies by situation
    front_tendencies: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Blitz tendencies
    blitz_rate_by_situation: Dict[str, float] = field(default_factory=dict)

    def get_coverage_prediction(self, situation: SituationInput) -> Dict[str, float]:
        """Predict coverage probabilities for a situation"""
        bucket = situation.situation_bucket

        if bucket in self.coverage_tendencies:
            return self.coverage_tendencies[bucket]

        # Fallback to general tendencies
        return self.coverage_tendencies.get("general", {
            "Cover 3": 0.4,
            "Cover 2": 0.3,
            "Cover 1": 0.2,
            "Cover 0": 0.1
        })


@dataclass
class Playbook:
    """
    Team's playbook - all available plays.
    """
    team_id: str
    plays: Dict[str, 'Play'] = field(default_factory=dict)

    def get_plays_for_situation(self, situation: SituationInput) -> List['Play']:
        """Filter plays that fit the current situation"""
        candidates = []
        for play in self.plays.values():
            # Check personnel
            if play.personnel_required != situation.personnel_on_field:
                continue
            # Check injuries
            if any(p in situation.injured_out for p in play.key_players):
                continue
            # Check situational fit
            bucket = situation.situation_bucket
            if play.situation_fit.get(bucket, 0.5) < 0.3:
                continue
            candidates.append(play)
        return candidates


@dataclass
class Play:
    """One play in the playbook"""
    play_id: str
    name: str                      # "WISCONSIN RED"
    formation: str                 # "Shotgun Trips"
    play_type: PlayType

    personnel_required: str = "11"
    key_players: List[str] = field(default_factory=list)

    # Situational fit scores (0-1)
    situation_fit: Dict[str, float] = field(default_factory=dict)

    # Historical performance vs each coverage
    # coverage -> {success_rate, avg_yards, times_called}
    performance_by_coverage: Dict[str, Dict[str, float]] = field(default_factory=dict)
