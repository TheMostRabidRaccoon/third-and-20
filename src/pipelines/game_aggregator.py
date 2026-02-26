#!/usr/bin/env python3
"""
Third & 20 - Game Aggregator v1.0

Aggregates per-play CV metrics into game-level insights WITHOUT requiring jersey OCR.
Uses position-from-geometry: D-Line, EDGE, LB, CB, S derived from snap alignment.

Key insight: Coaches don't need to know "Player #52 has 0.35s first step."
They need to know "Their MIKE LB is slow to flow, blitz him."

Outputs:
    - Defensive tendency distribution (Cover 3 %, etc.)
    - Positional metrics aggregated by role (D-Line avg first step, etc.)
    - Alignment tells (when SS < 8 yards, blitz 80%)

Usage:
    from game_aggregator import GameAggregator

    agg = GameAggregator()
    for clip in clips:
        result = pipeline.run(clip)
        agg.add_play(result)

    summary = agg.get_summary()
    print(summary)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum
import json


@dataclass
class GeometricPosition:
    """Position classification from snap geometry (no jersey required)"""
    track_id: int
    role: str  # "offense" or "defense"
    position: str  # DL, EDGE, LB, CB, S, SLOT, etc.
    alignment: str  # 0-tech, MIKE, Press, etc.

    # Snap geometry
    depth_from_los: float  # Signed depth (positive = downfield)
    width_from_center: float  # 0 = center, 0.5 = sideline

    # Post-snap metrics (may be None if not computed)
    first_step_sec: Optional[float] = None
    penetration: Optional[float] = None
    lateral_movement: Optional[float] = None


@dataclass
class PlayResult:
    """Standardized result from one play for aggregation"""
    clip_name: str
    play_number: int = 0

    # PlayState fields
    is_valid: bool = False
    los_x: Optional[float] = None
    drive_dir: Optional[int] = None

    # Defensive scheme
    coverage_shell: str = "Unknown"  # Cover 0, Cover 1, Cover 2, Cover 3, Cover 4
    front: str = "unknown"  # 3-down, 4-down, 5-down
    box_count: int = 0
    safety_alignment: str = "unknown"  # 1-high, 2-high, 0-high

    # Per-player positions (from geometry)
    defensive_positions: List[GeometricPosition] = field(default_factory=list)
    offensive_positions: List[GeometricPosition] = field(default_factory=list)

    # Optional context
    down: Optional[int] = None
    distance: Optional[int] = None
    field_position: Optional[int] = None  # Yard line
    quarter: Optional[int] = None


@dataclass
class AlignmentTell:
    """A detected pre-snap alignment tell"""
    description: str
    condition: str  # e.g., "SS depth < 0.10"
    outcome: str  # e.g., "blitz"
    occurrences: int = 0
    total_samples: int = 0

    @property
    def percentage(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return 100.0 * self.occurrences / self.total_samples


class GameAggregator:
    """
    Aggregates per-play results into game-level insights.

    Key principle: Position is inferred from geometry, not jersey OCR.
    We track tendencies by coverage shell and aggregate metrics by position group.
    """

    def __init__(self, home_team: str = "", away_team: str = ""):
        self.home_team = home_team
        self.away_team = away_team
        self.plays: List[PlayResult] = []

        # Tendency counters
        self.coverage_counts: Dict[str, int] = defaultdict(int)
        self.front_counts: Dict[str, int] = defaultdict(int)
        self.safety_counts: Dict[str, int] = defaultdict(int)

        # Metrics by position group
        # Structure: position_group -> metric_name -> list of values
        self.defensive_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Alignment tells (conditions we track)
        self.tell_trackers: Dict[str, Dict[str, int]] = {
            # "ss_shallow" -> {"blitz": 4, "no_blitz": 1, "total": 5}
        }

        # Down-and-distance tendencies
        self.coverage_by_down: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def add_play(self, result: PlayResult) -> None:
        """Add one play result to the aggregation"""
        if not result.is_valid:
            return  # Skip invalid plays

        self.plays.append(result)

        # Count coverage shell
        self.coverage_counts[result.coverage_shell] += 1

        # Count front
        self.front_counts[result.front] += 1

        # Count safety alignment
        self.safety_counts[result.safety_alignment] += 1

        # Track coverage by down if available
        if result.down is not None:
            self.coverage_by_down[result.down][result.coverage_shell] += 1

        # Aggregate defensive metrics by position group
        for pos in result.defensive_positions:
            group = self._get_position_group(pos.position)

            if pos.first_step_sec is not None:
                self.defensive_metrics[group]["first_step_sec"].append(pos.first_step_sec)

            if pos.penetration is not None:
                self.defensive_metrics[group]["penetration"].append(pos.penetration)

            if pos.lateral_movement is not None:
                self.defensive_metrics[group]["lateral_movement"].append(pos.lateral_movement)

            # Track depth at snap
            self.defensive_metrics[group]["depth_at_snap"].append(abs(pos.depth_from_los))

        # Check for alignment tells
        self._check_alignment_tells(result)

    def _get_position_group(self, position: str) -> str:
        """Map position to broader group for aggregation"""
        if position in ("DL",):
            return "D-Line"
        elif position in ("EDGE",):
            return "EDGE"
        elif position in ("LB",):
            return "LB"
        elif position in ("CB", "SLOT"):
            return "CB"
        elif position in ("S",):
            return "Safety"
        return "Unknown"

    def _check_alignment_tells(self, result: PlayResult) -> None:
        """
        Check for pre-snap tells that correlate with post-snap behavior.

        Examples:
        - When SS aligns < 8 yards deep, they blitz X%
        - When CBs are in press, it's man coverage X%
        """
        # Find safeties and check depth
        safeties = [p for p in result.defensive_positions if p.position == "S"]

        # Tell: Shallow safety -> likely blitz
        shallow_safeties = [s for s in safeties if abs(s.depth_from_los) < 0.10]

        if shallow_safeties:
            tell_key = "ss_shallow"
            if tell_key not in self.tell_trackers:
                self.tell_trackers[tell_key] = {"blitz": 0, "no_blitz": 0, "total": 0}

            self.tell_trackers[tell_key]["total"] += 1

            # Check if this looks like a blitz (box_count >= 7 or coverage is Cover 0)
            if result.box_count >= 7 or result.coverage_shell == "Cover 0":
                self.tell_trackers[tell_key]["blitz"] += 1
            else:
                self.tell_trackers[tell_key]["no_blitz"] += 1

        # Tell: Press corners -> likely man coverage
        corners = [p for p in result.defensive_positions if p.position in ("CB", "SLOT")]
        press_corners = [c for c in corners if "Press" in c.alignment]

        if len(press_corners) >= 2:  # Both corners in press
            tell_key = "both_press"
            if tell_key not in self.tell_trackers:
                self.tell_trackers[tell_key] = {"man": 0, "zone": 0, "total": 0}

            self.tell_trackers[tell_key]["total"] += 1

            # Cover 0, Cover 1, 2-Man are man coverages
            if result.coverage_shell in ("Cover 0", "Cover 1", "2-Man"):
                self.tell_trackers[tell_key]["man"] += 1
            else:
                self.tell_trackers[tell_key]["zone"] += 1

        # Tell: 1-high -> Cover 3 or Cover 1 (check which is more common)
        if result.safety_alignment == "1-high":
            tell_key = "single_high"
            if tell_key not in self.tell_trackers:
                self.tell_trackers[tell_key] = {"cover3": 0, "cover1": 0, "other": 0, "total": 0}

            self.tell_trackers[tell_key]["total"] += 1

            if result.coverage_shell == "Cover 3":
                self.tell_trackers[tell_key]["cover3"] += 1
            elif result.coverage_shell == "Cover 1":
                self.tell_trackers[tell_key]["cover1"] += 1
            else:
                self.tell_trackers[tell_key]["other"] += 1

    def get_defensive_tendencies(self) -> Dict[str, Any]:
        """Get coverage and front tendency breakdown"""
        total_plays = len(self.plays)
        if total_plays == 0:
            return {"error": "No valid plays"}

        # Coverage distribution
        coverage_dist = {
            shell: {
                "count": count,
                "percentage": 100.0 * count / total_plays
            }
            for shell, count in sorted(
                self.coverage_counts.items(),
                key=lambda x: -x[1]
            )
        }

        # Front distribution
        front_dist = {
            front: {
                "count": count,
                "percentage": 100.0 * count / total_plays
            }
            for front, count in sorted(
                self.front_counts.items(),
                key=lambda x: -x[1]
            )
        }

        # Safety alignment distribution
        safety_dist = {
            align: {
                "count": count,
                "percentage": 100.0 * count / total_plays
            }
            for align, count in sorted(
                self.safety_counts.items(),
                key=lambda x: -x[1]
            )
        }

        return {
            "total_plays": total_plays,
            "coverage": coverage_dist,
            "front": front_dist,
            "safety_alignment": safety_dist
        }

    def get_positional_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics aggregated by position group"""
        result = {}

        for group, metrics in self.defensive_metrics.items():
            result[group] = {}

            for metric_name, values in metrics.items():
                if not values:
                    continue

                # Convert to plain Python floats for JSON serialization
                values_float = [float(v) for v in values]
                avg = sum(values_float) / len(values_float)
                min_val = min(values_float)
                max_val = max(values_float)

                result[group][metric_name] = {
                    "avg": round(avg, 3),
                    "min": round(min_val, 3),
                    "max": round(max_val, 3),
                    "samples": len(values_float)
                }

        return result

    def get_alignment_tells(self) -> List[AlignmentTell]:
        """Get detected alignment tells with significance"""
        tells = []

        # SS shallow -> blitz
        if "ss_shallow" in self.tell_trackers:
            tracker = self.tell_trackers["ss_shallow"]
            if tracker["total"] >= 3:  # Minimum sample size
                blitz_pct = 100.0 * tracker["blitz"] / tracker["total"]
                if blitz_pct >= 60 or blitz_pct <= 20:  # Significant if skewed
                    tells.append(AlignmentTell(
                        description="Safety shallow alignment",
                        condition="SS depth < 10% field (~6 yards)",
                        outcome=f"blitz {blitz_pct:.0f}%",
                        occurrences=tracker["blitz"],
                        total_samples=tracker["total"]
                    ))

        # Press corners -> man
        if "both_press" in self.tell_trackers:
            tracker = self.tell_trackers["both_press"]
            if tracker["total"] >= 3:
                man_pct = 100.0 * tracker["man"] / tracker["total"]
                if man_pct >= 60:
                    tells.append(AlignmentTell(
                        description="Both corners in press",
                        condition="Both CBs < 3 yards off",
                        outcome=f"man coverage {man_pct:.0f}%",
                        occurrences=tracker["man"],
                        total_samples=tracker["total"]
                    ))

        # Single high -> Cover 3 vs Cover 1
        if "single_high" in self.tell_trackers:
            tracker = self.tell_trackers["single_high"]
            if tracker["total"] >= 5:
                cover3_pct = 100.0 * tracker["cover3"] / tracker["total"]
                cover1_pct = 100.0 * tracker["cover1"] / tracker["total"]

                if cover3_pct > cover1_pct and cover3_pct >= 60:
                    tells.append(AlignmentTell(
                        description="Single high safety",
                        condition="1-high alignment",
                        outcome=f"Cover 3 {cover3_pct:.0f}%",
                        occurrences=tracker["cover3"],
                        total_samples=tracker["total"]
                    ))
                elif cover1_pct > cover3_pct and cover1_pct >= 60:
                    tells.append(AlignmentTell(
                        description="Single high safety",
                        condition="1-high alignment",
                        outcome=f"Cover 1 {cover1_pct:.0f}%",
                        occurrences=tracker["cover1"],
                        total_samples=tracker["total"]
                    ))

        return tells

    def get_coverage_by_down(self) -> Dict[int, Dict[str, str]]:
        """Get coverage tendency by down"""
        result = {}

        for down, coverages in sorted(self.coverage_by_down.items()):
            total = sum(coverages.values())
            if total == 0:
                continue

            result[down] = {
                coverage: f"{100.0 * count / total:.0f}%"
                for coverage, count in sorted(
                    coverages.items(),
                    key=lambda x: -x[1]
                )
            }

        return result

    def get_summary(self) -> str:
        """Generate human-readable game summary"""
        lines = []

        # Header
        if self.home_team and self.away_team:
            lines.append(f"Game: {self.home_team} vs {self.away_team}")
        lines.append(f"Plays Analyzed: {len(self.plays)}")
        lines.append("")

        # Defensive tendencies
        tendencies = self.get_defensive_tendencies()
        if "error" not in tendencies:
            lines.append("=" * 50)
            lines.append("DEFENSIVE TENDENCIES")
            lines.append("=" * 50)

            lines.append("\nCoverage Distribution:")
            for shell, data in tendencies["coverage"].items():
                lines.append(f"  {shell}: {data['count']} ({data['percentage']:.0f}%)")

            lines.append("\nFront Distribution:")
            for front, data in tendencies["front"].items():
                lines.append(f"  {front}: {data['count']} ({data['percentage']:.0f}%)")

            lines.append("\nSafety Alignment:")
            for align, data in tendencies["safety_alignment"].items():
                lines.append(f"  {align}: {data['count']} ({data['percentage']:.0f}%)")

        # Positional metrics
        metrics = self.get_positional_metrics()
        if metrics:
            lines.append("")
            lines.append("=" * 50)
            lines.append("POSITIONAL METRICS (by geometry)")
            lines.append("=" * 50)

            for group, group_metrics in metrics.items():
                lines.append(f"\n{group}:")
                for metric_name, data in group_metrics.items():
                    if metric_name == "first_step_sec":
                        lines.append(f"  First step: avg {data['avg']:.3f}s (n={data['samples']})")
                    elif metric_name == "penetration":
                        lines.append(f"  Penetration: avg {data['avg']:.3f} (n={data['samples']})")
                    elif metric_name == "depth_at_snap":
                        lines.append(f"  Depth at snap: avg {data['avg']:.3f} (n={data['samples']})")

        # Alignment tells
        tells = self.get_alignment_tells()
        if tells:
            lines.append("")
            lines.append("=" * 50)
            lines.append("ALIGNMENT TELLS")
            lines.append("=" * 50)

            for tell in tells:
                lines.append(f"\nWhen {tell.condition}:")
                lines.append(f"  -> {tell.outcome} ({tell.occurrences}/{tell.total_samples})")

        # Coverage by down
        by_down = self.get_coverage_by_down()
        if by_down:
            lines.append("")
            lines.append("=" * 50)
            lines.append("COVERAGE BY DOWN")
            lines.append("=" * 50)

            for down, coverages in by_down.items():
                cov_str = ", ".join(f"{c} {p}" for c, p in coverages.items())
                lines.append(f"  {down}{'st' if down == 1 else 'nd' if down == 2 else 'rd' if down == 3 else 'th'} down: {cov_str}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export all aggregated data as dict (for JSON/CSV)"""
        return {
            "game_info": {
                "home_team": self.home_team,
                "away_team": self.away_team,
                "plays_analyzed": len(self.plays)
            },
            "tendencies": self.get_defensive_tendencies(),
            "positional_metrics": self.get_positional_metrics(),
            "alignment_tells": [
                {
                    "description": t.description,
                    "condition": t.condition,
                    "outcome": t.outcome,
                    "occurrences": t.occurrences,
                    "total": t.total_samples,
                    "percentage": t.percentage
                }
                for t in self.get_alignment_tells()
            ],
            "coverage_by_down": self.get_coverage_by_down()
        }


def create_play_result_from_pipeline(
    clip_name: str,
    play_state: 'PlayState',
    defensive_analysis: 'DefensiveAnalysis',
    tracking_data: 'PlayTracking',
    snap_frame: int,
    fps: float = 30.0,
    play_number: int = 0,
    down: int = None,
    distance: int = None
) -> PlayResult:
    """
    Create a PlayResult from pipeline outputs.

    This bridges the per-play pipeline output to the aggregator.
    """
    from defensive_inference import DefensiveInference

    result = PlayResult(
        clip_name=clip_name,
        play_number=play_number,
        is_valid=play_state.is_valid,
        los_x=play_state.los_x,
        drive_dir=play_state.drive_dir,
        coverage_shell=defensive_analysis.coverage_shell.value if hasattr(defensive_analysis.coverage_shell, 'value') else str(defensive_analysis.coverage_shell),
        front=defensive_analysis.front.value if hasattr(defensive_analysis.front, 'value') else str(defensive_analysis.front),
        box_count=defensive_analysis.box_count,
        safety_alignment=defensive_analysis.safety_alignment.value if hasattr(defensive_analysis.safety_alignment, 'value') else str(defensive_analysis.safety_alignment),
        down=down,
        distance=distance
    )

    # Get inference engine for position classification
    inference = DefensiveInference()
    inference.los_x = play_state.los_x
    inference.drive_dir = play_state.drive_dir

    # Build geometric positions for each player
    for player in defensive_analysis.player_positions:
        if player.team == "defense":
            position, alignment = inference.classify_player_position(
                player, defensive_analysis
            )

            # Get track for movement metrics
            track = tracking_data.players.get(player.track_id) if tracking_data else None
            first_step = None
            penetration = None
            lateral = None

            if track:
                first_step = track.calculate_first_step_time(snap_frame, fps)

                # Penetration for D-Line
                if position in ("DL", "EDGE"):
                    end_pos = track.get_position_at_frame(snap_frame + int(fps * 1.0))
                    if end_pos and play_state.los_x is not None:
                        penetration = abs(end_pos[0] - play_state.los_x)

                # Lateral movement for DBs
                if position in ("CB", "S", "SLOT"):
                    start_x = player.x
                    positions_after = [
                        f.x for f in track.frames
                        if f.frame_num > snap_frame and f.frame_num < snap_frame + int(fps)
                    ]
                    if positions_after:
                        lateral = max(abs(p - start_x) for p in positions_after)

            # Compute signed depth
            depth = 0.0
            if play_state.los_x is not None and play_state.drive_dir is not None:
                depth = (player.x - play_state.los_x) * play_state.drive_dir

            geo_pos = GeometricPosition(
                track_id=player.track_id,
                role="defense",
                position=position,
                alignment=alignment,
                depth_from_los=depth,
                width_from_center=abs(player.y - 0.5) if hasattr(player, 'y') else 0.0,
                first_step_sec=first_step,
                penetration=penetration,
                lateral_movement=lateral
            )
            result.defensive_positions.append(geo_pos)

    return result


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    print("Game Aggregator v1.0")
    print("=" * 40)
    print("This module aggregates per-play results into game-level insights.")
    print("\nUsage in code:")
    print("  from game_aggregator import GameAggregator, create_play_result_from_pipeline")
    print("  ")
    print("  agg = GameAggregator(home_team='BRUSH', away_team='BEDFORD')")
    print("  for clip in clips:")
    print("      result = create_play_result_from_pipeline(clip, play_state, defense, tracking, snap)")
    print("      agg.add_play(result)")
    print("  ")
    print("  print(agg.get_summary())")
