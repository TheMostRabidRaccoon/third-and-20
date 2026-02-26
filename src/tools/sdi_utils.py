"""
SDI Utilities - Shared grading and scoring functions for SDI pipeline.

Consolidates duplicated code from sdi_pipeline v1-v4.
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class SDIMetrics:
    """Standard SDI metrics container"""
    first_step_sec: Optional[float] = None
    recognition_latency_sec: Optional[float] = None
    initial_velocity: Optional[float] = None
    direction_deg: Optional[float] = None


def grade_recognition_latency(latency_sec: float) -> str:
    """Grade QB recognition latency"""
    if latency_sec is None:
        return ""
    if latency_sec <= 0.3:
        return "Elite"
    elif latency_sec <= 0.5:
        return "Above Average"
    elif latency_sec <= 0.7:
        return "Average"
    elif latency_sec <= 0.9:
        return "Below Average"
    else:
        return "Slow"


def grade_first_step(first_step_sec: float, position: str = "") -> str:
    """Grade first step quickness by position"""
    if first_step_sec is None:
        return ""

    # Skill positions (WR, RB, CB, etc.) have faster benchmarks
    skill_positions = ['WR', 'RB', 'CB', 'FS', 'SS', 'DB', 'QB', 'OLB']
    is_skill = any(p in position.upper() for p in skill_positions) if position else False

    if is_skill:
        if first_step_sec <= 0.15:
            return "Elite"
        elif first_step_sec <= 0.25:
            return "Above Average"
        elif first_step_sec <= 0.35:
            return "Average"
        elif first_step_sec <= 0.45:
            return "Below Average"
        else:
            return "Slow"
    else:
        # Linemen
        if first_step_sec <= 0.20:
            return "Elite"
        elif first_step_sec <= 0.30:
            return "Above Average"
        elif first_step_sec <= 0.40:
            return "Average"
        elif first_step_sec <= 0.50:
            return "Below Average"
        else:
            return "Slow"


def grade_sdi_score(score: Optional[float]) -> str:
    """Grade composite SDI score"""
    if score is None:
        return "Poor"
    if score >= 8000:
        return "Elite"
    elif score >= 6000:
        return "Above Average"
    elif score >= 4000:
        return "Average"
    elif score >= 2000:
        return "Below Average"
    else:
        return "Poor"


def calculate_sdi_score(metrics: SDIMetrics) -> Optional[float]:
    """Calculate composite SDI score (0-10000 scale)"""
    score = 0.0
    components = 0

    # First step component (40% weight)
    if metrics.first_step_sec and 0 < metrics.first_step_sec < 1.0:
        fs_score = max(0, 10000 * (1 - metrics.first_step_sec))
        score += fs_score * 0.4
        components += 1

    # Recognition latency component (30% weight) - offense only
    if metrics.recognition_latency_sec and 0 < metrics.recognition_latency_sec < 3.0:
        lat_score = max(0, 10000 * (1 - metrics.recognition_latency_sec / 3.0))
        score += lat_score * 0.3
        components += 1

    # Velocity component (30% weight)
    if metrics.initial_velocity and metrics.initial_velocity > 0:
        # Normalize velocity (assume max reasonable is 0.3 normalized units/frame)
        vel_score = min(10000, metrics.initial_velocity * 33333)
        score += vel_score * 0.3
        components += 1

    if components == 0:
        return None

    # Normalize by components used
    return score / (components * 0.333)  # Scale back up
