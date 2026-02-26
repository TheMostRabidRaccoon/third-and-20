#!/usr/bin/env python3
"""
Third & 20 - Full SDI Pipeline v5.0
Combined Offensive SDI + Defensive Front Inference

This pipeline runs CV analysis on game film to extract:
1. OFFENSIVE: Jersey OCR, player tracking, SDI metrics
2. DEFENSIVE: Pre-snap alignment inference (NEW)
   - Box count (6/7/8-man box)
   - Safety alignment (1-high vs 2-high)
   - Defensive front (4-down vs 3-down)
   - Coverage shell estimation
   - Per-player defensive metrics

Usage:
    python sdi_pipeline_v5_with_defense.py <video_files> -o output_prefix \\
        --roster roster.csv --home-color white --away-color red \\
        --home-team BRUSH --away-team SHAKER

Outputs:
    - {output_prefix}_offensive_sdi.csv - Offensive player metrics
    - {output_prefix}_defensive_analysis.csv - Per-play defensive schemes
    - {output_prefix}_defensive_players.csv - Per-player defensive metrics
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import csv
import json
import sys
import re
from collections import Counter

# OCR imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("WARNING: easyocr not installed. Run: pip install easyocr")

# Import our modules
from third_and_20_cv_v2_fixed import SnapDetectorV2, PlayAnalysis
from player_tracker import PlayerTracker, PlayTracking, calculate_player_metrics
from defensive_inference import (
    DefensiveInference,
    DefensiveAnalysis,
    DefensiveMetrics,
    analyze_defensive_from_tracking,
    build_defensive_metrics
)
from play_state import PlayState, infer_play_state


# =============================================================================
# DATA CLASSES (from v4, self-contained)
# =============================================================================

@dataclass
class JerseyDetection:
    """A detected jersey number with location and color"""
    number: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[float, float]  # normalized 0-1
    jersey_color_bgr: Tuple[int, int, int] = (0, 0, 0)
    jersey_color_name: str = ""
    team_name: str = ""


@dataclass
class IdentifiedPlayer:
    """A player with both tracking data and identity"""
    track_id: int
    jersey_number: Optional[int] = None
    player_name: Optional[str] = None
    position: Optional[str] = None
    team: str = "unknown"
    team_name: str = ""
    confidence: float = 0.0


@dataclass
class SDIMetrics:
    """Full SDI metrics for one player on one play"""
    clip_file: str
    track_id: int
    jersey_number: Optional[int] = None
    player_name: str = ""
    roster_position: str = ""
    team: str = ""
    team_name: str = ""
    jersey_color: str = ""
    inferred_position: str = ""
    recognition_latency_sec: Optional[float] = None
    recognition_latency_grade: str = ""
    first_step_sec: Optional[float] = None
    first_step_grade: str = ""
    post_snap_distance: Optional[float] = None
    initial_velocity: Optional[float] = None
    sdi_score: Optional[float] = None
    sdi_grade: str = ""

    def to_dict(self) -> dict:
        return {
            'clip_file': self.clip_file,
            'track_id': self.track_id,
            'jersey_number': self.jersey_number if self.jersey_number else "",
            'player_name': self.player_name,
            'roster_position': self.roster_position,
            'team': self.team,
            'team_name': self.team_name,
            'jersey_color': self.jersey_color,
            'inferred_position': self.inferred_position,
            'recognition_latency_sec': f"{self.recognition_latency_sec:.3f}" if self.recognition_latency_sec else "",
            'recognition_latency_grade': self.recognition_latency_grade,
            'first_step_sec': f"{self.first_step_sec:.3f}" if self.first_step_sec else "",
            'first_step_grade': self.first_step_grade,
            'post_snap_distance': f"{self.post_snap_distance:.3f}" if self.post_snap_distance else "",
            'initial_velocity': f"{self.initial_velocity:.3f}" if self.initial_velocity else "",
            'sdi_score': f"{self.sdi_score:.1f}" if self.sdi_score else "",
            'sdi_grade': self.sdi_grade
        }


# =============================================================================
# COLOR CLASSIFIER (HSV-based for lighting robustness)
# =============================================================================

def classify_jersey_color_hsv(bgr_color: Tuple[int, int, int]) -> str:
    """
    Classify BGR color into jersey color category using HSV color space.

    HSV is more robust to lighting variations than RGB:
    - H (Hue): Color type (0-180 in OpenCV)
    - S (Saturation): Color intensity (0-255)
    - V (Value): Brightness (0-255)

    Returns: 'white', 'black', 'brown', 'red', 'blue', 'yellow', 'green', 'gray'
    """
    import numpy as np

    b, g, r = [int(x) for x in bgr_color]

    # Convert single pixel BGR to HSV
    bgr_pixel = np.uint8([[[b, g, r]]])
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_pixel[0, 0]

    # Also compute RGB-based metrics for edge cases
    brightness = (b + g + r) / 3
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    chroma = max_channel - min_channel

    # === ACHROMATIC COLORS ===
    # White/gray/black are determined primarily by VALUE and SATURATION
    # Key insight: camera footage often adds blue tint to whites

    # White: high brightness, relatively low saturation
    # Very bright pixels (V > 170) can have higher saturation due to color cast
    # and still be white jerseys. Less bright needs lower saturation.
    if v > 170 and s < 120:  # Very bright with moderate saturation
        return 'white'
    if v > 130 and s < 80:  # Bright with low-medium saturation
        return 'white'

    # Black: very low brightness
    if v < 50:
        return 'black'

    # Gray: medium brightness, low saturation
    if s < 40 and 50 <= v <= 130:
        return 'gray'

    # === CHROMATIC COLORS (higher saturation) ===

    # Green (grass) - filter out early
    # Hue 35-85 is green range in OpenCV
    if 35 <= h <= 85 and s > 50:
        return 'green'

    # Brown/Maroon - hue in red-orange range but darker and less saturated than pure red
    # Brown has hue 0-25 or 165-180, medium saturation, medium-low value
    if (h < 25 or h > 165):
        if s > 80 and v < 160:  # Saturated but not too bright = brown/maroon
            if v < 120 or s > 120:  # Darker or very saturated
                return 'brown'

    # Red - high saturation, hue near 0 or 180, brighter than brown
    if (h < 10 or h > 170) and s > 120 and v > 100:
        return 'red'

    # Yellow - hue 15-35, high saturation and value
    if 15 <= h <= 35 and s > 100 and v > 120:
        return 'yellow'

    # Blue - hue 100-130, needs high saturation to be true blue
    # (low saturation blue-ish = gray/white)
    if 100 <= h <= 130 and s > 100:
        return 'blue'

    # === FALLBACKS based on brightness and RGB ratios ===

    # Catch remaining whites (bright but with some color cast)
    if brightness > 140 and s < 90:
        return 'white'

    # Catch remaining blacks
    if brightness < 55:
        return 'black'

    # Warm colors that didn't match brown/red above
    if h < 25 or h > 165:
        if brightness < 100:
            return 'brown'
        else:
            return 'red'

    # Cool colors in blue hue range
    # IMPORTANT: Many dark browns appear blue-ish on camera
    # True blue jerseys are relatively bright (v > 120) AND saturated
    if 90 <= h <= 140:
        if s > 100 and v > 120:  # High sat + bright = true blue
            return 'blue'
        elif v < 80:  # Dark with blue-ish tint = probably brown
            # Check if it's warmer (R > B)
            if r > b:
                return 'brown'
            return 'gray'
        elif s < 60:  # Low saturation = gray
            return 'gray'
        else:
            # Medium saturation, medium brightness in blue hue
            # Could be blue or gray depending on context
            return 'gray'

    return 'gray'


def classify_jersey_color(bgr_color: Tuple[int, int, int]) -> str:
    """
    Classify BGR color into jersey color category.

    Uses HSV-based classification for better lighting robustness.
    """
    return classify_jersey_color_hsv(bgr_color)


def get_team_from_color(color: str, home_color: str, away_color: str,
                        home_team: str, away_team: str) -> str:
    """
    Map detected color to team name with fuzzy matching.

    Key insight: If one team is in a light color (white/yellow) and the other
    is in a dark color (brown/black/red/blue), then gray detections likely
    belong to the dark team (shadowed/poorly lit dark jerseys appear gray).
    """
    # Grass is never a jersey
    if color == 'green':
        return 'unknown'

    # Exact match
    if color == home_color:
        return home_team
    if color == away_color:
        return away_team

    # Define light vs dark color groups
    light_colors = {'white', 'yellow'}
    dark_colors = {'brown', 'black', 'red', 'blue', 'maroon'}

    # Similar color matching
    similar_colors = {
        'white': ['white', 'yellow'],
        'yellow': ['yellow', 'white'],
        'brown': ['brown', 'black', 'red', 'maroon', 'gray'],  # gray only, not blue
        'black': ['black', 'brown', 'gray'],
        'red': ['red', 'brown', 'maroon'],
        'blue': ['blue', 'black', 'gray'],
        'maroon': ['maroon', 'brown', 'red'],
    }

    if color in similar_colors.get(home_color, []):
        return home_team
    if color in similar_colors.get(away_color, []):
        return away_team

    # SMART GRAY HANDLING:
    # If gray is detected and one team is light, one is dark,
    # gray likely belongs to the dark team (shadows make dark jerseys look gray)
    # NOTE: We DON'T include blue here because blue-ish whites are common
    # (camera white balance issues). Blue is only assigned via similar_colors.
    if color == 'gray':
        home_is_light = home_color in light_colors
        away_is_light = away_color in light_colors
        home_is_dark = home_color in dark_colors
        away_is_dark = away_color in dark_colors

        # If home is dark and away is light, gray -> home
        if home_is_dark and away_is_light:
            return home_team
        # If away is dark and home is light, gray -> away
        if away_is_dark and home_is_light:
            return away_team

    return 'unknown'


# =============================================================================
# JERSEY OCR
# =============================================================================

class JerseyOCR:
    """OCR for detecting jersey numbers"""

    def __init__(self):
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("easyocr not installed")
        print("  Loading EasyOCR model...")
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.jersey_patterns = [r'^[0-9]{1,2}$', r'^[0O][0-9]$']

    def detect_jerseys(self, frame: np.ndarray) -> List[JerseyDetection]:
        """Detect jersey numbers in a frame."""
        detections = []
        h, w = frame.shape[:2]

        results = self.reader.readtext(frame, detail=1)

        for bbox, text, conf in results:
            text = text.strip().replace(' ', '').upper()
            text = text.replace('O', '0').replace('I', '1').replace('L', '1')

            if self._is_jersey_number(text) and conf > 0.3:
                try:
                    number = int(text)
                    if 0 <= number <= 99:
                        pts = np.array(bbox)
                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)

                        cx = (x_min + x_max) / 2 / w
                        cy = (y_min + y_max) / 2 / h

                        jersey_bgr = self._get_jersey_color(
                            frame, int(x_min), int(y_min),
                            int(x_max), int(y_max)
                        )
                        color_name = classify_jersey_color(jersey_bgr)

                        detections.append(JerseyDetection(
                            number=number,
                            confidence=conf,
                            bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                            center=(cx, cy),
                            jersey_color_bgr=jersey_bgr,
                            jersey_color_name=color_name
                        ))
                except ValueError:
                    continue

        return detections

    def _is_jersey_number(self, text: str) -> bool:
        for pattern in self.jersey_patterns:
            if re.match(pattern, text):
                return True
        return False

    def _get_jersey_color(self, frame: np.ndarray, x1: int, y1: int,
                          x2: int, y2: int) -> Tuple[int, int, int]:
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.5)
        pad_y = int((y2 - y1) * 0.8)

        sample_y1 = max(0, y1 - pad_y * 2)
        sample_y2 = y1
        sample_x1 = max(0, x1 - pad_x)
        sample_x2 = min(w, x2 + pad_x)

        if sample_y2 <= sample_y1 or sample_x2 <= sample_x1:
            sample_y1 = max(0, y1 - pad_y)
            sample_y2 = min(h, y2 + pad_y)
            sample_x1 = max(0, x1 - pad_x)
            sample_x2 = min(w, x2 + pad_x)

        sample = frame[sample_y1:sample_y2, sample_x1:sample_x2]

        if sample.size == 0:
            return (128, 128, 128)

        mean_bgr = cv2.mean(sample)[:3]
        return (int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2]))


# =============================================================================
# MATCHING & GRADING
# =============================================================================

def match_jerseys_to_tracks(jerseys: List[JerseyDetection], tracking: PlayTracking,
                            snap_frame: int) -> Dict[int, int]:
    """Match jersey detections to tracked players based on position at snap"""
    matches = {}

    if not jerseys or not tracking.players:
        return matches

    track_positions = {}
    for track_id, player_track in tracking.players.items():
        pos = player_track.get_position_at_frame(snap_frame)
        if pos:
            track_positions[track_id] = (pos[0], pos[1])

    if not track_positions:
        return matches

    used_tracks = set()
    for jersey in sorted(jerseys, key=lambda j: -j.confidence):
        best_track = None
        best_dist = float('inf')

        jx, jy = jersey.center

        for track_id, (tx, ty) in track_positions.items():
            if track_id in used_tracks:
                continue

            dist = ((jx - tx) ** 2 + (jy - ty) ** 2) ** 0.5

            if dist < best_dist and dist < 0.15:
                best_dist = dist
                best_track = track_id

        if best_track is not None:
            matches[best_track] = jersey.number
            used_tracks.add(best_track)

    return matches


def grade_recognition_latency(latency_sec: float) -> str:
    if latency_sec is None:
        return ""
    if latency_sec < 0.5:
        return "Elite"
    elif latency_sec < 0.8:
        return "Above Average"
    elif latency_sec < 1.2:
        return "Average"
    elif latency_sec < 1.6:
        return "Below Average"
    else:
        return "Developmental"


def grade_first_step(first_step_sec: float) -> str:
    if first_step_sec is None:
        return ""
    if first_step_sec < 0.3:
        return "Elite"
    elif first_step_sec < 0.45:
        return "Above Average"
    elif first_step_sec < 0.6:
        return "Average"
    elif first_step_sec < 0.8:
        return "Below Average"
    else:
        return "Developmental"


def calculate_sdi_score(metrics: SDIMetrics) -> float:
    score = 5000

    if metrics.recognition_latency_sec is not None:
        lat = metrics.recognition_latency_sec
        if lat < 0.5:
            score += 2000
        elif lat < 0.8:
            score += 1000
        elif lat < 1.2:
            score += 0
        else:
            score -= 500

    if metrics.first_step_sec is not None:
        fs = metrics.first_step_sec
        if fs < 0.3:
            score += 2000
        elif fs < 0.45:
            score += 1000
        elif fs < 0.6:
            score += 0
        else:
            score -= 500

    if metrics.initial_velocity is not None:
        score += int(metrics.initial_velocity * 500)

    if metrics.post_snap_distance is not None:
        score += int(metrics.post_snap_distance * 200)

    return max(0, score)


def grade_sdi_score(score: float) -> str:
    if score >= 8000:
        return "Elite"
    elif score >= 6500:
        return "Above Average"
    elif score >= 5000:
        return "Average"
    elif score >= 3500:
        return "Below Average"
    else:
        return "Developmental"


def load_roster(csv_path: str) -> Dict[int, Dict]:
    """Load roster from CSV."""
    roster = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    jersey = int(float(row.get('Jersey', 0)))
                    if jersey > 0:
                        role = row.get('Role', '').lower()
                        if 'coach' in role or 'admin' in role or 'manager' in role:
                            continue
                        name = f"{row.get('FirstName', '')} {row.get('LastName', '')}".strip()
                        roster[jersey] = {
                            'name': name,
                            'position': row.get('Position', ''),
                            'grad_year': row.get('GraduationYear', ''),
                            'height': row.get('Height', ''),
                            'weight': row.get('Weight', '')
                        }
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"Warning: Could not load roster: {e}")

    return roster


# =============================================================================
# COMBINED PLAY ANALYSIS
# =============================================================================

@dataclass
class CombinedPlayAnalysis:
    """Full analysis of one play - offense + defense"""
    clip_file: str
    snap_frame: int
    latency_sec: float
    decision_type: str

    # PlayState (the keystone - needed for aggregation)
    play_state: Optional[PlayState] = None

    # Defensive analysis
    defensive_analysis: Optional[DefensiveAnalysis] = None

    # Player metrics
    offensive_metrics: List[SDIMetrics] = field(default_factory=list)
    defensive_metrics: List[DefensiveMetrics] = field(default_factory=list)

    def get_play_summary(self) -> dict:
        """Get summary row for play-level output"""
        summary = {
            'clip_file': self.clip_file,
            'snap_frame': self.snap_frame,
            'latency_sec': f"{self.latency_sec:.3f}",
            'decision_type': self.decision_type,
        }

        # Add PlayState info
        if self.play_state:
            ps = self.play_state
            summary.update({
                'playstate_valid': ps.is_valid,
                'los_x': f"{ps.los_x:.3f}" if ps.los_x else "",
                'drive_dir': ps.drive_dir,
                'offense_team': ps.offense_team_id or "",
                'defense_team': ps.defense_team_id or "",
            })

        if self.defensive_analysis:
            da = self.defensive_analysis
            summary.update({
                'box_count': da.box_count,
                'defenders_on_los': da.defenders_on_los,
                'front': da.front.value,
                'front_description': da.front_description,
                'safety_alignment': da.safety_alignment.value,
                'coverage_shell': da.coverage_shell.value,
                'potential_blitz': da.potential_blitz,
                'defensive_confidence': f"{da.confidence:.2f}"
            })

        return summary


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class SDIPipelineV5:
    """Full SDI Pipeline with Offensive + Defensive Analysis"""

    def __init__(self, roster_path: str = None, home_team: str = "HOME",
                 away_team: str = "AWAY", home_color: str = "white",
                 away_color: str = "red", all_players: bool = False,
                 camera_type: str = "auto"):
        """
        Initialize pipeline.

        Args:
            roster_path: Path to roster CSV (for home team)
            home_team: Home team name (e.g., "BRUSH")
            away_team: Away team name (e.g., "SHAKER")
            home_color: Home team jersey color
            away_color: Away team jersey color
            all_players: If True, output metrics for ALL tracked players
                         (bypass offense/defense filter - useful for pro film)
            camera_type: "endzone" (x-axis is depth), "sideline" (y-axis is depth),
                         or "auto" (detect from clip path)
        """
        print(f"=" * 60)
        print(f"Third & 20 SDI Pipeline v5.0")
        print(f"Offensive SDI + Defensive Front Inference")
        print(f"=" * 60)
        print(f"  Home: {home_team} ({home_color})")
        print(f"  Away: {away_team} ({away_color})")
        if all_players:
            print(f"  Mode: ALL PLAYERS (no team filter)")

        # Store team config
        self.home_team = home_team.upper()
        self.away_team = away_team.upper()
        self.home_color = home_color.lower()
        self.away_color = away_color.lower()
        self.all_players = all_players
        self.camera_type = camera_type.lower()

        # Load components
        print("  Loading snap detector...")
        self.snap_detector = SnapDetectorV2()

        print("  Loading jersey OCR...")
        self.jersey_ocr = JerseyOCR()

        print("  Loading player tracker...")
        self.player_tracker = PlayerTracker()

        print("  Loading defensive inference engine...")
        self.defensive_inference = DefensiveInference()

        # Load roster
        self.roster = {}
        if roster_path:
            print("  Loading roster...")
            self.roster = load_roster(roster_path)
            print(f"  Loaded {len(self.roster)} players from roster")

        # Color statistics
        self.color_counts = Counter()

        # Aggregated results
        self.all_plays: List[CombinedPlayAnalysis] = []

    def process_clip(self, clip_path: str) -> Optional[CombinedPlayAnalysis]:
        """
        Process a single clip through the full pipeline.

        Returns:
            CombinedPlayAnalysis with all metrics
        """
        clip_name = Path(clip_path).name

        # Step 1: Detect snap
        print(f"  [1/5] Detecting snap...")
        snap_result = self.snap_detector.analyze_clip(clip_path)

        if snap_result.snap_frame is None:
            print(f"  WARNING: No snap detected in {clip_name}")
            return None

        print(f"  Snap at frame {snap_result.snap_frame}, "
              f"latency: {snap_result.latency_sec:.3f}s ({snap_result.decision_type})")

        # Step 2: Read jersey numbers at snap frame + surrounding frames for color consensus
        print(f"  [2/5] Reading jersey numbers...")
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Read snap frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, snap_result.snap_frame)
        ret, snap_frame_img = cap.read()
        if not ret:
            cap.release()
            print(f"  WARNING: Could not read snap frame")
            return None

        # Read additional frames for multi-frame color consensus
        # Sample frames: snap-10, snap-5, snap, snap+5, snap+10
        consensus_frames = {}
        for offset in [-10, -5, 0, 5, 10]:
            frame_num = snap_result.snap_frame + offset
            if frame_num >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    consensus_frames[frame_num] = frame

        cap.release()

        if snap_result.snap_frame not in consensus_frames:
            consensus_frames[snap_result.snap_frame] = snap_frame_img

        jerseys = self.jersey_ocr.detect_jerseys(snap_frame_img)
        print(f"  Detected {len(jerseys)} jersey numbers")

        # Count colors
        color_counts = Counter(j.jersey_color_name for j in jerseys)
        for color, count in color_counts.items():
            self.color_counts[color] += count
        print(f"  Colors: {dict(color_counts)}")

        # Assign teams based on color
        for jersey in jerseys:
            jersey.team_name = get_team_from_color(
                jersey.jersey_color_name,
                self.home_color, self.away_color,
                self.home_team, self.away_team
            )

        # Step 3: Track players
        print(f"  [3/5] Tracking players...")
        tracking = self.player_tracker.track_clip(clip_path, snap_result.snap_frame)
        print(f"  Tracked {len(tracking.players)} players")

        # Step 4: Match jerseys to tracks and improve team classification
        print(f"  [4/5] Matching jerseys to tracks...")
        jersey_map = match_jerseys_to_tracks(jerseys, tracking, snap_result.snap_frame)
        print(f"  Matched {len(jersey_map)} players to jersey numbers")

        # Update team classification in tracking based on jersey colors + color fallback
        # Use multi-frame consensus for more robust color detection
        jersey_lookup = {j.number: j for j in jerseys}
        self._update_team_classification(
            tracking, jersey_map, jersey_lookup,
            consensus_frames, snap_result.snap_frame
        )

        # Step 5: INFER PLAY STATE (THE KEYSTONE)
        print(f"  [5/6] Inferring play state...")
        play_state = self._infer_play_state(tracking, snap_result.snap_frame, clip_path)
        print(f"  PlayState: LOS={play_state.los_x}, dir={play_state.drive_dir}, valid={play_state.is_valid}")

        # Step 6: DEFENSIVE ANALYSIS
        print(f"  [6/6] Analyzing defensive alignment...")
        defensive_analysis = self._analyze_defense(tracking, snap_result.snap_frame, clip_name, play_state)

        if defensive_analysis:
            print(f"  Defense: {defensive_analysis.summary()}")

        # Build offensive metrics
        offensive_metrics = self._build_offensive_metrics(
            clip_name, snap_result, tracking, jersey_map, jersey_lookup, play_state
        )

        # Build defensive metrics
        defensive_metrics = []
        if defensive_analysis:
            defensive_metrics = build_defensive_metrics(
                defensive_analysis, tracking, snap_result.snap_frame,
                fps, jersey_map, self.roster
            )

        # Create combined result
        play_analysis = CombinedPlayAnalysis(
            clip_file=clip_name,
            snap_frame=snap_result.snap_frame,
            latency_sec=snap_result.latency_sec,
            decision_type=snap_result.decision_type,
            play_state=play_state,
            defensive_analysis=defensive_analysis,
            offensive_metrics=offensive_metrics,
            defensive_metrics=defensive_metrics
        )

        return play_analysis

    def _update_team_classification(self, tracking: PlayTracking,
                                     jersey_map: Dict[int, int],
                                     jersey_lookup: Dict[int, JerseyDetection],
                                     consensus_frames: Dict[int, np.ndarray],
                                     snap_frame: int):
        """
        Update player team IDENTITY based on jersey colors (NOT role).

        Uses multi-frame consensus for more robust color detection.

        Priority:
        1. OCR identity (high confidence) - from jersey_map
        2. Color identity with multi-frame consensus (medium confidence)
        3. Unknown
        """
        # Get dimensions from snap frame
        snap_frame_img = consensus_frames.get(snap_frame)
        if snap_frame_img is None:
            snap_frame_img = list(consensus_frames.values())[0]
        h, w = snap_frame_img.shape[:2]

        ocr_count = 0
        color_count = 0
        unknown_count = 0

        for track_id, track in tracking.players.items():
            jersey_num = jersey_map.get(track_id)

            # Priority 1: OCR identity (high confidence)
            if jersey_num and jersey_num in jersey_lookup:
                jersey = jersey_lookup[jersey_num]
                track.team_id = jersey.team_name  # "BRUSH", "SHAKER", etc.
                track.identity_source = "ocr"
                ocr_count += 1
                continue

            # Priority 2: Color identity with multi-frame consensus
            color_team = self._get_team_from_track_color_consensus(
                track, consensus_frames, snap_frame, h, w
            )
            if color_team:
                track.team_id = color_team
                track.identity_source = "color"
                color_count += 1
                continue

            # Priority 3: Unknown
            track.team_id = None
            track.identity_source = "unknown"
            unknown_count += 1

        print(f"  Identity sources: {ocr_count} OCR, {color_count} color, {unknown_count} unknown")
        if hasattr(self, '_color_debug'):
            print(f"  Color distribution: {self._color_debug}")
            self._color_debug = {}  # Reset for next clip
        if hasattr(self, '_bgr_samples') and self._bgr_samples:
            # Print first 5 samples
            print(f"  BGR samples (first 5): {self._bgr_samples[:5]}")
            self._bgr_samples = []

    def _get_team_from_track_color_consensus(self, track, consensus_frames: Dict[int, np.ndarray],
                                              snap_frame: int, img_h: int, img_w: int) -> Optional[str]:
        """
        Sample torso ROI across multiple frames and use majority vote for team.

        More robust than single-frame sampling because it handles:
        - Momentary occlusions
        - Lighting flicker
        - Motion blur at snap
        """
        team_votes = []

        for frame_num, frame_img in consensus_frames.items():
            # Find track position at this frame (with tolerance)
            frame_data = None
            best_dist = float('inf')
            for f in track.frames:
                dist = abs(f.frame_num - frame_num)
                if dist < best_dist and dist <= 5:  # Within 5 frames
                    frame_data = f
                    best_dist = dist

            if not frame_data:
                continue

            # Sample color from this frame
            team = self._sample_color_at_position(
                frame_img, frame_data, img_h, img_w
            )
            if team and team != 'unknown':
                team_votes.append(team)

        # Use majority vote
        if not team_votes:
            return None

        vote_counts = Counter(team_votes)
        best_team, best_count = vote_counts.most_common(1)[0]

        # Require at least 2 votes or majority to be confident
        if best_count >= 2 or best_count == len(team_votes):
            return best_team

        return None

    def _sample_color_at_position(self, frame_img: np.ndarray, frame_data,
                                   img_h: int, img_w: int) -> Optional[str]:
        """Sample jersey color at a specific position in a frame."""
        # Convert normalized coords to pixel coords
        cx = int(frame_data.x * img_w)
        cy = int(frame_data.y * img_h)
        bbox_w = int(frame_data.width * img_w)
        bbox_h = int(frame_data.height * img_h)

        # Define torso ROI
        roi_half_w = int(bbox_w * 0.2)
        roi_half_h = int(bbox_h * 0.15)
        torso_offset_y = int(bbox_h * 0.1)

        roi_top = max(0, cy - torso_offset_y - roi_half_h)
        roi_bottom = min(img_h, cy - torso_offset_y + roi_half_h)
        roi_left = max(0, cx - roi_half_w)
        roi_right = min(img_w, cx + roi_half_w)

        if roi_bottom <= roi_top or roi_right <= roi_left:
            return None

        roi = frame_img[roi_top:roi_bottom, roi_left:roi_right]
        if roi.size == 0:
            return None

        mean_bgr = cv2.mean(roi)[:3]
        b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])

        color_name = classify_jersey_color((b, g, r))

        # Map color to team
        team = get_team_from_color(
            color_name,
            self.home_color, self.away_color,
            self.home_team, self.away_team
        )

        # Debug tracking
        if not hasattr(self, '_bgr_samples'):
            self._bgr_samples = []
        self._bgr_samples.append((b, g, r, color_name, team))

        if not hasattr(self, '_color_debug'):
            self._color_debug = {}
        self._color_debug[color_name] = self._color_debug.get(color_name, 0) + 1

        return team if team != 'unknown' else None

    def _get_team_from_track_color(self, track, snap_frame: int,
                                    frame_img: np.ndarray,
                                    img_h: int, img_w: int) -> Optional[str]:
        """
        Sample torso ROI from track bbox and classify jersey color to team.

        Returns team name (BRUSH, EUCLID) or None if color unclassifiable.
        """
        # Get track's frame data at snap
        frame_data = None
        for f in track.frames:
            if f.frame_num == snap_frame:
                frame_data = f
                break

        if not frame_data:
            return None

        # Convert normalized coords to pixel coords
        cx = int(frame_data.x * img_w)
        cy = int(frame_data.y * img_h)
        bbox_w = int(frame_data.width * img_w)
        bbox_h = int(frame_data.height * img_h)

        # Define torso ROI (upper-middle portion of bbox, where jersey is)
        # The bbox center (cx, cy) is the center of the person detection
        # For torso: go up from center (jersey is above center), narrow width
        # More conservative ROI to avoid grass/background
        roi_half_w = int(bbox_w * 0.2)  # Narrower (was 0.3)
        roi_half_h = int(bbox_h * 0.15)  # Shorter vertical span

        # Torso is above center (head at top, legs at bottom)
        torso_offset_y = int(bbox_h * 0.1)  # Slight upward offset from center

        roi_top = cy - torso_offset_y - roi_half_h
        roi_bottom = cy - torso_offset_y + roi_half_h
        roi_left = cx - roi_half_w
        roi_right = cx + roi_half_w

        # Clamp to image bounds
        roi_top = max(0, roi_top)
        roi_bottom = min(img_h, roi_bottom)
        roi_left = max(0, roi_left)
        roi_right = min(img_w, roi_right)

        if roi_bottom <= roi_top or roi_right <= roi_left:
            return None

        # Extract ROI and get mean color
        roi = frame_img[roi_top:roi_bottom, roi_left:roi_right]
        if roi.size == 0:
            return None

        mean_bgr = cv2.mean(roi)[:3]
        b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])

        color_name = classify_jersey_color((b, g, r))

        # Map color to team
        team = get_team_from_color(
            color_name,
            self.home_color, self.away_color,
            self.home_team, self.away_team
        )

        # Debug: track BGR values
        if not hasattr(self, '_bgr_samples'):
            self._bgr_samples = []
        self._bgr_samples.append((b, g, r, color_name, team))

        # Debug: track color distribution
        if not hasattr(self, '_color_debug'):
            self._color_debug = {}
        self._color_debug[color_name] = self._color_debug.get(color_name, 0) + 1

        return team if team != 'unknown' else None

    def _get_depth_axis(self, clip_path: str) -> str:
        """
        Determine which axis represents field depth based on camera type.

        Returns:
            'x' for endzone camera (field length along x-axis)
            'y' for sideline camera (field length along y-axis)
        """
        if self.camera_type == 'endzone':
            return 'x'
        elif self.camera_type == 'sideline':
            return 'y'
        else:
            # Auto-detect from clip path
            path_lower = clip_path.lower()
            if 'sideline' in path_lower or 'side' in path_lower:
                return 'y'
            elif 'endzone' in path_lower or 'end' in path_lower:
                return 'x'
            else:
                # Default to endzone (x-axis depth) if can't detect
                return 'x'

    def _infer_play_state(self, tracking: PlayTracking, snap_frame: int,
                          clip_path: str = "") -> PlayState:
        """Infer PlayState (LOS, drive_dir, possession) from geometry - THE KEYSTONE"""
        # Get positions at snap
        positions_at_snap = tracking.get_all_positions_at_frame(snap_frame)

        # Get team identities ONLY for tracks that have positions at snap
        team_ids = {}
        for track_id in positions_at_snap.keys():
            track = tracking.players.get(track_id)
            if track:
                team_ids[track_id] = getattr(track, 'team_id', None)

        # Debug: count known identities at snap
        known = {k: v for k, v in team_ids.items() if v not in (None, 'unknown', '')}
        teams_present = set(known.values())
        counts_per_team = {t: sum(1 for v in known.values() if v == t) for t in teams_present}
        print(f"  Identity at snap: {len(known)}/{len(positions_at_snap)} known, teams={counts_per_team}")

        # Determine axis based on camera type
        axis = self._get_depth_axis(clip_path)
        print(f"  Camera axis: {axis} (camera_type={self.camera_type})")

        # Try multiple post-snap windows for drive_dir detection
        # Some plays develop late (play action, screens) - need longer window
        # BOUNDS CHECK: Clamp to available frames for short clips
        max_frame = tracking.total_frames - 1 if hasattr(tracking, 'total_frames') else snap_frame + 60

        best_state = None
        for offset in [30, 40, 50]:
            post_snap_frame = min(snap_frame + offset, max_frame)
            positions_post_snap = tracking.get_all_positions_at_frame(post_snap_frame)

            if not positions_post_snap:
                continue

            overlap = len(set(positions_at_snap.keys()) & set(positions_post_snap.keys()))

            state = infer_play_state(positions_at_snap, positions_post_snap, team_ids, axis=axis)

            # If we got a valid state with drive_dir, use it
            if state.drive_dir is not None:
                print(f"  Post-snap offset: {offset} frames, overlap: {overlap}, drive_dir: {state.drive_dir}")
                return state

            # Keep the best attempt (for fallback)
            if best_state is None or state.los_confidence > best_state.los_confidence:
                best_state = state

        # Return best attempt even if drive_dir is None
        print(f"  Post-snap: tried offsets [30,40,50], no drive_dir detected")
        return best_state if best_state else infer_play_state(positions_at_snap, {}, team_ids, axis=axis)

    def _analyze_defense(self, tracking: PlayTracking, snap_frame: int,
                         clip_name: str, play_state: PlayState) -> Optional[DefensiveAnalysis]:
        """Run defensive alignment analysis using PlayState (no ego leakage)"""
        positions = tracking.get_all_positions_at_frame(snap_frame)
        if not positions:
            return None

        # Pass team IDENTITIES, not roles
        team_ids = {track_id: getattr(track, 'team_id', None)
                    for track_id, track in tracking.players.items()}

        return self.defensive_inference.analyze_positions_with_state(
            positions, team_ids, play_state, clip_name, snap_frame
        )

    def _build_offensive_metrics(self, clip_name: str, snap_result: PlayAnalysis,
                                  tracking: PlayTracking,
                                  jersey_map: Dict[int, int],
                                  jersey_lookup: Dict[int, JerseyDetection],
                                  play_state: PlayState) -> List[SDIMetrics]:
        """Build offensive player metrics - uses PlayState to determine offense"""
        # GATING: If PlayState invalid, skip role-dependent metrics entirely
        if not play_state.is_valid or play_state.offense_team_id is None:
            print(f"  SKIP offense metrics: invalid PlayState (possession unknown)")
            return []

        metrics = []

        for track_id, player_track in tracking.players.items():
            # Only process offensive players (unless all_players mode)
            if not self.all_players:
                track_team_id = getattr(player_track, 'team_id', None)
                # Use PlayState to determine if this player is on offense
                if track_team_id != play_state.offense_team_id:
                    continue

            # Get jersey info if matched
            jersey_num = jersey_map.get(track_id)
            jersey_info = jersey_lookup.get(jersey_num) if jersey_num else None

            # Determine team
            if jersey_info:
                team_name = jersey_info.team_name
                jersey_color = jersey_info.jersey_color_name
            else:
                team_name = play_state.offense_team_id or ""  # From PlayState, not assumption
                jersey_color = ""

            # Get roster info
            player_name = ""
            roster_position = ""
            if jersey_num and jersey_num in self.roster:
                player_info = self.roster[jersey_num]
                player_name = player_info['name']
                roster_position = player_info['position']

            # Calculate movement metrics
            first_step = player_track.calculate_first_step_time(
                snap_result.snap_frame, tracking.fps
            )
            distance = player_track.calculate_total_distance()
            velocity = player_track.get_velocity_at_frame(
                snap_result.snap_frame + 10, tracking.fps
            )

            # Build metrics object
            m = SDIMetrics(
                clip_file=clip_name,
                track_id=track_id,
                jersey_number=jersey_num,
                player_name=player_name,
                roster_position=roster_position,
                team=team_name.lower() if team_name else "",
                team_name=team_name,
                jersey_color=jersey_color,
                recognition_latency_sec=snap_result.latency_sec if jersey_num else None,
                first_step_sec=first_step,
                post_snap_distance=distance,
                initial_velocity=velocity[0] if velocity else None
            )

            # Add grades
            m.recognition_latency_grade = grade_recognition_latency(m.recognition_latency_sec)
            m.first_step_grade = grade_first_step(m.first_step_sec)
            m.sdi_score = calculate_sdi_score(m)
            m.sdi_grade = grade_sdi_score(m.sdi_score)

            metrics.append(m)

        return metrics

    def process_game(self, clip_paths: List[str], output_prefix: str):
        """
        Process all clips from a game.
        """
        self.all_plays = []

        for i, clip_path in enumerate(clip_paths, 1):
            print(f"\n[{i}/{len(clip_paths)}] Processing: {Path(clip_path).name}")

            try:
                play = self.process_clip(clip_path)
                if play:
                    self.all_plays.append(play)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Write results
        self._write_results(output_prefix)

        # Print summary
        self._print_summary()

    def _write_results(self, output_prefix: str):
        """Write all results to CSV files"""
        if not self.all_plays:
            print("\nNo plays to write")
            return

        # 1. Offensive SDI metrics
        offensive_csv = f"{output_prefix}_offensive_sdi.csv"
        all_offensive = []
        for play in self.all_plays:
            all_offensive.extend(play.offensive_metrics)

        if all_offensive:
            fieldnames = list(all_offensive[0].to_dict().keys())
            with open(offensive_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in all_offensive:
                    writer.writerow(m.to_dict())
            print(f"\nOffensive SDI saved to: {offensive_csv}")

        # 2. Defensive analysis (per-play)
        defense_plays_csv = f"{output_prefix}_defensive_analysis.csv"
        defense_plays = []
        for play in self.all_plays:
            if play.defensive_analysis:
                defense_plays.append(play.defensive_analysis.to_dict())

        if defense_plays:
            fieldnames = list(defense_plays[0].keys())
            with open(defense_plays_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(defense_plays)
            print(f"Defensive analysis saved to: {defense_plays_csv}")

        # 3. Defensive player metrics
        defense_players_csv = f"{output_prefix}_defensive_players.csv"
        all_defensive = []
        for play in self.all_plays:
            all_defensive.extend(play.defensive_metrics)

        if all_defensive:
            fieldnames = list(all_defensive[0].to_dict().keys())
            with open(defense_players_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in all_defensive:
                    writer.writerow(m.to_dict())
            print(f"Defensive players saved to: {defense_players_csv}")

        # 4. Play-by-play summary (offense + defense context)
        play_summary_csv = f"{output_prefix}_play_summary.csv"
        summaries = [play.get_play_summary() for play in self.all_plays]

        if summaries:
            fieldnames = list(summaries[0].keys())
            with open(play_summary_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summaries)
            print(f"Play summary saved to: {play_summary_csv}")

    def _print_summary(self):
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        if not self.all_plays:
            print("No plays analyzed")
            return

        print(f"\nTotal plays analyzed: {len(self.all_plays)}")

        # Offensive summary
        all_offensive = []
        for play in self.all_plays:
            all_offensive.extend(play.offensive_metrics)

        if all_offensive:
            with_jersey = [m for m in all_offensive if m.jersey_number]
            print(f"\nOFFENSE:")
            print(f"  Total player-plays: {len(all_offensive)}")
            print(f"  With jersey #: {len(with_jersey)}")

        # Defensive summary
        plays_with_defense = [p for p in self.all_plays if p.defensive_analysis]
        if plays_with_defense:
            print(f"\nDEFENSE:")
            print(f"  Plays with defensive analysis: {len(plays_with_defense)}")

            # Front distribution
            front_counts = Counter()
            safety_counts = Counter()
            box_counts = Counter()

            for play in plays_with_defense:
                da = play.defensive_analysis
                front_counts[da.front_description] += 1
                safety_counts[da.safety_alignment.value] += 1
                box_counts[da.box_count] += 1

            print(f"\n  Defensive Fronts:")
            for front, count in front_counts.most_common():
                pct = count / len(plays_with_defense) * 100
                print(f"    {front}: {count} plays ({pct:.1f}%)")

            print(f"\n  Safety Alignments:")
            for safety, count in safety_counts.most_common():
                pct = count / len(plays_with_defense) * 100
                print(f"    {safety}: {count} plays ({pct:.1f}%)")

            print(f"\n  Box Counts:")
            for box, count in sorted(box_counts.items()):
                pct = count / len(plays_with_defense) * 100
                print(f"    {box}-man box: {count} plays ({pct:.1f}%)")

            # Blitz tendency
            blitz_plays = [p for p in plays_with_defense if p.defensive_analysis.potential_blitz]
            blitz_pct = len(blitz_plays) / len(plays_with_defense) * 100
            print(f"\n  Potential blitz looks: {len(blitz_plays)} plays ({blitz_pct:.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Third & 20 SDI Pipeline v5.0 - Offensive + Defensive Analysis"
    )
    parser.add_argument("clips", nargs="+", help="Video clip files")
    parser.add_argument("-o", "--output", default="game_analysis",
                        help="Output file prefix (creates multiple CSVs)")
    parser.add_argument("--roster", help="Roster CSV file")
    parser.add_argument("--home-team", default="HOME", help="Home team name")
    parser.add_argument("--away-team", default="AWAY", help="Away team name")
    parser.add_argument("--home-color", default="white",
                        help="Home jersey color (white, yellow, brown, black, red, blue)")
    parser.add_argument("--away-color", default="red",
                        help="Away jersey color (white, yellow, brown, black, red, blue)")
    parser.add_argument("--all-players", action="store_true",
                        help="Output metrics for ALL tracked players (bypass team filter, useful for pro film)")

    args = parser.parse_args()

    # Expand glob patterns
    clips = []
    for pattern in args.clips:
        path = Path(pattern)
        if path.exists():
            clips.append(str(path))
        else:
            # Try glob
            clips.extend(str(p) for p in Path('.').glob(pattern))

    clips = sorted(set(clips))

    print(f"Third & 20 SDI Pipeline v5.0")
    print(f"Processing {len(clips)} clips...")
    print(f"Output prefix: {args.output}")
    print(f"Home: {args.home_team} ({args.home_color})")
    print(f"Away: {args.away_team} ({args.away_color})")
    print(f"Roster: {args.roster}")
    print("=" * 60)

    # Run pipeline
    pipeline = SDIPipelineV5(
        roster_path=args.roster,
        home_team=args.home_team,
        away_team=args.away_team,
        home_color=args.home_color,
        away_color=args.away_color,
        all_players=args.all_players
    )

    pipeline.process_game(clips, args.output)


if __name__ == "__main__":
    main()
