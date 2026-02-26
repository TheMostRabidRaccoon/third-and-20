#!/usr/bin/env python3
"""
Third & 20 - Full SDI Pipeline v4.1
Jersey OCR with COLOR-BASED team assignment

Pipeline:
1. Snap Detection - find the snap frame
2. Jersey OCR - read jersey numbers at snap
3. Player Tracking - track movement post-snap
4. Match jerseys to tracks - link identity to movement
5. Assign team by jersey COLOR (not text OCR)
6. Calculate SDI metrics per identified player

What's new in v4.1:
- Improved color classifier handles all jersey colors (white, yellow, brown, black, red, blue)
- Fuzzy color matching for similar colors
- Tuned for game film compression artifacts

Usage:
    python sdi_pipeline_v4.py <video_files> -o output.csv --roster roster.csv --home-color white --away-color red --home-team BRUSH --away-team SHAKER
    
Example for Shaker v Brush:
    python sdi_pipeline_v4.py ~/Games/Shaker/*.mp4 -o shaker_sdi.csv --roster roster.csv --home-color white --away-color red --home-team BRUSH --away-team SHAKER
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


@dataclass
class JerseyDetection:
    """A detected jersey number with location and color"""
    number: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[float, float]  # normalized 0-1
    jersey_color_bgr: Tuple[int, int, int] = (0, 0, 0)  # BGR mean color
    jersey_color_name: str = ""  # "white", "red", etc.
    team_name: str = ""  # Assigned based on color


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
    
    # Recognition Latency (QB only)
    recognition_latency_sec: Optional[float] = None
    recognition_latency_grade: str = ""
    
    # First Step Quickness
    first_step_sec: Optional[float] = None
    first_step_grade: str = ""
    
    # Movement metrics
    post_snap_distance: Optional[float] = None
    initial_velocity: Optional[float] = None
    
    # Composite SDI score
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
# COLOR CLASSIFIER v2.0 - Handles all jersey colors with game film compression
# =============================================================================

def classify_jersey_color(bgr_color: Tuple[int, int, int]) -> str:
    """
    Classify BGR color into jersey color category.
    Tuned for game film with compression artifacts and variable lighting.
    
    Args:
        bgr_color: Tuple of (Blue, Green, Red) values 0-255
        
    Returns:
        One of: 'white', 'yellow', 'brown', 'black', 'red', 'blue'
    """
    b, g, r = [int(x) for x in bgr_color]
    brightness = (b + g + r) / 3
    
    # Calculate color metrics
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    chroma = max_channel - min_channel
    saturation = chroma / max_channel if max_channel > 0 else 0
    
    # Warmth: how much warmer (red/yellow) vs cooler (blue)
    warmth = (r + g) / 2 - b if b > 0 else (r + g) / 2
    
    # === YELLOW ===
    # Yellow jerseys: R and G both elevated, B suppressed
    # In compressed video, appears as warm golden tone
    if brightness > 70:
        rg_avg = (r + g) / 2
        if rg_avg > b * 1.4 and g > b * 1.2 and r > b * 1.2:
            # Both R and G stronger than B = yellow/gold
            if abs(r - g) < 40:  # R and G relatively balanced
                return 'yellow'
    
    # === WHITE ===
    # High brightness, all channels similar (low saturation)
    if brightness > 82.5 and saturation < 0.3:
        return 'white'
    
    # === BLACK ===
    # Very low brightness
    if brightness < 55:
        return 'black'
    
    # === BLUE ===
    # B channel notably dominant
    if b > 65:
        if b > r * 1.15 and b > g * 1.1:
            return 'blue'
        # Also catch blue when B is highest even if margins are smaller
        if b == max_channel and b > r and b > g and chroma > 15:
            return 'blue'
    
    # === RED ===
    # R channel notably dominant, low G and B
    if r > 65:
        if r > g * 1.2 and r > b * 1.15:
            return 'red'
        # Red with less margin but clear dominance
        if r == max_channel and r > g * 1.1 and r > b * 1.1 and chroma > 15:
            return 'red'
    
    # === BROWN ===
    # Warm but dark: R >= G >= B pattern, moderate brightness
    if 55 <= brightness <= 100:
        if r >= g and g >= b:
            # Warm tone with R highest
            if r > b * 1.1:
                return 'brown'
    
    # === FALLBACK ===
    # Use brightness as tiebreaker
    if brightness > 82.5:
        # Light but didn't match white - likely yellow in bad lighting
        if warmth > 10:
            return 'yellow'
        return 'white'
    else:
        # Dark but didn't match black - likely brown
        if warmth > 5:
            return 'brown'
        # Could be dark blue
        if b >= r and b >= g:
            return 'blue'
        return 'brown'


def get_team_from_color(color: str, home_color: str, away_color: str, 
                        home_team: str, away_team: str) -> str:
    """
    Map detected color to team name with fuzzy matching.
    
    Args:
        color: Detected jersey color
        home_color: Expected home team color
        away_color: Expected away team color
        home_team: Home team name
        away_team: Away team name
        
    Returns:
        Team name or 'unknown'
    """
    # Direct match
    if color == home_color:
        return home_team
    if color == away_color:
        return away_team
    
    # Fuzzy matching for similar colors in compressed video
    similar_colors = {
        'white': ['white', 'yellow'],  # Both light
        'yellow': ['yellow', 'white'],
        'brown': ['brown', 'black', 'red'],  # All dark/warm
        'black': ['black', 'brown'],
        'red': ['red', 'brown'],  # Both warm
        'blue': ['blue', 'black'],  # Both dark/cool
    }
    
    # Check if color is similar to home
    if color in similar_colors.get(home_color, []):
        return home_team
    
    # Check if color is similar to away
    if color in similar_colors.get(away_color, []):
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
        # Jersey number regex patterns
        self.jersey_patterns = [
            r'^[0-9]{1,2}$',  # 1-99
            r'^[0O][0-9]$',   # 00-09 with O
        ]
    
    def detect_jerseys(self, frame: np.ndarray) -> List[JerseyDetection]:
        """
        Detect jersey numbers in a frame.
        Returns list of JerseyDetection objects.
        """
        detections = []
        h, w = frame.shape[:2]
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours that might be jersey numbers
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and process promising regions
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            aspect = cw / ch if ch > 0 else 0
            
            # Filter by size and aspect ratio
            if 200 < area < 10000 and 0.3 < aspect < 3.0:
                # Expand bounding box slightly
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + pad)
                candidates.append((x1, y1, x2, y2))
        
        # Run OCR on full frame (let EasyOCR find text)
        results = self.reader.readtext(frame, detail=1)
        
        for bbox, text, conf in results:
            # Clean text
            text = text.strip().replace(' ', '').upper()
            text = text.replace('O', '0').replace('I', '1').replace('L', '1')
            
            # Check if it's a jersey number
            if self._is_jersey_number(text) and conf > 0.3:
                try:
                    number = int(text)
                    if 0 <= number <= 99:
                        # Get bounding box
                        pts = np.array(bbox)
                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)
                        
                        # Calculate center (normalized)
                        cx = (x_min + x_max) / 2 / w
                        cy = (y_min + y_max) / 2 / h
                        
                        # Get jersey color from surrounding area
                        jersey_bgr = self._get_jersey_color(
                            frame, int(x_min), int(y_min), 
                            int(x_max), int(y_max)
                        )
                        
                        # Classify the color
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
        """Check if text looks like a jersey number"""
        for pattern in self.jersey_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def _get_jersey_color(self, frame: np.ndarray, x1: int, y1: int, 
                          x2: int, y2: int) -> Tuple[int, int, int]:
        """
        Get the dominant jersey color around the number.
        Samples area around the number bbox.
        """
        h, w = frame.shape[:2]
        
        # Expand box to get jersey area (above and around number)
        pad_x = int((x2 - x1) * 0.5)
        pad_y = int((y2 - y1) * 0.8)
        
        # Sample above the number (where jersey is)
        sample_y1 = max(0, y1 - pad_y * 2)
        sample_y2 = y1
        sample_x1 = max(0, x1 - pad_x)
        sample_x2 = min(w, x2 + pad_x)
        
        if sample_y2 <= sample_y1 or sample_x2 <= sample_x1:
            # Fallback to area around number
            sample_y1 = max(0, y1 - pad_y)
            sample_y2 = min(h, y2 + pad_y)
            sample_x1 = max(0, x1 - pad_x)
            sample_x2 = min(w, x2 + pad_x)
        
        # Get sample region
        sample = frame[sample_y1:sample_y2, sample_x1:sample_x2]
        
        if sample.size == 0:
            return (128, 128, 128)  # Default gray
        
        # Calculate mean color
        mean_bgr = cv2.mean(sample)[:3]
        return (int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2]))


# =============================================================================
# MATCHING & METRICS
# =============================================================================

def match_jerseys_to_tracks(jerseys: List[JerseyDetection], tracking: PlayTracking, snap_frame: int) -> Dict[int, int]:
    """
    Match jersey detections to tracked players based on position at snap
    Returns: {track_id: jersey_number}
    """
    matches = {}
    
    if not jerseys or not tracking.players:
        return matches
    
    # Get player positions at snap frame
    track_positions = {}
    for track_id, player_track in tracking.players.items():
        pos = player_track.get_position_at_frame(snap_frame)
        if pos:
            track_positions[track_id] = (pos[0], pos[1])
    
    if not track_positions:
        return matches
    
    # Match each jersey to nearest track
    used_tracks = set()
    for jersey in sorted(jerseys, key=lambda j: -j.confidence):
        best_track = None
        best_dist = float('inf')
        
        jx, jy = jersey.center
        
        for track_id, (tx, ty) in track_positions.items():
            if track_id in used_tracks:
                continue
            
            # Calculate distance (positions should be similar scale)
            dist = ((jx - tx) ** 2 + (jy - ty) ** 2) ** 0.5
            
            if dist < best_dist and dist < 0.15:  # Max distance threshold
                best_dist = dist
                best_track = track_id
        
        if best_track is not None:
            matches[best_track] = jersey.number
            used_tracks.add(best_track)
    
    return matches


# =============================================================================
# GRADING FUNCTIONS
# =============================================================================

def grade_recognition_latency(latency_sec: float) -> str:
    """Grade QB recognition latency"""
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
    """Grade first step quickness"""
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
    """
    Calculate composite SDI score.
    Weighted combination of available metrics.
    """
    score = 5000  # Base score
    
    # Recognition latency component (for QBs)
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
    
    # First step component
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
    
    # Initial velocity component
    if metrics.initial_velocity is not None:
        vel = metrics.initial_velocity
        # Higher velocity = better
        score += int(vel * 500)
    
    # Distance component
    if metrics.post_snap_distance is not None:
        dist = metrics.post_snap_distance
        # More distance (for skill players) = better
        score += int(dist * 200)
    
    return max(0, score)


def grade_sdi_score(score: float) -> str:
    """Grade composite SDI score"""
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


# =============================================================================
# ROSTER LOADING
# =============================================================================

def load_roster(csv_path: str) -> Dict[int, Dict]:
    """
    Load roster from CSV.
    Returns: {jersey_number: {name, position, ...}}
    """
    roster = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    jersey = int(float(row.get('Jersey', 0)))
                    if jersey > 0:
                        # Skip coaches/staff
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
# MAIN PIPELINE
# =============================================================================

class SDIPipelineV4:
    """Full SDI Pipeline with color-based team assignment"""
    
    def __init__(self, roster_path: str = None, home_team: str = "HOME",
                 away_team: str = "AWAY", home_color: str = "white",
                 away_color: str = "red"):
        """
        Initialize pipeline.
        
        Args:
            roster_path: Path to roster CSV (for home team)
            home_team: Home team name (e.g., "BRUSH")
            away_team: Away team name (e.g., "SHAKER")
            home_color: Home team jersey color
            away_color: Away team jersey color
        """
        print(f"Initializing SDI Pipeline v4.1 (Color-Based Team Detection)")
        print(f"  Home: {home_team} ({home_color})")
        print(f"  Away: {away_team} ({away_color})")
        
        # Store team config
        self.home_team = home_team.upper()
        self.away_team = away_team.upper()
        self.home_color = home_color.lower()
        self.away_color = away_color.lower()
        
        # Load components
        print("  Loading snap detector...")
        self.snap_detector = SnapDetectorV2()
        
        print("  Loading jersey OCR...")
        self.jersey_ocr = JerseyOCR()
        
        print("  Loading player tracker...")
        self.player_tracker = PlayerTracker()
        
        # Load roster
        self.roster = {}
        if roster_path:
            print("  Loading roster...")
            self.roster = load_roster(roster_path)
            print(f"  Loaded {len(self.roster)} players from roster")
        
        # Color statistics
        self.color_counts = Counter()
    
    def process_clip(self, clip_path: str) -> Tuple[PlayAnalysis, PlayTracking, List[SDIMetrics]]:
        """
        Process a single clip through the full pipeline.
        
        Returns:
            Tuple of (snap_analysis, tracking_data, player_metrics)
        """
        clip_name = Path(clip_path).name
        
        # Step 1: Detect snap
        print(f"  [1/4] Detecting snap...")
        snap_result = self.snap_detector.analyze_clip(clip_path)
        
        if snap_result.snap_frame is None:
            print(f"  WARNING: No snap detected in {clip_name}")
            return snap_result, None, []
        
        print(f"  Snap at frame {snap_result.snap_frame}, latency: {snap_result.latency_sec:.3f}s")
        
        # Step 2: Read jersey numbers at snap frame
        print(f"  [2/4] Reading jersey numbers...")
        cap = cv2.VideoCapture(clip_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, snap_result.snap_frame)
        ret, snap_frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"  WARNING: Could not read snap frame")
            return snap_result, None, []
        
        jerseys = self.jersey_ocr.detect_jerseys(snap_frame)
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
        print(f"  [3/4] Tracking players...")
        tracking = self.player_tracker.track_clip(clip_path, snap_result.snap_frame)
        print(f"  Tracked {len(tracking.players)} players")
        
        # Step 4: Match jerseys to tracks
        print(f"  [4/4] Matching jerseys to tracks...")
        jersey_map = match_jerseys_to_tracks(jerseys, tracking, snap_result.snap_frame)
        print(f"  Matched {len(jersey_map)} players to jersey numbers")
        
        # Build jersey lookup for team assignment
        jersey_lookup = {j.number: j for j in jerseys}
        
        # Calculate metrics for each tracked player
        metrics = []
        for track_id, player_track in tracking.players.items():
            # Get jersey info if matched
            jersey_num = jersey_map.get(track_id)
            jersey_info = jersey_lookup.get(jersey_num) if jersey_num else None
            
            # Determine team
            if jersey_info:
                team_name = jersey_info.team_name
                jersey_color = jersey_info.jersey_color_name
            else:
                team_name = "unknown"
                jersey_color = ""
            
            # Get roster info (only for home team)
            player_name = ""
            roster_position = ""
            if jersey_num and team_name == self.home_team and jersey_num in self.roster:
                player_info = self.roster[jersey_num]
                player_name = player_info['name']
                roster_position = player_info['position']
            
            # Calculate movement metrics
            first_step = player_track.calculate_first_step_time(snap_result.snap_frame, tracking.fps)
            distance = player_track.calculate_total_distance()
            velocity = player_track.get_velocity_at_frame(snap_result.snap_frame + 10, tracking.fps)
            
            # Build metrics object
            m = SDIMetrics(
                clip_file=clip_name,
                track_id=track_id,
                jersey_number=jersey_num,
                player_name=player_name,
                roster_position=roster_position,
                team=team_name.lower() if team_name != "unknown" else "",
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
        
        return snap_result, tracking, metrics
    
    def process_game(self, clip_paths: List[str], output_csv: str):
        """
        Process all clips from a game.
        """
        all_metrics = []
        
        for i, clip_path in enumerate(clip_paths, 1):
            print(f"\n[{i}/{len(clip_paths)}] Processing: {Path(clip_path).name}")
            
            try:
                snap, tracking, metrics = self.process_clip(clip_path)
                all_metrics.extend(metrics)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        # Write results
        if all_metrics:
            fieldnames = list(all_metrics[0].to_dict().keys())
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in all_metrics:
                    writer.writerow(m.to_dict())
            
            print(f"\nResults saved to: {output_csv}")
        
        # Print color summary
        print(f"\n{'='*60}")
        print("COLOR DETECTION SUMMARY:")
        for color, count in self.color_counts.most_common():
            team = get_team_from_color(color, self.home_color, self.away_color,
                                      self.home_team, self.away_team)
            print(f"  {color}: {count} detections -> {team}")
        
        # Print summary stats
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        if not all_metrics:
            print("No metrics collected")
            return
        
        print(f"\nTotal player-plays: {len(all_metrics)}")
        
        # Group by team
        by_team = {}
        for m in all_metrics:
            team = m.team_name or "Unknown"
            if team not in by_team:
                by_team[team] = []
            by_team[team].append(m)
        
        for team in sorted(by_team.keys()):
            team_metrics = by_team[team]
            with_jersey = [m for m in team_metrics if m.jersey_number]
            with_name = [m for m in team_metrics if m.player_name]
            
            print(f"\n{team}:")
            print(f"  Player-plays: {len(team_metrics)}")
            print(f"  With jersey #: {len(with_jersey)}")
            print(f"  With roster name: {len(with_name)}")
            
            if with_jersey:
                print(f"\n  Top Players by Play Count:")
                player_counts = Counter()
                player_info = {}
                for m in with_jersey:
                    key = m.jersey_number
                    player_counts[key] += 1
                    if key not in player_info:
                        player_info[key] = (m.player_name or f"#{key}", m.roster_position or "")
                
                for jersey, count in player_counts.most_common(10):
                    name, pos = player_info[jersey]
                    # Get average SDI
                    player_sdi = [m.sdi_score for m in with_jersey 
                                  if m.jersey_number == jersey and m.sdi_score]
                    avg_sdi = sum(player_sdi) / len(player_sdi) if player_sdi else 0
                    print(f"    #{jersey:2} {name:25} {pos:8} Plays: {count:2} SDI:{avg_sdi:.0f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Third & 20 SDI Pipeline v4.1 - Color-based team detection"
    )
    parser.add_argument("clips", nargs="+", help="Video clip files")
    parser.add_argument("-o", "--output", default="sdi_output.csv", help="Output CSV")
    parser.add_argument("--roster", help="Roster CSV file")
    parser.add_argument("--home-team", default="HOME", help="Home team name")
    parser.add_argument("--away-team", default="AWAY", help="Away team name")
    parser.add_argument("--home-color", default="white", 
                        help="Home jersey color (white, yellow, brown, black, red, blue)")
    parser.add_argument("--away-color", default="red",
                        help="Away jersey color (white, yellow, brown, black, red, blue)")
    
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
    
    print(f"Third & 20 SDI Pipeline v4.1 (Color-Based Team Detection)")
    print(f"Processing {len(clips)} clips...")
    print(f"Output: {args.output}")
    print(f"Home: {args.home_team} ({args.home_color})")
    print(f"Away: {args.away_team} ({args.away_color})")
    print(f"Roster: {args.roster}")
    print("=" * 60)
    
    # Run pipeline
    pipeline = SDIPipelineV4(
        roster_path=args.roster,
        home_team=args.home_team,
        away_team=args.away_team,
        home_color=args.home_color,
        away_color=args.away_color
    )
    
    pipeline.process_game(clips, args.output)


if __name__ == "__main__":
    main()
