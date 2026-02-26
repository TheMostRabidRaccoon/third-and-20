#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third & 20 - Highlight Reel Processor v1.0
Specialized pipeline for processing pro athlete highlight reels

Unlike game film (individual play clips), highlight reels are:
- Single long video with multiple plays spliced together
- Scene cuts between plays (no natural pre-snap cadence)
- Focus on ONE athlete we want to track

This processor:
1. Detects scene changes to find play boundaries
2. Tracks a specific target player (by jersey number or largest bbox)
3. Computes position-specific metrics (RB, DL, QB, WR, DB)
4. Outputs stats for just that player

Usage:
    python highlight_reel_processor.py <video_file> \\
        --player-name "Ricky Powers" \\
        --position RB \\
        --jersey 25 \\
        --team MICHIGAN \\
        -o ricky_powers_output

    python highlight_reel_processor.py <video_file> \\
        --player-name "Myles Garrett" \\
        --position DL \\
        --jersey 95 \\
        -o myles_garrett_output

    python highlight_reel_processor.py <video_file> \\
        --player-name "Lamar Jackson" \\
        --position QB \\
        --jersey 8 \\
        -o lamar_jackson_output
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import csv
import argparse
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("WARNING: easyocr not installed. Run: pip install easyocr")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlaySegment:
    """A detected play segment in the highlight reel"""
    play_number: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_sec: float

@dataclass
class PlayerFrame:
    """Single frame data for tracked player"""
    frame_num: int
    x: float  # center x (normalized 0-1)
    y: float  # center y (normalized 0-1)
    width: float
    height: float
    confidence: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0

@dataclass
class PlayMetrics:
    """Metrics for one play"""
    play_number: int
    start_frame: int
    end_frame: int
    duration_sec: float

    # Movement metrics
    first_movement_frame: Optional[int] = None
    first_step_sec: Optional[float] = None
    initial_velocity: Optional[float] = None
    max_velocity: Optional[float] = None
    avg_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None
    total_distance: Optional[float] = None

    # Position-specific (filled based on position)
    # RB metrics
    burst_speed: Optional[float] = None  # Speed 0.5s after first contact with line
    yards_after_contact_proxy: Optional[float] = None  # Distance after velocity dip (in frame %, multiply by ~53 for yard estimate)

    # DL metrics
    get_off_time: Optional[float] = None  # Snap to first step
    penetration_depth: Optional[float] = None  # Max Y displacement

    # QB metrics
    time_to_throw: Optional[float] = None  # First step back to arm motion
    pocket_movement: Optional[float] = None  # Distance moved in pocket

    # DB metrics
    break_reaction_time: Optional[float] = None
    closing_speed: Optional[float] = None

    # Tracking quality
    frames_tracked: int = 0
    tracking_coverage: float = 0.0  # % of play frames with tracking


@dataclass
class HighlightAnalysis:
    """Full analysis of a highlight reel"""
    video_path: str
    player_name: str
    position: str
    jersey_number: Optional[int]
    team: str

    total_plays: int = 0
    plays: List[PlaySegment] = field(default_factory=list)
    metrics: List[PlayMetrics] = field(default_factory=list)

    # Aggregate stats
    avg_first_step: Optional[float] = None
    avg_max_velocity: Optional[float] = None
    avg_acceleration: Optional[float] = None


# =============================================================================
# SCENE CHANGE DETECTOR
# =============================================================================

class SceneChangeDetector:
    """Detects scene changes (cuts) in highlight reels to find play boundaries"""

    def __init__(self, threshold: float = 30.0, min_play_frames: int = 30):
        """
        Args:
            threshold: Histogram difference threshold for scene change
            min_play_frames: Minimum frames for a valid play (filters glitches)
        """
        self.threshold = threshold
        self.min_play_frames = min_play_frames

    def detect_scenes(self, video_path: str) -> List[PlaySegment]:
        """Detect all scene changes and return play segments"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Analyzing {total_frames} frames for scene changes...")

        scene_changes = [0]  # Start with frame 0
        prev_hist = None
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and compute histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                # Compare histograms
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

                if diff > self.threshold:
                    scene_changes.append(frame_num)

            prev_hist = hist
            frame_num += 1

            if frame_num % 500 == 0:
                print(f"    Processed {frame_num}/{total_frames} frames...")

        scene_changes.append(total_frames)  # End with last frame
        cap.release()

        # Convert to play segments, filtering short ones
        plays = []
        play_num = 1

        for i in range(len(scene_changes) - 1):
            start = scene_changes[i]
            end = scene_changes[i + 1]
            duration = end - start

            if duration >= self.min_play_frames:
                plays.append(PlaySegment(
                    play_number=play_num,
                    start_frame=start,
                    end_frame=end,
                    duration_frames=duration,
                    duration_sec=duration / fps
                ))
                play_num += 1

        print(f"  Detected {len(plays)} plays (filtered {len(scene_changes) - 1 - len(plays)} short segments)")
        return plays


# =============================================================================
# TARGET PLAYER TRACKER
# =============================================================================

class TargetPlayerTracker:
    """Tracks a specific target player across frames"""

    def __init__(self, target_jersey: Optional[int] = None, model_path: str = "yolov8n.pt"):
        self.target_jersey = target_jersey
        self.model = YOLO(model_path) if YOLO_AVAILABLE else None
        self.ocr = easyocr.Reader(['en'], gpu=True) if EASYOCR_AVAILABLE else None

    def track_play(self, video_path: str, play: PlaySegment, fps: float) -> List[PlayerFrame]:
        """Track the target player through one play segment"""

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, play.start_frame)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_data = []
        prev_position = None
        prev_velocity = None

        for frame_offset in range(play.duration_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_num = play.start_frame + frame_offset

            # Detect people
            if self.model:
                results = self.model(frame, classes=[0], verbose=False)  # class 0 = person

                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes

                    # Find target player
                    target_box = self._find_target_player(frame, boxes, width, height)

                    if target_box is not None:
                        x1, y1, x2, y2 = target_box[:4]
                        conf = target_box[4] if len(target_box) > 4 else 1.0

                        cx = ((x1 + x2) / 2) / width
                        cy = ((y1 + y2) / 2) / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height

                        # Calculate velocity
                        vx, vy, speed = 0.0, 0.0, 0.0
                        accel = 0.0

                        if prev_position is not None:
                            dt = 1.0 / fps
                            vx = (cx - prev_position[0]) / dt
                            vy = (cy - prev_position[1]) / dt
                            speed = np.sqrt(vx**2 + vy**2)

                            if prev_velocity is not None:
                                accel = (speed - prev_velocity) / dt

                        pf = PlayerFrame(
                            frame_num=frame_num,
                            x=cx, y=cy,
                            width=w, height=h,
                            confidence=float(conf),
                            velocity_x=vx,
                            velocity_y=vy,
                            speed=speed,
                            acceleration=accel
                        )
                        frames_data.append(pf)

                        prev_position = (cx, cy)
                        prev_velocity = speed

        cap.release()
        return frames_data

    def _find_target_player(self, frame: np.ndarray, boxes, width: int, height: int) -> Optional[Tuple]:
        """Find the target player among detected people"""

        best_box = None
        best_score = -1

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0])

            # If we have a target jersey, try to match it
            if self.target_jersey and self.ocr:
                # Crop player region
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    # Try to read jersey number
                    jersey_num = self._read_jersey_number(crop)
                    if jersey_num == self.target_jersey:
                        return (x1, y1, x2, y2, conf)

            # Fallback: use largest/most central player (likely the highlight subject)
            box_area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2

            # Score based on size and centrality
            centrality = 1.0 - abs(cx - width/2) / (width/2)
            score = box_area * centrality * conf

            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2, conf)

        return best_box

    def _read_jersey_number(self, crop: np.ndarray) -> Optional[int]:
        """Try to read jersey number from player crop"""
        if self.ocr is None:
            return None

        try:
            results = self.ocr.readtext(crop, detail=0)
            for text in results:
                text = text.strip().replace('O', '0').replace('I', '1')
                if text.isdigit() and len(text) <= 2:
                    return int(text)
        except:
            pass
        return None


# =============================================================================
# POSITION-SPECIFIC METRICS CALCULATOR
# =============================================================================

class PositionMetricsCalculator:
    """Calculates position-specific metrics from tracking data"""

    MOVEMENT_THRESHOLD = 0.005  # Minimum movement to count as "moving"

    def calculate(self, position: str, frames: List[PlayerFrame],
                  play: PlaySegment, fps: float) -> PlayMetrics:
        """Calculate metrics based on player position"""

        metrics = PlayMetrics(
            play_number=play.play_number,
            start_frame=play.start_frame,
            end_frame=play.end_frame,
            duration_sec=play.duration_sec,
            frames_tracked=len(frames),
            tracking_coverage=len(frames) / play.duration_frames if play.duration_frames > 0 else 0
        )

        if len(frames) < 5:
            return metrics

        # Common metrics for all positions
        self._calculate_common_metrics(metrics, frames, fps)

        # Position-specific metrics
        if position == "RB":
            self._calculate_rb_metrics(metrics, frames, fps)
        elif position == "DL":
            self._calculate_dl_metrics(metrics, frames, fps)
        elif position == "QB":
            self._calculate_qb_metrics(metrics, frames, fps)
        elif position == "DB":
            self._calculate_db_metrics(metrics, frames, fps)
        elif position == "WR":
            self._calculate_wr_metrics(metrics, frames, fps)
        elif position == "LB":
            self._calculate_lb_metrics(metrics, frames, fps)

        return metrics

    def _calculate_common_metrics(self, metrics: PlayMetrics,
                                   frames: List[PlayerFrame], fps: float):
        """Calculate metrics common to all positions"""

        # Find first movement
        start_pos = (frames[0].x, frames[0].y)
        for i, f in enumerate(frames):
            dist = np.sqrt((f.x - start_pos[0])**2 + (f.y - start_pos[1])**2)
            if dist > self.MOVEMENT_THRESHOLD:
                metrics.first_movement_frame = f.frame_num
                metrics.first_step_sec = i / fps
                break

        # Velocity stats
        speeds = [f.speed for f in frames if f.speed > 0]
        if speeds:
            metrics.initial_velocity = speeds[0] if len(speeds) > 0 else None
            metrics.max_velocity = max(speeds)
            metrics.avg_velocity = np.mean(speeds)

        # Acceleration
        accels = [f.acceleration for f in frames]
        if accels:
            metrics.max_acceleration = max(accels)

        # Total distance
        total_dist = 0
        for i in range(1, len(frames)):
            dx = frames[i].x - frames[i-1].x
            dy = frames[i].y - frames[i-1].y
            total_dist += np.sqrt(dx**2 + dy**2)
        metrics.total_distance = total_dist

    def _calculate_rb_metrics(self, metrics: PlayMetrics,
                               frames: List[PlayerFrame], fps: float):
        """RB-specific: burst speed, yards after contact proxy"""

        # Burst speed: max speed in first 1 second after first movement
        if metrics.first_step_sec is not None:
            burst_window = int(fps * 1.0)  # 1 second
            first_move_idx = next((i for i, f in enumerate(frames)
                                   if f.frame_num == metrics.first_movement_frame), 0)

            burst_frames = frames[first_move_idx:first_move_idx + burst_window]
            if burst_frames:
                metrics.burst_speed = max(f.speed for f in burst_frames)

        # Yards after contact proxy: distance traveled after a velocity dip
        # (velocity dip suggests contact with defender)
        # Result is in "field widths" - multiply by 53.3 for approximate yards
        speeds = [f.speed for f in frames]
        if len(speeds) > 10:
            # Find significant velocity dip (>30% drop)
            for i in range(5, len(speeds) - 5):
                if speeds[i] < speeds[i-3] * 0.7 and speeds[i-3] > 0.01:
                    # Contact detected at frame i
                    # Only measure for 1 second after contact (not entire rest of play)
                    contact_window = int(fps * 1.0)
                    end_idx = min(i + contact_window, len(frames) - 1)

                    post_contact_dist = 0
                    for j in range(i, end_idx):
                        dx = frames[j+1].x - frames[j].x
                        dy = frames[j+1].y - frames[j].y
                        post_contact_dist += np.sqrt(dx**2 + dy**2)

                    # Convert to approximate yards (field width ~53.3 yards)
                    metrics.yards_after_contact_proxy = post_contact_dist * 53.3
                    break

    def _calculate_dl_metrics(self, metrics: PlayMetrics,
                               frames: List[PlayerFrame], fps: float):
        """DL-specific: get-off time, penetration depth"""

        # Get-off time = first step (already calculated)
        metrics.get_off_time = metrics.first_step_sec

        # Penetration depth: max forward (negative Y in most camera angles) movement
        if len(frames) >= 2:
            start_y = frames[0].y
            min_y = min(f.y for f in frames)
            metrics.penetration_depth = start_y - min_y  # Positive = good penetration

    def _calculate_qb_metrics(self, metrics: PlayMetrics,
                               frames: List[PlayerFrame], fps: float):
        """QB-specific: movement metrics only (time-to-throw requires ball tracking)"""

        # For highlight reels, QB metrics are limited to movement-based stats
        # Time-to-throw and pocket pressure require full game film with ball tracking

        # NOTE: For demo purposes, we only output common metrics (first_step, velocity, etc.)
        # QB-specific metrics like time_to_throw removed - not reliable from highlights
        pass

    def _calculate_db_metrics(self, metrics: PlayMetrics,
                               frames: List[PlayerFrame], fps: float):
        """DB-specific: closing speed, break reaction time"""

        # Closing speed: max speed during play (usually when breaking on ball)
        metrics.closing_speed = metrics.max_velocity

        # Break reaction time: similar to first step for DBs
        metrics.break_reaction_time = metrics.first_step_sec

    def _calculate_wr_metrics(self, metrics: PlayMetrics,
                               frames: List[PlayerFrame], fps: float):
        """WR-specific: release quickness, route speed"""
        # WR metrics similar to RB burst
        if metrics.first_step_sec is not None:
            burst_window = int(fps * 0.5)  # 0.5 second release window
            first_move_idx = next((i for i, f in enumerate(frames)
                                   if f.frame_num == metrics.first_movement_frame), 0)
            burst_frames = frames[first_move_idx:first_move_idx + burst_window]
            if burst_frames:
                metrics.burst_speed = max(f.speed for f in burst_frames)

    def _calculate_lb_metrics(self, metrics: PlayMetrics,
                               frames: List[PlayerFrame], fps: float):
        """LB-specific: read time, pursuit speed"""
        # Similar to DL
        metrics.get_off_time = metrics.first_step_sec
        metrics.closing_speed = metrics.max_velocity


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class HighlightReelProcessor:
    """Main processor for highlight reels"""

    def __init__(self, player_name: str, position: str,
                 jersey_number: Optional[int] = None, team: str = "UNKNOWN"):
        self.player_name = player_name
        self.position = position.upper()
        self.jersey_number = jersey_number
        self.team = team.upper()

        print(f"=" * 60)
        print(f"Third & 20 - Highlight Reel Processor v1.0")
        print(f"=" * 60)
        print(f"  Player: {player_name}")
        print(f"  Position: {position}")
        print(f"  Jersey: #{jersey_number}" if jersey_number else "  Jersey: Auto-detect")
        print(f"  Team: {team}")
        print(f"=" * 60)

        self.scene_detector = SceneChangeDetector()
        self.tracker = TargetPlayerTracker(target_jersey=jersey_number)
        self.metrics_calc = PositionMetricsCalculator()

    def process(self, video_path: str, output_prefix: str) -> HighlightAnalysis:
        """Process a highlight reel video"""

        print(f"\nProcessing: {video_path}")

        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        print(f"  Video: {total_frames} frames, {duration:.1f}s, {fps:.1f} fps")

        # Step 1: Detect scenes (play boundaries)
        print(f"\n[1/3] Detecting play boundaries...")
        plays = self.scene_detector.detect_scenes(video_path)

        # Step 2: Track target player through each play
        print(f"\n[2/3] Tracking {self.player_name} through {len(plays)} plays...")
        all_metrics = []

        for i, play in enumerate(plays):
            print(f"  Play {play.play_number}/{len(plays)} "
                  f"(frames {play.start_frame}-{play.end_frame}, {play.duration_sec:.1f}s)...")

            # Track player
            frames = self.tracker.track_play(video_path, play, fps)

            if len(frames) < 5:
                print(f"    WARNING: Only {len(frames)} frames tracked, skipping")
                continue

            # Calculate metrics
            metrics = self.metrics_calc.calculate(self.position, frames, play, fps)
            all_metrics.append(metrics)

            if metrics.first_step_sec:
                print(f"    First step: {metrics.first_step_sec:.3f}s, "
                      f"Max velocity: {metrics.max_velocity:.3f}" if metrics.max_velocity else "")

        # Step 3: Aggregate and output
        print(f"\n[3/3] Generating output...")

        analysis = HighlightAnalysis(
            video_path=video_path,
            player_name=self.player_name,
            position=self.position,
            jersey_number=self.jersey_number,
            team=self.team,
            total_plays=len(plays),
            plays=plays,
            metrics=all_metrics
        )

        # Calculate aggregates
        first_steps = [m.first_step_sec for m in all_metrics if m.first_step_sec]
        max_vels = [m.max_velocity for m in all_metrics if m.max_velocity]
        max_accels = [m.max_acceleration for m in all_metrics if m.max_acceleration]

        if first_steps:
            analysis.avg_first_step = np.mean(first_steps)
        if max_vels:
            analysis.avg_max_velocity = np.mean(max_vels)
        if max_accels:
            analysis.avg_acceleration = np.mean(max_accels)

        # Write outputs
        self._write_outputs(analysis, output_prefix)

        # Print summary
        self._print_summary(analysis)

        return analysis

    def _write_outputs(self, analysis: HighlightAnalysis, output_prefix: str):
        """Write CSV outputs"""

        # Per-play metrics
        plays_csv = f"{output_prefix}_play_metrics.csv"

        fieldnames = [
            'play_number', 'start_frame', 'end_frame', 'duration_sec',
            'first_step_sec', 'initial_velocity', 'max_velocity', 'avg_velocity',
            'max_acceleration', 'total_distance', 'frames_tracked', 'tracking_coverage'
        ]

        # Add position-specific fields
        if analysis.position == "RB":
            fieldnames.extend(['burst_speed', 'yards_after_contact_proxy'])
        elif analysis.position == "DL":
            fieldnames.extend(['get_off_time', 'penetration_depth'])
        elif analysis.position == "QB":
            pass  # QB-specific metrics removed - not reliable from highlights
        elif analysis.position == "DB":
            fieldnames.extend(['break_reaction_time', 'closing_speed'])

        with open(plays_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for m in analysis.metrics:
                row = {
                    'play_number': m.play_number,
                    'start_frame': m.start_frame,
                    'end_frame': m.end_frame,
                    'duration_sec': f"{m.duration_sec:.3f}",
                    'first_step_sec': f"{m.first_step_sec:.3f}" if m.first_step_sec else "",
                    'initial_velocity': f"{m.initial_velocity:.4f}" if m.initial_velocity else "",
                    'max_velocity': f"{m.max_velocity:.4f}" if m.max_velocity else "",
                    'avg_velocity': f"{m.avg_velocity:.4f}" if m.avg_velocity else "",
                    'max_acceleration': f"{m.max_acceleration:.4f}" if m.max_acceleration else "",
                    'total_distance': f"{m.total_distance:.4f}" if m.total_distance else "",
                    'frames_tracked': m.frames_tracked,
                    'tracking_coverage': f"{m.tracking_coverage:.2f}",
                    # Position specific
                    'burst_speed': f"{m.burst_speed:.4f}" if m.burst_speed else "",
                    'yards_after_contact_proxy': f"{m.yards_after_contact_proxy:.4f}" if m.yards_after_contact_proxy else "",
                    'get_off_time': f"{m.get_off_time:.3f}" if m.get_off_time else "",
                    'penetration_depth': f"{m.penetration_depth:.4f}" if m.penetration_depth else "",
                    'time_to_throw': f"{m.time_to_throw:.3f}" if m.time_to_throw else "",
                    'pocket_movement': f"{m.pocket_movement:.4f}" if m.pocket_movement else "",
                    'break_reaction_time': f"{m.break_reaction_time:.3f}" if m.break_reaction_time else "",
                    'closing_speed': f"{m.closing_speed:.4f}" if m.closing_speed else "",
                }
                writer.writerow(row)

        print(f"  Play metrics saved to: {plays_csv}")

        # Summary file
        summary_csv = f"{output_prefix}_summary.csv"
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Player Name', analysis.player_name])
            writer.writerow(['Position', analysis.position])
            writer.writerow(['Jersey Number', analysis.jersey_number or 'Auto'])
            writer.writerow(['Team', analysis.team])
            writer.writerow(['Video', Path(analysis.video_path).name])
            writer.writerow(['Total Plays Detected', analysis.total_plays])
            writer.writerow(['Plays with Tracking', len(analysis.metrics)])
            writer.writerow([''])
            writer.writerow(['Avg First Step (sec)', f"{analysis.avg_first_step:.3f}" if analysis.avg_first_step else "N/A"])
            writer.writerow(['Avg Max Velocity', f"{analysis.avg_max_velocity:.4f}" if analysis.avg_max_velocity else "N/A"])
            writer.writerow(['Avg Max Acceleration', f"{analysis.avg_acceleration:.4f}" if analysis.avg_acceleration else "N/A"])

            # Position-specific aggregates
            if analysis.position == "RB":
                bursts = [m.burst_speed for m in analysis.metrics if m.burst_speed]
                yac = [m.yards_after_contact_proxy for m in analysis.metrics if m.yards_after_contact_proxy]
                writer.writerow(['Avg Burst Speed', f"{np.mean(bursts):.4f}" if bursts else "N/A"])
                writer.writerow(['Avg YAC Proxy', f"{np.mean(yac):.4f}" if yac else "N/A"])
            elif analysis.position == "DL":
                getoffs = [m.get_off_time for m in analysis.metrics if m.get_off_time]
                pens = [m.penetration_depth for m in analysis.metrics if m.penetration_depth]
                writer.writerow(['Avg Get-Off Time', f"{np.mean(getoffs):.3f}" if getoffs else "N/A"])
                writer.writerow(['Avg Penetration', f"{np.mean(pens):.4f}" if pens else "N/A"])
            elif analysis.position == "QB":
                # QB-specific metrics removed - not reliable from highlights
                # For real game film, use the main SDI pipeline with ball tracking
                writer.writerow(['Note', 'QB metrics require full game film - use common metrics above'])

        print(f"  Summary saved to: {summary_csv}")

    def _print_summary(self, analysis: HighlightAnalysis):
        """Print summary to console"""
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {analysis.player_name} ({analysis.position})")
        print(f"{'=' * 60}")
        print(f"  Total plays detected: {analysis.total_plays}")
        print(f"  Plays with tracking: {len(analysis.metrics)}")

        if analysis.avg_first_step:
            print(f"\n  Avg First Step: {analysis.avg_first_step:.3f} sec")
        if analysis.avg_max_velocity:
            print(f"  Avg Max Velocity: {analysis.avg_max_velocity:.4f}")
        if analysis.avg_acceleration:
            print(f"  Avg Max Acceleration: {analysis.avg_acceleration:.4f}")

        # Position-specific
        if analysis.position == "RB":
            bursts = [m.burst_speed for m in analysis.metrics if m.burst_speed]
            if bursts:
                print(f"  Avg Burst Speed: {np.mean(bursts):.4f}")
        elif analysis.position == "DL":
            getoffs = [m.get_off_time for m in analysis.metrics if m.get_off_time]
            if getoffs:
                print(f"  Avg Get-Off Time: {np.mean(getoffs):.3f} sec")

        # Grade the player
        print(f"\n  SDI GRADE ESTIMATE:")
        if analysis.avg_first_step:
            if analysis.avg_first_step < 0.3:
                grade = "Elite"
            elif analysis.avg_first_step < 0.45:
                grade = "Above Average"
            elif analysis.avg_first_step < 0.6:
                grade = "Average"
            else:
                grade = "Below Average"
            print(f"    First Step: {grade}")


def main():
    parser = argparse.ArgumentParser(
        description="Third & 20 - Highlight Reel Processor"
    )
    parser.add_argument("video", help="Path to highlight reel video")
    parser.add_argument("--player-name", "-p", required=True, help="Player name")
    parser.add_argument("--position", required=True,
                        choices=["RB", "DL", "QB", "WR", "DB", "LB", "OL", "TE"],
                        help="Player position")
    parser.add_argument("--jersey", "-j", type=int, help="Jersey number (optional)")
    parser.add_argument("--team", "-t", default="UNKNOWN", help="Team name")
    parser.add_argument("-o", "--output", required=True, help="Output file prefix")

    args = parser.parse_args()

    processor = HighlightReelProcessor(
        player_name=args.player_name,
        position=args.position,
        jersey_number=args.jersey,
        team=args.team
    )

    processor.process(args.video, args.output)


if __name__ == "__main__":
    main()
