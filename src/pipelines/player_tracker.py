#!/usr/bin/env python3
"""
Third & 20 - Player Tracking Module v1.0
Adds player detection and tracking to extract per-player metrics

Dependencies:
    pip install ultralytics opencv-python numpy supervision

This module:
1. Detects players using YOLOv8
2. Tracks players across frames using ByteTrack
3. Extracts x,y positions per frame per player
4. Calculates per-player movement metrics (first step, total distance, etc.)
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import csv
import json

# Try importing YOLO - will fail gracefully if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")

# Try importing supervision for tracking
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("WARNING: supervision not installed. Run: pip install supervision")


@dataclass
class PlayerFrame:
    """Single frame data for one player"""
    frame_num: int
    x: float  # center x position (normalized 0-1)
    y: float  # center y position (normalized 0-1)
    width: float  # bbox width (normalized)
    height: float  # bbox height (normalized)
    confidence: float


@dataclass
class PlayerTrack:
    """Full track for one player across a clip"""
    track_id: int
    frames: List[PlayerFrame] = field(default_factory=list)
    team: str = "unknown"  # Will be classified by jersey color
    position_estimate: str = "unknown"  # Estimated from location/movement
    
    @property
    def first_frame(self) -> int:
        return self.frames[0].frame_num if self.frames else 0
    
    @property
    def last_frame(self) -> int:
        return self.frames[-1].frame_num if self.frames else 0
    
    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame
    
    def get_position_at_frame(self, frame_num: int, tolerance: int = 3) -> Optional[Tuple[float, float]]:
        """
        Get x,y position at specific frame.

        Args:
            frame_num: Target frame number
            tolerance: Max frames away to accept (default 3)

        Returns exact match first, then nearest within tolerance, then None.
        """
        # First try exact match
        for f in self.frames:
            if f.frame_num == frame_num:
                return (f.x, f.y)

        # Try nearest frame within tolerance
        best_frame = None
        best_dist = tolerance + 1
        for f in self.frames:
            dist = abs(f.frame_num - frame_num)
            if dist <= tolerance and dist < best_dist:
                best_frame = f
                best_dist = dist

        if best_frame:
            return (best_frame.x, best_frame.y)

        return None
    
    def get_velocity_at_frame(self, frame_num: int, fps: float, window: int = 3) -> Optional[Tuple[float, float]]:
        """Calculate velocity (pixels/sec) at frame"""
        positions = [(f.frame_num, f.x, f.y) for f in self.frames 
                     if abs(f.frame_num - frame_num) <= window]
        if len(positions) < 2:
            return None
        
        # Simple linear regression for velocity
        frames = [p[0] for p in positions]
        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]
        
        vx = (xs[-1] - xs[0]) / ((frames[-1] - frames[0]) / fps) if frames[-1] != frames[0] else 0
        vy = (ys[-1] - ys[0]) / ((frames[-1] - frames[0]) / fps) if frames[-1] != frames[0] else 0
        
        return (vx, vy)
    
    def calculate_first_step_time(self, snap_frame: int, fps: float, 
                                   movement_threshold: float = 0.01) -> Optional[float]:
        """
        Calculate time from snap to first significant movement
        This is key for D-line first step quickness
        """
        if not self.frames:
            return None
            
        # Find position at snap
        snap_pos = self.get_position_at_frame(snap_frame)
        if not snap_pos:
            # Find closest frame to snap
            closest = min(self.frames, key=lambda f: abs(f.frame_num - snap_frame))
            snap_pos = (closest.x, closest.y)
            
        # Find first frame with significant movement from snap position
        for f in self.frames:
            if f.frame_num <= snap_frame:
                continue
            dist = np.sqrt((f.x - snap_pos[0])**2 + (f.y - snap_pos[1])**2)
            if dist > movement_threshold:
                return (f.frame_num - snap_frame) / fps
                
        return None
    
    def calculate_total_distance(self, start_frame: int = None, end_frame: int = None) -> float:
        """Calculate total distance traveled (normalized units)"""
        if len(self.frames) < 2:
            return 0.0
            
        frames = self.frames
        if start_frame:
            frames = [f for f in frames if f.frame_num >= start_frame]
        if end_frame:
            frames = [f for f in frames if f.frame_num <= end_frame]
            
        total = 0.0
        for i in range(1, len(frames)):
            dx = frames[i].x - frames[i-1].x
            dy = frames[i].y - frames[i-1].y
            total += np.sqrt(dx*dx + dy*dy)
        return total


@dataclass
class PlayTracking:
    """Full tracking data for one play/clip"""
    clip_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    snap_frame: Optional[int] = None
    players: Dict[int, PlayerTrack] = field(default_factory=dict)
    
    def get_all_positions_at_frame(self, frame_num: int) -> Dict[int, Tuple[float, float]]:
        """Get all player positions at a specific frame"""
        positions = {}
        for track_id, track in self.players.items():
            pos = track.get_position_at_frame(frame_num)
            if pos:
                positions[track_id] = pos
        return positions
    
    def calculate_team_centroid(self, frame_num: int, team: str) -> Optional[Tuple[float, float]]:
        """Calculate center of mass for one team at a frame"""
        positions = []
        for track in self.players.values():
            if track.team == team:
                pos = track.get_position_at_frame(frame_num)
                if pos:
                    positions.append(pos)
        if not positions:
            return None
        return (np.mean([p[0] for p in positions]), np.mean([p[1] for p in positions]))
    
    def to_dict(self) -> dict:
        """Export to dictionary for JSON serialization"""
        return {
            'clip_path': self.clip_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'snap_frame': self.snap_frame,
            'num_players_tracked': len(self.players),
            'players': {
                track_id: {
                    'track_id': track.track_id,
                    'team': track.team,
                    'position_estimate': track.position_estimate,
                    'first_frame': track.first_frame,
                    'last_frame': track.last_frame,
                    'num_frames': len(track.frames)
                }
                for track_id, track in self.players.items()
            }
        }


class PlayerTracker:
    """
    Main player tracking class using YOLOv8 + ByteTrack
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.3):
        self.confidence = confidence
        self.model = None
        self.tracker = None
        
        if YOLO_AVAILABLE:
            # Use YOLOv8 nano for speed, can upgrade to yolov8s/m for accuracy
            self.model = YOLO(model_path)
        
        if SUPERVISION_AVAILABLE:
            # Tuned for HS football film - prioritize track continuity
            # - much longer buffer to survive occlusions at LOS
            # - keep matching threshold moderate to avoid false merges
            # - require more consecutive frames to reduce false tracks
            self.tracker = sv.ByteTrack(
                lost_track_buffer=90,            # frames before track deleted (was 30)
                minimum_matching_threshold=0.7,  # IoU for re-association (was 0.8)
                track_activation_threshold=0.25, # detection conf to start track
                minimum_consecutive_frames=2,    # require 2 frames to confirm track
                frame_rate=30
            )
    
    def track_clip(self, clip_path: str, snap_frame: Optional[int] = None,
                   start_frame: int = 0, max_frames: int = None) -> PlayTracking:
        """
        Process a clip and extract player tracking data
        
        Args:
            clip_path: Path to video file
            snap_frame: Frame number of snap (from snap detector)
            start_frame: Frame to start processing (skip pre-snap if known)
            max_frames: Maximum frames to process (None = all)
        """
        cap = cv2.VideoCapture(str(clip_path))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        result = PlayTracking(
            clip_path=str(clip_path),
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            snap_frame=snap_frame
        )
        
        if not self.model:
            print("ERROR: YOLO model not loaded")
            cap.release()
            return result
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_num = start_frame
        frames_processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and frames_processed >= max_frames:
                break
            
            # Run YOLO detection - filter for person class (0)
            results = self.model(frame, conf=self.confidence, classes=[0], verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                detections = self._results_to_detections(results[0], width, height)
                
                # Update tracker
                if self.tracker and SUPERVISION_AVAILABLE:
                    detections = self.tracker.update_with_detections(detections)
                
                # Store detections
                self._store_detections(result, detections, frame_num, width, height)
            
            frame_num += 1
            frames_processed += 1
            
            if frames_processed % 100 == 0:
                print(f"  Processed {frames_processed} frames...")
        
        cap.release()
        
        # Post-process: classify teams by jersey color, estimate positions
        self._classify_teams(result, clip_path)
        
        return result
    
    def _results_to_detections(self, results, width: int, height: int):
        """Convert YOLO results to supervision Detections"""
        if not SUPERVISION_AVAILABLE:
            return None
            
        boxes = results.boxes.xyxy.cpu().numpy()
        confidence = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        return sv.Detections(
            xyxy=boxes,
            confidence=confidence,
            class_id=class_ids
        )
    
    def _store_detections(self, result: PlayTracking, detections, 
                          frame_num: int, width: int, height: int):
        """Store detections in the PlayTracking structure"""
        if detections is None:
            return
            
        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            conf = detections.confidence[i] if detections.confidence is not None else 0.5
            
            # Get track ID (from ByteTrack) or use detection index
            if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                track_id = int(detections.tracker_id[i])
            else:
                track_id = i
            
            # Normalize coordinates
            x_center = ((box[0] + box[2]) / 2) / width
            y_center = ((box[1] + box[3]) / 2) / height
            box_width = (box[2] - box[0]) / width
            box_height = (box[3] - box[1]) / height
            
            # Create or update track
            if track_id not in result.players:
                result.players[track_id] = PlayerTrack(track_id=track_id)
            
            result.players[track_id].frames.append(PlayerFrame(
                frame_num=frame_num,
                x=x_center,
                y=y_center,
                width=box_width,
                height=box_height,
                confidence=conf
            ))
    
    def _classify_teams(self, result: PlayTracking, clip_path: str):
        """
        Classify players into teams based on position at snap.

        For endzone camera (typical game film):
        - Y-axis represents depth (offense at lower Y, defense at higher Y)
        - X-axis represents sideline to sideline

        We find the line of scrimmage (cluster of players) and classify
        players above/below it as defense/offense.
        """
        if result.snap_frame is None:
            return

        positions_at_snap = result.get_all_positions_at_frame(result.snap_frame)
        if not positions_at_snap:
            return

        # Find line of scrimmage using Y-coordinate clustering
        # The LOS should have a high density of players (O-line + D-line)
        y_positions = [pos[1] for pos in positions_at_snap.values()]

        if not y_positions:
            return

        # Estimate LOS as the median Y position (most players cluster here)
        los_y = np.median(y_positions)

        # Also look for the densest cluster of Y positions
        # This is where the lines are
        y_sorted = sorted(y_positions)
        best_los = los_y
        best_density = 0

        window = 0.1  # 10% of frame height
        for i, y in enumerate(y_sorted):
            # Count players within window
            count = sum(1 for yp in y_positions if abs(yp - y) < window)
            if count > best_density:
                best_density = count
                best_los = y

        los_y = best_los

        # Classify based on position relative to LOS
        # For endzone view: offense is typically at bottom (lower Y)
        # defense is at top (higher Y)
        for track_id, pos in positions_at_snap.items():
            if track_id not in result.players:
                continue

            x, y = pos

            # Calculate depth from LOS
            depth_from_los = y - los_y

            # Team classification
            # Players above LOS (higher Y) = defense
            # Players below LOS (lower Y) = offense
            # Players on LOS need additional context
            if depth_from_los > 0.02:  # More than 2% above LOS
                result.players[track_id].team = "defense"
            elif depth_from_los < -0.02:  # More than 2% below LOS
                result.players[track_id].team = "offense"
            else:
                # On the line - use x-position clustering
                # Typically 5 offensive linemen are more centered
                # than defensive linemen spread wider
                result.players[track_id].team = "unknown"  # Needs jersey color

            # Position estimate based on coordinates
            width_from_center = abs(x - 0.5)

            if width_from_center > 0.3:
                result.players[track_id].position_estimate = "wide"
            elif width_from_center < 0.15:
                result.players[track_id].position_estimate = "interior"
            else:
                result.players[track_id].position_estimate = "slot"


def calculate_player_metrics(tracking: PlayTracking) -> Dict[int, dict]:
    """
    Calculate per-player metrics from tracking data
    
    Returns dict of track_id -> metrics
    """
    metrics = {}
    
    for track_id, track in tracking.players.items():
        player_metrics = {
            'track_id': track_id,
            'team': track.team,
            'position': track.position_estimate,
            'frames_tracked': len(track.frames),
        }
        
        if tracking.snap_frame:
            # First step quickness (time to first movement after snap)
            first_step = track.calculate_first_step_time(
                tracking.snap_frame, tracking.fps
            )
            player_metrics['first_step_sec'] = first_step
            
            # Total distance post-snap
            post_snap_dist = track.calculate_total_distance(
                start_frame=tracking.snap_frame
            )
            player_metrics['post_snap_distance'] = post_snap_dist
            
            # Initial velocity at snap
            vel = track.get_velocity_at_frame(tracking.snap_frame + 5, tracking.fps)
            if vel:
                player_metrics['initial_velocity'] = np.sqrt(vel[0]**2 + vel[1]**2)
        
        metrics[track_id] = player_metrics
    
    return metrics


def batch_track(clip_paths: List[str], snap_data: Dict[str, int] = None,
                output_json: str = None) -> List[PlayTracking]:
    """
    Process multiple clips and extract tracking data
    
    Args:
        clip_paths: List of video file paths
        snap_data: Dict mapping clip filename to snap frame number
        output_json: Path to save results as JSON
    """
    tracker = PlayerTracker()
    results = []
    
    for path in clip_paths:
        filename = Path(path).name
        print(f"Tracking: {filename}...")
        
        snap_frame = None
        if snap_data and filename in snap_data:
            snap_frame = snap_data[filename]
        
        try:
            # Only process frames around the snap for efficiency
            start_frame = max(0, (snap_frame or 0) - 30)
            max_frames = 150  # ~5 seconds at 30fps
            
            tracking = tracker.track_clip(
                path, 
                snap_frame=snap_frame,
                start_frame=start_frame,
                max_frames=max_frames
            )
            results.append(tracking)
            
            print(f"  Found {len(tracking.players)} player tracks")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    if output_json and results:
        output_data = [r.to_dict() for r in results]
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_json}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python player_tracker.py <clip.mp4> [snap_frame]")
        print("Example: python player_tracker.py game_clip.mp4 150")
        sys.exit(1)
    
    clip_path = sys.argv[1]
    snap_frame = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"Processing: {clip_path}")
    if snap_frame:
        print(f"Snap frame: {snap_frame}")
    
    tracker = PlayerTracker()
    result = tracker.track_clip(clip_path, snap_frame=snap_frame)
    
    print(f"\nFound {len(result.players)} player tracks")
    
    # Calculate and display metrics
    metrics = calculate_player_metrics(result)
    
    print("\nPlayer Metrics:")
    print("-" * 60)
    for track_id, m in metrics.items():
        print(f"Track {track_id}: {m['team']} {m['position']}")
        if m.get('first_step_sec'):
            print(f"  First step: {m['first_step_sec']:.3f}s")
        if m.get('post_snap_distance'):
            print(f"  Distance: {m['post_snap_distance']:.3f}")
        if m.get('initial_velocity'):
            print(f"  Initial velocity: {m['initial_velocity']:.3f}")
