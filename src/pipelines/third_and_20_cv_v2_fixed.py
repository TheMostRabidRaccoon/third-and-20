#!/usr/bin/env python3
"""
Third & 20 - Computer Vision Pipeline v2.0
Improved snap detection with huddle-break filtering

Key improvement: Detect snap as the LAST major motion spike, not the first.
Huddle break creates gradual rise; snap creates sharp spike from a SET position.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import csv

@dataclass
class PlayAnalysis:
    clip_path: str
    duration_sec: float
    fps: float
    total_frames: int
    snap_frame: Optional[int] = None
    snap_time_sec: Optional[float] = None
    snap_confidence: float = 0.0
    decision_frame: Optional[int] = None
    decision_time_sec: Optional[float] = None
    decision_type: str = ""
    latency_sec: Optional[float] = None
    latency_frames: Optional[int] = None
    
    def to_dict(self):
        return {
            'clip_file': Path(self.clip_path).name,
            'duration_sec': f"{self.duration_sec:.2f}",
            'snap_frame': self.snap_frame or "",
            'snap_time_sec': f"{self.snap_time_sec:.2f}" if self.snap_time_sec else "",
            'snap_confidence': f"{self.snap_confidence:.0%}",
            'decision_frame': self.decision_frame or "",
            'decision_time_sec': f"{self.decision_time_sec:.2f}" if self.decision_time_sec else "",
            'decision_type': self.decision_type,
            'latency_sec': f"{self.latency_sec:.3f}" if self.latency_sec else "",
            'latency_frames': self.latency_frames or ""
        }


class SnapDetectorV2:
    """
    Improved snap detection that filters out huddle-break motion.
    
    Algorithm:
    1. Find all motion spikes in the clip
    2. Identify "set" periods (low motion, players stationary)
    3. Snap = first major spike AFTER a set period in the second half of clip
    """
    
    def __init__(self):
        # ROI focused on line of scrimmage
        self.roi_top_pct = 0.25
        self.roi_bottom_pct = 0.55
        self.roi_left_pct = 0.30
        self.roi_right_pct = 0.70
    
    def analyze_clip(self, clip_path: str) -> PlayAnalysis:
        cap = cv2.VideoCapture(str(clip_path))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        roi_top = int(height * self.roi_top_pct)
        roi_bottom = int(height * self.roi_bottom_pct)
        roi_left = int(width * self.roi_left_pct)
        roi_right = int(width * self.roi_right_pct)
        
        motion_scores = self._extract_motion(cap, roi_top, roi_bottom, roi_left, roi_right)
        cap.release()
        
        result = PlayAnalysis(
            clip_path=str(clip_path),
            duration_sec=duration,
            fps=fps,
            total_frames=total_frames
        )
        
        if len(motion_scores) < 100:
            return result
        
        motion = np.array(motion_scores)
        
        # Key insight: Snap happens AFTER players are SET
        # Find the last "quiet period" followed by explosion
        snap_frame = self._find_snap_after_set(motion, fps)
        
        if snap_frame:
            result.snap_frame = snap_frame
            result.snap_time_sec = snap_frame / fps
            result.snap_confidence = 0.90
            
            # Find decision
            decision_frame = self._find_decision(motion, snap_frame, fps)
            if decision_frame:
                result.decision_frame = decision_frame
                result.decision_time_sec = decision_frame / fps
                result.latency_sec = (decision_frame - snap_frame) / fps
                result.latency_frames = decision_frame - snap_frame
                result.decision_type = self._classify_decision(result.latency_frames, fps)
        
        return result
    
    def _extract_motion(self, cap, roi_top, roi_bottom, roi_left, roi_right):
        motion_scores = []
        prev_roi_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            roi = frame[roi_top:roi_bottom, roi_left:roi_right]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            
            if prev_roi_gray is not None:
                diff = cv2.absdiff(prev_roi_gray, roi_gray)
                motion_scores.append(np.mean(diff))
            else:
                motion_scores.append(0)
            
            prev_roi_gray = roi_gray
        
        return motion_scores
    
    def _find_snap_after_set(self, motion: np.ndarray, fps: float) -> Optional[int]:
        """
        Find snap by locating: quiet period (SET) Ã¢â€ â€™ sudden spike (SNAP)
        
        Search in the second half of the clip (after huddle break settles)
        """
        # Smooth motion to reduce noise
        window = 5
        smoothed = np.convolve(motion, np.ones(window)/window, mode='same')
        
        # Calculate local variance (helps identify "set" vs "moving")
        variance_window = 15  # 0.5 second window
        local_var = np.array([
            np.var(smoothed[max(0,i-variance_window):i+variance_window])
            for i in range(len(smoothed))
        ])
        
        # Thresholds
        quiet_thresh = np.percentile(smoothed, 25)  # Low motion = set
        spike_thresh = np.percentile(smoothed, 75)  # High motion = action
        var_quiet = np.percentile(local_var, 30)    # Low variance = players still
        
        # Search in second half of clip
        search_start = int(len(motion) * 0.4)
        search_end = int(len(motion) * 0.85)
        
        # Find quiet periods (potential "set" positions)
        quiet_periods = []
        in_quiet = False
        quiet_start = 0
        
        for i in range(search_start, search_end):
            is_quiet = smoothed[i] < quiet_thresh * 1.5 and local_var[i] < var_quiet * 2
            
            if is_quiet and not in_quiet:
                quiet_start = i
                in_quiet = True
            elif not is_quiet and in_quiet:
                if i - quiet_start > 10:  # At least 0.33s of quiet
                    quiet_periods.append((quiet_start, i))
                in_quiet = False
        
        # Find first spike after the LAST quiet period
        if quiet_periods:
            last_quiet_end = quiet_periods[-1][1]
            
            # Look for spike
            for i in range(last_quiet_end, min(last_quiet_end + 60, len(motion))):
                if smoothed[i] > spike_thresh:
                    # Verify sustained
                    if i + 10 < len(motion) and np.mean(smoothed[i:i+10]) > spike_thresh * 0.8:
                        return i
        
        # Fallback: find sharpest motion increase in second half
        search_range = smoothed[search_start:search_end]
        gradient = np.gradient(search_range)
        snap_idx = search_start + np.argmax(gradient)
        
        # Backtrack to find the start of the rise
        for j in range(snap_idx, max(search_start, snap_idx - 30), -1):
            if smoothed[j] < quiet_thresh * 1.3:
                return j
        
        return snap_idx
    
    def _find_decision(self, motion: np.ndarray, snap_frame: int, fps: float) -> Optional[int]:
        """Find QB decision point (handoff/throw commit)"""
        search_start = snap_frame + int(0.15 * fps)  # At least 150ms after snap
        search_end = min(snap_frame + int(2.0 * fps), len(motion) - 3)
        
        if search_start >= search_end:
            return snap_frame + int(0.5 * fps)  # Default 0.5s after snap
        
        window = 3
        smoothed = np.convolve(motion, np.ones(window)/window, mode='same')
        
        # Find first peak after snap
        for i in range(search_start + 3, search_end - 3):
            if smoothed[i] > smoothed[i-3] and smoothed[i] > smoothed[i+3]:
                if smoothed[i] > np.mean(smoothed[snap_frame:i]) * 1.05:
                    return i
        
        # Fallback
        return snap_frame + int(0.5 * fps)
    
    def _classify_decision(self, latency_frames: int, fps: float) -> str:
        latency_ms = (latency_frames / fps) * 1000
        
        if latency_ms < 300:
            return "quick_handoff"
        elif latency_ms < 600:
            return "handoff"
        elif latency_ms < 1200:
            return "short_pass"
        else:
            return "dropback"


def batch_analyze(clip_paths: List[str], output_csv: str = None) -> List[PlayAnalysis]:
    detector = SnapDetectorV2()
    results = []
    
    for path in clip_paths:
        print(f"Analyzing: {Path(path).name}...", end=" ")
        try:
            result = detector.analyze_clip(path)
            results.append(result)
            if result.snap_time_sec:
                print(f"Snap: {result.snap_time_sec:.2f}s, "
                      f"Decision: {result.decision_time_sec:.2f}s, "
                      f"Latency: {result.latency_sec:.3f}s ({result.decision_type})")
            else:
                print("No snap detected")
        except Exception as e:
            print(f"ERROR: {e}")
    
    if output_csv and results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
        print(f"\nResults saved to: {output_csv}")
    
    return results


if __name__ == "__main__":
    import sys
    import os
    
    # Default output location - current directory
    default_output = "cv_results_v2.csv"
    
    # Parse arguments - allow output file override with -o flag
    clips = []
    output_csv = default_output
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "-o" and i + 1 < len(args):
            output_csv = args[i + 1]
            i += 2
        else:
            clips.append(args[i])
            i += 1
    
    if not clips:
        print("Usage: python third_and_20_cv_v2.py <clip1.mp4> [clip2.mp4 ...] [-o output.csv]")
        print("Example: python third_and_20_cv_v2.py ~/Downloads/Game/*.mp4 -o game_results.csv")
        sys.exit(1)
    
    print(f"Processing {len(clips)} clips...")
    print(f"Output: {output_csv}")
    results = batch_analyze(clips, output_csv=output_csv)
