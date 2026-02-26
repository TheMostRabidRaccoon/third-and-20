#!/usr/bin/env python3
"""
Third & 20 - Batch Play Processor v2
CV is source of truth. Greg's data is sanity check only.

Usage:
    python batch_processor.py /path/to/clips/ --output results.csv
    python batch_processor.py /path/to/clips/ --manual greg_timing.csv --output results.csv
"""

import argparse
import csv
from pathlib import Path
from third_and_20_cv_v2 import SnapDetectorV2, PlayAnalysis
from typing import List, Dict, Optional
import sys

def load_manual_timing(csv_path: str) -> Dict[int, dict]:
    """
    Load Greg's manual timing data.
    Key by play number extracted from his data.
    """
    manual = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                play_num = int(row.get('play_num', 0))
                if play_num > 0:
                    manual[play_num] = {
                        'game': row.get('game', ''),
                        'formation': row.get('formation', ''),
                        'snap_time': parse_time(row.get('snap_time', '')),
                        'decision_time': parse_time(row.get('decision_time', '')),
                        'notes': row.get('notes', '')
                    }
            except ValueError:
                continue
    return manual


def parse_time(time_str: str) -> Optional[float]:
    """Parse time string to seconds"""
    if not time_str:
        return None
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[-1])
        return float(time_str)
    except:
        return None


def extract_play_number(clip_path: Path) -> Optional[int]:
    """Extract play number from clip filename like 'Wide - Clip 022.mp4'"""
    name = clip_path.stem
    for part in name.replace('_', ' ').replace('-', ' ').split():
        if part.isdigit():
            return int(part)
    return None


def find_clips(clip_dir: str) -> List[Path]:
    """Find all clip files, sorted by play number"""
    clip_path = Path(clip_dir)
    clips = []
    
    for ext in ['*.mp4', '*.mov', '*.MP4', '*.MOV']:
        clips.extend(clip_path.glob(ext))
        clips.extend(clip_path.glob(f"**/{ext}"))
    
    # Sort by play number
    return sorted(set(clips), key=lambda p: extract_play_number(p) or 0)


def batch_process(
    clip_dir: str,
    manual_csv: Optional[str] = None,
    output_csv: str = "batch_results.csv"
) -> List[dict]:
    """
    Process all clips. CV is source of truth.
    Manual data used for snap validation only.
    """
    
    # Load manual data if provided (for snap sanity check only)
    manual_data = {}
    if manual_csv and Path(manual_csv).exists():
        manual_data = load_manual_timing(manual_csv)
        print(f"Loaded manual timing for {len(manual_data)} plays (snap validation only)")
    
    # Find clips
    clips = find_clips(clip_dir)
    print(f"Found {len(clips)} clips to process\n")
    
    if not clips:
        print(f"No clips found in {clip_dir}")
        return []
    
    # Process each clip
    detector = SnapDetectorV2()
    results = []
    snap_deltas = []
    
    for clip in clips:
        play_num = extract_play_number(clip)
        print(f"Play {play_num or '?':3} | {clip.name}...", end=" ")
        
        try:
            analysis = detector.analyze_clip(str(clip))
            
            result = {
                'play_num': play_num or "",
                'clip_file': clip.name,
                'duration_sec': f"{analysis.duration_sec:.2f}",
                'snap_frame': analysis.snap_frame or "",
                'snap_sec': f"{analysis.snap_time_sec:.2f}" if analysis.snap_time_sec else "",
                'decision_frame': analysis.decision_frame or "",
                'decision_sec': f"{analysis.decision_time_sec:.2f}" if analysis.decision_time_sec else "",
                'latency_sec': f"{analysis.latency_sec:.3f}" if analysis.latency_sec else "",
                'latency_frames': analysis.latency_frames or "",
                'decision_type': analysis.decision_type,
                'confidence': f"{analysis.snap_confidence:.0%}"
            }
            
            # Add manual snap comparison (sanity check only)
            if play_num and play_num in manual_data:
                m = manual_data[play_num]
                result['greg_snap_sec'] = m['snap_time'] if m['snap_time'] else ""
                result['greg_formation'] = m['formation']
                result['greg_notes'] = m['notes']
                
                if m['snap_time'] and analysis.snap_time_sec:
                    delta = analysis.snap_time_sec - m['snap_time']
                    result['snap_delta'] = f"{delta:+.2f}"
                    snap_deltas.append(abs(delta))
                    
                    # Flag large mismatches
                    if abs(delta) > 1.0:
                        result['flag'] = "⚠️ SNAP MISMATCH >1s"
                else:
                    result['snap_delta'] = ""
            
            results.append(result)
            
            # Print summary
            latency_str = f"Latency: {result['latency_sec']}s" if result['latency_sec'] else "No decision"
            delta_str = f"(Δ {result.get('snap_delta', 'N/A')})" if result.get('snap_delta') else ""
            print(f"Snap: {result['snap_sec']}s {delta_str} | {latency_str} ({result['decision_type']})")
            
            if result.get('flag'):
                print(f"         {result['flag']}")
        
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'play_num': play_num or "",
                'clip_file': clip.name,
                'error': str(e)
            })
    
    # Save results
    if results:
        # Get all fieldnames
        fieldnames = []
        for r in results:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE")
        print(f"{'='*60}")
        print(f"Clips processed: {len(results)}")
        print(f"Results saved: {output_csv}")
        
        if snap_deltas:
            print(f"\nSnap Detection vs Greg (sanity check):")
            print(f"  Mean delta: {sum(snap_deltas)/len(snap_deltas):.2f}s")
            print(f"  Max delta: {max(snap_deltas):.2f}s")
            within_threshold = sum(1 for d in snap_deltas if d < 0.5)
            print(f"  Within 0.5s: {within_threshold}/{len(snap_deltas)} ({within_threshold/len(snap_deltas)*100:.0f}%)")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Third & 20 Batch Processor - CV is truth")
    parser.add_argument("clip_dir", help="Directory containing clip files")
    parser.add_argument("--manual", help="Greg's timing CSV (snap sanity check only)")
    parser.add_argument("--output", default="cv_results.csv", help="Output CSV")
    
    args = parser.parse_args()
    batch_process(args.clip_dir, args.manual, args.output)
