#!/usr/bin/env python3
"""
Jersey Number OCR Test
Tests OCR capability on game film frames

Usage:
    python jersey_ocr_test.py <video_file.mp4> [frame_number]
    
Example:
    python jersey_ocr_test.py "Tight - Clip 001.mp4" 379
    
Dependencies:
    pip install opencv-python easyocr pytesseract pillow
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Try importing OCR libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("easyocr not installed. Run: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("pytesseract not installed. Run: pip install pytesseract")


def extract_frame(video_path: str, frame_num: int = None) -> np.ndarray:
    """Extract a single frame from video"""
    cap = cv2.VideoCapture(video_path)
    
    if frame_num is None:
        # Get frame from middle of video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_num}")
    
    return frame


def preprocess_for_ocr(frame: np.ndarray) -> list:
    """
    Create multiple preprocessed versions of the frame
    Different preprocessing helps with different conditions
    """
    versions = []
    
    # Original
    versions.append(("original", frame))
    
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    versions.append(("grayscale", gray))
    
    # High contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    versions.append(("high_contrast", contrast))
    
    # Threshold (binary)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    versions.append(("threshold", thresh))
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    versions.append(("adaptive_threshold", adaptive))
    
    # Edge enhancement
    edges = cv2.Canny(gray, 50, 150)
    versions.append(("edges", edges))
    
    return versions


def run_easyocr(frame: np.ndarray, name: str = "") -> list:
    """Run EasyOCR on frame"""
    if not EASYOCR_AVAILABLE:
        return []
    
    reader = easyocr.Reader(['en'], gpu=False)
    
    # EasyOCR works best with RGB
    if len(frame.shape) == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb = frame
    
    results = reader.readtext(rgb)
    
    # Filter for likely jersey numbers (1-2 digit numbers)
    jersey_candidates = []
    for (bbox, text, confidence) in results:
        # Clean text
        clean = ''.join(c for c in text if c.isdigit())
        if clean and len(clean) <= 2 and confidence > 0.3:
            jersey_candidates.append({
                'text': clean,
                'original': text,
                'confidence': confidence,
                'bbox': bbox,
                'method': f'easyocr_{name}'
            })
    
    return jersey_candidates


def run_tesseract(frame: np.ndarray, name: str = "") -> list:
    """Run Tesseract OCR on frame"""
    if not TESSERACT_AVAILABLE:
        return []
    
    # Configure for digits only
    config = '--psm 11 -c tessedit_char_whitelist=0123456789'
    
    try:
        text = pytesseract.image_to_string(frame, config=config)
        
        # Also get detailed data
        data = pytesseract.image_to_data(frame, config=config, output_type=pytesseract.Output.DICT)
        
        jersey_candidates = []
        for i, txt in enumerate(data['text']):
            clean = ''.join(c for c in txt if c.isdigit())
            if clean and len(clean) <= 2:
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                if conf > 30:
                    jersey_candidates.append({
                        'text': clean,
                        'original': txt,
                        'confidence': conf / 100,
                        'bbox': (data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]),
                        'method': f'tesseract_{name}'
                    })
        
        return jersey_candidates
    except Exception as e:
        print(f"Tesseract error: {e}")
        return []


def detect_jersey_numbers(video_path: str, frame_num: int = None) -> dict:
    """
    Main function to detect jersey numbers in a video frame
    """
    print(f"\n{'='*60}")
    print(f"JERSEY OCR TEST")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    
    # Extract frame
    frame = extract_frame(video_path, frame_num)
    height, width = frame.shape[:2]
    print(f"Frame: {frame_num} ({width}x{height})")
    
    # Save original frame
    output_dir = Path("ocr_test_output")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "original_frame.jpg"), frame)
    print(f"\nSaved frame to: {output_dir / 'original_frame.jpg'}")
    
    # Preprocess
    preprocessed = preprocess_for_ocr(frame)
    
    # Run OCR on each version
    all_results = []
    
    for name, processed in preprocessed:
        print(f"\nProcessing: {name}...")
        
        # Save preprocessed version
        cv2.imwrite(str(output_dir / f"preprocessed_{name}.jpg"), processed)
        
        # EasyOCR
        if EASYOCR_AVAILABLE:
            easy_results = run_easyocr(processed, name)
            all_results.extend(easy_results)
            if easy_results:
                print(f"  EasyOCR found: {[r['text'] for r in easy_results]}")
        
        # Tesseract
        if TESSERACT_AVAILABLE:
            tess_results = run_tesseract(processed, name)
            all_results.extend(tess_results)
            if tess_results:
                print(f"  Tesseract found: {[r['text'] for r in tess_results]}")
    
    # Deduplicate and rank results
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Count occurrences of each number
    number_counts = {}
    for r in all_results:
        num = r['text']
        if num not in number_counts:
            number_counts[num] = {'count': 0, 'max_conf': 0, 'methods': set()}
        number_counts[num]['count'] += 1
        number_counts[num]['max_conf'] = max(number_counts[num]['max_conf'], r['confidence'])
        number_counts[num]['methods'].add(r['method'].split('_')[0])
    
    # Sort by count then confidence
    sorted_numbers = sorted(number_counts.items(), 
                           key=lambda x: (x[1]['count'], x[1]['max_conf']), 
                           reverse=True)
    
    print(f"\nDetected numbers (sorted by frequency):")
    print("-" * 40)
    for num, data in sorted_numbers[:20]:  # Top 20
        methods = ', '.join(data['methods'])
        print(f"  #{num:>2} - found {data['count']}x, conf: {data['max_conf']:.2f}, methods: {methods}")
    
    # Likely jersey numbers (valid range 1-99)
    jersey_numbers = [num for num, _ in sorted_numbers 
                      if num.isdigit() and 1 <= int(num) <= 99]
    
    print(f"\n✓ Likely jersey numbers: {jersey_numbers[:15]}")
    
    return {
        'frame_num': frame_num,
        'all_detections': all_results,
        'number_counts': number_counts,
        'likely_jerseys': jersey_numbers
    }


def annotate_frame_with_detections(video_path: str, frame_num: int, results: dict):
    """Draw bounding boxes around detected numbers"""
    frame = extract_frame(video_path, frame_num)
    
    for detection in results['all_detections']:
        bbox = detection['bbox']
        text = detection['text']
        conf = detection['confidence']
        
        # Handle different bbox formats
        if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            # Tesseract format: (left, top, width, height)
            x, y, w, h = [int(v) for v in bbox]
            pt1, pt2 = (x, y), (x + w, y + h)
        else:
            # EasyOCR format: list of 4 points
            pts = np.array(bbox, dtype=np.int32)
            pt1 = tuple(pts.min(axis=0))
            pt2 = tuple(pts.max(axis=0))
        
        # Draw rectangle and label
        color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
        cv2.rectangle(frame, pt1, pt2, color, 2)
        cv2.putText(frame, f"#{text} ({conf:.2f})", (pt1[0], pt1[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    output_path = Path("ocr_test_output") / "annotated_frame.jpg"
    cv2.imwrite(str(output_path), frame)
    print(f"\nSaved annotated frame to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python jersey_ocr_test.py <video_file.mp4> [frame_number]")
        print("\nExample:")
        print('  python jersey_ocr_test.py "Tight - Clip 001.mp4" 379')
        sys.exit(1)
    
    video_path = sys.argv[1]
    frame_num = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Check dependencies
    if not EASYOCR_AVAILABLE and not TESSERACT_AVAILABLE:
        print("\nERROR: No OCR library available!")
        print("Install at least one:")
        print("  pip install easyocr")
        print("  pip install pytesseract")
        print("\nFor pytesseract, also install Tesseract:")
        print("  brew install tesseract  # macOS")
        sys.exit(1)
    
    # Run detection
    results = detect_jersey_numbers(video_path, frame_num)
    
    # Create annotated frame
    annotate_frame_with_detections(video_path, frame_num, results)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Check the 'ocr_test_output' folder for images")
