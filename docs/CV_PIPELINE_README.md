# Third & 20 - Computer Vision Pipeline
## Installation & Usage Guide

---

## Files to Save

Copy these to `~/rri_server/` on your Mac:

| File | Purpose |
|------|---------|
| `third_and_20_cv_v2.py` | Core detection engine |
| `batch_processor.py` | Process multiple clips |
| `THIRD_AND_20_METRICS.md` | Metric definitions |

---

## Setup on Mac

```bash
# Navigate to your server directory
cd ~/rri_server

# Activate virtual environment
source venv/bin/activate

# Verify OpenCV is installed
python -c "import cv2; print(cv2.__version__)"

# If not installed:
pip install opencv-python numpy
```

---

## Usage

### Single Clip Analysis
```bash
python third_and_20_cv_v2.py ~/Downloads/Wide_-_Clip_022.mp4
```

Output:
```
Snap: 11.83s, Decision: 13.23s, Latency: 1.400s (dropback)
```

### Batch Processing (Full Game)
```bash
# Put all clips from one game in a folder
# Example: ~/Downloads/Erie_Clips/

python batch_processor.py ~/Downloads/Erie_Clips/ --output erie_results.csv

# With Greg's timing for snap sanity check:
python batch_processor.py ~/Downloads/Erie_Clips/ --manual greg_timing_erie.csv --output erie_results.csv
```

---

## Output CSV Columns

| Column | Description |
|--------|-------------|
| `play_num` | Extracted from clip filename |
| `clip_file` | Full filename |
| `snap_sec` | CV-detected snap time |
| `decision_sec` | CV-detected QB commit time |
| `latency_sec` | Decision - Snap (core metric) |
| `decision_type` | quick_handoff/handoff/short_pass/dropback |
| `snap_delta` | Difference from Greg's snap (sanity check) |
| `flag` | ⚠️ if snap mismatch > 1 second |

---

## Validation Results

**Erie Play 22:**
- CV Snap: 11.83s (Greg: 12.00s) = **0.17s delta** ✅
- CV Decision: 13.23s = **1.40s latency**
- Decision Type: handoff

---

## Troubleshooting

**"No snap detected"**
- Clip may be too short or have unusual camera angle
- Check if clip is corrupt: `ffprobe clip.mp4`

**Large snap delta (>1s)**
- Greg's play number may not match clip number
- CV uses clip filename as truth

**ModuleNotFoundError**
- Run: `pip install opencv-python numpy`

---

## Key Principle

**CV is source of truth.** Greg's data is sanity check only.

His decision timing is discarded (20/200 vision).
His play numbering may drift - clip filename is authoritative.
