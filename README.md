# Third & 20 🏈

**Rabid Raccoon Intelligence, LLC** — Sideline Coaching Intelligence System

A computer vision-driven scouting and play recommendation system for high school football. Converts game film into quantified scouting data and real-time play recommendations without human tagging.

**Core thesis:** Treat film as a physics signal, not a coaching clipboard. Measure geometry, timing, and movement. Let coaches interpret.

**Patent Status:** Pending

---

## What It Does

### CV Pipeline
- **Snap detection** — Identifies snap frame, QB decision point, and latency from raw game film
- **Player detection & tracking** — YOLOv8 + ByteTrack for persistent player identity across frames
- **Jersey color classification** — HSV-based team separation
- **Defensive scheme inference** — Box count, front alignment, safety depth, coverage shell — all from pre-snap geometry
- **SDI metrics** — First step timing, recognition latency, movement metrics per player

### Analytics
- **Game aggregation** — Cross-play, cross-game statistical rollups
- **Hudl integration** — Import and merge Hudl play-by-play data with CV pipeline output
- **Season analytics** — Longitudinal tracking across a full season

### Recommendation Engine
- **Play recommendations** — Top 3 plays with success probability based on opponent tendencies + game state
- **Formation parsing** — Automated offensive/defensive formation classification
- **Data merger** — Combines Hudl stats with CV metrics for richer analysis

---

## Architecture

```
Layer 1: SENSING        Video → raw detections (YOLO + ByteTrack)
Layer 2: CALIBRATION    Pixel coords → field coords (homography)
Layer 3: STATE          PlayState keystone (LOS, drive direction, possession)
Layer 4: INFERENCE      Geometric facts → scheme classification
Layer 5: PERSPECTIVE    Neutral data → team-specific view for coaches
```

**Key design decisions:**
- **Physics-first, not coaching-first.** The system outputs "1 deep safety at 12 yards, 7 in box, 4 on line" — not "Cover 3." Geometric facts, not scheme labels.
- **Neutral fact-finder.** All data is team-agnostic until the view layer. No ego-anchoring.
- **PlayState is the keystone.** Line of scrimmage, drive direction, and possession are inferred from geometry, not assumed from labels.
- **Pre-snap geometry, not post-snap coverage.** The system measures what it can see before the play. Coverage labels are coaching interpretations.

---

## Products

| Product | Description | Status |
|---------|-------------|--------|
| **Pre-Season Opponent Intelligence** | Run opponent's public film through pipeline → structural constraint profiles, no human tagging | Pipeline working, needs PlayState integration |
| **Player Development (SDI Cards)** | Longitudinal cognitive metrics per athlete — recognition latency, first step, assignment accuracy | Metrics compute, needs season-long pilot data |
| **In-Game Play Recommendations** | Real-time top 3 plays with success probability, <12 seconds from snap | Engine built, needs live integration |

---

## Tech Stack

Python · OpenCV · YOLOv8 (Ultralytics) · ByteTrack (Supervision) · EasyOCR · Pandas · NumPy

---

## Codebase

| Metric | Value |
|--------|-------|
| Python files | 26 |
| Lines of code | ~11,600 |
| Pipeline versions | v4 (legacy), v5 (current) |
| Pre-trained models | YOLOv8 nano |

---

## Setup

```bash
git clone https://github.com/TheMostRabidRaccoon/third-and-20.git
cd third-and-20
pip install -r requirements.txt
```

See `PROJECT_STATUS.md` for detailed component status and `VALIDATION_GUIDE.md` for output verification procedures.

---

## License

Proprietary — Rabid Raccoon Intelligence, LLC.

---

*Geometry over opinion. Measurement over intuition. Physics over clipboard.* 🦝
