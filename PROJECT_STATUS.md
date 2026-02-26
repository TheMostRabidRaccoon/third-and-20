# Third & 20 - Project Status

**Owner:** RRI LLC (Rabid Raccoon Intelligence)
**Last Updated:** 2026-02-25
**Patent Status:** Pending

---

## What This Is

A sideline coaching intelligence system for high school football. Converts game film into quantified scouting data and real-time play recommendations without human tagging.

**Core thesis:** Treat film as a physics signal, not a coaching clipboard. Measure geometry, timing, and movement. Let coaches interpret.

---

## Architecture (5 Layers)

```
Layer 1: SENSING        Video → raw detections (YOLO + ByteTrack)
Layer 2: CALIBRATION    Pixel coords → field coords (homography) [NOT BUILT]
Layer 3: STATE          PlayState keystone (LOS, drive_dir, possession)
Layer 4: INFERENCE      Geometric facts → scheme classification
Layer 5: PERSPECTIVE    Neutral data → team-specific view for coaches
```

---

## What Works

### CV Pipeline (v5)
| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Snap detection | `third_and_20_cv_v2_fixed.py` | **Working** | Detects snap frame, QB decision point, latency |
| Player detection | `player_tracker.py` (YOLO) | **Working** | YOLOv8 nano, detects players per frame |
| Player tracking | `player_tracker.py` (ByteTrack) | **Working** | Maintains identity across frames |
| Jersey color | `color_classifier.py` | **Working** | HSV-based team separation |
| Jersey OCR | `sdi_pipeline_v5_with_defense.py` | **Noisy** | EasyOCR struggles on compressed HS film |
| PlayState | `play_state.py` | **Working, not integrated** | Keystone module. Tested. Needs wiring into v5 |
| Defensive inference | `defensive_inference.py` | **Working** | Box count, front, safety alignment, coverage shell |
| SDI metrics | `sdi_pipeline_v5_with_defense.py` | **Working** | First step, recognition latency, movement metrics |

### Analytics
| Component | File | Status |
|-----------|------|--------|
| Game aggregation | `game_aggregator.py` | **Working** |
| Hudl import/merge | `hudl_analytics_engine.py`, `complete_analytics_merger.py` | **Working** |
| Season analytics | `analytics_engine.py` | **Working** |

### Recommendation Engine
| Component | File | Status |
|-----------|------|--------|
| Data models | `recommendation/models.py` | **Working** |
| Core engine | `recommendation/engine.py` | **Working** |
| Hudl+CV merger | `recommendation/data_merger.py` | **Working** |
| Formation parser | `recommendation/formation_parser.py` | **Working** |
| CLI interface | `recommendation/cli.py` | **Working** |

---

## What Doesn't Work Yet

### P0 - Blocking

| Gap | Why It Matters | Path Forward |
|-----|----------------|--------------|
| **PlayState not wired into v5 pipeline** | Old circular LOS logic still runs in player_tracker._classify_teams(). PlayState fix exists but isn't integrated. | Wire infer_play_state() into sdi_pipeline_v5. Replace _classify_teams() with assign_roles_from_play_state(). |
| **No field calibration (homography)** | All coords are normalized 0-1 pixels, not yards. Thresholds are arbitrary. Metrics aren't comparable across clips/games. | Build field_calibration.py: detect yard lines → compute homography → transform coords to yards. |
| **No ball detection** | Ball position is a stub returning (0.5, 0.5). | Needs CV model or heuristic (center/QB cluster). |

### P1 - High Priority

| Gap | Why It Matters |
|-----|----------------|
| Jersey OCR accuracy | Many players unidentified. EasyOCR poor on compressed film. |
| Validation against ground truth | No coach has verified corrected (PlayState) output yet. |
| Coverage shell accuracy | Heuristic only. Pre-snap geometry ≠ post-snap coverage call. System should output geometry, not scheme labels. |

### P2 - Future

| Gap | Notes |
|-----|-------|
| Real-time 12-second loop | Requires P0 complete. Offline/online split designed but not built. |
| Scoreboard OCR | Brush scoreboard too bright. Alternative: manual metadata. |
| Public film ingestion | YouTube/opponent Hudl processing. Same pipeline, different source. |

---

## Products (What We Sell)

### Product 1: Pre-Season Opponent Intelligence
- Run opponent's public film through pipeline
- Output: structural constraint profiles per opponent
- Automated, no human tagging required
- **Status:** Pipeline produces output. Needs validation with PlayState fix.

### Product 2: Player Development (SDI Cards)
- Longitudinal cognitive metrics per athlete
- Recognition latency, first step, assignment accuracy trends
- Recruitable data for college scouts
- **Status:** Metrics compute. Need season-long data from a pilot school.

### Product 3: In-Game Play Recommendations
- Real-time top 3 plays with success probability
- Based on pre-loaded opponent tendencies + live game state
- Target: <12 seconds from snap to recommendation
- **Status:** Recommendation engine built. Needs live integration + field calibration.

### Product 4: NIL Compliance (Ohio OHSAA)
- **Status:** Spec only. Not built.

---

## Data Assets

| Asset | Location | Size | Content |
|-------|----------|------|---------|
| Brush 2025 game film | `data/games/` | ~7 GB | Hudl video clips |
| Opponent analysis CSVs | `data/Brush Demo stats/` | 15 files | SDI, defensive, tendency data for Shaker, CCC, Hoban, Shaw, Garfield |
| Hudl play-by-play | `data/hudl_imports/` | Multiple CSVs | Traditional stats |
| Rosters | `data/rosters/` | Multiple CSVs | Player/jersey mapping |
| Manual defense tags | `data/defense_manual/` | Reference | Human-labeled defensive alignments |
| Formation playbooks | `data/formations/` | 11 PDFs | Brush offensive playbook |
| Pipeline output | `output/` | 9 CSVs (232K) | Generated analytics |
| Full game CV results | `data/full_game_cv_results.json` | 1 file | Complete pipeline output |

---

## Key Decisions Made

1. **Physics-first, not coaching-first.** System measures geometry. Coaches interpret. We don't output "Cover 3." We output "1 deep safety at 12 yards, 7 in box, 4 on line."

2. **Neutral fact-finder.** No ego-anchoring. No "our offense vs their defense." All data is team-agnostic until the view layer.

3. **PlayState is the keystone.** Nothing computes without LOS, drive_dir, and possession_team_id. These are inferred from geometry, not assumed from labels.

4. **High school market entry.** $3,500-5,000 hardware budget. Single vendor. No staff required.

5. **Pre-snap geometry, not post-snap coverage.** The system measures what it can see before the play. Coverage labels are coaching interpretations that belong in the view layer.

---

## Immediate Next Steps

1. **Wire PlayState into v5 pipeline** - Replace circular LOS logic
2. **Run corrected pipeline on 2-3 games** - Produce fresh output
3. **Self-validate output** - Use validation guide to check geometric accuracy
4. **Take the test school meeting** - Bring pre-season opponent profile as demo
5. **Initialize GitHub repo** - Version control, proper .gitignore

---

## File Map

```
third_and_20/
├── src/
│   ├── pipelines/           # 15 files, ~8,700 lines
│   │   ├── sdi_pipeline_v5_with_defense.py  ← MAIN PIPELINE
│   │   ├── play_state.py                    ← KEYSTONE MODULE
│   │   ├── player_tracker.py                ← CV detection/tracking
│   │   ├── defensive_inference.py           ← scheme inference
│   │   ├── third_and_20_cv_v2_fixed.py      ← snap detection
│   │   ├── analytics_engine.py              ← output aggregation
│   │   ├── game_aggregator.py               ← game-level stats
│   │   ├── complete_analytics_merger.py     ← Hudl + CV merge
│   │   ├── hudl_analytics_engine.py         ← Hudl import
│   │   ├── run_game_analysis.py             ← entry point
│   │   ├── batch_processor.py               ← multi-clip
│   │   ├── highlight_reel_processor.py      ← pro highlights
│   │   ├── playstate_diagnostic.py          ← debug tool
│   │   ├── sdi_pipeline_v4.py               ← legacy
│   │   └── test_aggregator.py               ← tests
│   ├── recommendation/      # 6 files, ~1,700 lines
│   │   ├── engine.py                        ← core rec engine
│   │   ├── models.py                        ← data models
│   │   ├── cli.py                           ← interactive CLI
│   │   ├── data_merger.py                   ← Hudl+CV merge
│   │   ├── formation_parser.py              ← formation analysis
│   │   └── ARCHITECTURE.md                  ← design spec
│   └── tools/               # 5 files, ~1,100 lines
│       ├── color_classifier.py              ← jersey color (HSV)
│       ├── jersey_ocr_test.py               ← OCR validation
│       ├── maxpreps_roster_scraper.py       ← roster scraping
│       ├── raccoon_pack_v3.py               ← multi-AI query
│       └── sdi_utils.py                     ← shared grading
├── data/                    # ~7.8 GB
├── output/                  # 9 CSV files
├── docs/                    # 15 files (specs, architecture, reports)
├── yolov8n.pt               # YOLO model weights
├── PROJECT_STATUS.md         ← THIS FILE
├── VALIDATION_GUIDE.md       ← How to check CV output
├── requirements.txt
└── .gitignore
```

---

## Dependencies

```
opencv-python
ultralytics
supervision
easyocr
pandas
numpy
```

---

## Codebase Stats

| Metric | Value |
|--------|-------|
| Total Python files | 26 |
| Total lines of code | ~11,600 |
| Data volume | ~7.8 GB |
| Pre-trained models | 1 (YOLOv8 nano) |
| Pipeline versions | v4 (legacy), v5 (current) |
| Last pipeline run | Jan 2026 (Brush demo) |
