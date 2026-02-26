# Third & 20: Neutral PlayState Architecture Analysis

**Date:** 2026-01-13
**Context:** Analysis of Grok's PlayState proposal against current pipeline, with recommendations for physics-first neutral sensing architecture.

---

## Executive Summary

The goal is to build a **neutral fact-finding system** that captures the physics of football plays without ego-anchoring (team bias). This enables:
1. Computing any stat from a single canonical data structure
2. Processing any film (opponent, public YouTube, etc.) with the same pipeline
3. Real-time decision support within ~12 seconds
4. Scouting that asks "what are they capable of?" not "what did they call?"

**Key insight:** Existing products (Hudl, etc.) are built on digitized clipboards—human workflows automated. Your approach starts from physics/sensing, which is fundamentally different and required for real-time.

---

## Part 1: Assessment of Grok's PlayState Proposal

### What Grok Proposed

A `PlayState` dataclass as a neutral sensing layer:

```python
@dataclass(frozen=True)
class PlayState:
    frame_num: int
    player_positions: List[Tuple[float, float]]  # Normalized 0-1
    ball_position: Optional[Tuple[float, float]]
    los_y: Optional[float]
    hash_marks: List[float]
    invariants: Dict[str, bool]
```

With invariant validation:
- `min_players`: At least 11 detected
- `positions_normalized`: All coordinates in valid range
- `los_valid`: Line of scrimmage detected

### Where Grok Is Right

1. **Invariants are valuable**: Current pipeline doesn't validate state before computing metrics. Bad detection produces garbage output silently. Explicit `is_valid()` checks would catch this.

2. **Separation of concerns**: Current pipeline mixes sensing (CV detection) with inference (offense/defense labeling) with metrics (SDI scores). Layered approach is cleaner.

3. **Immutability matters**: Frozen dataclass prevents downstream code from accidentally mutating state, which causes the "ego-state leakage" bug described in the incident report.

### Where Grok Falls Short

1. **Doesn't solve the hard problem**: PlayState is just a container. The real work is:
   - Homography estimation (video → field coordinates)
   - Ball detection (current code has stub returning `(0.5, 0.5)`)
   - LOS detection (player clustering is fragile)

2. **Doesn't address actual data outputs**: Metrics chart shows 30+ metrics. PlayState has 5 fields. It's a foundation, not a solution.

3. **Unnecessary dependency**: Adding sklearn for `get_cluster_centers()` when current code already does LOS estimation.

4. **Still uses normalized pixels**: The real neutral backbone needs **field coordinates in yards**, not camera-relative pixels.

---

## Part 2: The Coordinate System Problem

### Current State

| Component | Coordinate System | Problem |
|-----------|------------------|---------|
| PlayerTracker | Normalized 0-1 (video pixels) | Camera-dependent |
| DefensiveInference | Normalized 0-1 | Thresholds are arbitrary |
| SDI metrics | Normalized distances | Not comparable across clips |

### Why This Breaks

If camera zooms, pans, or differs between games:
- "0.1 distance" means different things
- Box depth thresholds (`BOX_DEPTH_SHALLOW = 0.08`) are meaningless
- Metrics aren't comparable across clips or games

### What We Need

Transform everything to **field coordinates**:
- X-axis: 0-100 yards (goal line to goal line), 0-120 with end zones
- Y-axis: 0-53.33 yards (sideline to sideline)
- Origin: Corner of field (left goal line, bottom sideline)

This makes all measurements absolute and comparable.

---

## Part 3: Homography / Field Calibration

### The Goal

Transform any video pixel `(u, v)` → field coordinates `(x, y)` in yards.

### Detectable Field Features

| Feature | Spacing | Detection Approach |
|---------|---------|-------------------|
| Yard lines | Every 5 yards | Edge detection + Hough lines |
| Hash marks | Fixed y-positions | Short line segment detection |
| Field numbers | Every 10 yards | OCR (10, 20, 30, 40, 50) |
| Sidelines | y=0 and y=53.33 | Long edge lines |
| Goal lines | x=0 and x=100 | End zone boundary |

### The Math

Homography is a 3x3 matrix H that maps:
```
[x_field]       [u_video]
[y_field] = H × [v_video]
[   1   ]       [   1   ]
```

Need **at least 4 corresponding points** between video and field. More points = more robust.

### Implementation Approach

**Phase 1: Detect field lines**
- Grayscale conversion
- Canny edge detection
- Hough transform for lines
- Filter horizontal (yard lines) and vertical (sidelines/hashes)

**Phase 2: Identify known points**
- Find intersections of yard lines with hash marks/sidelines
- OCR field numbers if visible for absolute position
- Use relative spacing (5 yards) if numbers not visible

**Phase 3: Compute homography**
- Match detected points to known field coordinates
- Use `cv2.findHomography()` with RANSAC
- Store matrix for clip (update if camera moves)

**Phase 4: Transform positions**
- Apply homography to every tracked position
- Output `(x, y)` in yards instead of `(u, v)` in pixels

### Handling Camera Movement

For broadcast/endzone film, camera may:
- Pan (follow play left/right)
- Zoom (tighter on action)
- Tilt (follow ball downfield)

Solutions:
1. Recompute homography periodically (every N frames or when detection changes)
2. Track field features across frames (optical flow on yard lines)
3. For fixed cameras (Brush setup), calibrate once per game

---

## Part 4: Game State Without Scoreboard OCR

### The Problem

Brush scoreboard is too bright for OCR. Can't rely on scoreboard for:
- Down & distance
- Score
- Quarter/time
- Possession

### Alternative Sources

| Data Point | Alternative Source |
|------------|-------------------|
| Down & Distance | Manual entry pre-clip, or infer from play outcome (risky) |
| Score | Not needed for physics layer—only game theory |
| Possession | Detect which team snaps ball (motion at center) |
| Field position | Yard line detection (homography solves this) |
| Quarter/Time | Manual metadata or filename convention |

### Key Insight

**You don't need score/down/distance for the physics layer.**

The neutral fact-finder only needs:
- Player positions in field coordinates
- Ball position
- Line of scrimmage
- Timing (snap frame, etc.)

Game state (down, distance, score) only matters for the **decision/game-theory layer**, which can be a separate enrichment step with manual metadata when OCR fails.

---

## Part 5: Comprehensive Play State Structure

### Design Principles

1. **All coordinates in field yards** (not video pixels)
2. **Team-agnostic** until explicitly requested
3. **Compute geometry once**, store results
4. **Validate with invariants** before downstream processing
5. **Enable any stat as a query** on the structure

### The Structure

```python
@dataclass
class CanonicalPlayState:
    """
    Complete neutral representation of one football play.
    All coordinates in field yards (0-100 x-axis, 0-53.33 y-axis).
    No team labels until explicitly requested.
    """

    # === METADATA ===
    play_id: str                          # Unique identifier
    source_clip: str                      # Video filename
    source_fps: float                     # Video frame rate
    calibration_confidence: float         # How good is our homography?

    # === TIMING (frame numbers) ===
    presnap_frame: int                    # First frame of analysis window
    snap_frame: int                       # Ball is snapped
    first_contact_frame: Optional[int]   # First blocking engagement
    throw_frame: Optional[int]            # Ball released (if pass)
    catch_frame: Optional[int]            # Ball caught/incomplete
    tackle_frame: Optional[int]           # Ball carrier down
    whistle_frame: int                    # Play ends

    # === FIELD GEOMETRY ===
    homography_matrix: List[List[float]]  # 3x3 transform matrix
    los_x: float                          # Line of scrimmage (yards from own goal)
    los_y: float                          # LOS y-position (should be ~26.67, middle)
    ball_spot: Tuple[float, float]        # Ball position at snap (x, y in yards)
    hash_position: str                    # "left", "middle", "right"

    # === GAME STATE (if known, else None) ===
    down: Optional[int]                   # 1-4
    distance: Optional[float]             # Yards to first down
    quarter: Optional[int]                # 1-4
    game_clock: Optional[str]             # "MM:SS"
    score_possession: Optional[int]       # Score of team with ball
    score_defense: Optional[int]          # Score of defending team

    # === RAW PLAYER DATA (team-agnostic) ===
    players: List[PlayerState]            # All detected players

    # === DERIVED GEOMETRY (computed once) ===
    los_cluster: List[int]                # Player IDs on/near LOS
    backfield_cluster: List[int]          # Player IDs behind LOS (offense side)
    secondary_cluster: List[int]          # Player IDs deep (defense side)

    # === INVARIANTS ===
    invariants: Dict[str, bool]           # Validation checks

    def is_valid(self) -> bool:
        """Check all invariants pass."""
        return all(self.invariants.values())


@dataclass
class PlayerState:
    """
    Single player's complete state for one play.
    All positions in field coordinates (yards).
    """

    # === IDENTITY ===
    track_id: int                         # Tracking ID for this clip
    jersey_number: Optional[int]          # If detected
    jersey_color_rgb: Tuple[int,int,int]  # Raw color detected

    # === POSITION AT SNAP ===
    snap_x: float                         # Field x-position at snap (yards)
    snap_y: float                         # Field y-position at snap (yards)
    snap_depth_from_los: float            # Signed: + = defense side, - = offense side
    snap_width_from_ball: float           # Signed: + = right of ball, - = left

    # === TRAJECTORY (post-snap) ===
    trajectory: List[TrajectoryPoint]     # Position at each frame

    # === COMPUTED METRICS (team-agnostic) ===
    first_movement_frame: int             # Frame of first significant movement
    first_movement_direction: float       # Angle in degrees (0 = toward offense endzone)
    max_velocity: float                   # Peak speed (yards/sec)
    total_distance: float                 # Total yards traveled

    # === CLASSIFICATION (computed, not assumed) ===
    side_of_ball: str                     # "los_plus" or "los_minus"
    alignment_zone: str                   # "box", "edge", "slot", "wide", "deep"


@dataclass
class TrajectoryPoint:
    """Single point in a player's trajectory."""
    frame: int
    x: float                              # Field x (yards)
    y: float                              # Field y (yards)
    vx: float                             # Velocity x (yards/sec)
    vy: float                             # Velocity y (yards/sec)
    ax: float                             # Acceleration x (yards/sec²)
    ay: float                             # Acceleration y (yards/sec²)
```

### What This Structure Enables

Any stat becomes a query:

| Stat | Query |
|------|-------|
| Box count | `len([p for p in players if p.alignment_zone == "box" and p.side_of_ball == "los_plus"])` |
| First step quickness | `(p.first_movement_frame - snap_frame) / fps` |
| Separation at catch | Distance between receiver and nearest defender at `catch_frame` |
| Pressure rate | Did any `los_plus` player reach QB position within 2.5 sec? |
| Coverage shell | Cluster `los_plus` players by depth |
| QB dropback depth | `qb.trajectory[throw_frame].x - los_x` |
| Pursuit angle | Compare actual path to optimal intercept vector |
| Time to throw | `(throw_frame - snap_frame) / fps` |

### What's NOT in This Structure

- "offense" or "defense" labels
- "home" or "away"
- Play call names
- Formation names
- Scheme terminology

All interpretation sits on top as a **view layer**. The raw data is neutral physics.

---

## Part 6: Architecture Layers

### Layer 1: Sensing (CV/Physics)
**Input:** Video frames
**Output:** Raw detections in video coordinates
**Components:**
- YOLO player detection
- ByteTrack tracking
- Jersey OCR
- Field line detection

### Layer 2: Calibration (Geometry)
**Input:** Raw detections + field features
**Output:** Field-coordinate positions
**Components:**
- Homography computation
- Coordinate transformation
- LOS detection
- Ball tracking

### Layer 3: State Resolution (Neutral)
**Input:** Field-coordinate positions
**Output:** `CanonicalPlayState`
**Components:**
- Player clustering (LOS, backfield, secondary)
- Invariant validation
- Trajectory computation
- Velocity/acceleration derivation

### Layer 4: Inference (Analysis)
**Input:** `CanonicalPlayState`
**Output:** Scheme classifications, metrics
**Components:**
- Formation recognition
- Coverage shell detection
- Blitz indicators
- SDI score computation

### Layer 5: Perspective (View)
**Input:** Analysis + team mapping
**Output:** Team-specific reports
**Components:**
- Jersey color → team mapping
- "Our offense" vs "their defense" framing
- Play recommendations
- Scouting reports

---

## Part 7: Implementation Roadmap

### Phase 1: Field Calibration Module
**Goal:** Detect yard lines, compute homography, transform coordinates

New file: `field_calibration.py`
- `detect_field_lines(frame)` → yard lines, hash marks
- `find_reference_points(lines)` → intersection points
- `compute_homography(video_points, field_points)` → 3x3 matrix
- `transform_position(u, v, H)` → (x_yards, y_yards)
- `FieldCalibrator` class to manage per-clip calibration

### Phase 2: Update PlayerTracker
**Goal:** Output field coordinates instead of normalized pixels

Changes to `player_tracker.py`:
- Accept `FieldCalibrator` in constructor
- Transform all positions through homography
- Output `PlayerState` with yard coordinates
- Add velocity/acceleration computation

### Phase 3: CanonicalPlayState Exporter
**Goal:** One JSON per play with all raw data

New file: `play_state_exporter.py`
- Build `CanonicalPlayState` from tracking + calibration
- Compute invariants
- Export to JSON
- Support batch export for full games

### Phase 4: Stat Query Layer
**Goal:** Any metric as a function on CanonicalPlayState

New file: `stat_queries.py`
- `box_count(state)` → int
- `first_step_time(state, player_id)` → float
- `separation_at_frame(state, p1, p2, frame)` → float
- `pressure_rate(state)` → bool
- `coverage_shell(state)` → CoverageShell enum

### Phase 5: View Layer
**Goal:** Team-specific rendering of neutral data

New file: `team_view.py`
- `TeamView` class with team mappings
- `render_offensive_report(state, view)` → formatted output
- `render_defensive_report(state, view)` → formatted output
- `render_scouting_report(states, opponent_view)` → opponent analysis

---

## Part 8: Key Differences from Current Pipeline

| Aspect | Current (v5) | Proposed |
|--------|--------------|----------|
| Coordinates | Normalized 0-1 pixels | Field yards (0-100 × 0-53.33) |
| Team assignment | Jersey color at detection | Neutral until view layer |
| LOS detection | Player clustering (fragile) | Field line detection (robust) |
| Validation | None (silent failures) | Explicit invariants |
| Data export | Multiple CSVs | Single canonical JSON per play |
| Stat computation | Hardcoded in pipeline | Queries on canonical state |
| Camera handling | Assumes fixed view | Homography handles movement |

---

## Part 9: Why This Matters for Your Product

### For Real-Time (12-second loop)
- Precompute homography per camera setup
- Canonical state enables fast lookup against precomputed opponent models
- No team-perspective logic in hot path

### For Scouting
- Public film becomes valuable (same pipeline, same output format)
- Opponent constraints are measurable, not guessed
- Multi-game analysis is just aggregation of canonical states

### For High School Market
- Works with bad scoreboards (no OCR dependency for physics)
- Works with varying camera setups (homography adapts)
- Single system, single truth source

### For Differentiation
- You're not selling "better Hudl"
- You're selling: "We convert any film into a quantitative opponent model that supports real-time decisions"
- The neutral physics foundation makes this credible

---

## Appendix A: Current Pipeline Files

| File | Purpose | Status |
|------|---------|--------|
| `sdi_pipeline_v5_with_defense.py` | Main pipeline | Needs homography integration |
| `player_tracker.py` | YOLO + ByteTrack | Needs field coordinate output |
| `defensive_inference.py` | Scheme classification | Needs field coordinate thresholds |
| `third_and_20_cv_v2_fixed.py` | Snap detection | OK as-is |

## Appendix B: Metrics from Analytics Chart

Full list of metrics that should be computable from CanonicalPlayState:

**CV/ML Layer:**
- Snap Detection
- Recognition Latency (RL)
- First Step Quickness (FSQ)
- Assignment Accuracy (AA)
- Leverage Integrity (LI)
- Coverage Win Rate (CWR)
- Formation Recognition
- Jersey OCR → Roster Match
- Pursuit Angle
- Decision Point Timing

**Data Layer:**
- Player Position (GPS/UWB or CV-derived)
- Player Load / High Speed Efforts
- Scoreboard OCR (when available)
- Play Clock Detection
- Public Film Scraping baselines

**Inference Layer:**
- Probability Chains
- Scheme Compatibility Scoring
- Coach DNA Pattern Matching
- Game State Weighting
- Contradiction Detection

**Output Layer:**
- SDI Composite Score
- Play Recommendations
- SDI Cards (Per-Player)
- Opponent Tendency Matrix
- One-Play Deep Dives

---

## Appendix C: Reference Documents

1. **"We need to be a neutral fact-finder viewing film (1).txt"** - Core philosophy document defining the physics-first approach
2. **"Grok code updates.txt"** - PlayState proposal with integration suggestions
3. **"Third_and_20_analytics.csv"** - Full metrics specification

---

*Document generated by Claude analysis of Third & 20 codebase and design documents.*
