# Third & 20: SDI Trajectory Feature
*Created: January 10, 2026*

---

## Origin

This feature came from a conversation with Greg about player performance tracking:

**Greg's Question:** "How do you not exclude a great player who starts slow but picks up mid-game?"

**The Problem:** A player might have poor early stats but trend upward throughout the game. Simple real-time stats would flag them as underperforming when they're actually recovering/improving.

---

## The Solution: Trajectory vs. Baseline

SDI (Situational Decision Index) should measure **performance relative to the player's own baseline**, not absolute performance.

### Two Metrics

| Metric | What It Measures |
|--------|------------------|
| **Baseline SDI** | Player's established performance level (from film history) |
| **Real-Time SDI** | Current game performance |

### The Trajectory Calculation

```
Trajectory = (Current Performance - Baseline) + Trend Direction

Where:
- Trend Direction = slope of performance over last N plays
- Positive slope = improving
- Negative slope = declining
- Flat slope = consistent
```

---

## Player Categories

Based on trajectory analysis:

| Pattern | Baseline | Current | Trend | Interpretation |
|---------|----------|---------|-------|----------------|
| **Hot** | High | High | Flat/Up | Feed them |
| **Cold** | High | Low | Flat/Down | Problem - investigate |
| **Warming Up** | High | Low | **Up** | Patient - they're coming back |
| **Fading** | High | High | **Down** | Watch closely |
| **Overperforming** | Low | High | Up | Ride the wave |
| **Consistent** | Any | =Baseline | Flat | Predictable |

---

## "Warming Up" Detection

This is the key insight from Greg's question:

**Scenario:** Star receiver has 4 targets, 2 catches in first half (below his 7/5 norm)

**Simple System:** Flags as underperforming → might reduce targets

**SDI Trajectory System:**
- Baseline: 7 targets, 5 catches per game
- Current: 4 targets, 2 catches
- BUT: Last 3 targets = 2 catches (improving)
- Trend: **Positive slope**
- Classification: **Warming Up**
- Recommendation: Maintain or increase targets

---

## Data Requirements

### For Baseline Calculation
- Historical game film (minimum 3 games recommended)
- Play-by-play tagging from Hudl
- Position-specific metrics (QB reads, WR routes, OL leverage, etc.)

### For Real-Time Tracking
- Current game play-by-play
- Same metrics as baseline
- Running calculation updated after each play

---

## Implementation Phases

### Phase 1: Baseline Establishment
- Ingest historical film
- Calculate per-player SDI baselines
- Identify variance (some players are consistent, some volatile)

### Phase 2: Real-Time Calculation
- Process current game data
- Compare to baseline
- Calculate trend direction

### Phase 3: Trajectory Alerts
- Flag significant deviations from baseline
- Identify "warming up" vs "cold" patterns
- Surface recommendations to coaching staff

### Phase 4: Prescriptive Output
- "Player X trending up from slow start - increase targets"
- "Player Y declining from hot start - check for injury/fatigue"
- "Player Z consistent with baseline - no adjustment needed"

---

## Technical Notes

### Trend Calculation
- Window: Last 5-10 plays (configurable)
- Method: Linear regression slope
- Threshold: Configurable sensitivity for "significant" trend

### Baseline Stability
- Recalculate baseline weekly during season
- Weight recent games more heavily
- Flag players with high variance (less predictable)

### Output Format
```json
{
  "player_id": "23",
  "player_name": "Smith, J",
  "position": "WR",
  "baseline_sdi": 7.2,
  "current_sdi": 5.1,
  "trend": "+0.8",
  "trend_direction": "up",
  "classification": "warming_up",
  "confidence": 0.78,
  "recommendation": "Maintain targets - trending toward baseline"
}
```

---

## Competitive Advantage

**What Hudl Does:** Shows you what happened (descriptive)

**What Modern Football Does:** Shows patterns in data (descriptive)

**What Third & 20 Does:** Tells you what to do next (prescriptive) with context for WHY a player might be underperforming temporarily

---

## Connection to Transfer Portal

This same logic applies to portal evaluation:

- Player's baseline from previous school
- Adjustment period at new school
- Trajectory toward (or away from) baseline
- "Is this player a bust or still adjusting?"

---

*RRI, LLC - Third & 20 Football Intelligence*
