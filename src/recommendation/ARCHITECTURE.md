# Third & 20 Play Recommendation Engine

## Overview
Real-time play recommendation for high school football coaches.
Target: 12-20 second response time from sideline input to ranked play suggestions.

## Input/Output Contract

### Input (Sideline Tablet)
```python
@dataclass
class SituationInput:
    # Required
    down: int                    # 1, 2, 3, 4
    distance: int                # Yards to first down
    yard_line: int               # -50 to +50 (negative = own territory)
    hash_mark: str               # "L", "M", "R"

    # Context
    quarter: int                 # 1, 2, 3, 4
    game_clock_seconds: int      # Seconds remaining in quarter
    score_differential: int      # Our score - their score
    timeouts_remaining: int      # 0, 1, 2, 3

    # Optional (pre-loaded or detected)
    personnel_on_field: str      # "11", "12", "21", "22", etc.
    fatigue_flags: List[str]     # Player IDs showing fatigue
    injured_out: List[str]       # Players unavailable
```

### Output (Coach Display)
```python
@dataclass
class PlayRecommendation:
    play_name: str               # From playbook: "WISCONSIN RED", "IOWA BLACK"
    success_probability: float   # 0.0 - 1.0
    expected_yards: float        # Average yards gained in similar situations
    confidence: str              # "HIGH", "MEDIUM", "LOW"

    # Reasoning (for coach buy-in)
    reasoning: List[str]         # ["Opponent shows Cover 2 68% on 2nd & long",
                                 #  "This play gains 7.2 yds avg vs Cover 2"]

    # Risk factors
    risk_flags: List[str]        # ["Requires healthy WR1", "Weather affects deep ball"]

@dataclass
class RecommendationResponse:
    situation: SituationInput
    timestamp: datetime
    response_time_ms: int

    recommendations: List[PlayRecommendation]  # Ranked, top 3-5

    # What defense we expect
    predicted_defense: str       # "Cover 2", "Cover 3", "Man"
    defense_confidence: float
```

## Data Model

### 1. Playbook Registry
```python
@dataclass
class Play:
    play_id: str
    name: str                    # "WISCONSIN RED"
    formation: str               # "Shotgun Trips"
    play_type: str               # "run", "pass", "screen", "play_action"
    primary_target: str          # "Outside zone left", "Slant to X"

    # Requirements
    personnel_required: str      # "11" = 1 RB, 1 TE, 3 WR
    key_players: List[str]       # ["WR1", "LT"] - plays that need specific players healthy

    # Situational fit
    down_distance_fit: Dict      # {"1st_10": 0.9, "3rd_long": 0.3}
    field_position_fit: Dict     # {"redzone": 0.8, "backed_up": 0.4}
```

### 2. Historical Outcomes (from Hudl + CV)
```python
@dataclass
class PlayOutcome:
    game_id: str
    play_number: int

    # Situation when called
    down: int
    distance: int
    yard_line: int
    hash_mark: str

    # What we called
    play_called: str             # "WISCONSIN RED"

    # What defense showed (from CV pipeline)
    defensive_front: str         # "3-down", "4-down"
    coverage_shell: str          # "Cover 2", "Cover 3", "Man"
    box_count: int               # 6, 7, 8
    safety_look: str             # "2-high", "1-high", "robber"

    # Result
    result: str                  # "complete", "incomplete", "rush", "sack", "int"
    yards_gained: int
    first_down: bool
    turnover: bool
```

### 3. Opponent Tendencies (aggregated from CV)
```python
@dataclass
class OpponentTendencies:
    opponent_id: str
    games_analyzed: int

    # Defensive tendencies by situation
    tendencies: Dict[str, Dict]  # situation_key -> {coverage: probability}

    # Example:
    # "1st_10_own_territory": {
    #     "Cover 3": 0.45,
    #     "Cover 2": 0.30,
    #     "Man": 0.25
    # }
    # "3rd_long": {
    #     "Cover 2": 0.55,
    #     "Cover 3": 0.35,
    #     "Blitz": 0.10
    # }
```

## Recommendation Algorithm

### Phase 1: Filter Playbook
```python
def filter_plays(situation: SituationInput, playbook: List[Play]) -> List[Play]:
    """Remove plays that don't fit situation or personnel"""
    candidates = []
    for play in playbook:
        # Personnel check
        if play.personnel_required != situation.personnel_on_field:
            continue
        # Injury check
        if any(p in situation.injured_out for p in play.key_players):
            continue
        # Situational fit (don't call deep pass on 3rd & 1)
        if play.down_distance_fit.get(situation_key(situation), 0) < 0.3:
            continue
        candidates.append(play)
    return candidates
```

### Phase 2: Predict Defense
```python
def predict_defense(situation: SituationInput,
                    opponent: OpponentTendencies) -> Dict[str, float]:
    """What coverage will opponent likely show?"""
    situation_key = make_situation_key(situation)

    # Look up historical tendencies
    if situation_key in opponent.tendencies:
        return opponent.tendencies[situation_key]

    # Fallback to general tendencies
    return opponent.general_tendencies
```

### Phase 3: Score Plays Against Predicted Defense
```python
def score_play(play: Play,
               predicted_defense: Dict[str, float],
               historical_outcomes: List[PlayOutcome]) -> float:
    """How well does this play perform against predicted defense?"""

    # Get historical performance of this play vs each coverage
    play_vs_coverage = get_play_vs_coverage_stats(play.name, historical_outcomes)

    # Weight by predicted defense probability
    expected_yards = 0.0
    for coverage, prob in predicted_defense.items():
        if coverage in play_vs_coverage:
            expected_yards += prob * play_vs_coverage[coverage]['avg_yards']

    # Factor in success rate
    success_rate = calculate_success_rate(play.name, historical_outcomes)

    return expected_yards * 0.6 + success_rate * 10 * 0.4
```

### Phase 4: Rank and Return
```python
def recommend(situation: SituationInput) -> RecommendationResponse:
    start = time.time()

    # 1. Filter playbook
    candidates = filter_plays(situation, PLAYBOOK)

    # 2. Predict defense
    predicted_defense = predict_defense(situation, OPPONENT)

    # 3. Score each play
    scored = [(play, score_play(play, predicted_defense, OUTCOMES))
              for play in candidates]

    # 4. Rank and take top 5
    scored.sort(key=lambda x: -x[1])
    top_plays = scored[:5]

    # 5. Build response with reasoning
    recommendations = []
    for play, score in top_plays:
        rec = PlayRecommendation(
            play_name=play.name,
            success_probability=calculate_success_prob(play, predicted_defense),
            expected_yards=calculate_expected_yards(play, predicted_defense),
            confidence=get_confidence_level(play, len(historical_outcomes)),
            reasoning=generate_reasoning(play, predicted_defense, OUTCOMES),
            risk_flags=get_risk_flags(play, situation)
        )
        recommendations.append(rec)

    return RecommendationResponse(
        situation=situation,
        timestamp=datetime.now(),
        response_time_ms=int((time.time() - start) * 1000),
        recommendations=recommendations,
        predicted_defense=max(predicted_defense, key=predicted_defense.get),
        defense_confidence=max(predicted_defense.values())
    )
```

## Data Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Hudl Export   │     │   CV Pipeline   │     │    Playbook     │
│  (play-by-play) │     │  (defensive     │     │   (formations,  │
│                 │     │   alignments)   │     │    play names)  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────┬───────────┘                       │
                     │                                   │
                     ▼                                   │
         ┌─────────────────────┐                         │
         │   Play Outcomes DB  │                         │
         │  (situation +       │                         │
         │   play + defense +  │◄────────────────────────┘
         │   result)           │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Opponent Tendency  │
         │     Aggregator      │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   Recommendation    │
         │      Engine         │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Coach Interface    │
         │  (tablet/sideline)  │
         └─────────────────────┘
```

## MVP Scope (Week 1)

### Must Have
- [ ] Playbook loader (from Hudl formation names)
- [ ] Historical outcome loader (Hudl CSV + CV JSON merger)
- [ ] Basic situation matching (down/distance buckets)
- [ ] Top 3 play recommendations with expected yards
- [ ] CLI interface for testing

### Nice to Have
- [ ] Opponent tendency tracking
- [ ] Confidence scores
- [ ] Reasoning explanations

### Future
- [ ] Tablet UI
- [ ] Real-time fatigue integration
- [ ] Weather adjustments
- [ ] Referee crew tendencies

## File Structure
```
src/recommendation/
├── __init__.py
├── models.py           # Dataclasses above
├── playbook.py         # Playbook registry
├── outcomes.py         # Historical outcome loader/merger
├── tendencies.py       # Opponent tendency aggregator
├── engine.py           # Core recommendation logic
├── cli.py              # Command-line interface
└── tests/
    └── test_engine.py
```
