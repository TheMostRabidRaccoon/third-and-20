"""
Third & 20 Play Recommendation Engine

Core logic for recommending plays based on:
1. Current situation (down, distance, field position)
2. Historical play outcomes
3. Predicted defensive alignment

Target: 12-20 second response time (actually <1s once data is loaded)
"""
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from models import (
    SituationInput, PlayRecommendation, RecommendationResponse,
    PlayOutcome, PlayType, PlayResult, DefensiveFront, CoverageShell,
    HashMark
)


@dataclass
class PlayStats:
    """Aggregated statistics for one play"""
    play_name: str
    formation: str
    play_type: PlayType

    times_called: int = 0
    total_yards: int = 0
    successes: int = 0  # Gained expected yards or converted
    first_downs: int = 0
    turnovers: int = 0

    # By coverage
    stats_by_coverage: Dict[str, Dict] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.times_called if self.times_called > 0 else 0.0

    @property
    def avg_yards(self) -> float:
        return self.total_yards / self.times_called if self.times_called > 0 else 0.0

    @property
    def turnover_rate(self) -> float:
        return self.turnovers / self.times_called if self.times_called > 0 else 0.0


class RecommendationEngine:
    """
    The core recommendation engine.

    Usage:
        engine = RecommendationEngine()
        engine.load_outcomes(play_outcomes)  # From merged Hudl + CV data

        situation = SituationInput(down=3, distance=7, yard_line=-35, hash_mark=HashMark.LEFT)
        response = engine.recommend(situation)

        for rec in response.recommendations:
            print(f"{rec.play_name}: {rec.success_probability:.0%} success, {rec.expected_yards:.1f} yds")
    """

    def __init__(self):
        self.outcomes: List[PlayOutcome] = []
        self.play_stats: Dict[str, PlayStats] = {}  # play_name -> stats

        # Situation-specific stats
        # situation_bucket -> play_name -> stats
        self.situation_stats: Dict[str, Dict[str, PlayStats]] = defaultdict(dict)

        # Defensive tendency tracking
        # situation_bucket -> {coverage: count}
        self.defensive_tendencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def load_outcomes(self, outcomes: List[PlayOutcome]):
        """Load historical play outcomes and build statistics"""
        self.outcomes = outcomes
        self._build_play_stats()
        self._build_defensive_tendencies()

    def _build_play_stats(self):
        """Aggregate statistics by play"""
        self.play_stats = {}
        self.situation_stats = defaultdict(dict)

        for outcome in self.outcomes:
            # Skip plays without formation (special teams, etc)
            if not outcome.formation:
                continue

            play_key = outcome.formation  # Use formation as play identifier

            # Initialize stats if needed
            if play_key not in self.play_stats:
                self.play_stats[play_key] = PlayStats(
                    play_name=play_key,
                    formation=outcome.formation,
                    play_type=outcome.play_type
                )

            stats = self.play_stats[play_key]
            stats.times_called += 1

            if outcome.yards_gained is not None:
                stats.total_yards += outcome.yards_gained

            if outcome.is_success:
                stats.successes += 1

            if outcome.first_down:
                stats.first_downs += 1

            if outcome.turnover:
                stats.turnovers += 1

            # Track by coverage (if we have CV data)
            if outcome.def_coverage:
                cov_key = outcome.def_coverage.value
                if cov_key not in stats.stats_by_coverage:
                    stats.stats_by_coverage[cov_key] = {
                        'times_called': 0, 'yards': 0, 'successes': 0
                    }
                stats.stats_by_coverage[cov_key]['times_called'] += 1
                if outcome.yards_gained:
                    stats.stats_by_coverage[cov_key]['yards'] += outcome.yards_gained
                if outcome.is_success:
                    stats.stats_by_coverage[cov_key]['successes'] += 1

            # Track by situation bucket
            bucket = self._get_situation_bucket(outcome.down, outcome.distance, outcome.yard_line)
            if play_key not in self.situation_stats[bucket]:
                self.situation_stats[bucket][play_key] = PlayStats(
                    play_name=play_key,
                    formation=outcome.formation,
                    play_type=outcome.play_type
                )

            sit_stats = self.situation_stats[bucket][play_key]
            sit_stats.times_called += 1
            if outcome.yards_gained:
                sit_stats.total_yards += outcome.yards_gained
            if outcome.is_success:
                sit_stats.successes += 1

    def _build_defensive_tendencies(self):
        """Build defensive tendency profiles by situation"""
        self.defensive_tendencies = defaultdict(lambda: defaultdict(int))

        for outcome in self.outcomes:
            if not outcome.cv_valid or not outcome.def_coverage:
                continue

            bucket = self._get_situation_bucket(outcome.down, outcome.distance, outcome.yard_line)
            cov_key = outcome.def_coverage.value
            self.defensive_tendencies[bucket][cov_key] += 1

    def _get_situation_bucket(self, down: int, distance: int, yard_line: int) -> str:
        """Convert situation to bucket key"""
        # Distance bucket
        if distance <= 2:
            dist_str = "short"
        elif distance <= 5:
            dist_str = "medium"
        elif distance <= 10:
            dist_str = "long"
        else:
            dist_str = "very_long"

        # Field position
        if yard_line <= -20:
            field_str = "backed_up"
        elif yard_line <= 0:
            field_str = "own"
        elif yard_line <= 35:
            field_str = "mid"
        elif yard_line <= 45:
            field_str = "plus"
        else:
            field_str = "redzone"

        return f"{down}_{dist_str}_{field_str}"

    def predict_defense(self, situation: SituationInput) -> Dict[str, float]:
        """Predict defensive coverage probabilities for a situation"""
        bucket = situation.situation_bucket

        if bucket in self.defensive_tendencies:
            counts = self.defensive_tendencies[bucket]
            total = sum(counts.values())
            if total > 0:
                return {cov: count / total for cov, count in counts.items()}

        # Fallback to general tendencies
        all_counts = defaultdict(int)
        for bucket_counts in self.defensive_tendencies.values():
            for cov, count in bucket_counts.items():
                all_counts[cov] += count

        total = sum(all_counts.values())
        if total > 0:
            return {cov: count / total for cov, count in all_counts.items()}

        # Ultimate fallback
        return {"Cover 3": 0.4, "Cover 2": 0.3, "Cover 1": 0.2, "Cover 0": 0.1}

    def _score_play(
        self,
        play_stats: PlayStats,
        predicted_defense: Dict[str, float],
        situation: SituationInput
    ) -> Tuple[float, float, List[str]]:
        """
        Score a play against predicted defense.

        Returns:
            (score, expected_yards, reasoning_list)
        """
        reasoning = []

        # Base score from overall stats
        base_success = play_stats.success_rate
        base_yards = play_stats.avg_yards

        # Weight by predicted defense
        weighted_yards = 0.0
        weighted_success = 0.0
        total_weight = 0.0

        for coverage, prob in predicted_defense.items():
            if coverage in play_stats.stats_by_coverage:
                cov_stats = play_stats.stats_by_coverage[coverage]
                times = cov_stats['times_called']
                if times > 0:
                    cov_yards = cov_stats['yards'] / times
                    cov_success = cov_stats['successes'] / times

                    weighted_yards += prob * cov_yards
                    weighted_success += prob * cov_success
                    total_weight += prob

                    if prob > 0.3:  # Significant coverage probability
                        reasoning.append(
                            f"vs {coverage} ({prob:.0%} likely): {cov_yards:.1f} yds, {cov_success:.0%} success"
                        )

        # If we have coverage-specific data, use it; otherwise use base stats
        if total_weight > 0.5:
            expected_yards = weighted_yards
            expected_success = weighted_success
        else:
            expected_yards = base_yards
            expected_success = base_success
            reasoning.append(f"Overall: {base_yards:.1f} yds, {base_success:.0%} success")

        # Turnover penalty
        if play_stats.turnover_rate > 0.1:
            reasoning.append(f"⚠️ {play_stats.turnover_rate:.0%} turnover rate")

        # Calculate composite score
        # Weight: 60% expected yards, 30% success rate, 10% sample size confidence
        sample_confidence = min(1.0, play_stats.times_called / 10)

        score = (
            expected_yards * 0.5 +
            expected_success * 10 * 0.4 +  # Scale success to ~same range as yards
            sample_confidence * 2 * 0.1
        )

        # Penalize high turnover plays
        score -= play_stats.turnover_rate * 5

        return score, expected_yards, reasoning

    def recommend(self, situation: SituationInput, top_n: int = 5) -> RecommendationResponse:
        """
        Generate play recommendations for a situation.

        Args:
            situation: Current game situation
            top_n: Number of recommendations to return

        Returns:
            RecommendationResponse with ranked plays
        """
        start_time = time.time()

        # 1. Predict defensive alignment
        predicted_defense = self.predict_defense(situation)
        top_coverage = max(predicted_defense, key=predicted_defense.get)
        defense_confidence = predicted_defense[top_coverage]

        # 2. Score all plays
        scored_plays = []

        for play_name, stats in self.play_stats.items():
            if stats.times_called < 2:  # Need minimum sample size
                continue

            score, expected_yards, reasoning = self._score_play(
                stats, predicted_defense, situation
            )

            scored_plays.append({
                'play_name': play_name,
                'stats': stats,
                'score': score,
                'expected_yards': expected_yards,
                'reasoning': reasoning
            })

        # 3. Rank by score
        scored_plays.sort(key=lambda x: -x['score'])

        # 4. Build recommendations
        recommendations = []
        for play_data in scored_plays[:top_n]:
            stats = play_data['stats']

            # Determine confidence based on sample size and consistency
            if stats.times_called >= 10:
                confidence = "HIGH"
            elif stats.times_called >= 5:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # Risk flags
            risk_flags = []
            if stats.turnover_rate > 0.1:
                risk_flags.append(f"Turnover risk ({stats.turnover_rate:.0%})")
            if stats.times_called < 5:
                risk_flags.append("Limited sample size")

            rec = PlayRecommendation(
                play_name=play_data['play_name'],
                formation=stats.formation,
                play_type=stats.play_type,
                success_probability=stats.success_rate,
                expected_yards=play_data['expected_yards'],
                confidence=confidence,
                reasoning=play_data['reasoning'],
                risk_flags=risk_flags,
                times_called=stats.times_called,
                success_rate=stats.success_rate,
                avg_yards=stats.avg_yards
            )
            recommendations.append(rec)

        # 5. Build response
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Map top coverage string to enum
        try:
            predicted_coverage_enum = CoverageShell(top_coverage)
        except ValueError:
            predicted_coverage_enum = CoverageShell.UNKNOWN

        return RecommendationResponse(
            situation=situation,
            response_time_ms=elapsed_ms,
            recommendations=recommendations,
            predicted_front=DefensiveFront.UNKNOWN,  # TODO: Add front prediction
            predicted_coverage=predicted_coverage_enum,
            defense_confidence=defense_confidence
        )

    def get_play_report(self, play_name: str) -> Optional[Dict]:
        """Get detailed stats for a specific play"""
        if play_name not in self.play_stats:
            return None

        stats = self.play_stats[play_name]
        return {
            'play_name': play_name,
            'formation': stats.formation,
            'play_type': stats.play_type.value,
            'times_called': stats.times_called,
            'avg_yards': stats.avg_yards,
            'success_rate': stats.success_rate,
            'first_down_rate': stats.first_downs / stats.times_called if stats.times_called > 0 else 0,
            'turnover_rate': stats.turnover_rate,
            'by_coverage': stats.stats_by_coverage
        }

    def get_tendency_report(self) -> Dict:
        """Get defensive tendency summary"""
        report = {}
        for bucket, coverages in self.defensive_tendencies.items():
            total = sum(coverages.values())
            if total > 0:
                report[bucket] = {
                    cov: f"{count}/{total} ({count/total:.0%})"
                    for cov, count in sorted(coverages.items(), key=lambda x: -x[1])
                }
        return report


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Test the recommendation engine"""
    import json
    import sys

    # Load merged data
    if len(sys.argv) < 2:
        print("Usage: python engine.py <merged_outcomes.json>")
        print("\nOr run interactively after loading data:")
        print("  engine = RecommendationEngine()")
        print("  engine.load_outcomes(outcomes)")
        print("  response = engine.recommend(SituationInput(down=3, distance=7, yard_line=-35, hash_mark=HashMark.LEFT))")
        sys.exit(1)

    outcomes_path = sys.argv[1]

    print(f"Loading outcomes from {outcomes_path}...")
    with open(outcomes_path, 'r') as f:
        data = json.load(f)

    # Convert dicts back to PlayOutcome objects
    outcomes = []
    for d in data:
        # Handle enum conversions
        if isinstance(d.get('play_type'), str):
            d['play_type'] = PlayType(d['play_type'])
        if isinstance(d.get('result'), str):
            d['result'] = PlayResult(d['result'])
        if isinstance(d.get('hash_mark'), str):
            d['hash_mark'] = HashMark(d['hash_mark'])
        if isinstance(d.get('def_front'), str):
            try:
                d['def_front'] = DefensiveFront(d['def_front'])
            except ValueError:
                d['def_front'] = None
        if isinstance(d.get('def_coverage'), str):
            try:
                d['def_coverage'] = CoverageShell(d['def_coverage'])
            except ValueError:
                d['def_coverage'] = None

        outcomes.append(PlayOutcome(**d))

    print(f"Loaded {len(outcomes)} play outcomes")

    # Build engine
    engine = RecommendationEngine()
    engine.load_outcomes(outcomes)

    print(f"\nBuilt stats for {len(engine.play_stats)} plays")
    print(f"Defensive tendencies for {len(engine.defensive_tendencies)} situations")

    # Test recommendation
    print("\n" + "="*60)
    print("TEST: 3rd & 7, own 35, left hash")
    print("="*60)

    situation = SituationInput(
        down=3,
        distance=7,
        yard_line=-35,
        hash_mark=HashMark.LEFT
    )

    response = engine.recommend(situation)

    print(f"\nPredicted defense: {response.predicted_coverage.value} ({response.defense_confidence:.0%} confidence)")
    print(f"Response time: {response.response_time_ms}ms")

    print("\nTop recommendations:")
    for i, rec in enumerate(response.recommendations, 1):
        print(f"\n{i}. {rec.play_name}")
        print(f"   Expected: {rec.expected_yards:.1f} yds, {rec.success_probability:.0%} success")
        print(f"   Confidence: {rec.confidence} (called {rec.times_called}x)")
        if rec.reasoning:
            for r in rec.reasoning[:2]:
                print(f"   - {r}")
        if rec.risk_flags:
            for r in rec.risk_flags:
                print(f"   ⚠️ {r}")


if __name__ == "__main__":
    main()
