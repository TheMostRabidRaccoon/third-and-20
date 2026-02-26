#!/usr/bin/env python3
"""
Third & 20 Recommendation Engine CLI

Interactive command-line interface for play recommendations.

Usage:
    python cli.py <merged_outcomes.json>

    Or with separate files:
    python cli.py --hudl <hudl.csv> --cv <cv_results.json>
"""
import json
import sys
import os
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    SituationInput, PlayOutcome, PlayType, PlayResult,
    HashMark, DefensiveFront, CoverageShell
)
from engine import RecommendationEngine
from data_merger import merge_data, load_hudl_csv, load_cv_json


def load_outcomes_from_json(path: str) -> list:
    """Load pre-merged outcomes from JSON"""
    with open(path, 'r') as f:
        data = json.load(f)

    outcomes = []
    for d in data:
        # Handle enum conversions
        if isinstance(d.get('play_type'), str):
            try:
                d['play_type'] = PlayType(d['play_type'])
            except ValueError:
                d['play_type'] = PlayType.RUN
        if isinstance(d.get('result'), str):
            try:
                d['result'] = PlayResult(d['result'])
            except ValueError:
                d['result'] = PlayResult.RUSH
        if isinstance(d.get('hash_mark'), str):
            try:
                d['hash_mark'] = HashMark(d['hash_mark'])
            except ValueError:
                d['hash_mark'] = HashMark.MIDDLE
        if isinstance(d.get('def_front'), str) and d['def_front']:
            try:
                d['def_front'] = DefensiveFront(d['def_front'])
            except ValueError:
                d['def_front'] = None
        if isinstance(d.get('def_coverage'), str) and d['def_coverage']:
            try:
                d['def_coverage'] = CoverageShell(d['def_coverage'])
            except ValueError:
                d['def_coverage'] = None

        try:
            outcomes.append(PlayOutcome(**d))
        except Exception as e:
            print(f"Warning: skipping malformed outcome: {e}")

    return outcomes


def interactive_loop(engine: RecommendationEngine):
    """Run interactive recommendation loop"""
    print("\n" + "="*60)
    print("THIRD & 20 PLAY RECOMMENDATION ENGINE")
    print("="*60)
    print("\nCommands:")
    print("  <down> <distance> <yard_line> [hash] - Get recommendations")
    print("  Example: 3 7 -35 L")
    print("  ")
    print("  plays              - List all plays in playbook")
    print("  play <name>        - Show stats for a play")
    print("  tendencies         - Show defensive tendencies")
    print("  quit               - Exit")
    print()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split()
        cmd = parts[0].lower()

        if cmd in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        elif cmd == 'plays':
            print("\nAvailable plays:")
            for name, stats in sorted(engine.play_stats.items(), key=lambda x: -x[1].times_called):
                print(f"  {name}: {stats.times_called}x called, {stats.avg_yards:.1f} avg yds, {stats.success_rate:.0%} success")

        elif cmd == 'play' and len(parts) > 1:
            play_name = ' '.join(parts[1:])
            report = engine.get_play_report(play_name)
            if report:
                print(f"\n{report['play_name']} ({report['play_type']})")
                print(f"  Called: {report['times_called']} times")
                print(f"  Avg yards: {report['avg_yards']:.1f}")
                print(f"  Success rate: {report['success_rate']:.0%}")
                print(f"  First down rate: {report['first_down_rate']:.0%}")
                print(f"  Turnover rate: {report['turnover_rate']:.0%}")
                if report['by_coverage']:
                    print(f"  By coverage:")
                    for cov, stats in report['by_coverage'].items():
                        times = stats['times_called']
                        avg = stats['yards'] / times if times > 0 else 0
                        succ = stats['successes'] / times if times > 0 else 0
                        print(f"    {cov}: {times}x, {avg:.1f} yds, {succ:.0%} success")
            else:
                print(f"Play '{play_name}' not found")

        elif cmd == 'tendencies':
            report = engine.get_tendency_report()
            print("\nDefensive tendencies by situation:")
            for bucket, coverages in sorted(report.items()):
                print(f"\n  {bucket}:")
                for cov, pct in coverages.items():
                    print(f"    {cov}: {pct}")

        else:
            # Try to parse as situation
            try:
                down = int(parts[0])
                distance = int(parts[1])
                yard_line = int(parts[2])
                hash_mark = HashMark.MIDDLE
                if len(parts) > 3:
                    h = parts[3].upper()
                    if h == 'L':
                        hash_mark = HashMark.LEFT
                    elif h == 'R':
                        hash_mark = HashMark.RIGHT

                situation = SituationInput(
                    down=down,
                    distance=distance,
                    yard_line=yard_line,
                    hash_mark=hash_mark
                )

                print(f"\nSituation: {down} & {distance} at {yard_line} ({hash_mark.value} hash)")

                response = engine.recommend(situation)

                print(f"\nPredicted defense: {response.predicted_coverage.value} ({response.defense_confidence:.0%})")
                print(f"Response time: {response.response_time_ms}ms")

                print("\n" + "-"*50)
                for i, rec in enumerate(response.recommendations, 1):
                    print(f"\n{i}. {rec.play_name}")
                    print(f"   {rec.expected_yards:.1f} yds expected | {rec.success_probability:.0%} success | {rec.confidence} confidence")
                    print(f"   (Called {rec.times_called}x, overall {rec.avg_yards:.1f} yds avg)")
                    if rec.reasoning:
                        for r in rec.reasoning[:2]:
                            print(f"   • {r}")
                    if rec.risk_flags:
                        for r in rec.risk_flags:
                            print(f"   ⚠️  {r}")

            except (ValueError, IndexError):
                print("Invalid input. Use: <down> <distance> <yard_line> [hash]")
                print("Example: 3 7 -35 L")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Third & 20 Play Recommendation Engine')
    parser.add_argument('outcomes', nargs='?', help='Merged outcomes JSON file')
    parser.add_argument('--hudl', help='Hudl CSV file')
    parser.add_argument('--cv', help='CV results JSON file')
    parser.add_argument('--game-id', default='game', help='Game identifier')

    args = parser.parse_args()

    # Load data
    if args.outcomes:
        print(f"Loading outcomes from {args.outcomes}...")
        outcomes = load_outcomes_from_json(args.outcomes)
    elif args.hudl:
        if args.cv:
            print(f"Merging {args.hudl} + {args.cv}...")
            outcomes = merge_data(args.hudl, args.cv, args.game_id)
        else:
            # Hudl only - no CV data
            print(f"Loading Hudl data from {args.hudl} (no CV data)...")
            hudl = load_hudl_csv(args.hudl)
            outcomes = []
            for play_num, data in hudl.items():
                if not data.get('down'):
                    continue
                outcomes.append(PlayOutcome(
                    game_id=args.game_id,
                    play_number=play_num,
                    clip_file=f"Clip {play_num:03d}.mp4",
                    down=data['down'],
                    distance=data['distance'],
                    yard_line=data['yard_line'],
                    hash_mark=HashMark(data['hash']) if data['hash'] in ('L', 'M', 'R') else HashMark.MIDDLE,
                    play_type=PlayType.RUN if data['play_type'].lower() == 'run' else PlayType.PASS,
                    formation=data['formation'],
                    play_direction=data['play_dir'],
                    def_front=None,
                    def_coverage=None,
                    def_box_count=None,
                    def_safety_look=None,
                    result=PlayResult.RUSH,  # Simplified
                    yards_gained=data['yards_gained'],
                    first_down=data['yards_gained'] >= data['distance'] if data['yards_gained'] else False,
                    turnover=data['result'].lower() in ('interception', 'fumble'),
                    cv_valid=False,
                    cv_confidence=0.0
                ))
    else:
        print("Usage: python cli.py <outcomes.json>")
        print("   or: python cli.py --hudl <hudl.csv> [--cv <cv.json>]")
        sys.exit(1)

    print(f"Loaded {len(outcomes)} play outcomes")

    # Build engine
    engine = RecommendationEngine()
    engine.load_outcomes(outcomes)

    print(f"Built stats for {len(engine.play_stats)} plays")

    # Run interactive loop
    interactive_loop(engine)


if __name__ == "__main__":
    main()
