"""
Third & 20 - Data Merger
Combines Hudl play-by-play data with CV pipeline defensive analysis.

Join key: PLAY # (Hudl) ↔ clip_num (CV)
  - "Wide - Clip 001.mp4" → clip_num = 1 → PLAY # = 1

Expected Hudl CSV columns:
  PLAY #, ODK, DN, DIST, HASH, YARD LN, PLAY TYPE, RESULT, OFF FORM, PLAY DIR

Expected CV JSON structure:
  [{"clip": "Wide - Clip 001.mp4", "clip_num": 1, "valid": true, ...}]
"""
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict

from models import (
    PlayOutcome, PlayType, PlayResult, HashMark,
    DefensiveFront, CoverageShell
)


def parse_yard_line(yard_ln_str: str) -> int:
    """
    Convert Hudl YARD LN to signed integer.
    Negative = own territory, Positive = opponent territory

    Examples:
        -35 → -35 (own 35)
        36 → 36 (opponent 36)
        45 → 45 (opponent 45, red zone)
    """
    try:
        return int(yard_ln_str)
    except ValueError:
        return 0


def parse_play_type(play_type_str: str) -> PlayType:
    """Convert Hudl PLAY TYPE to enum"""
    pt = play_type_str.lower().strip()
    if pt == "run":
        return PlayType.RUN
    elif pt == "pass":
        return PlayType.PASS
    elif pt == "punt":
        return PlayType.PUNT
    elif "screen" in pt:
        return PlayType.SCREEN
    else:
        return PlayType.RUN  # Default


def parse_result(result_str: str) -> PlayResult:
    """Convert Hudl RESULT to enum"""
    r = result_str.lower().strip()
    if r == "rush":
        return PlayResult.RUSH
    elif r == "complete":
        return PlayResult.COMPLETE
    elif r == "incomplete":
        return PlayResult.INCOMPLETE
    elif r == "interception":
        return PlayResult.INTERCEPTION
    elif r == "fumble":
        return PlayResult.FUMBLE
    elif r == "sack":
        return PlayResult.SACK
    elif r == "penalty":
        return PlayResult.PENALTY
    elif r == "touchdown" or r == "td":
        return PlayResult.TOUCHDOWN
    elif r == "downed":
        return PlayResult.DOWNED
    else:
        return PlayResult.RUSH  # Default


def parse_hash(hash_str: str) -> HashMark:
    """Convert Hudl HASH to enum"""
    h = hash_str.upper().strip()
    if h == "L":
        return HashMark.LEFT
    elif h == "R":
        return HashMark.RIGHT
    else:
        return HashMark.MIDDLE


def parse_def_front(front_str: Optional[str]) -> Optional[DefensiveFront]:
    """Convert CV def_front to enum"""
    if not front_str:
        return None
    f = front_str.lower()
    if "3" in f:
        return DefensiveFront.THREE_DOWN
    elif "4" in f:
        return DefensiveFront.FOUR_DOWN
    elif "5" in f:
        return DefensiveFront.FIVE_DOWN
    return DefensiveFront.UNKNOWN


def parse_coverage(cov_str: Optional[str]) -> Optional[CoverageShell]:
    """Convert CV def_coverage to enum"""
    if not cov_str:
        return None
    c = cov_str.lower()
    if "cover 0" in c or "0" == c:
        return CoverageShell.COVER_0
    elif "cover 1" in c or "1" == c:
        return CoverageShell.COVER_1
    elif "cover 2" in c:
        return CoverageShell.COVER_2
    elif "cover 3" in c:
        return CoverageShell.COVER_3
    elif "cover 4" in c:
        return CoverageShell.COVER_4
    elif "cover 6" in c:
        return CoverageShell.COVER_6
    elif "man" in c:
        return CoverageShell.MAN
    return CoverageShell.UNKNOWN


def extract_clip_num(clip_filename: str) -> int:
    """
    Extract play number from clip filename.
    "Wide - Clip 001.mp4" → 1
    "Wide - Clip 023.mp4" → 23
    """
    match = re.search(r'Clip\s*(\d+)', clip_filename)
    if match:
        return int(match.group(1))
    return 0


def load_hudl_csv(csv_path: str) -> Dict[int, Dict]:
    """
    Load Hudl play-by-play CSV.
    Returns dict keyed by PLAY #.

    Expected columns:
    PLAY #,ODK,DN,DIST,HASH,YARD LN,PLAY TYPE,RESULT,OFF FORM,OFF PLAY,PLAY DIR,DEF FRONT,COVERAGE,BLITZ,QTR,GN/LS
    """
    plays = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            play_num_str = row.get('PLAY #', '0')
            try:
                play_num = int(play_num_str)
            except ValueError:
                continue

            if play_num > 0:
                # Parse yards gained (can be negative or empty)
                yards_str = row.get('GN/LS', '')
                yards_gained = None
                if yards_str:
                    try:
                        yards_gained = int(yards_str)
                    except ValueError:
                        pass

                plays[play_num] = {
                    'odk': row.get('ODK', ''),
                    'down': int(row.get('DN', 0)) if row.get('DN') else 0,
                    'distance': int(row.get('DIST', 0)) if row.get('DIST') else 0,
                    'hash': row.get('HASH', 'M'),
                    'yard_line': parse_yard_line(row.get('YARD LN', '0')),
                    'play_type': row.get('PLAY TYPE', ''),
                    'result': row.get('RESULT', ''),
                    'formation': row.get('OFF FORM', ''),
                    'play_name': row.get('OFF PLAY', ''),  # Actual play name
                    'play_dir': row.get('PLAY DIR', ''),
                    'quarter': int(row.get('QTR', 1)) if row.get('QTR') else 1,
                    'yards_gained': yards_gained,
                    # Manual defensive tags (may be empty - CV will fill)
                    'hudl_def_front': row.get('DEF FRONT', ''),
                    'hudl_coverage': row.get('COVERAGE', ''),
                    'hudl_blitz': row.get('BLITZ', ''),
                }
    return plays


def load_cv_json(json_path: str) -> Dict[int, Dict]:
    """
    Load CV pipeline results JSON.
    Returns dict keyed by clip_num.
    """
    with open(json_path, 'r') as f:
        results = json.load(f)

    cv_data = {}
    for r in results:
        clip_num = r.get('clip_num', 0)
        if clip_num == 0:
            clip_num = extract_clip_num(r.get('clip', ''))
        if clip_num > 0:
            cv_data[clip_num] = r
    return cv_data


def merge_data(
    hudl_csv_path: str,
    cv_json_path: str,
    game_id: str = "unknown"
) -> List[PlayOutcome]:
    """
    Merge Hudl play-by-play with CV defensive analysis.

    Args:
        hudl_csv_path: Path to Hudl CSV export
        cv_json_path: Path to CV pipeline JSON output
        game_id: Identifier for this game

    Returns:
        List of PlayOutcome objects ready for recommendation engine
    """
    hudl_data = load_hudl_csv(hudl_csv_path)
    cv_data = load_cv_json(cv_json_path)

    outcomes = []

    # Merge on play number
    all_play_nums = set(hudl_data.keys()) | set(cv_data.keys())

    for play_num in sorted(all_play_nums):
        hudl = hudl_data.get(play_num, {})
        cv = cv_data.get(play_num, {})

        # Skip special teams plays without situation data
        if not hudl.get('down') or hudl.get('odk') in ('S', 'K'):
            continue

        # Determine if turnover
        result_str = hudl.get('result', '').lower()
        is_turnover = result_str in ('interception', 'fumble')

        # Determine first down
        # TODO: This would need next play's down to be accurate
        is_first_down = False

        # Determine first down from yards gained
        yards = hudl.get('yards_gained')
        distance = hudl.get('distance', 0)
        is_first_down = yards is not None and yards >= distance if distance > 0 else False

        outcome = PlayOutcome(
            game_id=game_id,
            play_number=play_num,
            clip_file=cv.get('clip', f'Wide - Clip {play_num:03d}.mp4'),

            # Situation from Hudl
            down=hudl.get('down', 0),
            distance=hudl.get('distance', 0),
            yard_line=hudl.get('yard_line', 0),
            hash_mark=parse_hash(hudl.get('hash', 'M')),

            # Play called from Hudl
            play_type=parse_play_type(hudl.get('play_type', '')),
            formation=hudl.get('formation', ''),
            play_direction=hudl.get('play_dir', ''),

            # Defense from CV (prefer CV, fallback to Hudl manual tags)
            def_front=parse_def_front(cv.get('def_front') or hudl.get('hudl_def_front')),
            def_coverage=parse_coverage(cv.get('def_coverage') or hudl.get('hudl_coverage')),
            def_box_count=cv.get('def_box_count'),
            def_safety_look=cv.get('def_safety_look'),

            # Result from Hudl
            result=parse_result(hudl.get('result', '')),
            yards_gained=yards,
            first_down=is_first_down,
            turnover=is_turnover,

            # CV quality
            cv_valid=cv.get('valid', False),
            cv_confidence=cv.get('confidence', 0.0)
        )

        outcomes.append(outcome)

    return outcomes


def save_merged_data(outcomes: List[PlayOutcome], output_path: str):
    """Save merged outcomes to JSON for later use"""
    data = [asdict(o) for o in outcomes]

    # Convert enums to strings
    for d in data:
        for key, val in d.items():
            if hasattr(val, 'value'):
                d[key] = val.value

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(outcomes)} play outcomes to {output_path}")


def print_merge_summary(outcomes: List[PlayOutcome]):
    """Print summary statistics of merged data"""
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)

    total = len(outcomes)
    valid_cv = sum(1 for o in outcomes if o.cv_valid)

    print(f"Total plays: {total}")
    print(f"With valid CV data: {valid_cv} ({100*valid_cv/total:.1f}%)")

    # By play type
    print("\nBy play type:")
    types = {}
    for o in outcomes:
        t = o.play_type.value if o.play_type else "unknown"
        types[t] = types.get(t, 0) + 1
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # By result
    print("\nBy result:")
    results = {}
    for o in outcomes:
        r = o.result.value if o.result else "unknown"
        results[r] = results.get(r, 0) + 1
    for r, count in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {r}: {count}")

    # Defensive fronts seen
    print("\nDefensive fronts detected:")
    fronts = {}
    for o in outcomes:
        if o.def_front:
            f = o.def_front.value
            fronts[f] = fronts.get(f, 0) + 1
    for f, count in sorted(fronts.items(), key=lambda x: -x[1]):
        print(f"  {f}: {count}")

    # Coverages seen
    print("\nCoverages detected:")
    covs = {}
    for o in outcomes:
        if o.def_coverage:
            c = o.def_coverage.value
            covs[c] = covs.get(c, 0) + 1
    for c, count in sorted(covs.items(), key=lambda x: -x[1]):
        print(f"  {c}: {count}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python data_merger.py <hudl_csv> <cv_json> [output_json]")
        print("\nExample:")
        print("  python data_merger.py shaker_hudl.csv full_game_cv_results.json merged_plays.json")
        sys.exit(1)

    hudl_path = sys.argv[1]
    cv_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "merged_play_outcomes.json"

    game_id = Path(hudl_path).stem

    outcomes = merge_data(hudl_path, cv_path, game_id)
    print_merge_summary(outcomes)
    save_merged_data(outcomes, output_path)
