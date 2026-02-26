"""
Third & 20 - Formation Report Parser

Parses Hudl Formation Reports to extract:
- Per-formation statistics
- Top plays within formations
- Situational tendencies
- Directional preferences
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Personnel(Enum):
    """Personnel groupings based on formation naming conventions"""
    IOWA = "12"      # 1 RB, 2 TE (Iowa Hawkeyes style)
    WISCONSIN = "21"  # 2 RB (FB), 1 TE (Wisconsin power running)
    FLA = "11"       # 1 RB, 1 TE, 3 WR (spread)
    TEXAS = "10"     # 1 RB, 0 TE, 4 WR (air raid)
    WYOMING = "22"   # Heavy/Jumbo package
    TWINS = "11"     # Standard 2x1 receiver set
    UNKNOWN = "11"


@dataclass
class PlayInfo:
    """Individual play within a formation"""
    name: str
    pct_of_plays: float
    is_run: bool
    avg_yards: float
    success_rate: float


@dataclass
class FormationTendencies:
    """Directional tendencies for a formation"""
    # Run tendencies
    run_right_pct: float = 0.5
    run_field_pct: float = 0.5
    run_strong_pct: float = 0.5
    run_with_motion_pct: float = 0.5

    # Pass tendencies
    pass_right_pct: float = 0.5
    pass_field_pct: float = 0.5
    pass_strong_pct: float = 0.5
    pass_with_motion_pct: float = 0.5


@dataclass
class FormationStats:
    """Complete statistics for one formation"""
    name: str
    personnel: Personnel

    # Overall stats
    pass_pct: float = 0.5
    run_pct: float = 0.5
    avg_yards: float = 0.0
    success_rate: float = 0.0
    motion_pct: float = 0.0

    # Situational usage
    first_down_pct: float = 0.33
    second_down_pct: float = 0.33
    third_down_pct: float = 0.33
    long_distance_pct: float = 0.0  # 7+ yards
    own_territory_pct: float = 0.5
    opp_territory_pct: float = 0.25
    redzone_pct: float = 0.25

    # Top plays
    top_plays: List[PlayInfo] = field(default_factory=list)

    # Tendencies
    tendencies: FormationTendencies = field(default_factory=FormationTendencies)

    # Raw takeaways
    takeaways: List[str] = field(default_factory=list)


def parse_percentage(text: str) -> float:
    """Extract percentage from text like '65%' or '65 percent'"""
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if match:
        return float(match.group(1)) / 100
    return 0.0


def parse_yards(text: str) -> float:
    """Extract yards from text like '7 yards' or '-5 yards'"""
    match = re.search(r'(-?\d+(?:\.\d+)?)\s*yard', text)
    if match:
        return float(match.group(1))
    return 0.0


def infer_personnel(formation_name: str) -> Personnel:
    """Infer personnel grouping from formation name"""
    name_upper = formation_name.upper()

    if 'IOWA' in name_upper:
        return Personnel.IOWA
    elif 'WISCONSIN' in name_upper:
        return Personnel.WISCONSIN
    elif 'FLA' in name_upper or 'FLORIDA' in name_upper:
        return Personnel.FLA
    elif 'TEXAS' in name_upper:
        return Personnel.TEXAS
    elif 'WYOMING' in name_upper:
        return Personnel.WYOMING
    elif 'TWINS' in name_upper:
        return Personnel.TWINS
    else:
        return Personnel.UNKNOWN


def parse_formation_block(text: str) -> Optional[FormationStats]:
    """Parse a single formation block from the report"""

    # Extract formation name
    name_match = re.search(r'Formation:\s*([A-Z\s\.]+)', text)
    if not name_match:
        return None

    name = name_match.group(1).strip()
    stats = FormationStats(
        name=name,
        personnel=infer_personnel(name)
    )

    # Parse overview section
    overview_match = re.search(r'Formation Overview.*?(?=🎯|Top Plays|$)', text, re.DOTALL)
    if overview_match:
        overview = overview_match.group(0)

        # Pass/Run split
        pass_match = re.search(r'pass\s+(\d+)%', overview, re.IGNORECASE)
        run_match = re.search(r'run\s+(\d+)%', overview, re.IGNORECASE)
        if pass_match:
            stats.pass_pct = float(pass_match.group(1)) / 100
        if run_match:
            stats.run_pct = float(run_match.group(1)) / 100

        # Average yards
        yards_match = re.search(r'averaging\s+(\d+)\s*yards?\s*per\s*play', overview, re.IGNORECASE)
        if yards_match:
            stats.avg_yards = float(yards_match.group(1))

        # Success rate
        success_match = re.search(r'(\d+)%\s*(?:success|efficiency)\s*rate', overview, re.IGNORECASE)
        if success_match:
            stats.success_rate = float(success_match.group(1)) / 100

        # Motion usage
        motion_match = re.search(r'Motion\s*appears\s*on\s*(\d+)%', overview, re.IGNORECASE)
        if motion_match:
            stats.motion_pct = float(motion_match.group(1)) / 100

    # Parse top plays table
    plays_section = re.search(r'Top Plays.*?Takeaway:', text, re.DOTALL)
    if plays_section:
        # Look for play entries
        play_pattern = r'([A-Z0-9\s]+)\n\s*(\d+)%\n\s*(Run|Pass)\n\s*(-?\d+)\s*yards?\n\s*(\d+)%'
        for match in re.finditer(play_pattern, plays_section.group(0), re.IGNORECASE):
            play = PlayInfo(
                name=match.group(1).strip(),
                pct_of_plays=float(match.group(2)) / 100,
                is_run=match.group(3).lower() == 'run',
                avg_yards=float(match.group(4)),
                success_rate=float(match.group(5)) / 100
            )
            stats.top_plays.append(play)

    # Parse situational usage
    when_section = re.search(r'When We Use It.*?(?=Tendencies|🔄|$)', text, re.DOTALL)
    if when_section:
        when_text = when_section.group(0)

        # Down usage
        first_match = re.search(r'(?:1st|first)\s*down.*?(\d+)%', when_text, re.IGNORECASE)
        second_match = re.search(r'(?:2nd|second)\s*down.*?(\d+)%', when_text, re.IGNORECASE)
        third_match = re.search(r'(?:3rd|third)\s*down.*?(\d+)%', when_text, re.IGNORECASE)

        if first_match:
            stats.first_down_pct = float(first_match.group(1)) / 100
        if second_match:
            stats.second_down_pct = float(second_match.group(1)) / 100
        if third_match:
            stats.third_down_pct = float(third_match.group(1)) / 100

        # Long distance
        long_match = re.search(r'long\s*(?:distance|yardage).*?(\d+)%', when_text, re.IGNORECASE)
        if long_match:
            stats.long_distance_pct = float(long_match.group(1)) / 100

        # Field position
        own_match = re.search(r'own\s*territory.*?(\d+)%', when_text, re.IGNORECASE)
        opp_match = re.search(r'opponent\s*territory.*?(\d+)%', when_text, re.IGNORECASE)
        rz_match = re.search(r'red\s*zone.*?(\d+)%', when_text, re.IGNORECASE)

        if own_match:
            stats.own_territory_pct = float(own_match.group(1)) / 100
        if opp_match:
            stats.opp_territory_pct = float(opp_match.group(1)) / 100
        if rz_match:
            stats.redzone_pct = float(rz_match.group(1)) / 100

    # Parse tendencies
    tend_section = re.search(r'Tendencies.*?(?=This formation|-----|\Z)', text, re.DOTALL)
    if tend_section:
        tend_text = tend_section.group(0)
        tendencies = FormationTendencies()

        # Run tendencies
        run_right = re.search(r'Runs:.*?right.*?(\d+)%', tend_text, re.IGNORECASE)
        run_field = re.search(r'Runs:.*?field.*?(\d+)%', tend_text, re.IGNORECASE)
        run_strong = re.search(r'Runs:.*?strong.*?(\d+)%', tend_text, re.IGNORECASE)

        if run_right:
            tendencies.run_right_pct = float(run_right.group(1)) / 100
        if run_field:
            tendencies.run_field_pct = float(run_field.group(1)) / 100
        if run_strong:
            tendencies.run_strong_pct = float(run_strong.group(1)) / 100

        # Pass tendencies
        pass_right = re.search(r'Passes:.*?right.*?(\d+)%', tend_text, re.IGNORECASE)
        pass_field = re.search(r'Passes:.*?field.*?(\d+)%', tend_text, re.IGNORECASE)
        pass_strong = re.search(r'Passes:.*?strong.*?(\d+)%', tend_text, re.IGNORECASE)

        if pass_right:
            tendencies.pass_right_pct = float(pass_right.group(1)) / 100
        if pass_field:
            tendencies.pass_field_pct = float(pass_field.group(1)) / 100
        if pass_strong:
            tendencies.pass_strong_pct = float(pass_strong.group(1)) / 100

        stats.tendencies = tendencies

    # Extract takeaways
    takeaway_matches = re.findall(r'Takeaway:\s*([^\n]+)', text)
    stats.takeaways = takeaway_matches

    return stats


def parse_formation_report(file_path: str) -> Dict[str, FormationStats]:
    """Parse entire formation report file"""
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    formations = {}

    # Split by formation blocks (marked by 🏈 or "Formation:")
    blocks = re.split(r'(?=🏈\s*Formation:|-----)', content)

    for block in blocks:
        if 'Formation:' in block:
            stats = parse_formation_block(block)
            if stats:
                formations[stats.name] = stats

    return formations


def print_formation_summary(formations: Dict[str, FormationStats]):
    """Print summary of parsed formations"""
    print("="*70)
    print("FORMATION REPORT SUMMARY")
    print("="*70)

    for name, stats in sorted(formations.items(), key=lambda x: -x[1].success_rate):
        print(f"\n{name} ({stats.personnel.value} personnel)")
        print(f"  Split: {stats.run_pct:.0%} run / {stats.pass_pct:.0%} pass")
        print(f"  Avg yards: {stats.avg_yards:.1f}")
        print(f"  Success rate: {stats.success_rate:.0%}")
        print(f"  Motion: {stats.motion_pct:.0%}")

        if stats.top_plays:
            print(f"  Top plays:")
            for play in stats.top_plays[:3]:
                play_type = "Run" if play.is_run else "Pass"
                print(f"    - {play.name}: {play.avg_yards:.0f} yds, {play.success_rate:.0%} ({play_type})")

        if stats.takeaways:
            print(f"  Key insight: {stats.takeaways[0][:80]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default to the downloaded file
        file_path = "/Users/rabidraccoon/Downloads/2025 Season Formation report .txt"
    else:
        file_path = sys.argv[1]

    print(f"Parsing {file_path}...")
    formations = parse_formation_report(file_path)
    print(f"Found {len(formations)} formations\n")
    print_formation_summary(formations)
