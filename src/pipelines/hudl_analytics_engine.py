#!/usr/bin/env python3
"""
Third & 20 - Hudl Analytics Engine
Ingests Hudl OC (Offensive Coordinator) export CSVs and generates comprehensive traditional stats.

This produces the "Import" stats from the Analytics Catalog:
- Passing: Completions, Attempts, Yards, TDs, INTs, Completion %, Yards/Attempt
- Rushing: Carries, Yards, TDs, Yards/Carry, Long Rush, Direction Breakdown
- Receiving: Receptions, Targets, Yards, TDs
- Total Offense: Total Yards, Plays, Yards/Play, First Downs, Turnovers
- Defense: Rushes Allowed, Yards Allowed, etc.
- Special Teams: FGs, XPs, Punt Yards, Returns
- Situational: Down/Distance, Red Zone, Formation tendencies

Usage:
    python hudl_analytics_engine.py --hudl-dir /path/to/hudl/csvs --output /path/to/output
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class HudlDataLoader:
    """Load and normalize Hudl OC export CSVs"""

    def __init__(self, hudl_dir: str):
        self.hudl_dir = Path(hudl_dir)
        self.games: Dict[str, pd.DataFrame] = {}
        self.all_plays = pd.DataFrame()

    def load_all_games(self) -> pd.DataFrame:
        """Load all Hudl CSVs from directory"""
        csv_files = list(self.hudl_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} Hudl CSV files")

        all_dfs = []
        for csv_file in csv_files:
            game_name = self._extract_game_name(csv_file.name)
            print(f"  Loading: {csv_file.name} -> {game_name}")

            try:
                df = self._load_single_game(csv_file, game_name)
                if df is not None and len(df) > 0:
                    self.games[game_name] = df
                    all_dfs.append(df)
            except Exception as e:
                print(f"    ERROR: {e}")

        if all_dfs:
            self.all_plays = pd.concat(all_dfs, ignore_index=True)
            print(f"\nTotal plays loaded: {len(self.all_plays)}")

        return self.all_plays

    def _extract_game_name(self, filename: str) -> str:
        """Extract opponent name from filename"""
        name = filename.replace('.csv', '').replace(' OC Hudl', '').replace(' Offense OC Hudl', '')
        name = name.replace(' by OC Hudl', '').replace(' OC', '')
        # Handle special cases
        mappings = {
            'Bedford': 'Bedford',
            'CCCH': 'CCC',
            'Erie': 'Erie',
            'Euclid': 'Euclid',
            'Garfield Heights': 'Garfield',
            'Hoban': 'Hoban',
            'Shaw': 'Shaw',
            'Warrensville': 'Warrensville',
            'WilloughbySouth': 'Willoughby',
            'Offense 2025 season': 'Season_Total'
        }
        for key, value in mappings.items():
            if key in name:
                return value
        return name

    def _load_single_game(self, csv_path: Path, game_name: str) -> Optional[pd.DataFrame]:
        """Load and normalize a single Hudl CSV"""
        # Try to read, handling potential header issues
        try:
            # First check if there's a "Table 1" header row
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()

            skiprows = 1 if first_line == 'Table 1' else 0
            df = pd.read_csv(csv_path, skiprows=skiprows)
        except Exception as e:
            print(f"    Read error: {e}")
            return None

        # Standardize column names
        df.columns = [c.strip().upper().replace(' ', '_') for c in df.columns]

        # Add game identifier
        df['GAME'] = game_name

        # Clean up data types
        if 'DN' in df.columns:
            df['DN'] = pd.to_numeric(df['DN'], errors='coerce')
        if 'DIST' in df.columns:
            df['DIST'] = pd.to_numeric(df['DIST'], errors='coerce')
        if 'GN/LS' in df.columns:
            df['YARDS'] = pd.to_numeric(df['GN/LS'], errors='coerce').fillna(0)
        if 'YARD_LN' in df.columns:
            df['YARD_LN'] = pd.to_numeric(df['YARD_LN'], errors='coerce')
        if 'QTR' in df.columns:
            df['QTR'] = pd.to_numeric(df['QTR'], errors='coerce')

        # Standardize ODK column
        if 'ODK' in df.columns:
            df['UNIT'] = df['ODK'].str.upper().str.strip()

        # Standardize result column
        if 'RESULT' in df.columns:
            df['RESULT'] = df['RESULT'].fillna('').str.strip()

        # Standardize play type
        if 'PLAY_TYPE' in df.columns:
            df['PLAY_TYPE'] = df['PLAY_TYPE'].fillna('').str.strip()

        return df


class TraditionalStatsCalculator:
    """Calculate traditional football statistics from Hudl data"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Filter to actual plays (not special teams markers)
        self.plays = df[df['UNIT'].isin(['O', 'D'])].copy()

    def calculate_passing_stats(self, unit: str = 'O') -> Dict:
        """Calculate passing statistics for offense (O) or defense allowed (D)"""
        plays = self.plays[self.plays['UNIT'] == unit]
        pass_plays = plays[plays['PLAY_TYPE'].str.contains('Pass', case=False, na=False)]

        completions = len(pass_plays[pass_plays['RESULT'].str.contains('Complete', case=False, na=False)])
        attempts = len(pass_plays[~pass_plays['RESULT'].str.contains('Sack|Scramble', case=False, na=False)])

        # Yards on completions only
        complete_plays = pass_plays[pass_plays['RESULT'].str.contains('Complete', case=False, na=False)]
        yards = complete_plays['YARDS'].sum()

        # TDs
        tds = len(pass_plays[pass_plays['RESULT'].str.contains('TD', case=False, na=False)])

        # Interceptions
        ints = len(pass_plays[pass_plays['RESULT'].str.contains('INT|Interception', case=False, na=False)])

        # Sacks
        sacks = len(pass_plays[pass_plays['RESULT'].str.contains('Sack', case=False, na=False)])
        sack_yards = pass_plays[pass_plays['RESULT'].str.contains('Sack', case=False, na=False)]['YARDS'].sum()

        # Incompletes
        incompletes = len(pass_plays[pass_plays['RESULT'].str.contains('Incomplete', case=False, na=False)])

        comp_pct = (completions / attempts * 100) if attempts > 0 else 0
        yards_per_att = yards / attempts if attempts > 0 else 0

        return {
            'completions': completions,
            'attempts': attempts,
            'passing_yards': yards,
            'passing_tds': tds,
            'interceptions': ints,
            'completion_pct': round(comp_pct, 1),
            'yards_per_attempt': round(yards_per_att, 1),
            'sacks': sacks,
            'sack_yards_lost': abs(sack_yards),
            'incompletes': incompletes
        }

    def calculate_rushing_stats(self, unit: str = 'O') -> Dict:
        """Calculate rushing statistics"""
        plays = self.plays[self.plays['UNIT'] == unit]
        rush_plays = plays[plays['PLAY_TYPE'].str.contains('Run', case=False, na=False)]

        carries = len(rush_plays)
        yards = rush_plays['YARDS'].sum()
        tds = len(rush_plays[rush_plays['RESULT'].str.contains('TD', case=False, na=False)])

        # Long rush
        long_rush = rush_plays['YARDS'].max() if len(rush_plays) > 0 else 0

        # Direction breakdown
        left = rush_plays[rush_plays['PLAY_DIR'] == 'L']['YARDS'].sum()
        middle = rush_plays[rush_plays['PLAY_DIR'] == 'M']['YARDS'].sum()
        right = rush_plays[rush_plays['PLAY_DIR'] == 'R']['YARDS'].sum()

        left_carries = len(rush_plays[rush_plays['PLAY_DIR'] == 'L'])
        middle_carries = len(rush_plays[rush_plays['PLAY_DIR'] == 'M'])
        right_carries = len(rush_plays[rush_plays['PLAY_DIR'] == 'R'])

        ypc = yards / carries if carries > 0 else 0

        return {
            'carries': carries,
            'rushing_yards': yards,
            'rushing_tds': tds,
            'yards_per_carry': round(ypc, 1),
            'long_rush': long_rush,
            'left_carries': left_carries,
            'left_yards': left,
            'middle_carries': middle_carries,
            'middle_yards': middle,
            'right_carries': right_carries,
            'right_yards': right
        }

    def calculate_total_offense(self, unit: str = 'O') -> Dict:
        """Calculate total offensive statistics"""
        plays = self.plays[self.plays['UNIT'] == unit]

        passing = self.calculate_passing_stats(unit)
        rushing = self.calculate_rushing_stats(unit)

        total_plays = len(plays[plays['PLAY_TYPE'].str.contains('Pass|Run', case=False, na=False)])
        total_yards = passing['passing_yards'] + rushing['rushing_yards']
        total_tds = passing['passing_tds'] + rushing['rushing_tds']

        # First downs (plays that resulted in new set of downs or TD)
        # Approximate: plays where yards >= distance needed
        first_downs = 0
        for _, play in plays.iterrows():
            if pd.notna(play.get('DIST')) and pd.notna(play.get('YARDS')):
                if play['YARDS'] >= play['DIST'] or 'TD' in str(play.get('RESULT', '')):
                    first_downs += 1

        # Turnovers
        turnovers = passing['interceptions']
        fumbles = len(plays[plays['RESULT'].str.contains('Fumble', case=False, na=False)])
        turnovers += fumbles

        ypp = total_yards / total_plays if total_plays > 0 else 0

        return {
            'total_plays': total_plays,
            'total_yards': total_yards,
            'total_tds': total_tds,
            'yards_per_play': round(ypp, 1),
            'first_downs': first_downs,
            'turnovers': turnovers,
            'fumbles': fumbles,
            **passing,
            **rushing
        }

    def calculate_situational_stats(self, unit: str = 'O') -> Dict:
        """Calculate situational statistics"""
        plays = self.plays[self.plays['UNIT'] == unit]

        # Down breakdown
        first_down_plays = plays[plays['DN'] == 1]
        second_down_plays = plays[plays['DN'] == 2]
        third_down_plays = plays[plays['DN'] == 3]
        fourth_down_plays = plays[plays['DN'] == 4]

        # Third down conversions
        third_conversions = 0
        for _, play in third_down_plays.iterrows():
            if pd.notna(play.get('DIST')) and pd.notna(play.get('YARDS')):
                if play['YARDS'] >= play['DIST'] or 'TD' in str(play.get('RESULT', '')):
                    third_conversions += 1

        third_down_pct = (third_conversions / len(third_down_plays) * 100) if len(third_down_plays) > 0 else 0

        # Red zone (inside 20)
        red_zone_plays = plays[plays['YARD_LN'].between(1, 20)]
        red_zone_tds = len(red_zone_plays[red_zone_plays['RESULT'].str.contains('TD', case=False, na=False)])
        red_zone_attempts = len(red_zone_plays['YARD_LN'].dropna().unique())  # Approximate drives

        # Run/Pass ratio
        run_plays = len(plays[plays['PLAY_TYPE'].str.contains('Run', case=False, na=False)])
        pass_plays = len(plays[plays['PLAY_TYPE'].str.contains('Pass', case=False, na=False)])
        total = run_plays + pass_plays

        return {
            'first_down_plays': len(first_down_plays),
            'second_down_plays': len(second_down_plays),
            'third_down_plays': len(third_down_plays),
            'third_down_conversions': third_conversions,
            'third_down_pct': round(third_down_pct, 1),
            'fourth_down_plays': len(fourth_down_plays),
            'red_zone_plays': len(red_zone_plays),
            'red_zone_tds': red_zone_tds,
            'run_plays': run_plays,
            'pass_plays': pass_plays,
            'run_pct': round(run_plays / total * 100, 1) if total > 0 else 0,
            'pass_pct': round(pass_plays / total * 100, 1) if total > 0 else 0
        }

    def calculate_formation_stats(self, unit: str = 'O') -> Dict:
        """Calculate formation tendency statistics"""
        plays = self.plays[self.plays['UNIT'] == unit]

        if 'OFF_FORM' not in plays.columns:
            return {}

        formation_counts = plays['OFF_FORM'].value_counts()
        formation_yards = plays.groupby('OFF_FORM')['YARDS'].agg(['sum', 'mean', 'count'])

        formations = {}
        for form in formation_counts.head(10).index:
            if pd.isna(form) or form == '':
                continue
            formations[f"form_{form}_plays"] = int(formation_counts.get(form, 0))
            formations[f"form_{form}_yards"] = int(formation_yards.loc[form, 'sum']) if form in formation_yards.index else 0
            formations[f"form_{form}_ypp"] = round(formation_yards.loc[form, 'mean'], 1) if form in formation_yards.index else 0

        return formations

    def calculate_special_teams_stats(self) -> Dict:
        """Calculate special teams statistics"""
        df = self.df

        # Kickoffs
        kickoffs = df[df['PLAY_TYPE'].str.contains('KO', case=False, na=False)]
        ko_returns = kickoffs[kickoffs['RESULT'].str.contains('Return', case=False, na=False)]

        # Punts
        punts = df[df['PLAY_TYPE'].str.contains('Punt', case=False, na=False)]
        punt_returns = punts[punts['RESULT'].str.contains('Return', case=False, na=False)]

        # Field Goals
        fgs = df[df['PLAY_TYPE'].str.contains('FG|Field Goal', case=False, na=False)]
        fg_made = len(fgs[fgs['RESULT'].str.contains('Good|Made', case=False, na=False)])
        fg_att = len(fgs)

        # Extra Points
        xps = df[df['PLAY_TYPE'].str.contains('Extra|XP|PAT', case=False, na=False)]
        xp_made = len(xps[xps['RESULT'].str.contains('Good|Made', case=False, na=False)])
        xp_att = len(xps)

        return {
            'kickoff_returns': len(ko_returns),
            'kickoff_return_yards': ko_returns['YARDS'].sum() if len(ko_returns) > 0 else 0,
            'punts': len(punts),
            'punt_returns': len(punt_returns),
            'punt_return_yards': punt_returns['YARDS'].sum() if len(punt_returns) > 0 else 0,
            'fg_made': fg_made,
            'fg_attempted': fg_att,
            'fg_pct': round(fg_made / fg_att * 100, 1) if fg_att > 0 else 0,
            'xp_made': xp_made,
            'xp_attempted': xp_att
        }


class HudlAnalyticsEngine:
    """Main analytics engine combining all Hudl statistics"""

    def __init__(self, hudl_dir: str):
        self.loader = HudlDataLoader(hudl_dir)
        self.all_plays = None
        self.game_stats: Dict[str, Dict] = {}
        self.season_stats: Dict = {}

    def run_analysis(self, output_prefix: str):
        """Run full analysis and generate output files"""
        print("=" * 60)
        print("THIRD & 20 - HUDL ANALYTICS ENGINE")
        print("=" * 60)

        # Load data
        print("\n[1/4] Loading Hudl data...")
        self.all_plays = self.loader.load_all_games()

        if self.all_plays.empty:
            print("ERROR: No data loaded")
            return

        # Calculate per-game stats
        print("\n[2/4] Calculating per-game statistics...")
        self._calculate_game_stats()

        # Calculate season totals
        print("\n[3/4] Calculating season totals...")
        self._calculate_season_stats()

        # Generate output files
        print("\n[4/4] Generating output files...")
        self._generate_outputs(output_prefix)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

    def _calculate_game_stats(self):
        """Calculate statistics for each game"""
        for game_name, game_df in self.loader.games.items():
            if game_name == 'Season_Total':
                continue

            print(f"  Calculating: {game_name}")
            calc = TraditionalStatsCalculator(game_df)

            self.game_stats[game_name] = {
                'game': game_name,
                'offense': calc.calculate_total_offense('O'),
                'defense_allowed': calc.calculate_total_offense('D'),
                'situational': calc.calculate_situational_stats('O'),
                'formations': calc.calculate_formation_stats('O'),
                'special_teams': calc.calculate_special_teams_stats()
            }

    def _calculate_season_stats(self):
        """Calculate season totals"""
        # Use Season_Total if available, otherwise aggregate
        if 'Season_Total' in self.loader.games:
            season_df = self.loader.games['Season_Total']
        else:
            season_df = self.all_plays

        calc = TraditionalStatsCalculator(season_df)

        self.season_stats = {
            'offense': calc.calculate_total_offense('O'),
            'defense_allowed': calc.calculate_total_offense('D'),
            'situational': calc.calculate_situational_stats('O'),
            'formations': calc.calculate_formation_stats('O'),
            'special_teams': calc.calculate_special_teams_stats()
        }

    def _generate_outputs(self, output_prefix: str):
        """Generate all output CSV files"""

        # 1. Per-game offensive stats
        game_rows = []
        for game_name, stats in self.game_stats.items():
            row = {'game': game_name}
            row.update(stats['offense'])
            row.update({f"def_{k}": v for k, v in stats['defense_allowed'].items()})
            row.update(stats['situational'])
            game_rows.append(row)

        if game_rows:
            df = pd.DataFrame(game_rows)
            path = f"{output_prefix}_game_traditional_stats.csv"
            df.to_csv(path, index=False)
            print(f"    Saved: {path}")

        # 2. Season totals
        season_row = {'category': 'Season Total'}
        season_row.update(self.season_stats['offense'])
        season_row.update({f"def_{k}": v for k, v in self.season_stats['defense_allowed'].items()})
        season_row.update(self.season_stats['situational'])

        path = f"{output_prefix}_season_traditional_stats.csv"
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for k, v in season_row.items():
                writer.writerow([k, v])
        print(f"    Saved: {path}")

        # 3. Formation tendencies
        form_rows = []
        for game_name, stats in self.game_stats.items():
            for k, v in stats['formations'].items():
                form_rows.append({'game': game_name, 'metric': k, 'value': v})

        if form_rows:
            df = pd.DataFrame(form_rows)
            path = f"{output_prefix}_formation_tendencies.csv"
            df.to_csv(path, index=False)
            print(f"    Saved: {path}")

        # 4. Detailed play-by-play export
        if self.all_plays is not None and len(self.all_plays) > 0:
            path = f"{output_prefix}_all_plays.csv"
            self.all_plays.to_csv(path, index=False)
            print(f"    Saved: {path}")

        # 5. Summary report
        self._generate_summary_report(output_prefix)

    def _generate_summary_report(self, output_prefix: str):
        """Generate human-readable summary report"""
        path = f"{output_prefix}_summary_report.txt"

        with open(path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("THIRD & 20 - TRADITIONAL STATS SUMMARY\n")
            f.write("Imported from Hudl OC Data\n")
            f.write("=" * 60 + "\n\n")

            # Season totals
            off = self.season_stats['offense']
            f.write("SEASON OFFENSIVE TOTALS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Plays: {off['total_plays']}\n")
            f.write(f"Total Yards: {off['total_yards']}\n")
            f.write(f"Yards/Play: {off['yards_per_play']}\n")
            f.write(f"Total TDs: {off['total_tds']}\n")
            f.write(f"Turnovers: {off['turnovers']}\n\n")

            f.write("PASSING\n")
            f.write(f"  {off['completions']}/{off['attempts']} for {off['passing_yards']} yards\n")
            f.write(f"  Completion %: {off['completion_pct']}%\n")
            f.write(f"  Yards/Attempt: {off['yards_per_attempt']}\n")
            f.write(f"  TDs: {off['passing_tds']}, INTs: {off['interceptions']}\n")
            f.write(f"  Sacks: {off['sacks']} for {off['sack_yards_lost']} yards\n\n")

            f.write("RUSHING\n")
            f.write(f"  {off['carries']} carries for {off['rushing_yards']} yards\n")
            f.write(f"  Yards/Carry: {off['yards_per_carry']}\n")
            f.write(f"  TDs: {off['rushing_tds']}, Long: {off['long_rush']}\n")
            f.write(f"  Direction: L={off['left_yards']}, M={off['middle_yards']}, R={off['right_yards']}\n\n")

            sit = self.season_stats['situational']
            f.write("SITUATIONAL\n")
            f.write(f"  3rd Down: {sit['third_down_conversions']}/{sit['third_down_plays']} ({sit['third_down_pct']}%)\n")
            f.write(f"  Red Zone TDs: {sit['red_zone_tds']}\n")
            f.write(f"  Run/Pass: {sit['run_pct']}% / {sit['pass_pct']}%\n\n")

            # Per-game breakdown
            f.write("=" * 60 + "\n")
            f.write("PER-GAME BREAKDOWN\n")
            f.write("=" * 60 + "\n\n")

            for game_name in sorted(self.game_stats.keys()):
                stats = self.game_stats[game_name]
                off = stats['offense']
                f.write(f"vs {game_name}\n")
                f.write(f"  Total: {off['total_yards']} yards on {off['total_plays']} plays ({off['yards_per_play']} ypp)\n")
                f.write(f"  Pass: {off['completions']}/{off['attempts']} for {off['passing_yards']} yards, {off['passing_tds']} TD\n")
                f.write(f"  Rush: {off['carries']} for {off['rushing_yards']} yards, {off['rushing_tds']} TD\n")
                f.write("\n")

        print(f"    Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Third & 20 - Hudl Analytics Engine")
    parser.add_argument("--hudl-dir", "-d", required=True, help="Directory containing Hudl CSV exports")
    parser.add_argument("--output", "-o", default="hudl_analytics", help="Output file prefix")

    args = parser.parse_args()

    engine = HudlAnalyticsEngine(args.hudl_dir)
    engine.run_analysis(args.output)


if __name__ == "__main__":
    main()
