#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third & 20 - Analytics Engine v1.0
Aggregates CV pipeline output into all catalog-promised analytics

This engine processes the raw SDI pipeline outputs and generates:
1. SECTION 4: Third & 20 Unique Metrics (CV-based)
2. SECTION 3: Position-Specific Metrics
3. Game/Season/Player Summaries
4. Defensive Tendency Reports

Input: *_offensive_sdi.csv, *_defensive_analysis.csv, *_defensive_players.csv
Output: Comprehensive analytics CSVs and summary reports

Usage:
    python analytics_engine.py --data-dir /path/to/Brush\ Demo\ stats --roster /path/to/roster.csv -o output_prefix
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import csv
import argparse
import json
from collections import defaultdict


# =============================================================================
# POSITION MAPPING
# =============================================================================

POSITION_GROUPS = {
    'QB': ['QB'],
    'RB': ['RB', 'FB', 'HB'],
    'WR': ['WR', 'SE', 'FL'],
    'TE': ['TE'],
    'OL': ['OL', 'T', 'G', 'C', 'OT', 'OG', 'LT', 'RT', 'LG', 'RG'],
    'DL': ['DL', 'DE', 'DT', 'NT', 'NG', 'LE', 'RE'],
    'LB': ['LB', 'MLB', 'OLB', 'ILB', 'WLB', 'SLB', 'MIKE', 'WILL', 'SAM'],
    'DB': ['DB', 'CB', 'S', 'FS', 'SS', 'NB', 'SAFETY'],
}

def get_position_group(position_str: str) -> str:
    """Map a position string to its group (QB, RB, WR, etc.)"""
    if not position_str or pd.isna(position_str):
        return 'UNKNOWN'

    # Handle comma-separated positions (e.g., "RB, MLB")
    positions = [p.strip().upper() for p in str(position_str).split(',')]

    for pos in positions:
        for group, variants in POSITION_GROUPS.items():
            if pos in variants:
                return group

    return 'UNKNOWN'


# =============================================================================
# DATA LOADERS
# =============================================================================

class GameDataLoader:
    """Loads and combines all game data from a directory"""

    def __init__(self, data_dir: str, roster_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.roster = self._load_roster(roster_path) if roster_path else {}

    def _load_roster(self, roster_path: str) -> Dict[int, Dict]:
        """Load roster CSV into lookup dict"""
        roster = {}
        try:
            df = pd.read_csv(roster_path)
            # Handle different column names
            jersey_col = '#' if '#' in df.columns else 'Jersey'
            name_col = 'Name' if 'Name' in df.columns else 'name'
            pos_col = 'Pos.' if 'Pos.' in df.columns else 'Position'

            for _, row in df.iterrows():
                try:
                    jersey = int(row[jersey_col])
                    roster[jersey] = {
                        'name': row.get(name_col, ''),
                        'position': row.get(pos_col, ''),
                        'position_group': get_position_group(row.get(pos_col, ''))
                    }
                except (ValueError, KeyError):
                    continue
        except Exception as e:
            print(f"Warning: Could not load roster: {e}")
        return roster

    def get_game_files(self) -> List[str]:
        """Get list of unique game prefixes"""
        games = set()
        for f in self.data_dir.glob("*_offensive_sdi.csv"):
            prefix = f.name.replace("_offensive_sdi.csv", "")
            games.add(prefix)
        return sorted(games)

    def load_game(self, game_prefix: str) -> Dict[str, pd.DataFrame]:
        """Load all data files for a single game"""
        data = {}

        offensive_path = self.data_dir / f"{game_prefix}_offensive_sdi.csv"
        if offensive_path.exists():
            data['offensive'] = pd.read_csv(offensive_path)

        defensive_path = self.data_dir / f"{game_prefix}_defensive_analysis.csv"
        if defensive_path.exists():
            data['defensive_plays'] = pd.read_csv(defensive_path)

        defensive_players_path = self.data_dir / f"{game_prefix}_defensive_players.csv"
        if defensive_players_path.exists():
            data['defensive_players'] = pd.read_csv(defensive_players_path)

        play_summary_path = self.data_dir / f"{game_prefix}_play_summary.csv"
        if play_summary_path.exists():
            data['play_summary'] = pd.read_csv(play_summary_path)

        return data

    def load_all_games(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all games in directory"""
        all_data = {}
        for game in self.get_game_files():
            # Skip pro athlete files
            if 'lamar' in game.lower() or 'ricky' in game.lower() or 'myles' in game.lower():
                continue
            all_data[game] = self.load_game(game)
        return all_data


# =============================================================================
# SECTION 4.1: COGNITIVE METRICS (CV-Based)
# =============================================================================

class CognitiveMetricsCalculator:
    """Calculates Third & 20 unique cognitive metrics"""

    @staticmethod
    def calculate_recognition_latency_stats(df: pd.DataFrame) -> Dict:
        """Aggregate Recognition Latency stats"""
        rl = df['recognition_latency_sec'].dropna()
        if len(rl) == 0:
            return {}

        return {
            'rl_mean': rl.mean(),
            'rl_median': rl.median(),
            'rl_std': rl.std(),
            'rl_min': rl.min(),
            'rl_max': rl.max(),
            'rl_elite_count': (rl < 0.5).sum(),
            'rl_elite_pct': (rl < 0.5).mean() * 100,
            'rl_developmental_count': (rl >= 1.6).sum(),
            'rl_developmental_pct': (rl >= 1.6).mean() * 100,
        }

    @staticmethod
    def calculate_first_step_stats(df: pd.DataFrame) -> Dict:
        """Aggregate First Step Quickness stats"""
        fs = df['first_step_sec'].dropna()
        if len(fs) == 0:
            return {}

        return {
            'fsq_mean': fs.mean(),
            'fsq_median': fs.median(),
            'fsq_std': fs.std(),
            'fsq_min': fs.min(),
            'fsq_max': fs.max(),
            'fsq_elite_count': (fs < 0.3).sum(),
            'fsq_elite_pct': (fs < 0.3).mean() * 100,
            'fsq_developmental_count': (fs >= 0.8).sum(),
            'fsq_developmental_pct': (fs >= 0.8).mean() * 100,
        }

    @staticmethod
    def calculate_sdi_stats(df: pd.DataFrame) -> Dict:
        """Aggregate SDI Composite stats"""
        sdi = df['sdi_score'].dropna()
        if len(sdi) == 0:
            return {}

        grades = df['sdi_grade'].value_counts()
        total = len(df['sdi_grade'].dropna())

        return {
            'sdi_mean': sdi.mean(),
            'sdi_median': sdi.median(),
            'sdi_std': sdi.std(),
            'sdi_min': sdi.min(),
            'sdi_max': sdi.max(),
            'sdi_elite_count': grades.get('Elite', 0),
            'sdi_elite_pct': grades.get('Elite', 0) / total * 100 if total > 0 else 0,
            'sdi_above_avg_count': grades.get('Above Average', 0),
            'sdi_above_avg_pct': grades.get('Above Average', 0) / total * 100 if total > 0 else 0,
            'sdi_average_count': grades.get('Average', 0),
            'sdi_average_pct': grades.get('Average', 0) / total * 100 if total > 0 else 0,
            'sdi_below_avg_count': grades.get('Below Average', 0),
            'sdi_below_avg_pct': grades.get('Below Average', 0) / total * 100 if total > 0 else 0,
            'sdi_developmental_count': grades.get('Developmental', 0),
            'sdi_developmental_pct': grades.get('Developmental', 0) / total * 100 if total > 0 else 0,
        }


# =============================================================================
# SECTION 4.2: DEFENSIVE SCHEME DETECTION
# =============================================================================

class DefensiveSchemeAnalyzer:
    """Analyzes defensive scheme tendencies"""

    @staticmethod
    def analyze_fronts(df: pd.DataFrame) -> Dict:
        """Analyze defensive front distribution"""
        if 'front_description' not in df.columns:
            return {}

        fronts = df['front_description'].value_counts()
        total = len(df)

        result = {
            'total_plays_analyzed': total,
            'front_distribution': {},
        }

        for front, count in fronts.items():
            safe_name = str(front).replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            result['front_distribution'][front] = {
                'count': int(count),
                'percentage': count / total * 100
            }
            result[f'front_{safe_name}_count'] = int(count)
            result[f'front_{safe_name}_pct'] = count / total * 100

        return result

    @staticmethod
    def analyze_coverage_shells(df: pd.DataFrame) -> Dict:
        """Analyze coverage shell distribution"""
        if 'coverage_shell' not in df.columns:
            return {}

        shells = df['coverage_shell'].value_counts()
        total = len(df)

        result = {
            'coverage_distribution': {},
        }

        for shell, count in shells.items():
            safe_name = str(shell).replace(' ', '_')
            result['coverage_distribution'][shell] = {
                'count': int(count),
                'percentage': count / total * 100
            }
            result[f'coverage_{safe_name}_count'] = int(count)
            result[f'coverage_{safe_name}_pct'] = count / total * 100

        return result

    @staticmethod
    def analyze_safety_alignment(df: pd.DataFrame) -> Dict:
        """Analyze safety alignment distribution"""
        if 'safety_alignment' not in df.columns:
            return {}

        alignments = df['safety_alignment'].value_counts()
        total = len(df)

        result = {
            'safety_distribution': {},
        }

        for align, count in alignments.items():
            safe_name = str(align).replace('-', '_')
            result['safety_distribution'][align] = {
                'count': int(count),
                'percentage': count / total * 100
            }
            result[f'safety_{safe_name}_count'] = int(count)
            result[f'safety_{safe_name}_pct'] = count / total * 100

        return result

    @staticmethod
    def analyze_box_counts(df: pd.DataFrame) -> Dict:
        """Analyze box count distribution"""
        if 'box_count' not in df.columns:
            return {}

        box = df['box_count']

        return {
            'box_count_mean': box.mean(),
            'box_count_median': box.median(),
            'box_count_min': int(box.min()),
            'box_count_max': int(box.max()),
            'light_box_count': int((box <= 5).sum()),  # 5 or fewer
            'light_box_pct': (box <= 5).mean() * 100,
            'stacked_box_count': int((box >= 7).sum()),  # 7 or more
            'stacked_box_pct': (box >= 7).mean() * 100,
        }

    @staticmethod
    def analyze_blitz_tendency(df: pd.DataFrame) -> Dict:
        """Analyze blitz tendencies"""
        if 'potential_blitz' not in df.columns:
            return {}

        blitz = df['potential_blitz']
        total = len(df)

        blitz_count = blitz.sum() if blitz.dtype == bool else (blitz == True).sum()

        return {
            'blitz_looks_count': int(blitz_count),
            'blitz_looks_pct': blitz_count / total * 100 if total > 0 else 0,
            'no_blitz_count': int(total - blitz_count),
            'no_blitz_pct': (total - blitz_count) / total * 100 if total > 0 else 0,
        }


# =============================================================================
# SECTION 3: POSITION-SPECIFIC METRICS
# =============================================================================

class PositionMetricsCalculator:
    """Calculates position-specific metrics from tracking data"""

    def __init__(self, roster: Dict[int, Dict]):
        self.roster = roster

    def enrich_with_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position_group column based on roster lookup"""
        df = df.copy()

        def get_pos_group(row):
            jersey = row.get('jersey_number')
            roster_pos = row.get('roster_position', '')

            # Try roster lookup first
            if pd.notna(jersey) and int(jersey) in self.roster:
                return self.roster[int(jersey)]['position_group']

            # Fall back to roster_position column
            if roster_pos:
                return get_position_group(roster_pos)

            return 'UNKNOWN'

        df['position_group'] = df.apply(get_pos_group, axis=1)
        return df

    def calculate_qb_metrics(self, df: pd.DataFrame) -> Dict:
        """QB-specific: Decision latency, recognition latency"""
        qb_df = df[df['position_group'] == 'QB']
        if len(qb_df) == 0:
            return {}

        result = {
            'qb_plays': len(qb_df),
            'qb_unique_players': qb_df['jersey_number'].nunique(),
        }

        # Recognition Latency (time to read defense) - UNIQUE TO THIRD & 20
        rl = qb_df['recognition_latency_sec'].dropna()
        if len(rl) > 0:
            result['qb_recognition_latency_avg'] = rl.mean()
            result['qb_recognition_latency_best'] = rl.min()
            result['qb_decision_speed_grade'] = 'Elite' if rl.mean() < 0.5 else 'Above Average' if rl.mean() < 0.8 else 'Average' if rl.mean() < 1.2 else 'Below Average'

        # First Step (for scrambling QBs)
        fs = qb_df['first_step_sec'].dropna()
        if len(fs) > 0:
            result['qb_first_step_avg'] = fs.mean()
            result['qb_mobility_grade'] = 'Elite' if fs.mean() < 0.3 else 'Above Average' if fs.mean() < 0.45 else 'Average'

        return result

    def calculate_wr_te_metrics(self, df: pd.DataFrame) -> Dict:
        """WR/TE-specific: First step quickness (release)"""
        wr_df = df[df['position_group'].isin(['WR', 'TE'])]
        if len(wr_df) == 0:
            return {}

        result = {
            'wr_te_plays': len(wr_df),
            'wr_te_unique_players': wr_df['jersey_number'].nunique(),
        }

        # First Step Quickness (release off line) - UNIQUE TO THIRD & 20
        fs = wr_df['first_step_sec'].dropna()
        if len(fs) > 0:
            result['wr_te_first_step_avg'] = fs.mean()
            result['wr_te_first_step_best'] = fs.min()
            result['wr_te_release_grade'] = 'Elite' if fs.mean() < 0.3 else 'Above Average' if fs.mean() < 0.45 else 'Average'

        # SDI for skill players
        sdi = wr_df['sdi_score'].dropna()
        if len(sdi) > 0:
            result['wr_te_sdi_avg'] = sdi.mean()

        return result

    def calculate_rb_metrics(self, df: pd.DataFrame) -> Dict:
        """RB-specific: First step, initial velocity"""
        rb_df = df[df['position_group'] == 'RB']
        if len(rb_df) == 0:
            return {}

        result = {
            'rb_plays': len(rb_df),
            'rb_unique_players': rb_df['jersey_number'].nunique(),
        }

        # First Step Quickness - UNIQUE TO THIRD & 20
        fs = rb_df['first_step_sec'].dropna()
        if len(fs) > 0:
            result['rb_first_step_avg'] = fs.mean()
            result['rb_first_step_best'] = fs.min()
            result['rb_burst_grade'] = 'Elite' if fs.mean() < 0.3 else 'Above Average' if fs.mean() < 0.45 else 'Average'

        # Initial Velocity - UNIQUE TO THIRD & 20
        vel = rb_df['initial_velocity'].dropna()
        if len(vel) > 0:
            result['rb_initial_velocity_avg'] = vel.mean()
            result['rb_initial_velocity_max'] = vel.max()

        # Post-snap distance (burst)
        dist = rb_df['post_snap_distance'].dropna()
        if len(dist) > 0:
            result['rb_post_snap_distance_avg'] = dist.mean()

        return result

    def calculate_dl_lb_metrics(self, df: pd.DataFrame) -> Dict:
        """DL/LB-specific: First step, recognition latency"""
        dl_lb_df = df[df['position_group'].isin(['DL', 'LB'])]
        if len(dl_lb_df) == 0:
            return {}

        result = {
            'dl_lb_plays': len(dl_lb_df),
            'dl_lb_unique_players': dl_lb_df['jersey_number'].nunique(),
        }

        # First Step Quickness (get-off) - UNIQUE TO THIRD & 20
        fs = dl_lb_df['first_step_sec'].dropna()
        if len(fs) > 0:
            result['dl_lb_first_step_avg'] = fs.mean()
            result['dl_lb_first_step_best'] = fs.min()
            result['dl_lb_get_off_grade'] = 'Elite' if fs.mean() < 0.3 else 'Above Average' if fs.mean() < 0.45 else 'Average'

        # Recognition Latency (read-react) - UNIQUE TO THIRD & 20
        rl = dl_lb_df['recognition_latency_sec'].dropna()
        if len(rl) > 0:
            result['dl_lb_recognition_latency_avg'] = rl.mean()
            result['dl_lb_read_react_grade'] = 'Elite' if rl.mean() < 0.5 else 'Above Average' if rl.mean() < 0.8 else 'Average'

        return result

    def calculate_db_metrics(self, df: pd.DataFrame) -> Dict:
        """DB-specific: Reaction time, coverage metrics"""
        db_df = df[df['position_group'] == 'DB']
        if len(db_df) == 0:
            return {}

        result = {
            'db_plays': len(db_df),
            'db_unique_players': db_df['jersey_number'].nunique(),
        }

        # Reaction Time (first step as proxy) - UNIQUE TO THIRD & 20
        fs = db_df['first_step_sec'].dropna()
        if len(fs) > 0:
            result['db_reaction_time_avg'] = fs.mean()
            result['db_reaction_time_best'] = fs.min()
            result['db_reaction_grade'] = 'Elite' if fs.mean() < 0.3 else 'Above Average' if fs.mean() < 0.45 else 'Average'

        # Recognition Latency
        rl = db_df['recognition_latency_sec'].dropna()
        if len(rl) > 0:
            result['db_recognition_latency_avg'] = rl.mean()

        return result

    def calculate_all_position_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate all position-specific metrics"""
        df = self.enrich_with_positions(df)

        result = {}
        result.update(self.calculate_qb_metrics(df))
        result.update(self.calculate_wr_te_metrics(df))
        result.update(self.calculate_rb_metrics(df))
        result.update(self.calculate_dl_lb_metrics(df))
        result.update(self.calculate_db_metrics(df))

        return result


# =============================================================================
# PLAYER RANKINGS
# =============================================================================

class PlayerRankingCalculator:
    """Calculates player rankings by various metrics"""

    @staticmethod
    def rank_by_sdi(df: pd.DataFrame, min_plays: int = 3) -> pd.DataFrame:
        """Rank players by SDI score"""
        # Group by player
        player_stats = df.groupby(['jersey_number', 'team_name']).agg({
            'sdi_score': ['mean', 'count', 'std'],
            'first_step_sec': 'mean',
            'recognition_latency_sec': 'mean',
            'sdi_grade': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()

        player_stats.columns = ['jersey_number', 'team', 'sdi_avg', 'plays', 'sdi_std',
                                'first_step_avg', 'recognition_latency_avg', 'most_common_grade']

        # Filter by minimum plays
        player_stats = player_stats[player_stats['plays'] >= min_plays]

        # Sort by SDI
        player_stats = player_stats.sort_values('sdi_avg', ascending=False)
        player_stats['sdi_rank'] = range(1, len(player_stats) + 1)

        return player_stats

    @staticmethod
    def rank_by_first_step(df: pd.DataFrame, min_plays: int = 3) -> pd.DataFrame:
        """Rank players by first step quickness"""
        player_stats = df.groupby(['jersey_number', 'team_name']).agg({
            'first_step_sec': ['mean', 'count', 'min'],
        }).reset_index()

        player_stats.columns = ['jersey_number', 'team', 'first_step_avg', 'plays', 'first_step_best']

        player_stats = player_stats[player_stats['plays'] >= min_plays]
        player_stats = player_stats.sort_values('first_step_avg', ascending=True)  # Lower is better
        player_stats['first_step_rank'] = range(1, len(player_stats) + 1)

        return player_stats


# =============================================================================
# MAIN ANALYTICS ENGINE
# =============================================================================

class AnalyticsEngine:
    """Main engine that orchestrates all analytics calculations"""

    def __init__(self, data_dir: str, roster_path: Optional[str] = None):
        print(f"=" * 60)
        print(f"Third & 20 - Analytics Engine v1.0")
        print(f"=" * 60)

        self.loader = GameDataLoader(data_dir, roster_path)
        self.cognitive_calc = CognitiveMetricsCalculator()
        self.defensive_analyzer = DefensiveSchemeAnalyzer()
        self.position_calc = PositionMetricsCalculator(self.loader.roster)
        self.ranking_calc = PlayerRankingCalculator()

        self.all_games = self.loader.load_all_games()
        print(f"  Loaded {len(self.all_games)} games")

    def run_full_analysis(self, output_prefix: str):
        """Run all analytics and output results"""

        # Combine all game data
        all_offensive = []
        all_defensive_plays = []
        all_defensive_players = []

        for game_name, game_data in self.all_games.items():
            if 'offensive' in game_data:
                df = game_data['offensive'].copy()
                df['game'] = game_name
                all_offensive.append(df)
            if 'defensive_plays' in game_data:
                df = game_data['defensive_plays'].copy()
                df['game'] = game_name
                all_defensive_plays.append(df)
            if 'defensive_players' in game_data:
                df = game_data['defensive_players'].copy()
                df['game'] = game_name
                all_defensive_players.append(df)

        combined_offensive = pd.concat(all_offensive, ignore_index=True) if all_offensive else pd.DataFrame()
        combined_defensive_plays = pd.concat(all_defensive_plays, ignore_index=True) if all_defensive_plays else pd.DataFrame()
        combined_defensive_players = pd.concat(all_defensive_players, ignore_index=True) if all_defensive_players else pd.DataFrame()

        print(f"\n  Total offensive records: {len(combined_offensive)}")
        print(f"  Total defensive play records: {len(combined_defensive_plays)}")
        print(f"  Total defensive player records: {len(combined_defensive_players)}")

        # Generate all reports
        print(f"\n[1/6] Generating cognitive metrics report...")
        self._generate_cognitive_report(combined_offensive, output_prefix)

        print(f"[2/6] Generating defensive scheme report...")
        self._generate_defensive_scheme_report(combined_defensive_plays, output_prefix)

        print(f"[3/6] Generating position-specific metrics...")
        self._generate_position_metrics_report(combined_offensive, output_prefix)

        print(f"[4/6] Generating player rankings...")
        self._generate_player_rankings(combined_offensive, output_prefix)

        print(f"[5/6] Generating per-game summaries...")
        self._generate_game_summaries(output_prefix)

        print(f"[6/6] Generating master summary...")
        self._generate_master_summary(combined_offensive, combined_defensive_plays, output_prefix)

        print(f"\n{'=' * 60}")
        print(f"Analytics complete! Output prefix: {output_prefix}")
        print(f"{'=' * 60}")

    def _generate_cognitive_report(self, df: pd.DataFrame, output_prefix: str):
        """Generate Section 4.1: Cognitive Metrics report"""
        if df.empty:
            return

        report = {}
        report.update(self.cognitive_calc.calculate_recognition_latency_stats(df))
        report.update(self.cognitive_calc.calculate_first_step_stats(df))
        report.update(self.cognitive_calc.calculate_sdi_stats(df))

        # Write to CSV
        output_path = f"{output_prefix}_cognitive_metrics.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['=== RECOGNITION LATENCY (RL) ===', ''])
            for k, v in report.items():
                if k.startswith('rl_'):
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])
            writer.writerow(['', ''])
            writer.writerow(['=== FIRST STEP QUICKNESS (FSQ) ===', ''])
            for k, v in report.items():
                if k.startswith('fsq_'):
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])
            writer.writerow(['', ''])
            writer.writerow(['=== SDI COMPOSITE ===', ''])
            for k, v in report.items():
                if k.startswith('sdi_'):
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])

        print(f"    Saved: {output_path}")

    def _generate_defensive_scheme_report(self, df: pd.DataFrame, output_prefix: str):
        """Generate Section 4.2: Defensive Scheme Detection report"""
        if df.empty:
            return

        report = {}
        report['total_plays'] = len(df)
        report.update(self.defensive_analyzer.analyze_fronts(df))
        report.update(self.defensive_analyzer.analyze_coverage_shells(df))
        report.update(self.defensive_analyzer.analyze_safety_alignment(df))
        report.update(self.defensive_analyzer.analyze_box_counts(df))
        report.update(self.defensive_analyzer.analyze_blitz_tendency(df))

        # Write to CSV
        output_path = f"{output_prefix}_defensive_schemes.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Plays Analyzed', report['total_plays']])
            writer.writerow(['', ''])

            writer.writerow(['=== DEFENSIVE FRONTS ===', ''])
            for k, v in report.items():
                if k.startswith('front_') and not k.endswith('distribution'):
                    writer.writerow([k, f"{v:.1f}" if isinstance(v, float) else v])

            writer.writerow(['', ''])
            writer.writerow(['=== COVERAGE SHELLS ===', ''])
            for k, v in report.items():
                if k.startswith('coverage_') and not k.endswith('distribution'):
                    writer.writerow([k, f"{v:.1f}" if isinstance(v, float) else v])

            writer.writerow(['', ''])
            writer.writerow(['=== SAFETY ALIGNMENT ===', ''])
            for k, v in report.items():
                if k.startswith('safety_') and not k.endswith('distribution'):
                    writer.writerow([k, f"{v:.1f}" if isinstance(v, float) else v])

            writer.writerow(['', ''])
            writer.writerow(['=== BOX COUNT ===', ''])
            for k, v in report.items():
                if k.startswith('box_') or k.startswith('light_box') or k.startswith('stacked_box'):
                    writer.writerow([k, f"{v:.1f}" if isinstance(v, float) else v])

            writer.writerow(['', ''])
            writer.writerow(['=== BLITZ TENDENCY ===', ''])
            for k, v in report.items():
                if k.startswith('blitz_') or k.startswith('no_blitz'):
                    writer.writerow([k, f"{v:.1f}" if isinstance(v, float) else v])

        print(f"    Saved: {output_path}")

    def _generate_position_metrics_report(self, df: pd.DataFrame, output_prefix: str):
        """Generate Section 3: Position-Specific Metrics report"""
        if df.empty:
            return

        report = self.position_calc.calculate_all_position_metrics(df)

        output_path = f"{output_prefix}_position_metrics.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])

            writer.writerow(['=== QUARTERBACK (QB) ===', ''])
            for k, v in report.items():
                if k.startswith('qb_'):
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])

            writer.writerow(['', ''])
            writer.writerow(['=== WIDE RECEIVER / TIGHT END (WR/TE) ===', ''])
            for k, v in report.items():
                if k.startswith('wr_te_'):
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])

            writer.writerow(['', ''])
            writer.writerow(['=== RUNNING BACK (RB) ===', ''])
            for k, v in report.items():
                if k.startswith('rb_'):
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])

            writer.writerow(['', ''])
            writer.writerow(['=== DEFENSIVE LINE / LINEBACKER (DL/LB) ===', ''])
            for k, v in report.items():
                if k.startswith('dl_lb_'):
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])

            writer.writerow(['', ''])
            writer.writerow(['=== DEFENSIVE BACK (DB) ===', ''])
            for k, v in report.items():
                if k.startswith('db_'):
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])

        print(f"    Saved: {output_path}")

    def _generate_player_rankings(self, df: pd.DataFrame, output_prefix: str):
        """Generate player ranking tables"""
        if df.empty:
            return

        # SDI Rankings
        sdi_rankings = self.ranking_calc.rank_by_sdi(df)
        sdi_path = f"{output_prefix}_player_rankings_sdi.csv"
        sdi_rankings.to_csv(sdi_path, index=False)
        print(f"    Saved: {sdi_path}")

        # First Step Rankings
        fs_rankings = self.ranking_calc.rank_by_first_step(df)
        fs_path = f"{output_prefix}_player_rankings_first_step.csv"
        fs_rankings.to_csv(fs_path, index=False)
        print(f"    Saved: {fs_path}")

    def _generate_game_summaries(self, output_prefix: str):
        """Generate per-game summary reports with Brush vs Opponent breakdown"""
        summaries = []

        for game_name, game_data in self.all_games.items():
            # Extract opponent name from game_name
            opponent = game_name.replace('_analysis', '').replace('_full', '').replace('brush_v_', '').title()

            summary = {'game': game_name, 'opponent': opponent}

            if 'offensive' in game_data:
                off_df = game_data['offensive']

                # Split by team
                brush_df = off_df[off_df['team'].str.lower() == 'brush']
                opp_df = off_df[off_df['team'].str.lower() != 'brush']

                # Total stats
                summary['total_plays'] = len(off_df)

                # ===== BRUSH OFFENSE STATS =====
                summary['brush_plays'] = len(brush_df)
                if len(brush_df) > 0:
                    # SDI metrics
                    summary['brush_avg_sdi'] = brush_df['sdi_score'].mean()
                    summary['brush_median_sdi'] = brush_df['sdi_score'].median()
                    summary['brush_max_sdi'] = brush_df['sdi_score'].max()
                    summary['brush_min_sdi'] = brush_df['sdi_score'].min()
                    summary['brush_std_sdi'] = brush_df['sdi_score'].std()

                    # First Step metrics
                    summary['brush_avg_first_step'] = brush_df['first_step_sec'].mean()
                    summary['brush_best_first_step'] = brush_df['first_step_sec'].min()
                    summary['brush_worst_first_step'] = brush_df['first_step_sec'].max()

                    # Recognition Latency metrics
                    summary['brush_avg_recognition'] = brush_df['recognition_latency_sec'].mean()
                    summary['brush_best_recognition'] = brush_df['recognition_latency_sec'].min()

                    # Post-snap distance
                    if 'post_snap_distance' in brush_df.columns:
                        summary['brush_avg_post_snap_dist'] = brush_df['post_snap_distance'].mean()

                    # Initial velocity
                    if 'initial_velocity' in brush_df.columns:
                        summary['brush_avg_initial_velocity'] = brush_df['initial_velocity'].mean()

                    # SDI Grade distribution
                    grades = brush_df['sdi_grade'].value_counts(normalize=True) * 100
                    summary['brush_elite_pct'] = grades.get('Elite', 0)
                    summary['brush_above_avg_pct'] = grades.get('Above Average', 0)
                    summary['brush_average_pct'] = grades.get('Average', 0)
                    summary['brush_below_avg_pct'] = grades.get('Below Average', 0)
                    summary['brush_developmental_pct'] = grades.get('Developmental', 0)

                    # Unique players tracked
                    summary['brush_unique_players'] = brush_df['jersey_number'].nunique()

                # ===== OPPONENT OFFENSE STATS =====
                summary['opponent_plays'] = len(opp_df)
                if len(opp_df) > 0:
                    summary['opponent_avg_sdi'] = opp_df['sdi_score'].mean()
                    summary['opponent_median_sdi'] = opp_df['sdi_score'].median()
                    summary['opponent_max_sdi'] = opp_df['sdi_score'].max()
                    summary['opponent_avg_first_step'] = opp_df['first_step_sec'].mean()
                    summary['opponent_best_first_step'] = opp_df['first_step_sec'].min()
                    summary['opponent_avg_recognition'] = opp_df['recognition_latency_sec'].mean()

                    grades = opp_df['sdi_grade'].value_counts(normalize=True) * 100
                    summary['opponent_elite_pct'] = grades.get('Elite', 0)
                    summary['opponent_above_avg_pct'] = grades.get('Above Average', 0)
                    summary['opponent_unique_players'] = opp_df['jersey_number'].nunique()

                # Differentials (positive = Brush better)
                if len(brush_df) > 0 and len(opp_df) > 0:
                    summary['sdi_differential'] = summary['brush_avg_sdi'] - summary['opponent_avg_sdi']
                    summary['first_step_differential'] = summary['opponent_avg_first_step'] - summary['brush_avg_first_step']

            # ===== OPPONENT DEFENSIVE SCHEME STATS =====
            if 'defensive_plays' in game_data:
                def_df = game_data['defensive_plays']
                summary['defensive_plays_analyzed'] = len(def_df)

                # Box count stats
                summary['opponent_avg_box_count'] = def_df['box_count'].mean()
                summary['opponent_max_box_count'] = def_df['box_count'].max()
                summary['opponent_light_box_pct'] = (def_df['box_count'] <= 5).mean() * 100
                summary['opponent_stacked_box_pct'] = (def_df['box_count'] >= 7).mean() * 100

                # Defenders on LOS
                if 'defenders_on_los' in def_df.columns:
                    summary['opponent_avg_defenders_on_los'] = def_df['defenders_on_los'].mean()

                # Second level count
                if 'second_level_count' in def_df.columns:
                    summary['opponent_avg_second_level'] = def_df['second_level_count'].mean()

                # Deep defenders
                if 'deep_defenders' in def_df.columns:
                    summary['opponent_avg_deep_defenders'] = def_df['deep_defenders'].mean()

                # Edge count
                if 'edge_count' in def_df.columns:
                    summary['opponent_avg_edge_count'] = def_df['edge_count'].mean()

                # Blitz tendency
                summary['opponent_blitz_pct'] = def_df['potential_blitz'].mean() * 100 if def_df['potential_blitz'].dtype == bool else 0

                # Coverage shell distribution
                if 'coverage_shell' in def_df.columns:
                    cov_dist = def_df['coverage_shell'].value_counts(normalize=True) * 100
                    summary['opponent_primary_coverage'] = def_df['coverage_shell'].mode()[0] if len(def_df['coverage_shell'].mode()) > 0 else 'Unknown'
                    summary['opponent_cover_0_pct'] = cov_dist.get('Cover 0', 0)
                    summary['opponent_cover_1_pct'] = cov_dist.get('Cover 1', 0)
                    summary['opponent_cover_2_pct'] = cov_dist.get('Cover 2', 0)
                    summary['opponent_cover_3_pct'] = cov_dist.get('Cover 3', 0)
                    summary['opponent_cover_4_pct'] = cov_dist.get('Cover 4', 0)

                # Front distribution
                if 'front_description' in def_df.columns:
                    front_dist = def_df['front_description'].value_counts(normalize=True) * 100
                    summary['opponent_primary_front'] = def_df['front_description'].mode()[0] if len(def_df['front_description'].mode()) > 0 else 'Unknown'
                    summary['opponent_3_4_pct'] = front_dist.get('3-4', 0)
                    summary['opponent_4_3_pct'] = front_dist.get('4-3', 0)
                    summary['opponent_nickel_pct'] = front_dist.get('4-2-5 Nickel', 0)
                    summary['opponent_dime_pct'] = front_dist.get('4-1-6 Dime', 0)
                    summary['opponent_goal_line_pct'] = front_dist.get('5-2 / Goal Line', 0)

                # Safety alignment distribution
                if 'safety_alignment' in def_df.columns:
                    safety_dist = def_df['safety_alignment'].value_counts(normalize=True) * 100
                    summary['opponent_0_high_pct'] = safety_dist.get('0-high', 0)
                    summary['opponent_1_high_pct'] = safety_dist.get('1-high', 0)
                    summary['opponent_2_high_pct'] = safety_dist.get('2-high', 0)
                    summary['opponent_robber_pct'] = safety_dist.get('robber', 0)

            summaries.append(summary)

        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_path = f"{output_prefix}_game_summaries.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"    Saved: {summary_path}")

    def _generate_master_summary(self, offensive_df: pd.DataFrame, defensive_df: pd.DataFrame, output_prefix: str):
        """Generate master summary with all key metrics, including with/without Hoban"""

        # Split data - Brush vs Opponents
        brush_off = offensive_df[offensive_df['team'].str.lower() == 'brush'] if not offensive_df.empty else pd.DataFrame()
        opp_off = offensive_df[offensive_df['team'].str.lower() != 'brush'] if not offensive_df.empty else pd.DataFrame()

        # Split data - WITH Hoban vs WITHOUT Hoban
        hoban_games = ['hoban_analysis']
        brush_off_no_hoban = brush_off[~brush_off['game'].isin(hoban_games)] if not brush_off.empty else pd.DataFrame()
        opp_off_no_hoban = opp_off[~opp_off['game'].isin(hoban_games)] if not opp_off.empty else pd.DataFrame()
        def_no_hoban = defensive_df[~defensive_df['game'].isin(hoban_games)] if not defensive_df.empty else pd.DataFrame()

        summary = {
            'report_type': 'Third & 20 Season Analytics Summary',
            'total_games': len(self.all_games),
            'total_plays': len(offensive_df),
        }

        # ========== ALL GAMES (WITH HOBAN) ==========
        summary['===_ALL_GAMES_INCL_HOBAN_==='] = ''

        if not brush_off.empty:
            summary['brush_total_plays'] = len(brush_off)
            summary['brush_avg_sdi'] = brush_off['sdi_score'].mean()
            summary['brush_avg_first_step'] = brush_off['first_step_sec'].mean()
            summary['brush_avg_recognition'] = brush_off['recognition_latency_sec'].mean()
            summary['brush_elite_pct'] = (brush_off['sdi_grade'] == 'Elite').mean() * 100

        if not opp_off.empty:
            summary['opponent_total_plays'] = len(opp_off)
            summary['opponent_avg_sdi'] = opp_off['sdi_score'].mean()
            summary['opponent_avg_first_step'] = opp_off['first_step_sec'].mean()
            summary['opponent_avg_recognition'] = opp_off['recognition_latency_sec'].mean()
            summary['opponent_elite_pct'] = (opp_off['sdi_grade'] == 'Elite').mean() * 100

        # Defensive tendencies (opponents against Brush)
        if not defensive_df.empty:
            summary['opponents_avg_box_count'] = defensive_df['box_count'].mean()
            summary['opponents_blitz_tendency_pct'] = defensive_df['potential_blitz'].mean() * 100 if defensive_df['potential_blitz'].dtype == bool else 0
            if 'coverage_shell' in defensive_df.columns and len(defensive_df['coverage_shell'].mode()) > 0:
                summary['opponents_most_common_coverage'] = defensive_df['coverage_shell'].mode()[0]
            if 'front_description' in defensive_df.columns and len(defensive_df['front_description'].mode()) > 0:
                summary['opponents_most_common_front'] = defensive_df['front_description'].mode()[0]

        # ========== WITHOUT HOBAN ==========
        summary['===_WITHOUT_HOBAN_==='] = ''
        summary['games_without_hoban'] = len(self.all_games) - 1

        if not brush_off_no_hoban.empty:
            summary['brush_plays_no_hoban'] = len(brush_off_no_hoban)
            summary['brush_avg_sdi_no_hoban'] = brush_off_no_hoban['sdi_score'].mean()
            summary['brush_avg_first_step_no_hoban'] = brush_off_no_hoban['first_step_sec'].mean()
            summary['brush_avg_recognition_no_hoban'] = brush_off_no_hoban['recognition_latency_sec'].mean()
            summary['brush_elite_pct_no_hoban'] = (brush_off_no_hoban['sdi_grade'] == 'Elite').mean() * 100

        if not opp_off_no_hoban.empty:
            summary['opponent_plays_no_hoban'] = len(opp_off_no_hoban)
            summary['opponent_avg_sdi_no_hoban'] = opp_off_no_hoban['sdi_score'].mean()
            summary['opponent_avg_first_step_no_hoban'] = opp_off_no_hoban['first_step_sec'].mean()
            summary['opponent_avg_recognition_no_hoban'] = opp_off_no_hoban['recognition_latency_sec'].mean()
            summary['opponent_elite_pct_no_hoban'] = (opp_off_no_hoban['sdi_grade'] == 'Elite').mean() * 100

        if not def_no_hoban.empty:
            summary['opponents_avg_box_count_no_hoban'] = def_no_hoban['box_count'].mean()
            summary['opponents_blitz_pct_no_hoban'] = def_no_hoban['potential_blitz'].mean() * 100 if def_no_hoban['potential_blitz'].dtype == bool else 0

        # ========== HOBAN ONLY ==========
        brush_off_hoban = brush_off[brush_off['game'].isin(hoban_games)] if not brush_off.empty else pd.DataFrame()
        opp_off_hoban = opp_off[opp_off['game'].isin(hoban_games)] if not opp_off.empty else pd.DataFrame()
        def_hoban = defensive_df[defensive_df['game'].isin(hoban_games)] if not defensive_df.empty else pd.DataFrame()

        summary['===_HOBAN_GAME_ONLY_==='] = ''

        if not brush_off_hoban.empty:
            summary['brush_plays_hoban'] = len(brush_off_hoban)
            summary['brush_avg_sdi_hoban'] = brush_off_hoban['sdi_score'].mean()
            summary['brush_avg_first_step_hoban'] = brush_off_hoban['first_step_sec'].mean()
            summary['brush_elite_pct_hoban'] = (brush_off_hoban['sdi_grade'] == 'Elite').mean() * 100

        if not opp_off_hoban.empty:
            summary['hoban_plays'] = len(opp_off_hoban)
            summary['hoban_avg_sdi'] = opp_off_hoban['sdi_score'].mean()
            summary['hoban_avg_first_step'] = opp_off_hoban['first_step_sec'].mean()
            summary['hoban_elite_pct'] = (opp_off_hoban['sdi_grade'] == 'Elite').mean() * 100

        if not def_hoban.empty:
            summary['hoban_defense_box_count'] = def_hoban['box_count'].mean()
            summary['hoban_defense_blitz_pct'] = def_hoban['potential_blitz'].mean() * 100 if def_hoban['potential_blitz'].dtype == bool else 0

        output_path = f"{output_prefix}_master_summary.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for k, v in summary.items():
                if '===' in k:
                    writer.writerow(['', ''])
                    writer.writerow([k.replace('_', ' '), ''])
                else:
                    writer.writerow([k, f"{v:.3f}" if isinstance(v, float) else v])

        print(f"    Saved: {output_path}")

        # Also print to console
        print(f"\n{'=' * 60}")
        print("MASTER SUMMARY")
        print(f"{'=' * 60}")
        for k, v in summary.items():
            if '===' in k:
                print(f"\n{k.replace('_', ' ')}")
            else:
                print(f"  {k}: {f'{v:.3f}' if isinstance(v, float) else v}")


def main():
    parser = argparse.ArgumentParser(
        description="Third & 20 - Analytics Engine"
    )
    parser.add_argument("--data-dir", "-d", required=True,
                        help="Directory containing game data CSVs")
    parser.add_argument("--roster", "-r", help="Path to roster CSV")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file prefix")

    args = parser.parse_args()

    engine = AnalyticsEngine(args.data_dir, args.roster)
    engine.run_full_analysis(args.output)


if __name__ == "__main__":
    main()
