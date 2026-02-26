#!/usr/bin/env python3
"""
Complete Analytics Merger for Third & 20
Combines:
1. Hudl Traditional Stats (offense/defense totals)
2. CV Cognitive Metrics (SDI, First Step, Recognition Latency)
3. Defensive Individual Stats (from CV + DC files)
4. Offensive Tendencies (from CV)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_PATH = Path("/sessions/exciting-friendly-curie/mnt/rri_server")
HUDL_STATS = BASE_PATH / "data/Brush Demo stats/brush_hudl_game_traditional_stats.csv"
CV_OUTPUT = BASE_PATH / "projects/third_and_20/output"
DEFENSE_MANUAL = BASE_PATH / "data/defense_manual"
OUTPUT_DIR = BASE_PATH / "data/Brush Demo stats"

# Game name mappings (standardize between sources)
GAME_MAPPING = {
    'CCC': 'CCCH',
    'Cleveland Central Catholic': 'CCCH',
    'CCCHS': 'CCCH',
    'Will South': 'Willoughby South',
    'WilloughbySouth': 'Willoughby South',
    'Willoughby': 'Willoughby South',
    'GARFIELD HTS': 'Garfield Heights',
    'Garfield': 'Garfield Heights',
    'WARRENSVILLE HTS': 'Warrensville',
    'Warrensville Heights': 'Warrensville',
    'HOBAN': 'Hoban',
    'Shaker': 'Shaker Heights',
    'SHAKER': 'Shaker Heights',
}

def standardize_game_name(name):
    """Standardize game/opponent names"""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return GAME_MAPPING.get(name, name)

def load_hudl_traditional_stats():
    """Load and process Hudl traditional stats"""
    print("Loading Hudl traditional stats...")
    df = pd.read_csv(HUDL_STATS)
    df['game'] = df['game'].apply(standardize_game_name)
    df = df.rename(columns={'game': 'opponent'})
    return df

def load_cv_sdi_rankings():
    """Load CV SDI player rankings"""
    print("Loading CV SDI rankings...")
    sdi_path = CV_OUTPUT / "01_SDI_Player_Rankings.csv"
    if sdi_path.exists():
        df = pd.read_csv(sdi_path)
        return df
    return pd.DataFrame()

def load_cv_offensive_tendencies():
    """Load CV offensive tendencies by opponent"""
    print("Loading CV offensive tendencies...")
    tend_path = CV_OUTPUT / "02_Offensive_Tendencies_By_Opponent.csv"
    if tend_path.exists():
        df = pd.read_csv(tend_path)
        df['Opponent'] = df['Opponent'].apply(standardize_game_name)
        # Get OVERALL rows only for game-level summary
        overall = df[df['Situation'] == 'OVERALL'].copy()
        overall = overall.rename(columns={
            'Opponent': 'opponent',
            'Run %': 'cv_run_pct',
            'Pass %': 'cv_pass_pct',
            'Avg Gain': 'cv_avg_gain',
            'Top Formation': 'cv_top_formation',
            'Total Plays': 'cv_total_plays'
        })
        return overall[['opponent', 'cv_run_pct', 'cv_pass_pct', 'cv_avg_gain', 'cv_top_formation', 'cv_total_plays']]
    return pd.DataFrame()

def load_cv_defense_by_game():
    """Load CV defensive stats by game and aggregate to game totals"""
    print("Loading CV defensive stats...")
    def_path = CV_OUTPUT / "07_Defense_By_Game.csv"
    if def_path.exists():
        df = pd.read_csv(def_path)
        df['Game'] = df['Game'].apply(standardize_game_name)

        # Aggregate to game totals
        game_totals = df.groupby('Game').agg({
            'Tackles': 'sum',
            'TFLs': 'sum',
            'Missed Tackles': 'sum',
            'Sacks': 'sum',
            'INTs': 'sum',
            'PBUs': 'sum',
            'FF': 'sum',
            'FR': 'sum',
            'TD': 'sum'
        }).reset_index()

        game_totals = game_totals.rename(columns={
            'Game': 'opponent',
            'Tackles': 'cv_tackles',
            'TFLs': 'cv_tfls',
            'Missed Tackles': 'cv_missed_tackles',
            'Sacks': 'cv_def_sacks',
            'INTs': 'cv_def_ints',
            'PBUs': 'cv_pbus',
            'FF': 'cv_ff',
            'FR': 'cv_fr',
            'TD': 'cv_def_tds'
        })

        # Calculate tackle efficiency
        game_totals['cv_tackle_efficiency'] = (
            game_totals['cv_tackles'] /
            (game_totals['cv_tackles'] + game_totals['cv_missed_tackles'])
        ).round(3) * 100

        return game_totals
    return pd.DataFrame()

def extract_opponent_from_game(game_name):
    """Extract opponent name from game format like 'SHAKER_V_BRUSH' or 'BRUSH_V_HOBAN'"""
    if pd.isna(game_name):
        return game_name
    game_name = str(game_name).upper()
    # Format is typically OPPONENT_V_BRUSH or BRUSH_V_OPPONENT
    parts = game_name.replace('_V_', ' V ').replace(' V ', '|').split('|')
    if len(parts) == 2:
        team1, team2 = parts[0].strip(), parts[1].strip()
        # Return the non-Brush team as opponent
        if 'BRUSH' in team1:
            opponent = team2
        else:
            opponent = team1
        # Replace underscores with spaces, title case
        opponent = opponent.replace('_', ' ').title()
        return standardize_game_name(opponent)
    return standardize_game_name(game_name.replace('_', ' ').title())

def load_cv_sdi_from_game_files():
    """Load SDI from individual per-game analysis files"""
    print("Loading CV SDI from per-game files...")

    game_files = {
        'hoban': 'Hoban',
        'bedford': 'Bedford',
        'brush_v_erie': 'Erie',
        'shaw': 'Shaw',
        'warrensville': 'Warrensville',
        'willoughby': 'Willoughby South',
        'garfield': 'Garfield Heights',
        'euclid': 'Euclid',
        'ccc': 'CCCH',
        'shaker': 'Shaker Heights',
    }

    sdi_data = []
    data_dir = BASE_PATH / "data/Brush Demo stats"

    for file in data_dir.glob("*_offensive_sdi.csv"):
        fname = file.stem.lower()
        opponent = None
        for prefix, opp_name in game_files.items():
            if prefix in fname:
                opponent = opp_name
                break

        if opponent:
            try:
                df = pd.read_csv(file)
                sdi_col = 'sdi_score' if 'sdi_score' in df.columns else 'sdi_calibrated'
                if sdi_col in df.columns:
                    sdi_vals = df[sdi_col].dropna()
                    if len(sdi_vals) > 0:
                        if sdi_vals.mean() > 100:
                            sdi_vals = sdi_vals / 100
                        sdi_data.append({
                            'opponent': opponent,
                            'avg_sdi': round(sdi_vals.mean(), 2),
                            'sdi_std': round(sdi_vals.std(), 2),
                            'max_sdi': round(sdi_vals.max(), 2),
                            'min_sdi': round(sdi_vals.min(), 2),
                            'sdi_plays': len(sdi_vals)
                        })
                        print(f"  {opponent}: {len(sdi_vals)} plays, avg SDI={sdi_vals.mean():.1f}")
            except Exception as e:
                print(f"  Error {file.name}: {e}")

    if sdi_data:
        return pd.DataFrame(sdi_data)
    return pd.DataFrame()

def load_cv_sdi_play_by_play():
    """Load SDI play-by-play and calculate game averages"""
    print("Loading CV SDI...")
    df_game_files = load_cv_sdi_from_game_files()
    if len(df_game_files) > 0:
        print(f"SDI games: {list(df_game_files['opponent'].unique())}")
        return df_game_files
    return pd.DataFrame()

def merge_all_data():
    """Merge all data sources into complete analytics"""
    print("\n=== MERGING ALL ANALYTICS DATA ===\n")

    # Load all sources
    hudl = load_hudl_traditional_stats()
    cv_tendencies = load_cv_offensive_tendencies()
    cv_defense = load_cv_defense_by_game()
    cv_sdi = load_cv_sdi_play_by_play()

    print(f"\nHudl games: {list(hudl['opponent'].unique())}")
    print(f"CV tendencies games: {list(cv_tendencies['opponent'].unique()) if len(cv_tendencies) > 0 else 'None'}")
    print(f"CV defense games: {list(cv_defense['opponent'].unique()) if len(cv_defense) > 0 else 'None'}")

    # Start with Hudl as base (most complete traditional stats)
    combined = hudl.copy()

    # Merge CV offensive tendencies
    if len(cv_tendencies) > 0:
        combined = pd.merge(combined, cv_tendencies, on='opponent', how='left')
        print(f"After CV tendencies merge: {len(combined)} rows")

    # Merge CV defensive stats
    if len(cv_defense) > 0:
        combined = pd.merge(combined, cv_defense, on='opponent', how='left')
        print(f"After CV defense merge: {len(combined)} rows")

    # Merge CV SDI game averages
    if len(cv_sdi) > 0:
        combined = pd.merge(combined, cv_sdi, on='opponent', how='left')
        print(f"After CV SDI merge: {len(combined)} rows")

    return combined

def generate_complete_output():
    """Generate the complete analytics output file"""
    combined = merge_all_data()

    # Order columns logically
    column_order = [
        # Game ID
        'opponent',

        # Total Offense
        'total_plays', 'total_yards', 'total_tds', 'yards_per_play', 'first_downs', 'turnovers', 'fumbles',

        # Passing
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
        'completion_pct', 'yards_per_attempt', 'sacks', 'sack_yards_lost', 'incompletes',

        # Rushing
        'carries', 'rushing_yards', 'rushing_tds', 'yards_per_carry', 'long_rush',
        'left_carries', 'left_yards', 'middle_carries', 'middle_yards', 'right_carries', 'right_yards',

        # Defense Totals (from Hudl)
        'def_total_plays', 'def_total_yards', 'def_total_tds', 'def_yards_per_play', 'def_first_downs',
        'def_turnovers', 'def_fumbles',
        'def_completions', 'def_attempts', 'def_passing_yards', 'def_passing_tds', 'def_interceptions',
        'def_completion_pct', 'def_yards_per_attempt', 'def_sacks', 'def_sack_yards_lost', 'def_incompletes',
        'def_carries', 'def_rushing_yards', 'def_rushing_tds', 'def_yards_per_carry', 'def_long_rush',
        'def_left_carries', 'def_left_yards', 'def_middle_carries', 'def_middle_yards', 'def_right_carries', 'def_right_yards',

        # Situational
        'first_down_plays', 'second_down_plays', 'third_down_plays', 'third_down_conversions', 'third_down_pct',
        'fourth_down_plays', 'red_zone_plays', 'red_zone_tds', 'run_plays', 'pass_plays', 'run_pct', 'pass_pct',

        # CV Offensive Tendencies
        'cv_run_pct', 'cv_pass_pct', 'cv_avg_gain', 'cv_top_formation', 'cv_total_plays',

        # CV Defensive Individual Totals
        'cv_tackles', 'cv_tfls', 'cv_missed_tackles', 'cv_def_sacks', 'cv_def_ints',
        'cv_pbus', 'cv_ff', 'cv_fr', 'cv_def_tds', 'cv_tackle_efficiency',

        # CV SDI Metrics
        'avg_sdi', 'sdi_std', 'max_sdi', 'min_sdi', 'sdi_plays'
    ]

    # Only include columns that exist
    final_columns = [c for c in column_order if c in combined.columns]
    remaining = [c for c in combined.columns if c not in final_columns]
    final_columns.extend(remaining)

    combined = combined[final_columns]

    # Save output
    output_path = OUTPUT_DIR / "brush_COMPLETE_game_analytics.csv"
    combined.to_csv(output_path, index=False)
    print(f"\n=== SAVED: {output_path} ===")

    # Print summary
    print(f"\nTotal games: {len(combined)}")
    print(f"Total columns: {len(combined.columns)}")

    # Categorize columns
    hudl_cols = [c for c in combined.columns if not c.startswith('cv_') and c not in ['avg_sdi', 'sdi_std', 'max_sdi', 'min_sdi', 'sdi_plays']]
    cv_cols = [c for c in combined.columns if c.startswith('cv_') or c in ['avg_sdi', 'sdi_std', 'max_sdi', 'min_sdi', 'sdi_plays']]

    print(f"\nHudl Traditional Stats: {len(hudl_cols)} columns")
    print(f"CV Cognitive/Defense Metrics: {len(cv_cols)} columns")

    return combined

def generate_defense_individual_complete():
    """Generate complete individual defensive stats merging CV + DC files"""
    print("\n=== GENERATING INDIVIDUAL DEFENSE STATS ===\n")

    # Load CV defense by game
    cv_def_path = CV_OUTPUT / "07_Defense_By_Game.csv"
    cv_individual = CV_OUTPUT / "06_Defense_Individual_Stats.csv"

    if cv_individual.exists():
        cv_ind = pd.read_csv(cv_individual)
        output_path = OUTPUT_DIR / "brush_COMPLETE_defense_individual.csv"
        cv_ind.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        return cv_ind
    return pd.DataFrame()

if __name__ == "__main__":
    # Generate main complete analytics
    complete = generate_complete_output()

    # Generate complete individual defense
    defense_ind = generate_defense_individual_complete()

    print("\n=== COMPLETE ANALYTICS GENERATION DONE ===")
    print(f"\nFiles generated:")
    print(f"  1. brush_COMPLETE_game_analytics.csv - {len(complete)} games, {len(complete.columns)} metrics")
    if len(defense_ind) > 0:
        print(f"  2. brush_COMPLETE_defense_individual.csv - {len(defense_ind)} player records")
