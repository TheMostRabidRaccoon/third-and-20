#!/usr/bin/env python3
"""
MaxPreps Historical Roster Scraper
From Gemini's Football Data Pipeline (01/11/26)

Scrapes rosters from MaxPreps.com and outputs CSVs formatted for Third & 20's CV pipeline.
Supports historical rosters (2023, 2024, etc.) for matching YouTube game film to correct players.

Usage:
    python maxpreps_roster_scraper.py

Or import and use programmatically:
    from maxpreps_roster_scraper import scrape_roster
    scrape_roster("Massillon_Washington", "https://www.maxpreps.com/oh/massillon/washington-tigers/football/roster/?season=2023", "2023")
"""

import requests
import pandas as pd
import time
import random
import os
from datetime import datetime

# CONFIG: Masquerade as a real browser to avoid 403 blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

# Output directory for roster CSVs
OUTPUT_DIR = "rosters"


def scrape_roster(team_name: str, url: str, year: str, output_dir: str = OUTPUT_DIR) -> str | None:
    """
    Scrapes a specific MaxPreps roster page and outputs a CSV for Third & 20 CV pipeline.
    
    Args:
        team_name: Name of the team (used in filename)
        url: MaxPreps roster URL (include ?season=YYYY for historical)
        year: Season year (e.g., "2023", "2024")
        output_dir: Directory to save CSV files
        
    Returns:
        Path to saved CSV file, or None if failed
    """
    print(f"--> Fetching {year} roster for: {team_name}...")
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        # MaxPreps stores data in tables
        dfs = pd.read_html(response.text)
        
        if not dfs:
            print(f"❌ Error: No tables found on page: {url}")
            return None
        
        # Find the table with jersey numbers
        roster_df = None
        for df in dfs:
            cols = [str(c).lower() for c in df.columns]
            if '#' in cols or 'jersey' in cols or any('#' in str(c) for c in df.columns):
                roster_df = df
                break
        
        if roster_df is None:
            # Fallback: take the largest table (likely the roster)
            roster_df = max(dfs, key=len)
            print(f"⚠️  Warning: No jersey column found, using largest table ({len(roster_df)} rows)")
        
        # --- CLEANING FOR THIRD & 20 CV PIPELINE ---
        
        # 1. Standardize headers
        roster_df.columns = [str(c).lower().strip() for c in roster_df.columns]
        
        # 2. Rename keys to match pipeline requirements
        rename_map = {
            '#': 'jersey',
            'jersey': 'jersey',
            'no.': 'jersey',
            'number': 'jersey',
            'athlete': 'name',
            'player': 'name',
            'pos': 'position',
            'pos.': 'position',
            'gr': 'class_year',
            'grade': 'class_year',
            'yr': 'class_year',
            'class': 'class_year',
            'ht': 'height',
            'ht.': 'height',
            'wt': 'weight',
            'wt.': 'weight'
        }
        roster_df = roster_df.rename(columns=rename_map)
        
        # 3. Clean Jersey numbers (remove '#' symbol if present)
        if 'jersey' in roster_df.columns:
            roster_df['jersey'] = (
                roster_df['jersey']
                .astype(str)
                .str.replace('#', '', regex=False)
                .str.strip()
            )
            # Remove rows without valid jersey numbers
            roster_df = roster_df[roster_df['jersey'].str.match(r'^\d+$', na=False)]
        
        # 4. Add Metadata
        roster_df['team'] = team_name.replace('_', ' ')
        roster_df['season'] = year
        roster_df['scraped_at'] = datetime.now().isoformat()
        
        # 5. Select only columns needed for CV pipeline
        desired_cols = ['jersey', 'name', 'position', 'height', 'weight', 'class_year', 'team', 'season']
        final_cols = [c for c in desired_cols if c in roster_df.columns]
        final_df = roster_df[final_cols].copy()
        
        # 6. Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # 7. Save CSV
        filename = f"roster_{team_name.replace(' ', '_')}_{year}.csv"
        filepath = os.path.join(output_dir, filename)
        final_df.to_csv(filepath, index=False)
        
        print(f"✅ Success! Saved {filepath} ({len(final_df)} players)")
        print(f"   Columns: {', '.join(final_cols)}")
        
        # Be polite to the server
        time.sleep(random.uniform(2, 4))
        
        return filepath

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   URL may be invalid or MaxPreps may be blocking requests.")
        return None
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return None


def scrape_multiple(targets: list[tuple[str, str, str]], output_dir: str = OUTPUT_DIR) -> list[str]:
    """
    Scrape multiple rosters from a list of targets.
    
    Args:
        targets: List of (team_name, url, year) tuples
        output_dir: Directory to save CSV files
        
    Returns:
        List of successfully saved file paths
    """
    saved_files = []
    
    print(f"\n{'='*60}")
    print(f"MaxPreps Roster Scraper - Third & 20 Pipeline")
    print(f"{'='*60}")
    print(f"Targets: {len(targets)} teams")
    print(f"Output: {output_dir}/")
    print(f"{'='*60}\n")
    
    for team, url, year in targets:
        filepath = scrape_roster(team, url, year, output_dir)
        if filepath:
            saved_files.append(filepath)
    
    print(f"\n{'='*60}")
    print(f"Complete! Saved {len(saved_files)}/{len(targets)} rosters")
    print(f"{'='*60}\n")
    
    return saved_files


# =============================================================================
# TARGET LIST - ADD YOUR MAXPREPS URLS HERE
# =============================================================================
# 
# To find URLs:
#   1. Go to MaxPreps.com
#   2. Search for the school (e.g., "Massillon Washington")
#   3. Go to Football > Roster
#   4. For historical rosters, add ?season=YYYY to the URL
#
# Competition Tiers (for tagging in your database):
#   Tier 1 (National Elite): IMG, Mater Dei, St. Edward, Duncanville
#   Tier 2 (Regional Power): Massillon, Hoban, Glenville, Avon
#   Tier 3 (Varsity Standard): Mentor, Walsh Jesuit, playoff qualifiers
#   Tier 4 (Developmental): <.500 teams (use for CV testing, not validation)
#
# YouTube Sources for matching video:
#   - WHS-TV Ohio: Massillon (gold standard, wide-angle press box)
#   - NEO Productions: St. Edward, Glenville, Mentor
#   - OhioSportsNet1: Hoban, Avon
# =============================================================================

TARGETS = [
    # --- TIER 2: Northeast Ohio Powers ---
    # ("Massillon_Washington", "https://www.maxpreps.com/oh/massillon/washington-tigers/football/roster/?season=2024", "2024"),
    # ("Massillon_Washington", "https://www.maxpreps.com/oh/massillon/washington-tigers/football/roster/?season=2023", "2023"),
    
    # ("St_Edward", "https://www.maxpreps.com/oh/lakewood/st-edward-eagles/football/roster/?season=2024", "2024"),
    # ("St_Edward", "https://www.maxpreps.com/oh/lakewood/st-edward-eagles/football/roster/?season=2023", "2023"),
    
    # ("Archbishop_Hoban", "https://www.maxpreps.com/oh/akron/archbishop-hoban-knights/football/roster/?season=2024", "2024"),
    
    # ("Glenville", "https://www.maxpreps.com/oh/cleveland/glenville-tarblooders/football/roster/?season=2024", "2024"),
    
    # ("Avon", "https://www.maxpreps.com/oh/avon/avon-eagles/football/roster/?season=2024", "2024"),
    
    # ("Mentor", "https://www.maxpreps.com/oh/mentor/mentor-cardinals/football/roster/?season=2024", "2024"),
    
    # --- TIER 1: National Elite (for 99th percentile baselines) ---
    # ("IMG_Academy", "https://www.maxpreps.com/fl/bradenton/img-academy-ascenders/football/roster/?season=2024", "2024"),
    
    # ("Duncanville", "https://www.maxpreps.com/tx/duncanville/duncanville-panthers/football/roster/?season=2024", "2024"),
    
    # ("St_John_Bosco", "https://www.maxpreps.com/ca/bellflower/st-john-bosco-braves/football/roster/?season=2024", "2024"),
    
    # --- TIER 4: Local opponents (for testing, not validation) ---
    # ("Shaker_Heights", "https://www.maxpreps.com/oh/shaker-heights/shaker-heights-red-raiders/football/roster/?season=2024", "2024"),
    
    # ("Brush", "https://www.maxpreps.com/oh/lyndhurst/brush-arcs/football/roster/?season=2024", "2024"),
]


if __name__ == "__main__":
    if not TARGETS:
        print("⚠️  No targets configured!")
        print("")
        print("To use this scraper:")
        print("  1. Open this file in a text editor")
        print("  2. Uncomment the teams you want in the TARGETS list")
        print("  3. Or add your own URLs following the same format")
        print("  4. Run: python maxpreps_roster_scraper.py")
        print("")
        print("Example:")
        print('  ("Massillon_Washington", "https://www.maxpreps.com/oh/massillon/washington-tigers/football/roster/?season=2024", "2024"),')
    else:
        scrape_multiple(TARGETS)
