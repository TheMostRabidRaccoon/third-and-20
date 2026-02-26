"""
Third & 20 - Jersey Color Classifier v2.0
Handles: white, yellow, brown, black, red, blue

Tuned for game film with compression artifacts and variable lighting.
"""

from typing import Tuple


def classify_jersey_color(bgr_color: Tuple[int, int, int]) -> str:
    """
    Classify BGR color into jersey color category.
    
    Args:
        bgr_color: Tuple of (Blue, Green, Red) values 0-255
        
    Returns:
        One of: 'white', 'yellow', 'brown', 'black', 'red', 'blue'
    """
    b, g, r = [int(x) for x in bgr_color]
    brightness = (b + g + r) / 3
    
    # Calculate color metrics
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    chroma = max_channel - min_channel
    saturation = chroma / max_channel if max_channel > 0 else 0
    
    # Warmth: how much warmer (red/yellow) vs cooler (blue)
    warmth = (r + g) / 2 - b if b > 0 else (r + g) / 2
    
    # === YELLOW ===
    # Yellow jerseys: R and G both elevated, B suppressed
    # In compressed video, appears as warm golden tone
    if brightness > 70:
        rg_avg = (r + g) / 2
        if rg_avg > b * 1.4 and g > b * 1.2 and r > b * 1.2:
            # Both R and G stronger than B = yellow/gold
            if abs(r - g) < 40:  # R and G relatively balanced
                return 'yellow'
    
    # === WHITE ===
    # High brightness, all channels similar (low saturation)
    if brightness > 95 and saturation < 0.3:
        return 'white'
    
    # === BLACK ===
    # Very low brightness
    if brightness < 55:
        return 'black'
    
    # === BLUE ===
    # B channel notably dominant
    if b > 65:
        if b > r * 1.15 and b > g * 1.1:
            return 'blue'
        # Also catch blue when B is highest even if margins are smaller
        if b == max_channel and b > r and b > g and chroma > 15:
            return 'blue'
    
    # === RED ===
    # R channel notably dominant, low G and B
    if r > 65:
        if r > g * 1.2 and r > b * 1.15:
            return 'red'
        # Red with less margin but clear dominance
        if r == max_channel and r > g * 1.1 and r > b * 1.1 and chroma > 15:
            return 'red'
    
    # === BROWN ===
    # Warm but dark: R >= G >= B pattern, moderate brightness
    if 55 <= brightness <= 100:
        if r >= g and g >= b:
            # Warm tone with R highest
            if r > b * 1.1:
                return 'brown'
    
    # === FALLBACK ===
    # Use brightness as tiebreaker
    if brightness > 85:
        # Light but didn't match white - likely yellow in bad lighting
        if warmth > 10:
            return 'yellow'
        return 'white'
    else:
        # Dark but didn't match black - likely brown
        if warmth > 5:
            return 'brown'
        # Could be dark blue
        if b >= r and b >= g:
            return 'blue'
        return 'brown'


def get_team_from_color(color: str, home_color: str, away_color: str, 
                        home_team: str, away_team: str) -> str:
    """
    Map detected color to team name.
    
    Args:
        color: Detected jersey color
        home_color: Expected home team color
        away_color: Expected away team color
        home_team: Home team name
        away_team: Away team name
        
    Returns:
        Team name or 'unknown'
    """
    # Direct match
    if color == home_color:
        return home_team
    if color == away_color:
        return away_team
    
    # Fuzzy matching for similar colors
    similar_colors = {
        'white': ['white', 'yellow'],  # Both light
        'yellow': ['yellow', 'white'],
        'brown': ['brown', 'black'],   # Both dark warm
        'black': ['black', 'brown'],
        'red': ['red', 'brown'],       # Both warm
        'blue': ['blue', 'black'],     # Both dark cool
    }
    
    # Check if color is similar to home
    if color in similar_colors.get(home_color, []):
        return home_team
    
    # Check if color is similar to away
    if color in similar_colors.get(away_color, []):
        return away_team
    
    return 'unknown'


# === TEST CASES ===
if __name__ == "__main__":
    # Test with values we've seen from actual game film
    test_cases = [
        # Shaker v Brush game (brightness ~80-108)
        ((131, 104, 90), "Expected: white/yellow (Brush)"),
        ((99, 100, 64), "Expected: brown/dark"),
        ((88, 73, 79), "Expected: dark"),
        ((90, 70, 80), "Expected: dark"),
        
        # Synthetic test cases
        ((200, 200, 200), "Expected: white"),
        ((50, 50, 50), "Expected: black"),
        ((60, 80, 150), "Expected: yellow"),
        ((100, 60, 60), "Expected: blue"),
        ((60, 60, 120), "Expected: red"),
        ((70, 85, 100), "Expected: brown"),
    ]
    
    print("=== Color Classifier Test ===\n")
    for bgr, expected in test_cases:
        result = classify_jersey_color(bgr)
        b, g, r = bgr
        brightness = (b + g + r) / 3
        print(f"BGR={bgr} (brightness={brightness:.0f}) -> {result:8} | {expected}")
