import sys
from pathlib import Path

# Pipeline modules use local imports (from play_state import ...), so tests
# import them the same way the scripts do until the repo is packaged.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "pipelines"))
