import sys
from pathlib import Path

# Matches the sys.path convention scripts/exp_e10_multisequence_benchmark.py already uses
# (src/ isn't installed as an editable package in either local venv).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
