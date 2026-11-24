import sys
from pathlib import Path

# Make src/app.py importable as `app`, matching how `python src/app.py` resolves imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
