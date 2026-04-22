from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for relative in [".vendor", "src"]:
    candidate = ROOT / relative
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

