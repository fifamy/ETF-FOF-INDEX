from pathlib import Path
import sys


def bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    for relative in [".vendor", "src"]:
        candidate = root / relative
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root

