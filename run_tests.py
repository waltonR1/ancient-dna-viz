import sys
from pathlib import Path
import pytest

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    raise SystemExit(pytest.main(["-v", "src/ancient_dna/tests"]))
