import sys
from pathlib import Path
import pytest

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    src_path = root / "src"
    sys.path.insert(0, str(src_path))

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Added to sys.path: {src_path}")
    print("[INFO] Running pytest...\n")

    errno = pytest.main(["--disable-warnings"])
    raise SystemExit(errno)


#查看HTML报告
#start htmlcov\index.html  (Windows)
#open htmlcov/index.html (macOS/Linux)
