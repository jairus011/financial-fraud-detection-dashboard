"""Run the dashboard with the port and interface expected by a web host."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        port = int(os.environ.get("PORT", "8501"))
    except ValueError as exc:
        raise SystemExit("PORT must be an integer between 1 and 65535.") from exc
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be an integer between 1 and 65535.")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py",
                            "--server.address=0.0.0.0", f"--server.port={port}",
                            "--server.headless=true", "--server.fileWatcherType=none"])


if __name__ == "__main__":
    main()
