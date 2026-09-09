"""Platform runtime setup using the verified dependency snapshots."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "phase1/setup.py"), run_name="__main__")
