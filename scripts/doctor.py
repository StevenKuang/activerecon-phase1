"""Check the installed platform simulator, evaluator and method environments."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "phase1/doctor.py"), run_name="__main__")
