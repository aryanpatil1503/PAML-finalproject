#!/usr/bin/env python3
"""
Full pipeline orchestrator for hospital readmission project.
Calls preprocessing, EDA, model training with hyperparameter tuning, and evaluation in sequence.
"""
import subprocess
import sys
from pathlib import Path


def run_script(path):
    print(f"\n--- Running {path.name} ---")
    res = subprocess.run([sys.executable, str(path)], check=True)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    root = Path(__file__).parent
    scripts = [
        root / 'src' / 'data' / 'run_preprocessing.py',
        root / 'src' / 'data' / 'eda.py',
        root / 'src' / 'models' / 'train.py',
        root / 'src' / 'models' / 'evaluate.py'
    ]
    for script in scripts:
        run_script(script)
    print("\nAll pipeline steps completed successfully.")


if __name__ == '__main__':
    main()
