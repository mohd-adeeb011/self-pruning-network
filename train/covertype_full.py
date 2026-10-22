"""Reproduce the saved Covertype curve with one command.

Run: python -m train.covertype_full
The original exploratory protocol ran 90% first, then the standard full grid.
"""
import subprocess
import sys


def main():
    subprocess.run([sys.executable, '-m', 'train.covertype_experiment'], check=True)
    subprocess.run([sys.executable, '-m', 'train.covertype_sweep'], check=True)


if __name__ == '__main__':
    main()
