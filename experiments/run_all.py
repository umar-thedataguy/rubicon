"""Run everything: detection benchmark, showcase pipeline, figures (~8 min on a laptop)."""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
for script in ("run_testbed_eval.py", "run_showcase.py", "make_figures.py"):
    print(f"\n===== {script}")
    if subprocess.call([sys.executable, str(HERE / script)]) != 0:
        sys.exit(f"{script} failed")
