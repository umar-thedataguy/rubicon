"""
tools.py — the two small utilities the agents share.

profile():        a compact JSON description of a CSV (schema, gaps, samples),
                  which is what the LLM sees instead of the raw rows.
run_in_sandbox(): executes a Python script in a separate OS process with a
                  timeout, so model-written code cannot crash or hang the app.
"""
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd


def profile(csv_path, target, max_sample_values=3):
    df = pd.read_csv(csv_path)
    cols = {}
    for c in df.columns:
        s = df[c]
        cols[c] = {
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique()),
            "samples": [str(v) for v in s.dropna().unique()[:max_sample_values]],
        }
    prof = {"rows": int(len(df)), "target": target, "columns": cols}
    text = json.dumps(prof)
    # a rough size hint only; real token counts come back from the API
    prof["approx_tokens"] = len(text) // 4
    return prof


def run_in_sandbox(code, timeout_sec=60):
    """Run `code` as its own Python process. Returns {success, stdout, stderr}."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout_sec, check=False)
        return {"success": r.returncode == 0, "stdout": r.stdout.strip(),
                "stderr": r.stderr.strip()[-2000:]}
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"timed out after {timeout_sec}s"}
    finally:
        if os.path.exists(path):
            os.remove(path)
