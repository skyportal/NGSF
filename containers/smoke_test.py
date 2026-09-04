"""Fit one spectrum inside the built image.

Mirrors how a caller must drive it: copy the tree somewhere writable (pkg_dir is
also the output root), point the config at the baked bank, and run ``run.py``.

    docker run --rm -v "$PWD:/repo" -w /tmp ngsf:ci python /repo/containers/smoke_test.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NGSF_DIR = Path(os.environ.get("NGSF_DIR", "/opt/NGSF"))
BANK_DIR = Path(os.environ.get("NGSF_BANK_DIR", "/opt/ngsf-bank"))
SPECTRUM = "NGSF/tests/data/SN2021urb_2021-08-06_00-00-00_Keck1_LRIS_TNS.flm"
REDSHIFT = 0.127


def main() -> int:
    if not BANK_DIR.is_dir():
        print(f"FAIL: no template bank at {BANK_DIR}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "NGSF"
        shutil.copytree(NGSF_DIR, tree, ignore=shutil.ignore_patterns("__pycache__", ".git"))
        for sub in ("fit_results", "fit_results_z"):
            (tree / sub).mkdir(exist_ok=True)

        config_path = tree / "config" / "parameters.json"
        config = json.loads(config_path.read_text())
        config.update({"pkg_dir": f"{tree}/", "bank_dir": f"{BANK_DIR}/", "show_plot": 0})
        config_path.write_text(json.dumps(config))

        spectrum = tree / SPECTRUM
        if not spectrum.exists():
            print(f"FAIL: test spectrum missing at {spectrum}", file=sys.stderr)
            return 1

        proc = subprocess.run(
            [sys.executable, "run.py", str(spectrum), str(REDSHIFT), "4000", "9500"],
            cwd=tree,
            env={
                **os.environ,
                "NGSFCONFIG": str(config_path),
                "PYTHONPATH": str(tree),
                "MPLBACKEND": "Agg",
            },
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-3000:], file=sys.stderr)
            print(proc.stderr[-3000:], file=sys.stderr)
            print(f"FAIL: NGSF exited {proc.returncode}", file=sys.stderr)
            return 1

        stem = Path(SPECTRUM).stem
        results = tree / "fit_results_z" / f"{stem}.csv"
        if not results.exists():
            print(f"FAIL: no results at {results}", file=sys.stderr)
            return 1

        rows = list(csv.DictReader(results.open()))
        if not rows:
            print("FAIL: results table is empty", file=sys.stderr)
            return 1
        if not all(abs(float(r["Z"]) - REDSHIFT) < 1e-6 for r in rows):
            print("FAIL: fit did not honour the fixed redshift", file=sys.stderr)
            return 1
        chi2 = [float(r["CHI2/dof"]) for r in rows]
        if not all(c == c for c in chi2):  # NaN check without numpy
            print("FAIL: NaN in CHI2/dof", file=sys.stderr)
            return 1

        plots = sorted((tree / "fit_results_z").glob(f"{stem}_ngsf*.png"))
        if not plots:
            print("FAIL: no fit plots written", file=sys.stderr)
            return 1

        print(f"OK: {len(rows)} ranked matches, best chi2/dof={min(chi2):.3f}, {len(plots)} plots")
        return 0


if __name__ == "__main__":
    sys.exit(main())
