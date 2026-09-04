"""End-to-end fit against the real template bank.

Marked `integration`: it needs the 74 MB WISeREP bank, which is downloaded once
into <repo>/bank (cached in CI). Deselect with `-m "not integration"`.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

REPO = Path(__file__).resolve().parents[2]
BANK_PATH = REPO / "bank"
# WISeREP rate-limits; set NGSF_BANK_URL to a mirror to avoid depending on it.
BANK_URL = os.environ.get(
    "NGSF_BANK_URL", "https://www.wiserep.org/sites/default/files/supyfit_bank.zip"
)
SPECTRUM = REPO / "NGSF/tests/data/SN2021urb_2021-08-06_00-00-00_Keck1_LRIS_TNS.flm"
REDSHIFT = 0.127

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def bank():
    if BANK_PATH.is_dir():
        return BANK_PATH
    archive = REPO / "supyfit_bank.zip"
    try:
        # WISeREP serves an error page unless the request looks like a browser.
        request = Request(BANK_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=300) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as z:
            z.extractall(REPO)
    except Exception as e:  # noqa: BLE001 — no bank means the fit can't run at all
        # Skipping keeps a WISeREP outage from blocking unrelated PRs, but a
        # green run then proves nothing about the fit; set
        # NGSF_REQUIRE_INTEGRATION=1 (CI, once the bank has a reliable mirror)
        # to make an unavailable bank a failure instead.
        message = f"template bank unavailable: {e}"
        if os.environ.get("NGSF_REQUIRE_INTEGRATION"):
            pytest.fail(message)
        pytest.skip(message)
    finally:
        archive.unlink(missing_ok=True)
    return BANK_PATH


@pytest.fixture
def tree(tmp_path, bank):
    """A writable NGSF tree: pkg_dir is both the code root and the output root."""
    root = tmp_path / "NGSF"
    shutil.copytree(
        REPO,
        root,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".git", "bank", "fit_results", "fit_results_z"
        ),
    )
    for sub in ("fit_results", "fit_results_z"):
        (root / sub).mkdir(exist_ok=True)

    config_path = root / "config" / "parameters.json"
    config = json.loads(config_path.read_text())
    config.update({"pkg_dir": f"{root}/", "bank_dir": f"{bank}/", "show_plot": 0})
    config_path.write_text(json.dumps(config))
    return root


def run_fit(tree, redshift):
    proc = subprocess.run(
        [sys.executable, "run.py", str(SPECTRUM), str(redshift), "4000", "9500"],
        cwd=tree,
        env={
            **os.environ,
            "NGSFCONFIG": str(tree / "config" / "parameters.json"),
            "PYTHONPATH": str(tree),
            "MPLBACKEND": "Agg",
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"NGSF failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    out_dir = tree / ("fit_results" if float(redshift) == 100 else "fit_results_z")
    results = out_dir / f"{SPECTRUM.stem}.csv"
    assert results.exists(), f"no results at {results}"
    return list(csv.DictReader(results.open())), out_dir


def test_fit_at_fixed_redshift(tree):
    rows, out_dir = run_fit(tree, REDSHIFT)

    assert rows, "results table is empty"
    assert all(abs(float(r["Z"]) - REDSHIFT) < 1e-6 for r in rows)
    chi2 = [float(r["CHI2/dof"]) for r in rows]
    assert all(c == c for c in chi2), "NaN in CHI2/dof"
    assert chi2 == sorted(chi2), "results are not ranked by chi2"

    # The ranked fit plots are what gets posted back to SkyPortal/Fritz.
    assert sorted(out_dir.glob(f"{SPECTRUM.stem}_ngsf*.png"))


def test_fit_records_the_parameters_it_used(tree):
    run_fit(tree, REDSHIFT)
    used = json.loads((tree / "fit_results_z" / f"{SPECTRUM.stem}_pars_used.json").read_text())
    assert used["use_exact_z"] == 1
    assert used["z_exact"] == REDSHIFT
    assert used["lower_lam"] == 4000 and used["upper_lam"] == 9500


def test_fit_rejects_a_spectrum_that_is_too_short(tree, tmp_path):
    short = tmp_path / "short.ascii"
    short.write_text("".join(f"{4000.0 + i} 1.0\n" for i in range(40)))
    proc = subprocess.run(
        [sys.executable, "run.py", str(short), str(REDSHIFT), "4000", "9500"],
        cwd=tree,
        env={
            **os.environ,
            "NGSFCONFIG": str(tree / "config" / "parameters.json"),
            "PYTHONPATH": str(tree),
            "MPLBACKEND": "Agg",
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "too short to fit" in proc.stdout + proc.stderr
