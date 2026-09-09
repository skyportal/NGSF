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
# Mirror of the WISeREP archive; wiserep.org 403s under repeated automated pulls.
BANK_URL = os.environ.get(
    "NGSF_BANK_URL",
    "https://github.com/skyportal/NGSF/releases/download/template-bank-v1/supyfit_bank.zip",
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


def test_fit_writes_the_model_spectrum_for_each_ranked_match(tree):
    # Consumers overlay this on the observed spectrum, so the name and the
    # two-column shape are a contract, not an implementation detail.
    rows, out_dir = run_fit(tree, REDSHIFT)

    models = sorted(out_dir.glob(f"{SPECTRUM.stem}_ngsf*_model.txt"))
    assert len(models) == len(sorted(out_dir.glob(f"{SPECTRUM.stem}_ngsf*.png")))

    for model in models:
        lam, flux = [], []
        for line in model.read_text().splitlines():
            if line.startswith("#"):
                continue
            w, f = line.split()
            lam.append(float(w))
            flux.append(float(f))
        assert len(lam) > 100
        assert lam == sorted(lam)
        assert lam[0] >= 4000 and lam[-1] <= 9500  # the fitted range
        # Median-normalized like the binned observation, so an overlay lines up.
        finite = [f for f in flux if f == f]
        assert finite, f"{model.name} is all NaN"
        assert 0.1 < sum(finite) / len(finite) < 10


def test_fit_handles_a_spectrum_narrower_than_the_fitted_range(tree, tmp_path):
    """A spectrum that stops short of the fitted range leaves the error spectrum
    undefined past its last point. Those wavelengths must drop out of chi2, not
    poison it -- getting this wrong makes every chi2 inf and the ranking junk,
    and it is invisible on spectra that span the range. Cut at 9000 A, which
    still clears minimum_overlap; cutting further would make inf correct."""
    narrow = tmp_path / "narrow.ascii"
    kept = [
        line
        for line in SPECTRUM.read_text().splitlines()
        if line.strip() and not line.startswith("#") and float(line.split()[0]) < 9000
    ]
    narrow.write_text("\n".join(kept) + "\n")

    proc = subprocess.run(
        [sys.executable, "run.py", str(narrow), str(REDSHIFT), "4000", "9500"],
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

    rows = list(csv.DictReader((tree / "fit_results_z" / "narrow.csv").open()))
    assert rows, "results table is empty"
    chi2 = [float(r["CHI2/dof"]) for r in rows]
    assert all(c == c for c in chi2), "NaN in CHI2/dof"
    assert any(c != float("inf") for c in chi2), "every chi2 is inf; the fit found nothing"
    # The table is ranked on CHI2/dof2, which is the column NGSF sorts by.
    ranked = [float(r["CHI2/dof2"]) for r in rows]
    assert ranked == sorted(ranked), "results are not ranked"


def test_fit_refuses_a_spectrum_that_barely_overlaps_the_range(tree, tmp_path):
    """Too little overlap must fail loudly. It used to write ten rows of
    infinities that sort into an arbitrary order and read like a real ranking,
    so callers would annotate a source with a classification built on nothing."""
    sparse = tmp_path / "sparse.ascii"
    kept = [
        line
        for line in SPECTRUM.read_text().splitlines()
        if line.strip() and not line.startswith("#") and float(line.split()[0]) < 7000
    ]
    sparse.write_text("\n".join(kept) + "\n")

    proc = subprocess.run(
        [sys.executable, "run.py", str(sparse), str(REDSHIFT), "4000", "9500"],
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
    assert proc.returncode != 0, "a spectrum with too little overlap was fit anyway"
    assert "cannot be fit over this range" in proc.stdout + proc.stderr
    # and no results table is left behind for a caller to read as a real answer
    assert not (tree / "fit_results_z" / "sparse.csv").exists()


def test_free_redshift_fit_writes_the_chi2_profile(tree):
    """The results table keeps only the best row per template, so the shape of
    chi2 in z is otherwise computed and discarded. Kept as a diagnostic; note
    the width of the minimum does not by itself indicate a trustworthy z."""
    # A short grid keeps this quick; the profile behaviour is the same.
    config_path = tree / "config" / "parameters.json"
    config = json.loads(config_path.read_text())
    config.update({"z_range_begin": 0.0, "z_range_end": 0.01, "z_int": 0.001})
    config_path.write_text(json.dumps(config))

    run_fit(tree, 100)  # 100 is NGSF's free-redshift sentinel

    profile = tree / "fit_results" / f"{SPECTRUM.stem}_chi2_vs_z.csv"
    assert profile.exists(), "no chi2 profile written"
    rows = list(csv.DictReader(profile.open()))
    assert len(rows) == 11, f"expected one row per redshift step, got {len(rows)}"

    zs = [float(r["Z"]) for r in rows]
    assert zs == sorted(zs), "profile is not ordered by redshift"
    assert abs(zs[0]) < 1e-6 and abs(zs[-1] - 0.01) < 1e-6, "profile does not span the grid"

    chi2 = [float(r["CHI2/dof2"]) for r in rows]
    assert all(c == c for c in chi2), "NaN in the profile"
    # The best profile value must agree with the best of the results table.
    best_rows = list(csv.DictReader((tree / "fit_results" / f"{SPECTRUM.stem}.csv").open()))
    assert min(chi2) <= float(best_rows[0]["CHI2/dof2"]) * 1.0001


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
