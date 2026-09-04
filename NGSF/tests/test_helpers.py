"""Error-spectrum estimation, template selection, and version resolution."""

import numpy as np
import pytest

import NGSF_version
from NGSF.auxiliary import select_templates
from NGSF.error_routines import linear_error, savitzky_golay


def _noisy_spectrum(n=500, scale=0.05, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.linspace(4000.0, 9000.0, n)
    flux = 1.0 + rng.normal(0.0, scale, n)
    return np.array([lam, flux]).T


def test_savitzky_golay_returns_an_error_per_wavelength():
    spectrum = _noisy_spectrum()
    error = savitzky_golay(spectrum)
    assert error.shape == spectrum.shape
    assert np.array_equal(error[:, 0], spectrum[:, 0])
    assert (error[:, 1] > 0).all()


def test_savitzky_golay_error_scales_with_the_noise():
    quiet = savitzky_golay(_noisy_spectrum(scale=0.01))[:, 1].mean()
    loud = savitzky_golay(_noisy_spectrum(scale=0.10))[:, 1].mean()
    assert loud > quiet * 3


def test_savitzky_golay_never_returns_a_zero_error():
    # A zero error would divide by zero in the chi2; the routine floors it.
    lam = np.linspace(4000.0, 9000.0, 500)
    flat = np.array([lam, np.ones_like(lam)]).T
    assert (savitzky_golay(flat)[:, 1] > 0).all()


def test_linear_error_returns_an_error_per_wavelength():
    spectrum = _noisy_spectrum(n=500)
    error = linear_error(spectrum)
    assert error.shape == spectrum.shape
    assert np.array_equal(error[:, 0], spectrum[:, 0])


def test_linear_error_pads_a_length_that_is_not_a_multiple_of_ten():
    # The routine fits in blocks of 10 and back-fills the remainder.
    spectrum = _noisy_spectrum(n=505)
    assert linear_error(spectrum).shape == (505, 2)


def test_select_templates_matches_on_substring():
    database = np.array(["path/Ia-norm/sn1.dat", "path/Ic/sn2.dat", "path/II/sn3.dat"])
    assert sorted(select_templates(database, ["Ic", "II"])) == [
        "path/II/sn3.dat",
        "path/Ic/sn2.dat",
    ]


def test_select_templates_is_empty_for_an_absent_type():
    database = np.array(["path/Ia-norm/sn1.dat"])
    assert len(select_templates(database, ["TDE He"])) == 0


@pytest.mark.parametrize("attr", ["ROOT_DIR", "CONFIG_DIR"])
def test_version_module_exposes_its_paths(attr):
    assert getattr(NGSF_version, attr)


def test_version_is_a_string_even_without_git(monkeypatch, tmp_path):
    # Every NGSF module imports NGSF_version, and git fails inside a container
    # (no .git, or a tree owned by another uid), so this must never raise.
    monkeypatch.setattr(NGSF_version, "VERSION_FILE", str(tmp_path / "missing"))
    monkeypatch.setattr(
        NGSF_version, "_git_version", lambda: (_ for _ in ()).throw(OSError("no git"))
    )
    assert NGSF_version._get_version() == "unknown"


def test_version_falls_back_to_the_baked_file(monkeypatch, tmp_path):
    baked = tmp_path / ".version"
    baked.write_text("abc1234 container\n")
    monkeypatch.setattr(NGSF_version, "VERSION_FILE", str(baked))
    monkeypatch.setattr(
        NGSF_version, "_git_version", lambda: (_ for _ in ()).throw(OSError("no git"))
    )
    assert NGSF_version._get_version() == "abc1234 container"
