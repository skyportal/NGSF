"""Spectrum reading and binning. No template bank needed."""

import numpy as np
import pytest

from NGSF.Header_Binnings import bin_spectrum, kill_header, kill_header_and_bin


def _write(tmp_path, text, name="spec.ascii"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_kill_header_reads_two_columns(tmp_path):
    path = _write(tmp_path, "4000.0 1.0\n4010.0 2.0\n4020.0 3.0\n")
    spectrum = kill_header(path)
    assert spectrum.shape == (3, 2)
    assert spectrum[0].tolist() == [4000.0, 1.0]


@pytest.mark.parametrize("prefix", ["#", "%", "@", "SIMPLE"])
def test_kill_header_strips_comment_and_keyword_lines(tmp_path, prefix):
    path = _write(tmp_path, f"{prefix} header junk\n4000.0 1.0\n4010.0 2.0\n")
    assert kill_header(path).shape == (2, 2)


def test_kill_header_drops_nan_and_none_fluxes(tmp_path):
    path = _write(tmp_path, "4000.0 1.0\n4010.0 nan\n4020.0 None\n4030.0 2.0\n")
    spectrum = kill_header(path)
    assert spectrum[:, 0].tolist() == [4000.0, 4030.0]


def test_kill_header_accepts_tab_and_multi_space_separators(tmp_path):
    path = _write(tmp_path, "4000.0\t1.0\n4010.0     2.0\n")
    assert kill_header(path).shape == (2, 2)


def test_kill_header_keeps_extra_columns_out_of_the_result(tmp_path):
    # Many archival spectra carry a third error column.
    path = _write(tmp_path, "4000.0 1.0 0.1\n4010.0 2.0 0.2\n")
    spectrum = kill_header(path)
    assert spectrum.shape == (2, 2)
    assert spectrum[:, 1].tolist() == [1.0, 2.0]


def test_bin_spectrum_bins_and_median_normalizes():
    lam = np.arange(4000.0, 5000.0, 1.0)
    spectrum = np.array([lam, np.full_like(lam, 7.0)]).T
    binned = bin_spectrum(spectrum, 10)

    # An astropy table, ~1 bin per 10 A, with the flux divided by its median.
    assert binned.colnames == ["lam_bin", "bin_flux"]
    assert len(binned) == pytest.approx(99, abs=2)
    assert np.allclose(binned["bin_flux"], 1.0)
    assert binned["lam_bin"][0] > lam[0]


def test_bin_spectrum_drops_empty_bins():
    # Far coarser sampling than the bin width leaves most bins empty (NaN); they
    # are masked out rather than carried through as NaN flux.
    lam = np.arange(4000.0, 5000.0, 50.0)
    spectrum = np.array([lam, np.ones_like(lam)]).T
    binned = bin_spectrum(spectrum, 10)
    assert len(binned) <= len(lam)
    assert not np.isnan(np.asarray(binned["bin_flux"])).any()


def test_bin_spectrum_keeps_the_error_column_when_given_one():
    lam = np.arange(4000.0, 5000.0, 1.0)
    spectrum = np.array([lam, np.ones_like(lam), np.full_like(lam, 0.1)]).T
    assert bin_spectrum(spectrum, 10).colnames == [
        "lam_bin",
        "bin_flux",
        "bin_fluxerror",
    ]


def test_kill_header_and_bin_writes_the_binned_file(tmp_path):
    lam = np.arange(4000.0, 5000.0, 1.0)
    path = _write(tmp_path, "".join(f"{w} 1.0\n" for w in lam))
    out = tmp_path / "binned.ascii"

    binned, saved = kill_header_and_bin(path, 10, save_bin=str(out))
    assert saved == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert len(binned) < len(lam)


def test_kill_header_and_bin_rejects_a_resolution_finer_than_the_sampling(tmp_path):
    # Binning below the native sampling would invent structure.
    lam = np.arange(4000.0, 5000.0, 50.0)
    path = _write(tmp_path, "".join(f"{w} 1.0\n" for w in lam))
    with pytest.raises(Exception, match="resolution you chose"):
        kill_header_and_bin(path, 10, save_bin=str(tmp_path / "b.ascii"))
