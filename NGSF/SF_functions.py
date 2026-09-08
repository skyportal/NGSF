import itertools
import json
import os

import extinction
import numpy as np
from astropy import table
from astropy.io import ascii
from extinction import apply
from PyAstronomy import pyasl
from scipy import interpolate

import NGSF_version
from NGSF.error_routines import linear_error, savitzky_golay
from NGSF.get_metadata import Metadata
from NGSF.Header_Binnings import bin_spectrum_bank, kill_header, mask_lines_bank
from NGSF.params import Parameters

try:
    configfile = os.environ["NGSFCONFIG"]
except KeyError:
    configfile = os.path.join(NGSF_version.CONFIG_DIR, "parameters.json")
with open(configfile) as config_file:
    ngsf_cfg = json.load(config_file)

np.seterr(divide="ignore", invalid="ignore")


def sn_hg_arrays(
    z,
    extcon,
    lam,
    templates_sn_trunc,
    templates_sn_trunc_dict,
    templates_gal_trunc,
    templates_gal_trunc_dict,
    alam_dict,
):

    sn = []
    gal = []
    for i in range(0, len(templates_sn_trunc)):
        one_sn = templates_sn_trunc_dict[templates_sn_trunc[i]]
        a_lam_sn = alam_dict[templates_sn_trunc[i]]
        redshifted_sn = one_sn[:, 0] * (z + 1)
        extinct_excon = one_sn[:, 1] * 10 ** (-0.4 * extcon * a_lam_sn) / (1 + z)
        sn_interp = np.interp(lam, redshifted_sn, extinct_excon, left=np.nan, right=np.nan)

        sn.append(sn_interp)

    for i in range(0, len(templates_gal_trunc)):
        one_gal = templates_gal_trunc_dict[templates_gal_trunc[i]]
        gal_interp = np.interp(
            lam,
            one_gal[:, 0] * (z + 1),
            one_gal[:, 1] / (1 + z),
            left=np.nan,
            right=np.nan,
        )
        gal.append(gal_interp)

    # Redefine sn and gal by adding a new axis

    sn = np.array(sn)
    gal = np.array(gal)

    gal = gal[:, np.newaxis, :]
    sn = sn[np.newaxis, :, :]

    return sn, gal


def remove_telluric(spectrum):

    lam = spectrum[:, 0]
    flux = spectrum[:, 1]

    flux_no_tell = 0
    for i in range(0, len(lam)):
        if 7594 <= lam[i] <= 7680:
            flux[i] = -10000

        array1 = flux
        flux_no_tell = np.where(array1 == -10000, np.nan, array1)

    return np.array([lam, flux_no_tell]).T


def Alam(lamin, A_v=1, R_v=3.1):
    """
    Add extinction with R_v = 3.1 and A_v = 1, A_v = 1 in order
    to find the constant of proportionality for
    the extinction law.
    """

    flux = np.ones(len(lamin), dtype=float)
    # flux = [float(x) for x in flux]
    lamin = np.array([float(i) for i in lamin])
    redreturn = apply(extinction.ccm89(lamin, A_v, R_v), flux)

    return redreturn


def error_obj(kind, lam, object_to_fit):
    """
    This function gives an error based on user input. The error can be obtained
    by either a Savitzky-Golay filter, a linear error approximation or it can
    come with the file itself.

    parameters
    ----------
    It takes a "kind" of error (linear, SG or included), a lambda range and an
    object whose error we want to obtain

    returns
    -------
    Error.

    """

    object_spec = np.loadtxt(object_to_fit)
    object_spec[:, 1] = object_spec[:, 1] / np.nanmedian(object_spec[:, 1])
    sigma = None

    if kind == "included" and len(object_spec[1, :]) > 2:
        error = object_spec[:, 2]

        object_err_interp = interpolate.interp1d(
            object_spec[:, 0], error, bounds_error=False, fill_value="nan"
        )

        sigma = object_err_interp(lam)

    if kind == "linear":
        error = linear_error(object_spec)

        object_err_interp = interpolate.interp1d(
            error[:, 0], error[:, 1], bounds_error=False, fill_value="nan"
        )

        sigma = object_err_interp(lam)

    if kind == "sg":
        error = savitzky_golay(object_spec)

        object_err_interp = interpolate.interp1d(
            error[:, 0], error[:, 1], bounds_error=False, fill_value="nan"
        )

        sigma = object_err_interp(lam)

    return sigma


# Result columns. Module-level so rows can be accumulated cheaply in the hot
# loop and turned into a single table once, instead of one table per row.
RESULT_NAMES = (
    "SPECTRUM",
    "GALAXY",
    "SN",
    "CONST_SN",
    "CONST_GAL",
    "Z",
    "A_v",
    "Phase",
    "Band",
    "Frac(SN)",
    "Frac(gal)",
    "CHI2/dof",
    "CHI2/dof2",
)
RESULT_DTYPES = (
    "S200",
    "S200",
    "S200",
    "f",
    "f",
    "f",
    "f",
    "S200",
    "S200",
    "f",
    "f",
    "f",
    "f",
)


def core(
    int_obj,
    z,
    extcon,
    templates_sn_trunc,
    templates_sn_trunc_dict,
    templates_gal_trunc,
    templates_gal_trunc_dict,
    alam_dict,
    lam,
    resolution,
    iterations,
    **kwargs,
):
    """

    Inputs:
    ------
    z - an array of redshifts
    extcon - array of values of A_v

    Outputs:
    --------
    Astropy table with the names for the best fit supernova and host galaxy,
    constants of proportionality for both the host galaxy and supernova
    templates, the value of chi2, the corresponding redshift and A_v.

    """

    if resolution:
        pass
    kind = kwargs["kind"]
    original = kwargs["original"]
    minimum_overlap = kwargs["minimum_overlap"]

    name = os.path.basename(original)

    sigma = error_obj(kind, lam, original)

    sn, gal = sn_hg_arrays(
        z,
        extcon,
        lam,
        templates_sn_trunc,
        templates_sn_trunc_dict,
        templates_gal_trunc,
        templates_gal_trunc_dict,
        alam_dict,
    )

    # Apply linear algebra witchcraft

    # Each of these five sums was being recomputed two or three times over; every
    # np.nansum copies the whole cube to strip NaNs, so they are worth holding.
    sn_sq = np.nansum(sn**2, 2)
    gal_sq = np.nansum(gal**2, 2)
    sn_gal = np.nansum(gal * sn, 2)
    sn_obj = np.nansum(sn * int_obj, 2)
    gal_obj = np.nansum(gal * int_obj, 2)

    c = 1 / (sn_sq * gal_sq - sn_gal**2)
    b = c * (gal_sq * sn_obj - sn_gal * gal_obj)
    d = c * (sn_sq * gal_obj - sn_gal * sn_obj)

    b[b < 0] = np.nan
    d[d < 0] = np.nan

    # Add new axis in order to compute chi2
    sn_b = b[:, :, np.newaxis]
    gal_d = d[:, :, np.newaxis]

    # Obtain number of degrees of freedom

    # The residual was built twice, once for the overlap count and once for chi2.
    residual = int_obj - (sn_b * sn + gal_d * gal)

    a = np.isnan((residual / sigma) ** 2)
    # a is boolean, so there is nothing for nansum to strip.
    times = len(lam) - np.sum(a, 2)

    overlap = times / len(lam) > minimum_overlap

    # Obtain and reduce chi2
    chi2 = np.nansum((residual**2 / sigma**2), 2)

    # avoid short overlaps
    chi2[~overlap] = np.inf

    reduchi2 = chi2 / (times - 2) ** 2
    reduchi2 = np.where(reduchi2 == 0, 1e10, reduchi2)

    reduchi2_once = chi2 / (times - 2)
    reduchi2_once = np.where(reduchi2_once == 0, 1e10, reduchi2_once)

    # Flatten the matrix out and obtain indices corresponding values of
    # proportionality constants
    reduchi2_1d = reduchi2.ravel()

    index = np.argsort(reduchi2_1d)

    redchi2 = []
    rows = []

    for i in range(iterations):
        idx = np.unravel_index(index[i], reduchi2.shape)
        rchi2 = reduchi2[idx]

        redchi2.append(rchi2)

        supernova_file = templates_sn_trunc[idx[1]]
        host_galaxy_file = templates_gal_trunc[idx[0]]

        host_galaxy_file = str(host_galaxy_file)
        idxx = host_galaxy_file.rfind("/")
        host_galaxy_file = host_galaxy_file[idxx + 1 :]

        bb = b[idx[0]][idx[1]]

        dd = d[idx[0]][idx[1]]
        sn_flux = sn[0, idx[1], :]
        gal_flux = gal[idx[0], 0, :]
        sn_cont = bb * np.nanmean(sn_flux * 10 ** (-0.4 * extcon * Alam(lam)))
        gal_cont = dd * np.nanmean(gal_flux)
        sum_cont = sn_cont + gal_cont
        sn_cont = sn_cont / sum_cont
        gal_cont = gal_cont / sum_cont

        ii = supernova_file.rfind(":")
        the_phase = supernova_file[ii + 1 : -1]
        the_band = supernova_file[-1]

        # Same np.array(...) as the table used to be built from, so the values
        # go through the identical string coercion and the output is unchanged.
        rows.append(
            np.array(
                [
                    os.path.basename(name),
                    host_galaxy_file,
                    supernova_file,
                    bb,
                    dd,
                    z,
                    extcon,
                    the_phase,
                    the_band,
                    sn_cont,
                    gal_cont,
                    reduchi2_once[idx],
                    reduchi2[idx],
                ]
            )
        )

    return rows, redchi2


def mask_gal_lines(data, z_obj):

    host_lines = np.array(
        [
            6564.61,
            4862.69,
            3726.09,
            3729.88,
            5008.24,
            4960.30,
            6549.84,
            6585.23,
            6718.32,
            6732.71,
        ]
    )

    host_lines_air = (1 + z_obj) * pyasl.airtovac2(host_lines)
    host_range_air = np.column_stack([host_lines_air, host_lines_air])
    z_disp = 4e2 / 3e5
    host_range_air[:, 0] = host_range_air[:, 0] * (1 - z_disp)
    host_range_air[:, 1] = host_range_air[:, 1] * (1 + z_disp)

    def func(x, y):
        return (x < y[1]) & (x > y[0])

    cum_mask = np.array([True] * len(data[:, 0]))
    for i in range(len(host_lines_air)):
        mask = np.array(list(map(lambda x: ~func(x, host_range_air[i]), data[:, 0])))
        cum_mask = cum_mask & mask

    data_masked = data[cum_mask]

    return data_masked


def all_parameter_space(
    int_obj,
    redshift,
    extconstant,
    templates_sn_trunc,
    templates_gal_trunc,
    lam,
    resolution,
    iterations,
    **kwargs,
):
    """

    This function loops the core function of superfit over two user given
    arrays, one for redshift and one for the extinction constant, it then sorts
    all the chi2 values obtained and plots the curve that corresponds to the
    smallest one. This is not the recommended method to use, since it takes the
    longest time, it is rather a method to check results if there are any doubts
    with the two recommended methods.

    Parameters
    ----------
    Truncated SN and HG template libraries, extinction array and redshift array,
    lambda axis and **kwargs for the object path.

    Returns
    -------
    Astropy table with the best fit parameters: Host Galaxy and Supernova
    proportionality constants, redshift, extinction law constant and chi2 value,
    plots are optional. In this version for the fit the same SN can appear with
    two different redshifts (since it is a brute-force method in which we go
    over the whole parameter space we don't want to eliminate any results).

    For plotting: in order not to plot every single result the user can choose
    how many to plot, default set to the first three.

    """

    import time

    parameters = Parameters(ngsf_cfg)

    verbose = parameters.verbose == 1

    metadata = Metadata()

    print("NGSF started")
    start = time.time()

    save = kwargs["save"]

    if templates_sn_trunc is not None:
        pass
    templates_sn_trunc_dict = {}
    templates_gal_trunc_dict = {}
    alam_dict = {}
    # sn_spec_files = [str(x) for x in metadata.shorhand_dict.values()]
    path_dict = {}

    all_bank_files = [str(x) for x in metadata.dictionary_all_trunc_objects.values()]

    print("Reading SN templates", flush=True)
    if resolution == 10 or resolution == 30:
        for i in range(0, len(all_bank_files)):
            a = all_bank_files[i]

            full_name = a[a.find("sne") :]
            one_sn = os.path.join(
                parameters.bank_dir, "binnings", str(resolution) + "A/", str(full_name)
            )

            if parameters.mask_galaxy_lines == 1:
                one_sn = np.loadtxt(one_sn)
                one_sn = mask_lines_bank(one_sn)
            else:
                one_sn = np.loadtxt(one_sn)

            idx = all_bank_files[i].rfind("/") + 1
            filename = all_bank_files[i][idx:]

            short_name = str(metadata.shorhand_dict[filename])

            path_dict[short_name] = all_bank_files[i]

            templates_sn_trunc_dict[short_name] = one_sn
            alam_dict[short_name] = Alam(one_sn[:, 0])

    elif parameters.resolution != 30 or parameters.resolution != 10:
        for i in range(0, len(all_bank_files)):
            if parameters.mask_galaxy_lines == 1:
                one_sn = kill_header(all_bank_files[i])
                one_sn = mask_lines_bank(one_sn)
                one_sn = bin_spectrum_bank(one_sn, resolution)

            elif parameters.mask_galaxy_lines == 0:
                one_sn = kill_header(all_bank_files[i])
                one_sn = bin_spectrum_bank(one_sn, resolution)
            else:
                one_sn = None

            idx = all_bank_files[i].rfind("/") + 1
            filename = all_bank_files[i][idx:]

            short_name = str(metadata.shorhand_dict[filename])

            path_dict[short_name] = all_bank_files[i]
            templates_sn_trunc_dict[short_name] = one_sn
            alam_dict[short_name] = Alam(one_sn[:, 0])

    print("Reading Galaxy templates", flush=True)
    for i in range(0, len(templates_gal_trunc)):
        one_gal = np.loadtxt(templates_gal_trunc[i])
        one_gal = bin_spectrum_bank(one_gal, resolution)
        templates_gal_trunc_dict[templates_gal_trunc[i]] = one_gal

    sn_spec_files = [x for x in path_dict.keys()]
    results = []

    if not verbose:
        print(
            f"Probing A_v: {extconstant[0]:.2f} to {extconstant[-1]:.2f} "
            f"with {extconstant[1] - extconstant[0]:.2f} sampling",
            flush=True,
        )
    old_z = -999.0
    for element in itertools.product(redshift, extconstant):
        if element[0] != old_z and verbose:
            print(f"\nProbing z={element[0]:.2f}", flush=True)
            old_z = element[0]
            print("A_v = ", end=" ", flush=True)

        if verbose:
            print(f"{element[1]:.2f}", end=" ", flush=True)

        a, _ = core(
            int_obj,
            element[0],
            element[1],
            sn_spec_files,
            templates_sn_trunc_dict,
            templates_gal_trunc,
            templates_gal_trunc_dict,
            alam_dict,
            lam,
            resolution,
            iterations,
            **kwargs,
        )

        results.extend(a)

    if verbose:
        print("\nDone.")

    result = table.Table(rows=results, names=RESULT_NAMES, dtype=RESULT_DTYPES)

    result.sort("CHI2/dof2")

    result = table.unique(result, keys="SN", keep="first")

    result.sort("CHI2/dof2")

    ascii.write(result, save + ".csv", format="csv", fast_writer=False, overwrite=True)

    end = time.time()
    print(f"Runtime: {end - start: .2f}s ")

    return
