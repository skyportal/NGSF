import glob
import os

import numpy as np

from NGSF.auxiliary import select_templates
from NGSF.Header_Binnings import kill_header


class Parameters:
    def __init__(self, data):

        self.pkg_dir = data["pkg_dir"]
        self.bank_dir = data["bank_dir"]
        self.object_to_fit = data["object_to_fit"]
        self.save_results_path = data["saving_results_path"]

        self.use_exact_z = data["use_exact_z"]
        self.z_exact = data["z_exact"]
        self.z_range_begin = data["z_range_begin"]
        self.z_range_end = data["z_range_end"]
        self.z_int = data["z_int"]

        if self.use_exact_z:
            self.redshift = np.array([self.z_exact])
        else:
            z_num = int((self.z_range_end - self.z_range_begin) / self.z_int) + 1
            self.redshift = np.linspace(self.z_range_begin, self.z_range_end, z_num)

        self.mask_galaxy_lines = data["mask_galaxy_lines"]
        self.mask_telluric = data["mask_telluric"]

        # Redshift at which to place the host emission lines. Normally the fit's
        # own redshift, but during a free-redshift scan there isn't one yet, so a
        # caller with a catalog redshift can supply it here and have the lines
        # masked while z is being measured.
        self.mask_host_lines_z = data.get("mask_host_lines_z", None)
        if self.mask_host_lines_z is not None:
            self.mask_host_lines_z = float(self.mask_host_lines_z)

        if (
            self.mask_galaxy_lines == 1
            and len(self.redshift) != 1
            and self.mask_host_lines_z is None
        ):
            raise Exception(
                "Make sure to pick an exact value for z in order to mask the "
                "host lines accordingly, or set mask_host_lines_z."
            )

        # Epochs
        self.epoch_high = data["epoch_high"]
        self.epoch_low = data["epoch_low"]

        # Chose minimum overlap
        self.minimum_overlap = data["minimum_overlap"]

        # Width, in resolution elements, of the running mean divided out of the
        # object and templates before fitting. 0 disables it and fits the flux
        # directly, which lets the galaxy component and A_v absorb continuum
        # mismatch and match at the wrong redshift.
        self.continuum_width = int(data.get("continuum_width", 0))
        # Order of the log-wavelength polynomial continuum (SNID-style); takes
        # precedence over continuum_width. 0 disables it, which is the default:
        # measured over 25 spectra, flattening gained one redshift (11->12 within
        # 0.02 of catalog) but lost three classifications (7->4), because the
        # continuum shape itself carries typing information. Useful when the
        # redshift is what you want, not a better fit overall.
        self.continuum_order = int(data.get("continuum_order", 0))

        # Number of steps for A_v (do not change)
        self.Alam_high = data["Alam_high"]
        self.Alam_low = data["Alam_low"]
        self.Alam_interval = data["Alam_interval"]

        alam_num = int((self.Alam_high - self.Alam_low) / self.Alam_interval) + 1
        self.extconstant = np.linspace(self.Alam_low, self.Alam_high, alam_num)

        # Library to look at
        self.temp_gal_tr = data["temp_gal_tr"]
        self.temp_sn_tr = data["temp_sn_tr"]

        self.resolution = data["resolution"]
        self.upper = data["upper_lam"]
        self.lower = data["lower_lam"]
        self.lam = None

        # Kind of error spectrum ('SG', 'linear' or 'included')
        self.kind = data["error_spectrum"]

        # Show plot?
        self.show = data["show_plot"]

        # Allow png output.
        if "show_plot_png" in data:
            self.show_plot_png = data["show_plot_png"]
        else:
            self.show_plot_png = False

        # How many results to plot?
        self.n = data["how_many_plots"]

        # Verbose?
        if "verbose" in data:
            self.verbose = data["verbose"]
        else:
            self.verbose = 0

        self.iterations = 10

        # Template library

        if self.resolution == 10 or self.resolution == 30:
            templates_gal = glob.glob(
                os.path.join(self.bank_dir, "binnings", str(self.resolution) + "A", "gal", "*")
            )
            templates_gal = [x for x in templates_gal if "CVS" not in x and "README" not in x]
            templates_gal = np.array(templates_gal)

            templates_sn = glob.glob(
                os.path.join(self.bank_dir, "binnings", str(self.resolution) + "A", "sne/**/**/*")
            )
            templates_sn = [
                x
                for x in templates_sn
                if "wiserep_spectra.csv" not in x
                and "info" not in x
                and "photometry" not in x
                and "photometry.pdf" not in x
            ]
            templates_sn = np.array(templates_sn)

        else:
            templates_gal = glob.glob(
                os.path.join(self.bank_dir, "original_resolution", "gal", "*")
            )
            templates_gal = [x for x in templates_gal if "CVS" not in x and "README" not in x]
            templates_gal = np.array(templates_gal)

            templates_sn = glob.glob(
                os.path.join(self.bank_dir, "original_resolution", "sne/**/**/*")
            )
            templates_sn = [
                x
                for x in templates_sn
                if "wiserep_spectra.csv" not in x
                and "info" not in x
                and "photometry" not in x
                and "photometry.pdf" not in x
            ]
            templates_sn = np.array(templates_sn)

        self.templates_sn_trunc = select_templates(templates_sn, self.temp_sn_tr)
        self.templates_gal_trunc = select_templates(templates_gal, self.temp_gal_tr)

    def calc_lam(self):

        if self.upper == self.lower:
            self.lower = kill_header(self.object_to_fit)[1][0] - 300
            self.upper = kill_header(self.object_to_fit)[-1][0] + 300

            interval = int((self.upper - self.lower) / self.resolution)
            self.lam = np.linspace(self.lower, self.upper, interval)

        else:
            interval = int((self.upper - self.lower) / self.resolution)
            self.lam = np.linspace(self.lower, self.upper, interval)
