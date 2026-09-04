import argparse
import json
import os
from datetime import datetime

import pandas as pd

import NGSF_version
from fritz_func import *

parser = argparse.ArgumentParser(
    prog="download_fit_post",
    description="Downloads an ascii spectrum file for the provided specid from Fritz,"
    "runs NGSF and uploads a comment to the source page with the results of the fit",
    epilog="The --fritz_z flag trumps a specified redshift using the -z flag and --INSTRUMENT trumps --wav_range",
)

parser.add_argument("specid", help="specid of spectrum of interest on Fritz")
parser.add_argument(
    "--fritz_z",
    help="Use this option to fetch the value of the redshift on Fritz and use that for fitting.",
    action="store_true",
)
parser.add_argument("-z", "--redshift", help="Use this option to fix the redshift")
parser.add_argument(
    "-n",
    "--ngps",
    action="store_true",
    help="Use this option to modify wavelength range for fitting based on NGPS R and I channels",
)
parser.add_argument(
    "--ghts",
    action="store_true",
    help="Use this option to modify wavelength range for fitting based on the SOAR GHTS bandpass",
)
parser.add_argument(
    "--wav_range",
    nargs=2,
    help="Specify min and max wavelengths to use for fitting in angstroms",
    metavar=("MIN_WAV", "MAX_WAV"),
)

args = parser.parse_args()

try:
    configfile = os.environ["NGSFCONFIG"]
except KeyError:
    configfile = os.path.join(NGSF_version.CONFIG_DIR, "parameters.json")
with open(configfile) as config_file:
    ngsf_cfg = json.load(config_file)

to_fit_dir = str(ngsf_cfg["pkg_dir"] + "spectra_to_fit/")
z_fit_dir = str(ngsf_cfg["pkg_dir"] + "fit_results_z/")
fit_dir = str(ngsf_cfg["pkg_dir"] + "fit_results/")

if args.wav_range is None:
    wav_min, wav_max = None, None

elif args.wav_range is not None:
    wav_min, wav_max = args.wav_range

if args.ngps:
    wav_min = "5900"  # in angstroms
    wav_max = "10000"  # in angstroms

if args.ghts:
    wav_min = "4000"  # in angstroms
    wav_max = "7000"  # in angstroms

if wav_min is None:
    wav_min = "4000"

if wav_max is None:
    wav_max = "9500"


if not os.path.exists(to_fit_dir):  # if folder doesn't exist
    os.system("mkdir " + to_fit_dir)  # creates the folder

if not os.path.exists(z_fit_dir):  # if folder doesn't exist
    os.system("mkdir " + z_fit_dir)  # creates the folder

if not os.path.exists(fit_dir):  # if folder doesn't exist
    os.system("mkdir " + fit_dir)  # creates the folder

specid = args.specid

spec_filename, specid, ztfname = write_ascii_file_from_specid(specid, to_fit_dir)

if args.fritz_z:
    z = get_redshift(ztfname)

else:
    z = args.redshift

if z is None:
    z = 100
    print("No redshift specified.")
    cmd = str(
        "python run.py "
        + str(to_fit_dir)
        + str(spec_filename)
        + " "
        + "100"
        + " "
        + wav_min
        + " "
        + wav_max
    )
    os.system(cmd)
    spec = str(spec_filename)
    name = spec.split(".")[0]
    fitted_csv = fit_dir + name + ".csv"
    df = pd.read_csv(fitted_csv)
    fit_z = df["Z"][0]
    cmd_z = str(
        "python run.py "
        + str(to_fit_dir)
        + str(spec_filename)
        + " "
        + str(fit_z)
        + " "
        + wav_min
        + " "
        + wav_max
    )
    os.system(cmd_z)
    z = fit_z
    images = []
    prefix = spec_filename.split(".")[0]
    for i in range(0, 3):
        fit_png_name = str(prefix + "_ngsf" + str(i) + ".png")
        fit_png_file = str(fit_dir + fit_png_name)
        if os.path.exists(fit_png_file):
            images.append(fit_png_file)
        else:
            images.append("blank.png")

    for i in range(0, 3):
        fit_png_name = str(prefix + "_ngsf" + str(i) + ".png")
        fit_png_file_z = str(z_fit_dir + fit_png_name)
        if os.path.exists(fit_png_file_z):
            images.append(fit_png_file_z)
        else:
            images.append("blank.png")

    comb_fit_png_name = str(prefix + "_ngsf.png")
    comb_fit_png_file = str(fit_dir + comb_fit_png_name)
    combine_images(columns=3, space=10, images=images, savepath=comb_fit_png_file)

    now_str = str(datetime.now())
    text = str(
        "Top 3 matches from superfit with redshift a free parameter (row 1) and with redshift fixed (row 2) for "
        + prefix
        + " fritz specid "
        + specid
        + " ran at "
        + now_str
    )

elif float(z) != 100:
    cmd = str(
        "python run.py "
        + str(to_fit_dir)
        + str(spec_filename)
        + " "
        + str(z)
        + " "
        + wav_min
        + " "
        + wav_max
    )
    os.system(cmd)
    images = []
    prefix = spec_filename.split(".")[0]

    for i in range(0, 3):
        fit_png_name = str(prefix + "_ngsf" + str(i) + ".png")
        fit_png_file_z = str(z_fit_dir + fit_png_name)
        if os.path.exists(fit_png_file_z):
            images.append(fit_png_file_z)
        else:
            images.append("blank.png")

    comb_fit_png_name = str(prefix + "_ngsf.png")
    comb_fit_png_file = str(fit_dir + comb_fit_png_name)
    combine_images(columns=2, space=10, images=images, savepath=comb_fit_png_file)

    now_str = str(datetime.now())
    text = str(
        "Top 3 matches from superfit with redshift fixed to the Fritz value z = "
        + str(z)
        + " for "
        + prefix
        + " fritz specid "
        + specid
        + " ran at "
        + now_str
    )

response = post_comment(ztfname, text, comb_fit_png_file, comb_fit_png_name)
print(response)
