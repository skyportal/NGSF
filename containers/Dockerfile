# NGSF runtime image for OSG/CVMFS: the code, its deps, and the template bank.
# osg-skyportal-plugin ships its wrapper + bridge per-job, so nothing plugin- or
# Fritz-specific is baked in here.
#
#   docker buildx build --platform linux/amd64 -f containers/Dockerfile \
#     -t docker.io/michaelwcoughlin/ngsf:latest --push .

# Staged so neither .git (38 MB) nor the bank zip lands in a final-image layer.
FROM debian:bookworm-slim AS assets

# WISeREP serves an error page unless the request looks like a browser.
ARG NGSF_BANK_URL=https://www.wiserep.org/sites/default/files/supyfit_bank.zip

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl unzip git \
    && rm -rf /var/lib/apt/lists/*

COPY . /src
RUN cd /src \
    && (git describe --always 2>/dev/null || echo container) > NGSF/.version \
    && rm -rf .git __pycache__ NGSF/__pycache__ SF_class.ipynb \
              fit_results fit_results_z spectra_to_fit

# WISeREP rate-limits (403s under repeated pulls), so retry; point
# NGSF_BANK_URL at a mirror to stop depending on it during a build.
RUN for attempt in 1 2 3 4 5; do \
        curl -fsSL -A "Mozilla/5.0" -o /tmp/bank.zip "$NGSF_BANK_URL" && break; \
        echo "bank download attempt $attempt failed; retrying in $((attempt * 20))s"; \
        rm -f /tmp/bank.zip; sleep $((attempt * 20)); \
    done \
    && test -s /tmp/bank.zip \
    && unzip -q /tmp/bank.zip -d /tmp/unpacked \
    && mv /tmp/unpacked/bank /bank \
    && rm -rf /tmp/bank.zip /tmp/unpacked


FROM python:3.13-slim

ENV NGSF_DIR=/opt/NGSF \
    NGSF_BANK_DIR=/opt/ngsf-bank \
    PYTHONPATH=/opt/NGSF \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

# Floors, not pins: NGSF has no upper bound on any of these (tested on numpy 2).
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache \
        "numpy>=2.0" \
        "scipy>=1.13" \
        "astropy>=6" \
        "pandas>=2.2" \
        "matplotlib>=3.8" \
        "PyAstronomy>=0.21" \
        "extinction>=0.4.7"

COPY --from=assets /src /opt/NGSF
COPY --from=assets /bank /opt/ngsf-bank

# Fail the build, not a job, on a broken runtime or an empty bank.
RUN python -c "\
import glob, os, NGSF_version; \
from NGSF.Header_Binnings import kill_header; \
from NGSF.SF_functions import all_parameter_space; \
b = os.environ['NGSF_BANK_DIR']; \
n = len(glob.glob(os.path.join(b, 'binnings', '10A', 'sne', '*', '*', '*'))); \
g = len(glob.glob(os.path.join(b, 'binnings', '10A', 'gal', '*'))); \
print('NGSF', NGSF_version.__version__, '| sn templates', n, '| gal templates', g); \
assert n > 500 and g >= 10, 'template bank looks empty'"

WORKDIR /srv
CMD ["python3"]
