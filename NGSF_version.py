import os
import subprocess
from pathlib import Path

ROOT_DIR = os.path.dirname(str(Path(__file__)))
_ROOT_CONFIG = os.path.join(ROOT_DIR, "config")
# An installed copy carries the config inside the package, not beside it.
CONFIG_DIR = (
    _ROOT_CONFIG if os.path.isdir(_ROOT_CONFIG) else os.path.join(ROOT_DIR, "NGSF", "config")
)
VERSION_FILE = os.path.join(ROOT_DIR, "NGSF", ".version")


def _git_version():
    cwd = os.getcwd()
    os.chdir(ROOT_DIR)
    try:
        git_hash = subprocess.check_output(
            ["git", "describe", "--always"], stderr=subprocess.DEVNULL
        ).strip()
        git_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%cd"], stderr=subprocess.DEVNULL
        ).strip()
        return git_hash.decode("utf8") + " " + git_date.decode("utf8")
    finally:
        os.chdir(cwd)


def _get_version():
    # An installed copy has no .git, and git rejects a tree owned by another
    # uid ("dubious ownership") — the normal case inside a container.
    try:
        return _git_version()
    except Exception:
        pass
    try:
        with open(VERSION_FILE) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


__version__ = _get_version()
