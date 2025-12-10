"""
NEBULA_PATH_CONFIG.py

Path and logging configuration for the NEBULA
(Neuromorphic Event-Based Luminance Asset-tracking) simulation framework.

This module centralizes:
- The base directories for input and output data.
- The locations and names of default input files (e.g., TLEs).
- Common subdirectory and filename conventions for NEBULA outputs.
- Helper functions to create/check the output directory and to configure
  logging (console + file) in a single, explicit place.

It is designed to live under:

    NEBULA/Configuration/NEBULA_PATH_CONFIG.py

so that the NEBULA root directory can be inferred relative to this file.
"""

from pathlib import Path
import logging
import tempfile


# ---------------------------------------------------------------------------
# Base project and data directories
# ---------------------------------------------------------------------------

# NEBULA_ROOT:
#   The root directory of the NEBULA project.
#   If this file is located at:
#       NEBULA/Configuration/NEBULA_PATH_CONFIG.py
#   then:
#       Path(__file__).resolve()          -> .../NEBULA/Configuration/NEBULA_PATH_CONFIG.py
#       Path(__file__).resolve().parent   -> .../NEBULA/Configuration
#       Path(__file__).resolve().parent.parent -> .../NEBULA
NEBULA_ROOT = Path(__file__).resolve().parent.parent

# Base directory for all input data used by NEBULA (TLEs, etc.).
# You can change "input" to something else if you prefer, as long as
# the corresponding folder exists in the NEBULA root.
NEBULA_INPUT_DIR = NEBULA_ROOT / "Input"

# Directory specifically for TLE files under the input directory.
NEBULA_TLE_DIR = NEBULA_INPUT_DIR / "NEBULA_TLES"

# Default path to the observer TLE file (from your original AMOS setup).
OBS_TLE_FILE = NEBULA_TLE_DIR / "NEBULA_GEO_OBSERVER.txt"

# Default path to the target TLE file.
TAR_TLE_FILE = NEBULA_TLE_DIR / "NEBULA_GEO_TARGETS.txt"


# ---------------------------------------------------------------------------
# Output directories and file naming conventions
# ---------------------------------------------------------------------------

# Base directory for NEBULA outputs.  All run-specific folders and files
# should be created somewhere under this root.
NEBULA_OUTPUT_DIR = NEBULA_ROOT / "NEBULA_OUTPUT"

# Subdirectory name for star-frame products within a given run directory.
RUN_SUBDIR_STAR_FRAMES = "star_frames"

# Subdirectory name for individual frame images or per-frame products.
RUN_SUBDIR_FRAMES = "frames"

# Subdirectory name for animations (MP4, GIF, etc.).
RUN_SUBDIR_ANIMATION = "animation"

# Subdirectory name for CSV-based time series and diagnostic tables.
RUN_SUBDIR_CSV = "csv"

# Filename for the Gaia catalog cache used by star-query routines.
GAIA_CACHE_FILENAME = "gaia_cache.csv"

# Filename for the visibility time-series CSV (combined, LOS, illum, etc.).
VIS_TIMESERIES_CSV = "visibility_timeseries.csv"

# Filename for the pointing-vector CSV (time-tagged LOS vectors and RA/Dec).
POINTING_VECTORS_CSV = "pointing_vectors.csv"

# Filename for the per-star visibility summary CSV.
VISIBLE_STARS_CSV = "visible_stars.csv"

# Filename for the per-target visibility CSV used by sidereal/survey modes.
TARGET_VISIBILITY_CSV = "target_visibility.csv"

# Filename for the boresight window summary CSV in sidereal modes.
BORESIGHT_WINDOWS_CSV = "boresight_windows.csv"

# Filename for the visibility time-series plot (PNG).
VIS_TIMESERIES_PNG = "visibility_timeseries.png"

# Filename for the main orbital visibility animation (MP4).
VIS_ANIMATION_MP4 = "visibility_animation.mp4"

# Filename for the HDF5 file storing star frames for a given run.
STAR_FRAMES_HDF5 = "star_frames.hdf5"


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

# Directory where log files will be written.  We keep logs under the
# main NEBULA output directory in a dedicated "logs" subfolder.
LOG_DIR = NEBULA_OUTPUT_DIR / "logs"

# Default log file path used by NEBULA.  Individual runs can override
# this if they want a different log file name.
LOG_FILE = LOG_DIR / "nebula_run.log"


def ensure_output_directory() -> Path:
    """
    Ensure that the base NEBULA output directory exists and is writable.

    This function:
    - Creates NEBULA_OUTPUT_DIR (and any missing parents) if it does not exist.
    - Resolves the directory path to an absolute path.
    - Attempts to create and write to a temporary file inside that directory
      to verify write permissions.

    Returns
    -------
    Path
        The resolved absolute path to NEBULA_OUTPUT_DIR.

    Raises
    ------
    RuntimeError
        If the directory cannot be created or is not writable.
    """
    # Create the output directory (and parents) if it does not already exist.
    try:
        NEBULA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create NEBULA output directory '{NEBULA_OUTPUT_DIR}': {exc}"
        ) from exc

    # Resolve the directory path to an absolute path on disk.
    resolved_output = NEBULA_OUTPUT_DIR.resolve()

    # Attempt to open a temporary file in the resolved directory to
    # test that we have write permission.
    try:
        with tempfile.TemporaryFile(dir=resolved_output) as tf:
            tf.write(b"")  # simple write to confirm access
    except Exception as exc:
        raise RuntimeError(
            f"Cannot write to NEBULA output directory '{resolved_output}': {exc}"
        ) from exc

    # Return the resolved, verified output directory path to the caller.
    return resolved_output


def configure_logging(level: int = logging.INFO) -> Path:
    """
    Configure the root logger to write to both the console and a log file.

    This function:
    - Ensures that the log directory exists.
    - Installs a StreamHandler (console) and FileHandler (LOG_FILE).
    - Sets the logging level and a standard timestamped format.

    It is meant to be called once near the start of your main script.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG).  Defaults to INFO.

    Returns
    -------
    Path
        The path to the log file being used.

    Notes
    -----
    If logging has already been configured elsewhere, this will replace
    existing handlers on the root logger to avoid duplicate output.
    """
    # Make sure the log directory exists before we create a FileHandler.
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Compute the path to the log file that we will attach to the logger.
    log_path = LOG_FILE

    # Remove any existing handlers on the root logger to prevent
    # messages from being duplicated if configure_logging is called
    # more than once.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Create a console handler that writes log messages to stderr/stdout.
    console_handler = logging.StreamHandler()

    # Create a file handler that writes log messages to the specified log file.
    file_handler = logging.FileHandler(log_path, encoding="utf-8")

    # Configure the basic logging setup with our chosen handlers, level,
    # message format, and timestamp format.
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[console_handler, file_handler],
    )

    # Optionally log an initial message so that the log file clearly
    # shows where logging was initialized and which file is being used.
    logging.getLogger(__name__).info("Logging initialized. Log file: %s", log_path)

    # Return the log file path to the caller for reference.
    return log_path
