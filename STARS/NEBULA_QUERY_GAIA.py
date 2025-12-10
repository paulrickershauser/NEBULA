"""
NEBULA_QUERY_GAIA
=================

Catalog-level Gaia query module for NEBULA.

This module consumes per-window sky footprints produced by NEBULA_SKY_SELECTOR
and builds a Gaia DR3 catalog cache per observer / per window. The resulting
cache can be used later by NEBULA_STAR_PROJECTION (or similar) to project
catalog stars into the detector and compute flux contributions.

---------------------------------------------------------------------------
Purpose
---------------------------------------------------------------------------

- Load per-observer, per-window frames with sky footprints from:

    NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked_with_sky.pkl

- For each *eligible* window (see Assumptions / Invariants), build a Gaia
  cone-search parameter set and query Gaia DR3 either via astroquery TAP
  or via local Gaia shards.

- Compress the Gaia results into compact numpy arrays and metadata, grouped
  per observer and per window.

- Save the result to:

    NEBULA_OUTPUT/STARS/<catalog_name>/obs_gaia_cones.pkl

---------------------------------------------------------------------------
Assumptions
---------------------------------------------------------------------------

- sky_center_ra_deg / sky_center_dec_deg are ICRS RA/Dec in *degrees*,
  as produced by NEBULA_SKY_SELECTOR.

- sky_radius_deg already includes:
    FOV half-angle + max great-circle slew + a safety margin,
  so there is no *implicit* under-query at the catalog level.

- Circular cone approximation:
    The sensor FOV half-angle is ≲ a few degrees in the current NEBULA use,
    so a circular cone in ICRS is an adequate approximation to the true
    projected sensor FOV + slew footprint. Differences between an exact
    projected polygon and a cone are negligible compared to pixel scale
    and pointing errors.

- t_mid_utc is an ISO-8601 string in UTC (no leap seconds), derived from
  the same timescale as the window start_time / end_time in the frames
  pickle.

- Gaia positions and proper motions are referenced to
  NEBULA_STAR_CATALOG.reference_epoch (e.g., 2016.0 for DR3). No epoch
  propagation is performed in this module.

- Only windows with:
    sky_selector_status == "ok"  AND  n_targets > 0
  are *queried* for Gaia stars. Other windows are skipped and only appear
  in summary counts.

---------------------------------------------------------------------------
Invariants
---------------------------------------------------------------------------

For each observer in the resulting Gaia cache:

- window_index values are a subset of the window_index values in the
  corresponding frames file.

- Within a given observer, window_index values are unique (no duplicates).

---------------------------------------------------------------------------
Scope
---------------------------------------------------------------------------

This module is purely:

- geometric (ICRS cone queries) and
- catalog-level photometric (Gaia G, BP, RP, etc.)

It explicitly does NOT:

- compute sky footprints (that is NEBULA_SKY_SELECTOR’s job),
- project stars into pixels, or
- compute sensor-band flux or time-dependent RA/Dec propagation.

All bandpass/conversion, flux-in-sensor, and time propagation logic lives
downstream in NEBULA_STAR_PROJECTION and radiometry modules.
"""

# -------------------------------------------------------------------------
# Standard library imports
# -------------------------------------------------------------------------

# Import logging for module-wide logging functionality.
import logging

# Import os for filesystem path manipulations.
import os

# Import pickle for reading and writing NEBULA pickle files.
import pickle

# Import datetime for timestamps in metadata.
import datetime as dt

# Import typing helpers for type annotations.
from typing import Any, Dict, List, Optional, Tuple, Set

# Import dataclass utilities for compact structured containers.
from dataclasses import dataclass, asdict

# -------------------------------------------------------------------------
# Third-party imports
# -------------------------------------------------------------------------

# Import numpy for numerical arrays and simple statistics.
import numpy as np

# Import astropy Table for Gaia query results.
from astropy.table import Table

# Import SkyCoord for ICRS coordinate handling.
from astropy.coordinates import SkyCoord

# Import units for specifying angular radii in cone searches.
import astropy.units as u

# Import Time to compute mid-times and MJD.
from astropy.time import Time

# -------------------------------------------------------------------------
# NEBULA configuration imports
# (adjust package paths if your tree is slightly different)
# -------------------------------------------------------------------------

# Import NEBULA output directory root.
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

# Import active sensor config (for star_mag_limit_G).
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR

# Import star catalog + query config and helpers.
from Configuration.NEBULA_STAR_CONFIG import (
    NEBULA_STAR_CATALOG,
    NEBULA_STAR_QUERY,
    get_gaia_query_columns,
    VARIABLE_FLAG_MAPPING,
)

# Progress bar import 
from tqdm import tqdm 

# -------------------------------------------------------------------------
# Astroquery Gaia import (for online mode)
# -------------------------------------------------------------------------

# Import Gaia TAP interface for cone_search_async.
from astroquery.gaia import Gaia

# -------------------------------------------------------------------------
# Module-level globals
# -------------------------------------------------------------------------
FRAMES_WITH_SKY_FILENAME = "obs_target_frames_ranked_with_sky.pkl"
GAIA_CACHE_FILENAME = "obs_gaia_cones.pkl"

# Create a module-level logger placeholder; initialized by get_logger().
_logger: Optional[logging.Logger] = None

# Track which optional Gaia columns we have already warned about once.
_missing_column_warnings: Set[str] = set()


def get_logger(name: str = "NEBULA_QUERY_GAIA") -> logging.Logger:
    """
    Get or create a module-level logger.

    Parameters
    ----------
    name : str, optional
        Name of the logger to retrieve or create. Defaults to
        "NEBULA_QUERY_GAIA".

    Returns
    -------
    logging.Logger
        Configured logger instance for this module.

    Notes
    -----
    - This helper ensures a single logger instance is reused across the
      module, avoiding duplicate handlers if imported multiple times.
    """
    # Declare _logger as global so we can assign to it.
    global _logger

    # If the logger is already initialized, just return it.
    if _logger is not None:
        return _logger

    # Otherwise, create a new logger with the given name.
    logger = logging.getLogger(name)

    # If the logger has no handlers, configure a simple StreamHandler.
    if not logger.handlers:
        # Create a StreamHandler that logs to stderr by default.
        handler = logging.StreamHandler()
        # Create a minimal formatter with time, name, and level.
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        # Attach the formatter to the handler.
        handler.setFormatter(formatter)
        # Attach the handler to the logger.
        logger.addHandler(handler)

    # Set a default logging level if not already set.
    logger.setLevel(logging.INFO)

    # Store the logger in the module-level variable.
    _logger = logger

    # Return the configured logger.
    return logger


# -------------------------------------------------------------------------
# Dataclasses for query parameters and results
# -------------------------------------------------------------------------


@dataclass
class GaiaWindowQueryParams:
    """
    Per-window Gaia cone-search parameters.

    This dataclass captures the geometric and photometric policy for a
    single window for a single observer.

    Parameters
    ----------
    obs_name : str
        Name of the observer satellite (e.g., "SBSS (USA 216)").

    window_index : int
        Integer index of the window within that observer's frames.
        Must match the window_index in the frames pickle.

    ra_center_deg : float
        RA of the cone center in ICRS degrees.

    dec_center_deg : float
        Dec of the cone center in ICRS degrees.

    sky_radius_deg : float
        Radius of the sky footprint in degrees, as produced by
        NEBULA_SKY_SELECTOR. This already includes:
        - FOV half-angle,
        - max great-circle slew, and
        - a safety margin.

    query_radius_deg : float
        Catalog cone radius in degrees. This is typically:

            query_radius_deg = sky_radius_deg + NEBULA_STAR_QUERY.cone_padding_deg

        so that the Gaia cone is slightly larger than the sky footprint.

    mag_limit_G : float
        Gaia G-band limiting magnitude for the query, i.e.:

            mag_limit_G = mag_limit_sensor_G + NEBULA_STAR_QUERY.mag_buffer

        where mag_limit_sensor_G is the sensor's canonical star limit.

    n_targets_in_window : int or None
        Number of GEO targets contributing photons in this window (from
        the target-photon frames), not the number of Gaia stars. Used
        only for metadata / sanity checks.

    t_mid_utc : str or None
        Mid-time of the window in ISO-8601 UTC string form, derived from
        (start_time + end_time)/2. None if times are unavailable.

    t_mid_mjd_utc : float or None
        Mid-time of the window as MJD(UTC) float. None if time parsing
        fails or times are unavailable.

    Notes
    -----
    - In the underlying observer / target pickles, per-sample times are
      stored as timezone-aware Python datetime objects in UTC.
    - NEBULA_QUERY_GAIA converts each window's start/end times to
      astropy.time.Time, computes the midpoint, and stores:
          * t_mid_utc      as an ISO-8601 UTC string (midpoint in UTC), and
          * t_mid_mjd_utc  as a float MJD(UTC).
    - For a valid query parameter set, if t_mid_utc is not None then
      t_mid_mjd_utc should also be non-None. Downstream code can
      convert these UTC times to TT/TDB as needed.

    """

    obs_name: str
    window_index: int
    ra_center_deg: float
    dec_center_deg: float
    sky_radius_deg: float
    query_radius_deg: float
    mag_limit_G: float
    n_targets_in_window: Optional[int] = None
    t_mid_utc: Optional[str] = None
    t_mid_mjd_utc: Optional[float] = None


@dataclass
class GaiaWindowResult:
    """
    Per-window Gaia query result, compressed into numpy arrays.

    Parameters
    ----------
    window_index : int
        Index of the window in the original frames for this observer.

    t_mid_utc : str or None
        Mid-time of the observation window in ISO-8601 UTC.

    t_mid_mjd_utc : float or None
        Mid-time of the observation window as MJD(UTC).

    sky_center_ra_deg : float
        RA of the cone center in ICRS degrees.

    sky_center_dec_deg : float
        Dec of the cone center in ICRS degrees.

    sky_radius_deg : float
        Original sky footprint radius in degrees.

    query_radius_deg : float
        Catalog cone radius in degrees (sky radius + padding).

    mag_limit_G : float
        Gaia G-band limiting magnitude used for the query.

    status : {"ok", "error"}
        Query status:
        - "ok": query completed successfully and compression succeeded.
        - "error": query failed; no stars stored for this window.

    error_message : str or None
        String representation of the exception if status == "error";
        otherwise None.

    n_rows : int
        Number of Gaia sources in this cone:
        - For status == "ok": n_rows == len(gaia_source_id).
        - For status == "error": n_rows == 0.

    gaia_source_id : numpy.ndarray
        int64 array of Gaia source IDs.

    ra_deg : numpy.ndarray
        float32 array of RA in ICRS degrees.

    dec_deg : numpy.ndarray
        float32 array of Dec in ICRS degrees.

    mag_G : numpy.ndarray
        float32 array of Gaia G magnitudes.

    mag_BP : numpy.ndarray or None
        float32 array of Gaia BP magnitudes, or None if not requested
        or not available.

    mag_RP : numpy.ndarray or None
        float32 array of Gaia RP magnitudes, or None if not requested
        or not available.

    pm_ra_masyr : numpy.ndarray or None
        float32 array of proper motion in RA (Gaia convention μ_α cos δ)
        in mas/yr, or None if not requested / not available.

    pm_dec_masyr : numpy.ndarray or None
        float32 array of proper motion in Dec in mas/yr, or None.

    parallax_mas : numpy.ndarray or None
        float32 array of parallax in mas, or None if not requested /
        not available.

    ruwe : numpy.ndarray or None
        float32 array of RUWE values, or None if not requested /
        not available.

    phot_variable_flag : numpy.ndarray or None
        int8 array encoding variability classification mapped via
        VARIABLE_FLAG_MAPPING, or None if not requested / not available.

    query_meta : dict
        Dictionary of metadata about this query, e.g.:

        - "mode": "astroquery" or "local"
        - "columns": list of requested column names
        - "row_limit": max_rows used in query
        - "timestamp_utc": ISO time when query/compression ran
        - "catalog_table": underlying Gaia table name
        - "error_type": exception class name if status == "error"
        - "faintest_mag_G": faintest G in this cone (or None)
        - "star_surface_density_per_deg2": star density (if computed)

    Notes
    -----
    - Gaia NaNs are preserved as NaN in the float32 arrays. Entirely
      missing optional columns result in the corresponding field being
      None. No cleaning or imputation is performed here.

    - pm_ra_masyr is Gaia's μ_α cos δ in mas/yr. No conversion to a
      different convention is performed in this module. Any propagation
      to other epochs must respect this definition and is handled
      downstream.

    - t_mid_utc / t_mid_mjd_utc are mid-times of the observation window,
      not the Gaia reference epoch. Gaia positions/PM are referenced to
      run_meta["gaia_reference_epoch"] in the cache.
    """

    window_index: int
    t_mid_utc: Optional[str]
    t_mid_mjd_utc: Optional[float]
    sky_center_ra_deg: float
    sky_center_dec_deg: float
    sky_radius_deg: float
    query_radius_deg: float
    mag_limit_G: float
    status: str
    error_message: Optional[str]
    n_rows: int
    gaia_source_id: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag_G: np.ndarray
    mag_BP: Optional[np.ndarray]
    mag_RP: Optional[np.ndarray]
    pm_ra_masyr: Optional[np.ndarray]
    pm_dec_masyr: Optional[np.ndarray]
    parallax_mas: Optional[np.ndarray]
    ruwe: Optional[np.ndarray]
    phot_variable_flag: Optional[np.ndarray]
    query_meta: Dict[str, Any]


# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------


def load_obs_frames_with_sky(
    frames_dir: str, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Load the obs_target_frames_ranked_with_sky.pkl frames pickle.

    Parameters
    ----------
    frames_dir : str
        Directory containing the TARGET_PHOTON_FRAMES outputs, typically:

            os.path.join(NEBULA_OUTPUT_DIR, "TARGET_PHOTON_FRAMES")

    logger : logging.Logger
        Logger instance for informative messages and error reporting.

    Returns
    -------
    Dict[str, Any]
        Dictionary keyed by observer name, each containing at least
        a "windows" list with per-window dictionaries (including
        sky_center_ra_deg, sky_center_dec_deg, sky_radius_deg, etc.).

    Raises
    ------
    FileNotFoundError
        If obs_target_frames_ranked_with_sky.pkl does not exist.

    Notes
    -----
    - This is the canonical loader for NEBULA_QUERY_GAIA; downstream
      callers should not guess the filename directly.
    """
    # Build the full path to the frames pickle.
    frames_path = os.path.join(frames_dir, FRAMES_WITH_SKY_FILENAME)

    # If the file does not exist, raise a FileNotFoundError with context.
    if not os.path.exists(frames_path):
        raise FileNotFoundError(
            "Frames file not found: {}".format(frames_path)
        )

    # Log the path being loaded.
    logger.info("NEBULA_QUERY_GAIA: loading frames with sky from '%s'.", frames_path)

    # Open the frames file in binary read mode.
    with open(frames_path, "rb") as f:
        # Load the pickled dictionary.
        obs_frames = pickle.load(f)

    # Compute simple counts for logging: number of observers and windows.
    n_obs = len(obs_frames)
    n_windows_total = 0
    for obs_entry in obs_frames.values():
        # Sum the number of windows in each observer entry if present.
        n_windows_total += len(obs_entry.get("windows", []))

    # Log a summary of what was loaded.
    logger.info(
        "NEBULA_QUERY_GAIA: loaded frames for %d observers, %d windows total.",
        n_obs,
        n_windows_total,
    )

    # Return the loaded frames dictionary.
    return obs_frames


def save_gaia_cache(
    gaia_cache: Dict[str, Any],
    stars_dir: str,
    logger: logging.Logger,
) -> str:
    """
    Save the per-observer Gaia cache dictionary to disk.

    Parameters
    ----------
    gaia_cache : dict
        Dictionary keyed by observer name, with each value containing
        per-observer metadata and a "windows" list of GaiaWindowResult
        dictionaries (as produced by asdict()).

    stars_dir : str
        Directory under which the Gaia cache file will be written,
        typically:

            os.path.join(NEBULA_OUTPUT_DIR, "STARS", NEBULA_STAR_CATALOG.name)

    logger : logging.Logger
        Logger instance for informative messages and error reporting.

    Returns
    -------
    str
        Full path to the written cache file.

    Notes
    -----
    - The file is always named "obs_gaia_cones.pkl" within stars_dir.
    - The caller is expected to ensure stars_dir reflects the current
      catalog (e.g., Gaia DR3).
    """
    # Ensure the output directory exists.
    os.makedirs(stars_dir, exist_ok=True)

    # Construct the full path for the Gaia cache file.
    cache_path = os.path.join(stars_dir, GAIA_CACHE_FILENAME)

    # Log that we are writing the Gaia cache.
    logger.info("NEBULA_QUERY_GAIA: saving Gaia cache to '%s'.", cache_path)

    # Open the file in binary write mode.
    with open(cache_path, "wb") as f:
        # Pickle the gaia_cache dictionary.
        pickle.dump(gaia_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Compute simple counts for logging.
    n_obs = len(gaia_cache)
    n_windows_total = 0
    for obs_entry in gaia_cache.values():
        n_windows_total += len(obs_entry.get("windows", []))

    # Log a summary of what was written.
    logger.info(
        "NEBULA_QUERY_GAIA: saved Gaia cache for %d observers, %d queried windows.",
        n_obs,
        n_windows_total,
    )

    # Return the path to the written file.
    return cache_path


def load_gaia_cache(
    stars_dir: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Load a precomputed Gaia cache from disk.

    Parameters
    ----------
    stars_dir : str
        Directory containing "obs_gaia_cones.pkl", typically:

            os.path.join(NEBULA_OUTPUT_DIR, "STARS", NEBULA_STAR_CATALOG.name)

    logger : logging.Logger
        Logger instance for informative messages and warnings.

    Returns
    -------
    Dict[str, Any]
        gaia_cache dictionary as written by save_gaia_cache().

    Raises
    ------
    FileNotFoundError
        If the obs_gaia_cones.pkl file does not exist.

    Notes
    -----
    - This helper also warns if the cached catalog name / release does
      not match the current NEBULA_STAR_CATALOG configuration, or if the
      cache predates the current schema (no query_gaia_version field).
    """
    # Build the full path to the Gaia cache file.
    cache_path = os.path.join(stars_dir, GAIA_CACHE_FILENAME)

    # If the file is missing, raise FileNotFoundError.
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            "Gaia cache file not found: {}".format(cache_path)
        )

    # Log that we are loading the Gaia cache.
    logger.info("NEBULA_QUERY_GAIA: loading Gaia cache from '%s'.", cache_path)

    # Load the pickled gaia_cache dictionary.
    with open(cache_path, "rb") as f:
        gaia_cache = pickle.load(f)

    # If the cache is non-empty, perform consistency checks using a representative observer.
    if gaia_cache:
        # Pick an arbitrary observer key.
        any_obs_name = next(iter(gaia_cache.keys()))
        # Grab the top-level entry for that observer (metadata + windows).
        obs_entry = gaia_cache[any_obs_name]
    
        # Extract catalog name and release from the cache.
        cache_name = obs_entry.get("catalog_name")
        cache_release = obs_entry.get("release")
    
        # Compare against current NEBULA_STAR_CATALOG configuration.
        if cache_name != NEBULA_STAR_CATALOG.name:
            logger.warning(
                "NEBULA_QUERY_GAIA: loaded Gaia cache for catalog '%s', but "
                "NEBULA_STAR_CATALOG.name is '%s'.",
                cache_name,
                NEBULA_STAR_CATALOG.name,
            )
    
        if cache_release != NEBULA_STAR_CATALOG.release:
            logger.warning(
                "NEBULA_QUERY_GAIA: loaded Gaia cache for release '%s', but "
                "NEBULA_STAR_CATALOG.release is '%s'.",
                cache_release,
                NEBULA_STAR_CATALOG.release,
            )
    
        # Check that the top-level entry has the core keys we expect.
        required_header_keys = {"catalog_name", "release", "run_meta", "windows"}
        missing_keys = required_header_keys.difference(obs_entry.keys())
        if missing_keys:
            logger.warning(
                "NEBULA_QUERY_GAIA: Gaia cache entry for observer '%s' is missing "
                "expected keys: %s",
                any_obs_name,
                sorted(missing_keys),
            )
    
        # Check for schema version presence.
        run_meta = obs_entry.get("run_meta", {})
        version = run_meta.get("query_gaia_version")
        if version is None:
            logger.warning(
                "NEBULA_QUERY_GAIA: loaded Gaia cache without 'query_gaia_version'; "
                "it may predate the current NEBULA_QUERY_GAIA schema."
            )


    # Compute simple counts for logging.
    n_obs = len(gaia_cache)
    n_windows_total = 0
    for obs_entry in gaia_cache.values():
        n_windows_total += len(obs_entry.get("windows", []))

    # Log summary.
    logger.info(
        "NEBULA_QUERY_GAIA: loaded Gaia cache for %d observers, %d queried windows.",
        n_obs,
        n_windows_total,
    )

    # Return the cached dictionary.
    return gaia_cache


# -------------------------------------------------------------------------
# Helpers for building query parameters
# -------------------------------------------------------------------------


def _compute_window_mid_time(
    window: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Compute mid-time of a window as ISO-UTC string and MJD(UTC).

    Parameters
    ----------
    window : dict
        Per-window dictionary from the frames pickle. Should contain
        "start_time" and "end_time" entries that astropy.time.Time
        can parse (e.g. string, datetime, or Time instances).

    logger : logging.Logger
        Logger instance for warnings if time parsing fails.

    Returns
    -------
    t_mid_utc : str or None
        ISO-8601 UTC mid-time string, or None if unavailable.

    t_mid_mjd_utc : float or None
        Mid-time as MJD(UTC) float, or None if unavailable.

    Notes
    -----
    - If either start_time or end_time is missing or cannot be parsed,
      both return values are None and a warning is logged.
    """
    # Extract start and end times from the window dictionary.
    start = window.get("start_time")
    end = window.get("end_time")

    # If either is missing, we cannot compute a mid-time.
    if start is None or end is None:
        logger.debug(
            "NEBULA_QUERY_GAIA: window has no start_time/end_time; "
            "t_mid will be None."
        )
        return None, None

    try:
        # Construct Time objects from start and end (string, datetime, etc.).
        t_start = Time(start)
        t_end = Time(end)

        # Compute the midpoint in the same scale, then convert to UTC.
        t_mid = t_start + 0.5 * (t_end - t_start)
        t_mid_utc = t_mid.utc.isot
        t_mid_mjd_utc = float(t_mid.utc.mjd)

        # Return both representations.
        return t_mid_utc, t_mid_mjd_utc

    except Exception as exc:
        # If anything goes wrong, log a warning and return None values.
        logger.warning(
            "NEBULA_QUERY_GAIA: failed to parse mid-time for window_index=%s: %r",
            window.get("window_index", -1),
            exc,
        )
        return None, None


def build_query_params_for_window(
    obs_name: str,
    window: Dict[str, Any],
    mag_limit_sensor_G: float,
    logger: logging.Logger,
) -> GaiaWindowQueryParams:
    """
    Build GaiaWindowQueryParams for a single window.

    Parameters
    ----------
    obs_name : str
        Name of the observer satellite.

    window : dict
        Dictionary describing a single window from the frames pickle.
        Must already satisfy:
        - window["sky_selector_status"] == "ok"
        - window["n_targets"] > 0
        and contain sky footprint keys:
        - "sky_center_ra_deg"
        - "sky_center_dec_deg"
        - "sky_radius_deg"

    mag_limit_sensor_G : float
        Sensor G-band limiting magnitude (ACTIVE_SENSOR.star_mag_limit_G
        or an override), before adding NEBULA_STAR_QUERY.mag_buffer.

    logger : logging.Logger
        Logger instance for warnings and diagnostics.

    Returns
    -------
    GaiaWindowQueryParams
        Fully-populated query parameter dataclass for this window.

    Raises
    ------
    KeyError
        If required keys (e.g., sky_center_ra_deg) are missing.

    ValueError
        If any required value (e.g., sky_radius_deg) is non-finite or
        otherwise invalid.

    Notes
    -----
    - n_targets_in_window is taken from window["n_targets"] and refers to
      GEO targets contributing photons, not Gaia stars.
    """
    # Ensure required sky footprint keys are present.
    required_keys = ("sky_center_ra_deg", "sky_center_dec_deg", "sky_radius_deg")
    for key in required_keys:
        if key not in window:
            raise KeyError(
                "Window is missing required key '{}' for Gaia query params.".format(
                    key
                )
            )

    # Extract sky footprint center and radius (ICRS degrees).
    ra_center_deg = float(window["sky_center_ra_deg"])
    dec_center_deg = float(window["sky_center_dec_deg"])
    sky_radius_deg = float(window["sky_radius_deg"])

    # Validate that the sky radius is finite and positive.
    if not np.isfinite(sky_radius_deg) or sky_radius_deg <= 0.0:
        raise ValueError(
            "Invalid sky_radius_deg '{}' for obs='{}', window_index='{}'.".format(
                sky_radius_deg,
                obs_name,
                window.get("window_index", -1),
            )
        )

    # Compute the catalog cone radius: sky radius + padding.
    query_radius_deg = sky_radius_deg + float(NEBULA_STAR_QUERY.cone_padding_deg)
    
    # Sanity-check that the catalog cone radius is still positive.
    if query_radius_deg <= 0.0 or not np.isfinite(query_radius_deg):
        raise ValueError(
            "Invalid query_radius_deg '{}' for obs='{}', window_index='{}'. "
            "Check NEBULA_STAR_QUERY.cone_padding_deg and sky_radius_deg.".format(
                query_radius_deg,
                obs_name,
                window.get("window_index", -1),
            )
        )


    # Compute the Gaia query G-band limit as sensor limit + buffer.
    mag_limit_G = float(mag_limit_sensor_G) + float(NEBULA_STAR_QUERY.mag_buffer)

    # Extract number of targets in this window (not Gaia stars).
    n_targets_in_window = int(window.get("n_targets", 0))

    # Compute mid-time in both ISO-UTC string and MJD(UTC).
    t_mid_utc, t_mid_mjd_utc = _compute_window_mid_time(window, logger)

    # Extract window_index from the window dict (required for join-back).
    window_index = int(window.get("window_index", -1))

    # Build and return the GaiaWindowQueryParams dataclass.
    params = GaiaWindowQueryParams(
        obs_name=obs_name,
        window_index=window_index,
        ra_center_deg=ra_center_deg,
        dec_center_deg=dec_center_deg,
        sky_radius_deg=sky_radius_deg,
        query_radius_deg=query_radius_deg,
        mag_limit_G=mag_limit_G,
        n_targets_in_window=n_targets_in_window,
        t_mid_utc=t_mid_utc,
        t_mid_mjd_utc=t_mid_mjd_utc,
    )

    return params


# -------------------------------------------------------------------------
# Gaia query logic
# -------------------------------------------------------------------------


def run_gaia_cone_search(
    params: GaiaWindowQueryParams,
    columns: List[str],
    mode: str,
    logger: logging.Logger,
) -> Table:
    """
    Run a Gaia cone search for a single window and return an astropy Table.

    Parameters
    ----------
    params : GaiaWindowQueryParams
        Cone-search parameters for this window, including center RA/Dec,
        query radius in degrees, and mag_limit_G.

    columns : list of str
        List of Gaia column names to request from the catalog. Must
        include at least:
        - "source_id", "ra", "dec", NEBULA_STAR_CATALOG.mag_column
        plus any additional columns indicated by NEBULA_STAR_QUERY
        flags.

    mode : {"astroquery", "local"}
        Query mode:
        - "astroquery": use astroquery.gaia.Gaia.cone_search_async
        - "local": read from local Gaia shards (not implemented here)

    logger : logging.Logger
        Logger for diagnostics and warnings.

    Returns
    -------
    astropy.table.Table
        Table containing the raw Gaia results for this cone. Columns
        must include those requested in `columns`.

    Raises
    ------
    NotImplementedError
        If mode == "local" and local querying is not implemented.

    Exception
        Any lower-level exception from astroquery or local I/O will
        propagate to the caller and be handled by the safe wrapper.

    Notes
    -----
    - This function does not apply the magnitude cut; that is done
      after querying to enforce mag_limit_G at the Python level.
    - If NEBULA_STAR_QUERY.max_rows is non-zero, the TAP job's row
      limit is set accordingly and a warning is logged if len(table)
      == max_rows (indicating possible truncation).
    """
    # Ensure mode is valid.
    if mode not in ("astroquery", "local"):
        raise ValueError(
            "Invalid mode '{}'; expected 'astroquery' or 'local'.".format(mode)
        )

    # If using astroquery mode, go through Gaia TAP.
    if mode == "astroquery":
        # Build an ICRS SkyCoord for the cone center.
        coord = SkyCoord(
            ra=params.ra_center_deg * u.deg,
            dec=params.dec_center_deg * u.deg,
            frame="icrs",
        )

        # Configure Gaia row limit from NEBULA_STAR_QUERY.max_rows.
        max_rows = getattr(NEBULA_STAR_QUERY, "max_rows", None)
        if max_rows is not None and max_rows > 0:
            Gaia.ROW_LIMIT = int(max_rows)
        else:
            # None or 0 => no explicit TAP row limit.
            Gaia.ROW_LIMIT = -1

        # Log the query parameters at debug level for traceability.
        logger.debug(
            "NEBULA_QUERY_GAIA: running Gaia cone for obs='%s', window_index=%d, "
            "RA=%.6f deg, Dec=%.6f deg, radius=%.6f deg.",
            params.obs_name,
            params.window_index,
            params.ra_center_deg,
            params.dec_center_deg,
            params.query_radius_deg,
        )

        # Launch the asynchronous cone search job.
        job = Gaia.cone_search_async(
            coord,
            radius=params.query_radius_deg * u.deg,
            table_name=NEBULA_STAR_CATALOG.table,
            columns=columns,
        )


        # Retrieve the results as an astropy Table.
        table = job.get_results()

        # If we have a positive row limit and hit that limit, log a warning.
        if max_rows is not None and max_rows > 0 and len(table) == max_rows:
            logger.warning(
                "NEBULA_QUERY_GAIA: (obs='%s', window_index=%d) Gaia result hit "
                "row_limit=%d; star list may be truncated at the faint end.",
                params.obs_name,
                params.window_index,
                max_rows,
            )

        # Return the raw Table.
        return table

    # Otherwise, mode == "local" and you would implement local shards here.
    # For now, raise a clear NotImplementedError so it fails loudly.
    raise NotImplementedError(
        "Local Gaia mode ('local') is not yet implemented in NEBULA_QUERY_GAIA."
    )


def compress_gaia_table(
    params: GaiaWindowQueryParams,
    table: Table,
    columns: List[str],
    mode: str,
    logger: logging.Logger,
) -> GaiaWindowResult:
    """
    Compress a Gaia result Table into a GaiaWindowResult dataclass.

    Parameters
    ----------
    params : GaiaWindowQueryParams
        Query parameters for this window (geometry, mag limit, times).

    table : astropy.table.Table
        Gaia result table for this cone, containing at least:
        - "source_id", "ra", "dec", NEBULA_STAR_CATALOG.mag_column,
        plus any additional columns requested.

    columns : list of str
        Column names that were requested from Gaia. Stored into
        query_meta["columns"] for reproducibility.

    mode : {"astroquery", "local"}
        Mode used for the query (stored in query_meta["mode"]).

    logger : logging.Logger
        Logger instance for one-time warnings on missing optional
        columns.

    Returns
    -------
    GaiaWindowResult
        Dataclass with compressed arrays and metadata.

    Notes
    -----
    - All floating-point arrays are stored as float32 for memory
      efficiency (IDs as int64, variability flags as int8).
    - Optional columns (BP/RP, parallax, RUWE, variability) are only
      read if the corresponding NEBULA_STAR_QUERY.include_* flag is
      True. If the column is missing from the table, the field is set
      to None and a one-time warning is emitted.
    """
    # Number of rows in the raw Gaia table.
    n_rows = len(table)

    # Initialize query_meta dictionary with basic information.
    query_meta = {
        "mode": mode,
        "columns": list(columns),
        "row_limit": getattr(NEBULA_STAR_QUERY, "max_rows", None),
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "catalog_table": NEBULA_STAR_CATALOG.table,
    }

    # If there are no rows, create empty arrays of the correct dtypes.
    if n_rows == 0:
        gaia_source_id = np.array([], dtype=np.int64)
        ra_deg = np.array([], dtype=np.float32)
        dec_deg = np.array([], dtype=np.float32)
        mag_G = np.array([], dtype=np.float32)
        mag_BP = None
        mag_RP = None
        pm_ra_masyr = None
        pm_dec_masyr = None
        parallax_mas = None
        ruwe = None
        phot_variable_flag = None

        # Faintest magnitude is undefined when there are no rows.
        query_meta["faintest_mag_G"] = None
    else:
        # ------------------------------------------------------------------
        # Apply a magnitude cut at the *query* limit before compressing.
        #
        # This keeps only stars with G <= params.mag_limit_G, where
        #   params.mag_limit_G = mag_limit_sensor_G + NEBULA_STAR_QUERY.mag_buffer
        #
        # We still query potentially deeper from Gaia, but we do not store
        # stars that are fainter than the configured limit in the cache.
        # ------------------------------------------------------------------

        # Extract the configured G-band magnitude column.
        g_col = NEBULA_STAR_CATALOG.mag_column
        g_all = np.array(table[g_col], dtype=np.float32)

        # Build a mask for finite mags brighter than or equal to the limit.
        mag_limit = float(params.mag_limit_G)
        finite_mask = np.isfinite(g_all)
        bright_mask = g_all <= mag_limit
        keep_mask = finite_mask & bright_mask

        # Apply the mask to the table so all following columns are consistent.
        table = table[keep_mask]
        mag_G = g_all[keep_mask]

        # Update row count after filtering.
        n_rows = len(table)

        # If the cut removed everything, we still want consistent empty arrays.
        if n_rows == 0:
            gaia_source_id = np.array([], dtype=np.int64)
            ra_deg = np.array([], dtype=np.float32)
            dec_deg = np.array([], dtype=np.float32)
            mag_G = np.array([], dtype=np.float32)
            mag_BP = None
            mag_RP = None
            pm_ra_masyr = None
            pm_dec_masyr = None
            parallax_mas = None
            ruwe = None
            phot_variable_flag = None
            query_meta["faintest_mag_G"] = None
        else:
            # Extract core required columns from the filtered table.
            gaia_source_id = np.array(table["source_id"], dtype=np.int64)
            ra_deg = np.array(table["ra"], dtype=np.float32)
            dec_deg = np.array(table["dec"], dtype=np.float32)

            # Compute the faintest (largest) stored G magnitude, ignoring NaNs.
            if np.any(np.isfinite(mag_G)):
                query_meta["faintest_mag_G"] = float(np.nanmax(mag_G))
            else:
                query_meta["faintest_mag_G"] = None

            # Initialize optional arrays to None; fill conditionally below.
            mag_BP = None
            mag_RP = None
            pm_ra_masyr = None
            pm_dec_masyr = None
            parallax_mas = None
            ruwe = None
            phot_variable_flag = None

        # Convenience function to log missing optional columns only once.
        def _warn_missing(col_name):
            # Use the module-level set to avoid repeated warnings.
            if col_name not in _missing_column_warnings:
                logger.warning(
                    "NEBULA_QUERY_GAIA: optional Gaia column '%s' not present "
                    "in table; corresponding field will be None.",
                    col_name,
                )
                _missing_column_warnings.add(col_name)

        # Optionally extract BP/RP magnitudes if enabled.
        if getattr(NEBULA_STAR_QUERY, "include_bp_rp", False):
            bp_col = getattr(NEBULA_STAR_CATALOG, "bp_mag_column", None)
            rp_col = getattr(NEBULA_STAR_CATALOG, "rp_mag_column", None)

            if bp_col is not None and bp_col in table.colnames:
                mag_BP = np.array(table[bp_col], dtype=np.float32)
            else:
                mag_BP = None
                if bp_col is not None:
                    _warn_missing(bp_col)

            if rp_col is not None and rp_col in table.colnames:
                mag_RP = np.array(table[rp_col], dtype=np.float32)
            else:
                mag_RP = None
                if rp_col is not None:
                    _warn_missing(rp_col)

        # Optionally extract proper motions if enabled.
        if getattr(NEBULA_STAR_QUERY, "use_proper_motion", False):
            if "pmra" in table.colnames:
                pm_ra_masyr = np.array(table["pmra"], dtype=np.float32)
            else:
                pm_ra_masyr = None
                _warn_missing("pmra")

            if "pmdec" in table.colnames:
                pm_dec_masyr = np.array(table["pmdec"], dtype=np.float32)
            else:
                pm_dec_masyr = None
                _warn_missing("pmdec")

        # Optionally extract parallax if enabled.
        if getattr(NEBULA_STAR_QUERY, "include_parallax", False):
            if "parallax" in table.colnames:
                parallax_mas = np.array(table["parallax"], dtype=np.float32)
            else:
                parallax_mas = None
                _warn_missing("parallax")

        # Optionally extract RUWE if enabled.
        if getattr(NEBULA_STAR_QUERY, "include_ruwe", False):
            if "ruwe" in table.colnames:
                ruwe = np.array(table["ruwe"], dtype=np.float32)
            else:
                ruwe = None
                _warn_missing("ruwe")

        # Optionally extract variability flag if enabled.
        if getattr(NEBULA_STAR_QUERY, "include_variability_flag", False):
            if "phot_variable_flag" in table.colnames:
                # Map string flags to int8 via VARIABLE_FLAG_MAPPING.
                raw_flags = np.array(table["phot_variable_flag"], dtype=object)
                mapped = np.empty(raw_flags.shape, dtype=np.int8)
                for i, val in enumerate(raw_flags):
                    # Default to -1 if value not in mapping.
                    mapped[i] = VARIABLE_FLAG_MAPPING.get(val, -1)
                phot_variable_flag = mapped
            else:
                phot_variable_flag = None
                _warn_missing("phot_variable_flag")

    # Build and return the GaiaWindowResult.
    result = GaiaWindowResult(
        window_index=params.window_index,
        t_mid_utc=params.t_mid_utc,
        t_mid_mjd_utc=params.t_mid_mjd_utc,
        sky_center_ra_deg=params.ra_center_deg,
        sky_center_dec_deg=params.dec_center_deg,
        sky_radius_deg=params.sky_radius_deg,
        query_radius_deg=params.query_radius_deg,
        mag_limit_G=params.mag_limit_G,
        status="ok",
        error_message=None,
        n_rows=n_rows,
        gaia_source_id=gaia_source_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        mag_G=mag_G,
        mag_BP=mag_BP,
        mag_RP=mag_RP,
        pm_ra_masyr=pm_ra_masyr,
        pm_dec_masyr=pm_dec_masyr,
        parallax_mas=parallax_mas,
        ruwe=ruwe,
        phot_variable_flag=phot_variable_flag,
        query_meta=query_meta,
    )

    return result


def safe_gaia_cone_search_for_window(
    params: GaiaWindowQueryParams,
    columns: List[str],
    mode: str,
    logger: logging.Logger,
) -> GaiaWindowResult:
    """
    Run a Gaia cone search with error handling for a single window.

    Parameters
    ----------
    params : GaiaWindowQueryParams
        Cone-search parameters for this window.

    columns : list of str
        Column names to request from Gaia.

    mode : {"astroquery", "local"}
        Mode used for the query. Passed through to run_gaia_cone_search
        and stored in query_meta["mode"].

    logger : logging.Logger
        Logger for warnings and diagnostics.

    Returns
    -------
    GaiaWindowResult
        Compressed result for this window, with:
        - status = "ok" on success, or
        - status = "error" with empty arrays on failure.

    Notes
    -----
    - This function is responsible for catching all exceptions from
      run_gaia_cone_search and producing a GaiaWindowResult with
      status="error" and descriptive error_message.
    """
    try:
        # Run the actual Gaia cone search (may raise).
        table = run_gaia_cone_search(params, columns, mode, logger)

        # Compress the table into a GaiaWindowResult with status="ok".
        result = compress_gaia_table(params, table, columns, mode, logger)

        # Return the successful result.
        return result

    except Exception as exc:
        # On any exception, log a warning with context.
        logger.warning(
            "NEBULA_QUERY_GAIA: Gaia query failed for obs='%s', window_index=%d: %r",
            params.obs_name,
            params.window_index,
            exc,
        )

        # Build empty arrays for the error case.
        gaia_source_id = np.array([], dtype=np.int64)
        ra_deg = np.array([], dtype=np.float32)
        dec_deg = np.array([], dtype=np.float32)
        mag_G = np.array([], dtype=np.float32)

        query_meta = {
            "mode": mode,  # "astroquery" or "local"
            "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
            "columns": list(columns),
            "row_limit": getattr(NEBULA_STAR_QUERY, "max_rows", None),
            "catalog_table": NEBULA_STAR_CATALOG.table,
            "cone_center_ra_deg": params.ra_center_deg,
            "cone_center_dec_deg": params.dec_center_deg,
            "cone_radius_deg": params.query_radius_deg,
            "mag_limit_G": params.mag_limit_G,
            # Identify the exception type so build_gaia_cones_for_all_windows()
            # can populate error_type_counts meaningfully.
            "error_type": type(exc).__name__,
            # Optionally:
            # "adql": "<ADQL string>" if you add that later
        }



        # Return a GaiaWindowResult with status="error".
        error_result = GaiaWindowResult(
            window_index=params.window_index,
            t_mid_utc=params.t_mid_utc,
            t_mid_mjd_utc=params.t_mid_mjd_utc,
            sky_center_ra_deg=params.ra_center_deg,
            sky_center_dec_deg=params.dec_center_deg,
            sky_radius_deg=params.sky_radius_deg,
            query_radius_deg=params.query_radius_deg,
            mag_limit_G=params.mag_limit_G,
            status="error",
            error_message=repr(exc),
            n_rows=0,
            gaia_source_id=gaia_source_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            mag_G=mag_G,
            mag_BP=None,
            mag_RP=None,
            pm_ra_masyr=None,
            pm_dec_masyr=None,
            parallax_mas=None,
            ruwe=None,
            phot_variable_flag=None,
            query_meta=query_meta,
        )

        return error_result


# -------------------------------------------------------------------------
# Main builder over all observers and windows
# -------------------------------------------------------------------------


def build_gaia_cones_for_all_windows(
    obs_target_frames_with_sky: Dict[str, Any],
    mag_limit_sensor_G: float,
    columns: List[str],
    mode: str,
    logger: logging.Logger,
    max_windows_per_obs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build Gaia cone queries and cache for all observers and eligible windows.

    Parameters
    ----------
    obs_target_frames_with_sky : dict
        Dictionary keyed by observer name, as loaded from
        obs_target_frames_ranked_with_sky.pkl. Each value should
        contain a "windows" list of per-window dictionaries.

    mag_limit_sensor_G : float
        Sensor G-band limiting magnitude used as the base star limit
        for all observers in this run.

    columns : list of str
        Column names to request from Gaia. Must satisfy the required
        column contract enforced in main().

    mode : {"astroquery", "local"}
        Query mode used for all observers/windows in this run.

    logger : logging.Logger
        Logger instance for informational logging and error messages.

    max_windows_per_obs : int or None, optional
        If not None, limits the number of *original* windows inspected
        per observer (including skipped ones). Useful for debugging.

    Returns
    -------
    Dict[str, Any]
        gaia_cache dictionary keyed by observer name, with each entry
        containing:
        - top-level catalog / mode info
        - "run_meta" summary (counts, densities, etc.)
        - "windows" list of GaiaWindowResult dictionaries.

    Notes
    -----
    - Windows with sky_selector_status != "ok" or n_targets <= 0 are
      *not* stored in the "windows" list; they are accounted for in
      run_meta["window_counts"].
    - No deduplication of Gaia sources is performed across windows;
      a given Gaia source may appear in multiple window results.
    """
    # Initialize the output Gaia cache dictionary.
    gaia_cache = {}

    # Iterate over all observers in the frames dictionary.
    for obs_name, obs_entry in obs_target_frames_with_sky.items():
        # Extract the list of windows for this observer (default empty).
        windows = obs_entry.get("windows", [])

        # Determine how many original windows we will inspect.
        if max_windows_per_obs is not None:
            n_original_windows = min(len(windows), int(max_windows_per_obs))
        else:
            n_original_windows = len(windows)

        # Initialize counters for skip reasons and status.
        n_queried = 0
        n_skipped_zero_targets = 0
        n_skipped_bad_sky = 0
        n_skipped_broken = 0
        n_ok = 0
        n_error = 0
        total_stars = 0

        # Initialize error-type histogram.
        error_type_counts = {}  # type: Dict[str, int]

        # Collect per-window GaiaWindowResult dictionaries.
        window_results = []

        # Collect per-window star densities for this observer.
        densities = []

        # Track time range across all windows (if start/end times present).
        min_start_time = None  # type: Optional[Time]
        max_end_time = None  # type: Optional[Time]

        # Log at INFO that we are processing this observer.
        logger.info(
            "NEBULA_QUERY_GAIA: processing observer '%s' with %d windows "
            "(capped at %d for this run).",
            obs_name,
            len(windows),
            n_original_windows,
        )

        # Loop over windows up to n_original_windows.
        for window in tqdm(
            windows[:n_original_windows],
            total=n_original_windows,
            desc=f"Gaia cones [{obs_name}]",
        ):
            # Extract sky selector status and number of targets.
            status = window.get("sky_selector_status", None)
            n_targets = int(window.get("n_targets", 0))

            # If sky selector status is not "ok" or required keys are missing,
            # increment skipped_bad_sky and continue.
            if status != "ok" or not all(
                k in window
                for k in ("sky_center_ra_deg", "sky_center_dec_deg", "sky_radius_deg")
            ):
                n_skipped_bad_sky += 1
                continue

            # If there are zero targets, we do not query Gaia for this window.
            if n_targets <= 0:
                n_skipped_zero_targets += 1
                continue

            # Try to build query parameters; malformed windows count as skipped_broken.
            try:
                params = build_query_params_for_window(
                    obs_name=obs_name,
                    window=window,
                    mag_limit_sensor_G=mag_limit_sensor_G,
                    logger=logger,
                )
            except (KeyError, ValueError) as exc:
                logger.error(
                    "NEBULA_QUERY_GAIA: malformed window for obs='%s', "
                    "window_index=%s: %r",
                    obs_name,
                    window.get("window_index", -1),
                    exc,
                )
                n_skipped_broken += 1
                continue


            # Update global time range using start_time/end_time if present.
            start_time = window.get("start_time")
            end_time = window.get("end_time")
            if start_time is not None:
                try:
                    t_start = Time(start_time)
                    if min_start_time is None or t_start < min_start_time:
                        min_start_time = t_start
                except Exception:
                    # Ignore parse errors for global range; already warned inside helper.
                    pass
            if end_time is not None:
                try:
                    t_end = Time(end_time)
                    if max_end_time is None or t_end > max_end_time:
                        max_end_time = t_end
                except Exception:
                    pass

            # We are about to perform a Gaia query for this window.
            n_queried += 1

            # Run the safe Gaia cone search and get a GaiaWindowResult.
            result = safe_gaia_cone_search_for_window(
                params=params,
                columns=columns,
                mode=mode,
                logger=logger,
            )

            # Update counts based on result status.
            if result.status == "ok":
                n_ok += 1
                total_stars += result.n_rows
            
                # Compute star surface density if area is positive.
                # Use the spherical-cap solid angle for radius theta (in deg),
                # then convert steradians to deg^2.
                theta_deg = float(result.query_radius_deg)
                if theta_deg > 0.0:
                    theta_rad = np.deg2rad(theta_deg)
                    omega_sr = 2.0 * np.pi * (1.0 - np.cos(theta_rad))
                    area_deg2 = omega_sr * (180.0 / np.pi) ** 2
                    density = result.n_rows / area_deg2
                    result.query_meta["star_surface_density_per_deg2"] = density
                    densities.append(density)

            else:
                n_error += 1
                # Record the error type into the histogram.
                error_type = result.query_meta.get("error_type", "UnknownError")
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

            # Append the dataclass as a plain dictionary using asdict().
            window_results.append(asdict(result))

        # Compute star density summary statistics for this observer.
        if densities:
            median_density = float(np.median(densities))
            p10_density = float(np.percentile(densities, 10))
            p90_density = float(np.percentile(densities, 90))
        else:
            median_density = float("nan")
            p10_density = float("nan")
            p90_density = float("nan")

        # Compute frames time range ISO strings, if we successfully parsed any times.
        if min_start_time is not None and max_end_time is not None:
            frames_time_range_utc = (
                min_start_time.utc.isot,
                max_end_time.utc.isot,
            )
        else:
            frames_time_range_utc = (None, None)

        # Build the per-observer cache entry.
        obs_cache = {
            "observer_name": obs_name,
            "mode": mode,
            "catalog_name": NEBULA_STAR_CATALOG.name,
            "catalog_table": NEBULA_STAR_CATALOG.table,
            "band": NEBULA_STAR_CATALOG.band,
            "release": NEBULA_STAR_CATALOG.release,
            "query_config": asdict(NEBULA_STAR_QUERY),
            "mag_limit_sensor_G": float(mag_limit_sensor_G),
            "run_meta": {
                "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                "query_gaia_version": "v1.0.0",
                "source_frames_file": FRAMES_WITH_SKY_FILENAME,
                "frames_time_range_utc": frames_time_range_utc,
                # Number of windows we actually inspected in this run
                "original_windows": n_original_windows,
                # Total windows present in the frames pickle (before any cap)
                "total_windows_in_frames": len(windows),
                "sky_selector_version": "<set_in_NEBULA_config_or_logs>",
                "gaia_reference_epoch": NEBULA_STAR_CATALOG.reference_epoch,
                "query_mag_limit_G": float(mag_limit_sensor_G)
                + float(NEBULA_STAR_QUERY.mag_buffer),
                "columns_requested": list(columns),
                "window_counts": {
                    "queried": n_queried,
                    "skipped_zero_targets": n_skipped_zero_targets,
                    "skipped_bad_sky": n_skipped_bad_sky,
                    "skipped_broken": n_skipped_broken,
                    "ok": n_ok,
                    "error": n_error,
                },
                "total_stars_ok_windows": total_stars,
                "error_type_counts": error_type_counts,
                "star_density_stats": {
                    "median_per_deg2": median_density,
                    "p10_per_deg2": p10_density,
                    "p90_per_deg2": p90_density,
                },
            },
            "windows": window_results,
        }

        # Attach this observer's cache entry to the overall gaia_cache.
        gaia_cache[obs_name] = obs_cache

    # Return the full Gaia cache dictionary.
    return gaia_cache


# -------------------------------------------------------------------------
# main() entry point
# -------------------------------------------------------------------------


def main(
    mag_limit_sensor_G: Optional[float] = None,
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Entry point for NEBULA_QUERY_GAIA.

    Parameters
    ----------
    mag_limit_sensor_G : float or None, optional
        Sensor G-band limiting magnitude to use for all observers in
        this run. If None, the function attempts to use
        ACTIVE_SENSOR.star_mag_limit_G. If that attribute is missing,
        a RuntimeError is raised.

    mode : {"astroquery", "local"} or None, optional
        Query mode. If None, NEBULA_STAR_QUERY.mode is used. If
        provided, it overrides the config for this run and is stored
        in the cache.

    logger : logging.Logger or None, optional
        Logger to use. If None, a default logger from get_logger()
        is created.

    Returns
    -------
    Dict[str, Any]
        gaia_cache dictionary keyed by observer name, which is also
        written to disk via save_gaia_cache(). Callers should treat
        this dictionary as read-only (it reflects what was written).

    Raises
    ------
    RuntimeError
        If mag_limit_sensor_G is None and ACTIVE_SENSOR.star_mag_limit_G
        is not defined.

    ValueError
        If mode is not None and is not one of {"astroquery", "local"}.

    Notes
    -----
    - In this version, all observers share the same Gaia G-band star
      limit mag_limit_sensor_G. Any observer-level differences in optics
      or exposure are handled by downstream radiometry, not by this
      catalog query module.
    """
    # Initialize logger if needed.
    if logger is None:
        logger = get_logger("NEBULA_QUERY_GAIA")

    # Resolve sensor magnitude limit for Gaia G-band stars.
    #
    # Priority:
    #   1) Explicit argument mag_limit_sensor_G (most specific, per-run override)
    #   2) ACTIVE_SENSOR.star_mag_limit_G  (if you define it in the sensor config)
    #   3) ACTIVE_SENSOR.mag_limit        (generic limiting mag from SENSOR_CONFIG)
    #
    # This avoids duplicating the calibrated number: right now your
    # GEN3_VGA_CD sensor already has mag_limit=9.8, so Gaia will use
    # that unless you later introduce a more specific star_mag_limit_G.
    if mag_limit_sensor_G is None:
        # Most specific: a dedicated GAIA G-band star limit on the sensor
        star_limit = getattr(ACTIVE_SENSOR, "star_mag_limit_G", None)

        if star_limit is not None:
            mag_limit_sensor_G = float(star_limit)

        else:
            # Fall back to the generic limiting magnitude on the sensor
            generic_limit = getattr(ACTIVE_SENSOR, "mag_limit", None)
            if generic_limit is not None:
                mag_limit_sensor_G = float(generic_limit)
            else:
                raise RuntimeError(
                    "NEBULA_QUERY_GAIA: no magnitude limit available. "
                    "Please define ACTIVE_SENSOR.star_mag_limit_G or "
                    "ACTIVE_SENSOR.mag_limit in NEBULA_SENSOR_CONFIG, "
                    "or pass mag_limit_sensor_G to main()."
                )


    # Log the sensor mag limit and buffer.
    logger.info(
        "NEBULA_QUERY_GAIA: sensor G-band limit = %.3f, mag_buffer = %.3f.",
        mag_limit_sensor_G,
        float(NEBULA_STAR_QUERY.mag_buffer),
    )

    # Resolve mode (astroquery / local).
    effective_mode = mode or getattr(NEBULA_STAR_QUERY, "mode", "astroquery")
    
    # Validate whichever mode we will actually use.
    if effective_mode not in ("astroquery", "local"):
        raise ValueError(
            "Invalid mode '{}'; expected 'astroquery' or 'local'.".format(effective_mode)
        )
    
    # If overriding config mode, log that fact.
    if mode is not None and mode != getattr(NEBULA_STAR_QUERY, "mode", "<unset>"):
        logger.info(
            "NEBULA_QUERY_GAIA: overriding NEBULA_STAR_QUERY.mode='%s' with mode='%s' "
            "for this run.",
            getattr(NEBULA_STAR_QUERY, "mode", "<unset>"),
            mode,
        )


    # Obtain the list of Gaia columns from config.
    columns = get_gaia_query_columns(
        use_proper_motion=getattr(NEBULA_STAR_QUERY, "use_proper_motion", False)
    )

    # Enforce required column contract (source_id, ra, dec, G).
    required = {
        "source_id",
        "ra",
        "dec",
        NEBULA_STAR_CATALOG.mag_column,
    }
    if getattr(NEBULA_STAR_QUERY, "use_proper_motion", False):
        required |= {"pmra", "pmdec"}

    missing = required.difference(columns)
    if missing:
        raise RuntimeError(
            "NEBULA_STAR_CONFIG / get_gaia_query_columns is inconsistent with "
            "NEBULA_QUERY_GAIA expectations. Missing columns: {}".format(
                sorted(missing)
            )
        )

    # Compute Gaia query magnitude limit for logging (sensor + buffer).
    query_mag_limit_G = float(mag_limit_sensor_G) + float(NEBULA_STAR_QUERY.mag_buffer)
    logger.info(
        "NEBULA_QUERY_GAIA: Gaia query limit (G) = %.3f (sensor limit + buffer).",
        query_mag_limit_G,
    )
    logger.info(
        "NEBULA_QUERY_GAIA: using same G-band star limit for all observers."
    )

    # Determine frames directory and load frames with sky.
    frames_dir = os.path.join(NEBULA_OUTPUT_DIR, "TARGET_PHOTON_FRAMES")
    obs_frames = load_obs_frames_with_sky(frames_dir=frames_dir, logger=logger)

    # Build Gaia cones and cache for all observers and eligible windows.
    gaia_cache = build_gaia_cones_for_all_windows(
        obs_target_frames_with_sky=obs_frames,
        mag_limit_sensor_G=mag_limit_sensor_G,
        columns=columns,
        mode=effective_mode,
        logger=logger,
        max_windows_per_obs=None,  # full run by default
    )

    # Determine STARS directory and save the Gaia cache.
    stars_dir = os.path.join(NEBULA_OUTPUT_DIR, "STARS", NEBULA_STAR_CATALOG.name)
    save_gaia_cache(gaia_cache=gaia_cache, stars_dir=stars_dir, logger=logger)

    # Log per-observer summary.
    for obs_name, obs_entry in gaia_cache.items():
        run_meta = obs_entry.get("run_meta", {})
        counts = run_meta.get("window_counts", {})
        density_stats = run_meta.get("star_density_stats", {})
        logger.info(
            "NEBULA_QUERY_GAIA: observer '%s' -> "
            "windows: %d queried, %d skipped_zero_targets, %d skipped_bad_sky, %d skipped_broken; "
            "status: %d ok, %d error; "
            "total Gaia stars (ok windows): %d; "
            "median star density: %.3f stars/deg^2 (p10=%.3f, p90=%.3f).",
            obs_name,
            counts.get("queried", 0),
            counts.get("skipped_zero_targets", 0),
            counts.get("skipped_bad_sky", 0),
            counts.get("skipped_broken", 0),
            counts.get("ok", 0),
            counts.get("error", 0),
            run_meta.get("total_stars_ok_windows", 0),
            density_stats.get("median_per_deg2", float("nan")),
            density_stats.get("p10_per_deg2", float("nan")),
            density_stats.get("p90_per_deg2", float("nan")),
        )

    # Return the Gaia cache (same object that was written to disk).
    return gaia_cache


# If you want this module to be runnable as a script:
if __name__ == "__main__":
    # Run main() with default arguments when invoked as a script.
    main()
