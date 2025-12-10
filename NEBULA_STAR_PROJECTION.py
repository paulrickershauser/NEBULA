"""
NEBULA_STAR_PROJECTION
======================

This module implements the *star projection* stage for the NEBULA pipeline.

High-level role
---------------
Given:

    1) Windowed target frames with sky footprints
       (obs_target_frames_ranked_with_sky.pkl),

    2) Gaia cone queries per observer + window (obs_gaia_cones.pkl),

    3) Observer tracks with pointing (observer_tracks_with_pointing.pkl),

this module:

    * Propagates Gaia star positions from the catalog reference epoch
      to each window's mid-time using astrometric proper motion (when enabled).

    * Builds / selects a NebulaWCS instance representing the sensor pointing
      at each window's mid-time, using the *same WCS semantics* as for targets.

    * Projects Gaia RA/Dec at mid-time into sensor pixel coordinates
      (x_pix_mid, y_pix_mid).

    * Applies a simple on-detector mask to determine which stars fall
      inside the active sensor rows × cols at mid-time.

    * Aggregates per-window star metadata into a new pickle:

          obs_star_projections.pkl

      keyed by observer name, with one "StarWindowProjection" dict per window.

Important constraints
---------------------
* This module is **read-only** with respect to upstream NEBULA stages:

    - It does **not** re-query Gaia.
    - It does **not** recompute LOS, illumination, flux, or pointing.
    - It does **not** rebuild target photon frames.

* It only consumes existing pickles and writes a new one.

* It uses the **same WCS + projection stack** as your target projection
  (NEBULA_WCS / NEBULA_SENSOR_PROJECTION) to ensure stars and targets
  align on the detector.

* Stars are treated as **mid-window only**:

    - One propagated RA/Dec per star per window: ra_deg_mid, dec_deg_mid.
    - One projected pixel position per star per window: x_pix_mid, y_pix_mid.
    - A boolean on_detector flag per star.

* WCS selection per window uses a **single snapshot**:

    - For each observer, build WCS either as a single NebulaWCS (static pointing)
      or a sequence aligned with observer_track["t_mjd_utc"].
    - For each window, select the WCS whose t_mjd_utc is *closest* to
      window["t_mid_mjd_utc"].

Outputs and schema
------------------
This module writes a pickle, typically at:

    NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl

with the structure:

    obs_star_projections[obs_name] = {
        "observer_name": str,
        "rows": int,
        "cols": int,
        "catalog_name": str,
        "catalog_band": str,
        "mag_limit_sensor_G": float,
        "run_meta": {...},
        "windows": [StarWindowProjection, ...],
    }

Where each StarWindowProjection is a dict with:

    * window metadata (indices, times, sky center / radius, etc.)
    * Gaia cone status per window (gaia_status, gaia_error_message)
    * counts (n_stars_input, n_stars_on_detector)
    * density estimate: n_stars_on_detector / (π * query_radius_deg²)
    * a "stars" dict keyed by Gaia source_id (as string), with:

          {
              "gaia_source_id": int,
              "source_id": str,         # string alias of gaia_source_id
              "source_type": "star",
              "ra_deg_catalog": float,
              "dec_deg_catalog": float,
              "ra_deg_mid": float,
              "dec_deg_mid": float,
              "mag_G": float,
              "x_pix_mid": float,
              "y_pix_mid": float,
              "on_detector": bool,
              # optional: pm_ra_masyr, pm_dec_masyr, parallax_mas,
              #           radial_velocity_km_s
          }

See individual function docstrings for more detail.

NOTE ABOUT IMPORT PATHS
-----------------------
Some imports in this module are **best-effort guesses** based on the NEBULA
logging and your written spec. In particular:

    * NEBULA_TARGET_PHOTONS loader function for frames_with_sky
    * NEBULA_QUERY_GAIA loader function for Gaia cones
    * Pointing / observer_tracks_with_pointing loader
    * NEBULA_WCS / build_wcs_for_observer / project_radec_to_pixels
    * NEBULA_PATH_CONFIG attributes for default paths

These are wrapped in try/except and checked at runtime. If a required
loader or helper is missing, the code raises a RuntimeError with a clear
message indicating what needs to be wired to the actual function in your
codebase.

You should only need to fix those import/loader wiring points; the core
projection logic and schema are fully implemented.
"""

# Typing utilities for type hints and generic container types.
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Standard library modules for logging, file paths, pickling, and timestamps.
import logging
import os
import pickle
import datetime

# Numerical arrays for vectorized operations.
import numpy as np

# Astropy time and coordinates for proper-motion propagation.
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord

# NEBULA configuration: base output directory for all products.
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

# NEBULA sensor configuration: sensor model and active sensor selection.
from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig, ACTIVE_SENSOR

# NEBULA star configuration: Gaia catalog metadata and query policy.
from Configuration.NEBULA_STAR_CONFIG import NEBULA_STAR_CATALOG, NEBULA_STAR_QUERY

# WCS helpers for building per-observer WCS series and projecting RA/Dec to pixels.
from Utility.SENSOR.NEBULA_WCS import (
    NebulaWCS,
    build_wcs_for_observer,
    project_radec_to_pixels,
)

# Optional support for raw Astropy WCS objects in isinstance checks.
from astropy.wcs import WCS

# Gaia cache loader and metadata from the NEBULA_QUERY_GAIA module.
from Utility.STARS import NEBULA_QUERY_GAIA

# Pixel pickler, used for locating observer tracks with pointing + pixel geometry.
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER


# Run-meta version string for this star projection stage.
STAR_PROJECTION_RUN_META_VERSION: str = "0.1"

# Alias for the per-window star projection dictionary type.
StarWindowProjection = Dict[str, Any]



def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger instance for this module.

    Parameters
    ----------
    logger : logging.Logger or None, optional
        If provided, this logger instance will be returned unchanged.
        If None, this function returns ``logging.getLogger(__name__)``
        without modifying handlers or levels.

    Returns
    -------
    logging.Logger
        The logger to use inside this module.
    """
    # If the caller supplied a logger, simply return it.
    if logger is not None:
        return logger

    # Otherwise, obtain a module-level logger using the standard pattern.
    return logging.getLogger(__name__)

def _resolve_default_frames_path() -> str:
    """
    Resolve the default path to ``obs_target_frames_ranked_with_sky.pkl``.

    This helper assumes that the NEBULA pipeline has been run via
    ``sim_test.py`` with both ``BUILD_TARGET_PHOTON_FRAMES`` and
    ``RUN_GAIA_PIPELINE`` (and thus ``NEBULA_SKY_SELECTOR``) enabled.

    In that workflow, ``NEBULA_TARGET_PHOTONS`` writes
    ``obs_target_frames_ranked.pkl`` under::

        NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES"

    and ``NEBULA_SKY_SELECTOR`` reads that file, attaches sky footprints,
    and writes::

        NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES" /
            "obs_target_frames_ranked_with_sky.pkl"

    This function simply reconstructs that default path using the
    configured ``NEBULA_OUTPUT_DIR`` from ``Configuration.NEBULA_PATH_CONFIG``.

    Returns
    -------
    str
        Absolute path to the frames-with-sky pickle
        (``obs_target_frames_ranked_with_sky.pkl``).
    """
    # Construct a default path under the TARGET_PHOTON_FRAMES subdirectory,
    # matching the layout used by NEBULA_TARGET_PHOTONS and NEBULA_SKY_SELECTOR.
    default_path = os.path.join(
        NEBULA_OUTPUT_DIR,
        "TARGET_PHOTON_FRAMES",
        "obs_target_frames_ranked_with_sky.pkl",
    )
    return default_path

def _resolve_default_obs_tracks_path() -> str:
    """
    Resolve the default path to the observer tracks pickle used for WCS.

    This helper assumes that the NEBULA pixel pipeline has been run via
    ``NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs`` (for example,
    through ``sim_test.py``). In that workflow, the pixel pickler writes
    the observer tracks with pointing and pixel geometry to::

        NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH

    where ``OBS_PIXEL_PICKLE_PATH`` is defined in
    :mod:`Utility.SAT_OBJECTS.NEBULA_PIXEL_PICKLER` as::

        NEBULA_OUTPUT_DIR / "PIXEL_SatPickles" /
            "observer_tracks_with_pixels.pkl"

    STAR projection only needs the observer time series and pointing
    information; reusing the pixel-augmented observer pickle ensures that
    the WCS construction is consistent with the rest of the pixel
    pipeline.

    Returns
    -------
    str
        Absolute path to the observer tracks pickle
        (``observer_tracks_with_pixels.pkl``).
    """
    # Use the same path that NEBULA_PIXEL_PICKLER uses when it writes
    # observer tracks with pixel geometry. This guarantees that
    # NEBULA_STAR_PROJECTION is aligned with the upstream pixel pipeline.
    return NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH

def _resolve_default_output_path() -> str:
    """
    Resolve the default path for writing 'obs_star_projections.pkl'.

    This helper uses the global NEBULA_OUTPUT_DIR constant and
    NEBULA_STAR_CATALOG to determine a default location:

        <NEBULA_OUTPUT_DIR>/STARS/<catalog_name>/obs_star_projections.pkl
    """
    # Determine catalog name (e.g., "GaiaDR3_G").
    catalog_name = getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")

    # Build a default path under STARS/<catalog_name>.
    default_path = os.path.join(
        NEBULA_OUTPUT_DIR,
        "STARS",
        catalog_name,
        "obs_star_projections.pkl",
    )

    return default_path

def _load_frames_with_sky_from_disk(
    frames_path: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Load ``obs_target_frames_ranked_with_sky.pkl`` from disk.

    This helper is used after the NEBULA pipeline has been run via
    ``sim_test.py`` with::

        BUILD_TARGET_PHOTON_FRAMES = True
        RUN_GAIA_PIPELINE = True

    In that workflow:

    * :mod:`Utility.FRAMES.NEBULA_TARGET_PHOTONS` writes
      ``obs_target_frames_ranked.pkl`` under
      ``NEBULA_OUTPUT_DIR / "TARGET_PHOTON_FRAMES"``.
    * :mod:`Utility.STARS.NEBULA_SKY_SELECTOR` reads that file,
      attaches sky footprints, and writes
      ``obs_target_frames_ranked_with_sky.pkl`` in the same directory.

    This function simply resolves the expected path (if not provided),
    loads the pickle with :mod:`pickle`, and logs a short summary.

    Parameters
    ----------
    frames_path : str or None
        Path to ``obs_target_frames_ranked_with_sky.pkl``. If None, a
        default is computed via :func:`_resolve_default_frames_path`.
    logger : logging.Logger
        Logger for emitting informational messages.

    Returns
    -------
    dict
        Dictionary keyed by observer name, each entry being the
        frames-with-sky structure produced by NEBULA_SKY_SELECTOR.
    """
    # If no explicit path is provided, use the standard TARGET_PHOTON_FRAMES
    # location where NEBULA_SKY_SELECTOR writes the frames-with-sky pickle.
    if frames_path is None:
        frames_path = _resolve_default_frames_path()

    logger.info("Loading frames-with-sky from '%s'.", frames_path)

    # Directly load the pickle written by NEBULA_SKY_SELECTOR.
    with open(frames_path, "rb") as f:
        frames_with_sky: Dict[str, Dict[str, Any]] = pickle.load(f)

    # Compute basic statistics for logging.
    n_observers = len(frames_with_sky)
    total_windows = sum(
        len(entry.get("windows", [])) for entry in frames_with_sky.values()
    )

    logger.info(
        "Loaded frames-with-sky for %d observers (%d windows total).",
        n_observers,
        total_windows,
    )

    return frames_with_sky

def _coerce_window_mid_time(window_entry: Dict[str, Any],
                            logger: Optional[logging.Logger] = None) -> Time:
    """
    Robustly construct an astropy Time object for the mid-window epoch.

    This helper is intentionally tolerant of older pickles where the mid-time
    may be stored in slightly different formats:

      * New schema (preferred):
          - window_entry["t_mid_mjd_utc"] : float (MJD, UTC)
          - window_entry["t_mid_utc"]     : ISO string (optional)

      * Older schemas we want to support:
          - t_mid_utc as a plain float (assumed MJD)
          - t_mid_utc as a datetime.datetime
          - t_mid_utc as a numpy.datetime64
          - t_mid_utc as an astropy.time.Time
          - t_mid_utc as an ISO / ISOT string

    Parameters
    ----------
    window_entry : dict
        Per-window dictionary from frames-with-sky.
    logger : logging.Logger, optional
        Logger for debug/warning messages.

    Returns
    -------
    t_mid : astropy.time.Time
        Mid-window time as a Time object in UTC scale.

    Raises
    ------
    ValueError
        If no usable mid-time representation can be inferred.
    """
    # 1) Preferred: explicit MJD field
    if "t_mid_mjd_utc" in window_entry:
        val = window_entry["t_mid_mjd_utc"]
        try:
            return Time(float(val), format="mjd", scale="utc")
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning(
                    "Failed to interpret 't_mid_mjd_utc'=%r as MJD (UTC): %s. "
                    "Falling back to 't_mid_utc' if available.",
                    val,
                    exc,
                )

    # 2) Fallbacks based on t_mid_utc
    val = window_entry.get("t_mid_utc", None)

    # Already a Time object
    if isinstance(val, Time):
        return val

    # Python datetime
    if isinstance(val, datetime.datetime):
        return Time(val, scale="utc")

    # Numpy datetime64
    if isinstance(val, np.datetime64):
        # Convert to Python datetime via numpy, then to Time
        py_dt = val.astype("datetime64[ns]").astype(datetime.datetime)
        return Time(py_dt, scale="utc")

    # Numeric types: interpret as MJD
    if isinstance(val, (float, int, np.floating, np.integer)):
        return Time(float(val), format="mjd", scale="utc")

    # Strings: try ISOT then ISO
    if isinstance(val, str):
        # Try ISOT first (YYYY-MM-DDThh:mm:ss.sss)
        try:
            return Time(val, format="isot", scale="utc")
        except Exception:  # noqa: BLE001
            # Fall back to more generic ISO
            return Time(val, format="iso", scale="utc")

    # Nothing worked: emit a useful error
    msg = (
        "Could not infer mid-window time from window_entry; "
        f"t_mid_mjd_utc={window_entry.get('t_mid_mjd_utc', None)!r}, "
        f"t_mid_utc={val!r}"
    )
    if logger is not None:
        logger.error(msg)
    raise ValueError(msg)

def _load_observer_tracks_with_pointing_from_disk(
    obs_tracks_path: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Load observer tracks with pointing from disk.

    This helper assumes that the NEBULA pixel pipeline has already been
    run via :func:`NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs`
    (for example, through ``sim_test.py``). In that workflow,
    :mod:`Utility.SAT_OBJECTS.NEBULA_PIXEL_PICKLER` writes the observer
    tracks with pointing and pixel geometry to::

        NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH

    The tracks are keyed by observer name and each entry includes, at a
    minimum:

        * ``t_mjd_utc`` : 1D array of float
        * ``pointing_boresight_ra_deg`` : 1D array of float
        * ``pointing_boresight_dec_deg`` : 1D array of float

    STAR projection uses these time series and pointing fields to build
    WCS solutions for each observer.

    Parameters
    ----------
    obs_tracks_path : str or None
        Path to the observer tracks pickle. If None, a default is
        computed via :func:`_resolve_default_obs_tracks_path`, which
        returns :data:`NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH`.
    logger : logging.Logger
        Logger for emitting informational messages.

    Returns
    -------
    dict
        Dictionary keyed by observer name, each entry being a track dict
        that includes at least the required time and pointing fields.
    """
    # If no explicit path is supplied, use the observer tracks pickle
    # written by the NEBULA pixel pipeline.
    if obs_tracks_path is None:
        obs_tracks_path = _resolve_default_obs_tracks_path()

    logger.info(
        "Loading observer tracks with pointing from '%s'.",
        obs_tracks_path,
    )

    # Directly load the pickle produced by NEBULA_PIXEL_PICKLER.
    with open(obs_tracks_path, "rb") as f:
        obs_tracks: Dict[str, Dict[str, Any]] = pickle.load(f)

    # Compute the number of observers contained in the tracks dictionary.
    n_observers = len(obs_tracks)

    # Determine, for each observer, whether the pointing fields are present.
    for obs_name, track in obs_tracks.items():
        has_pointing = all(
            key in track
            for key in (
                "pointing_boresight_ra_deg",
                "pointing_boresight_dec_deg",
            )
        )
        logger.debug(
            "Observer '%s': pointing fields present = %s.",
            obs_name,
            has_pointing,
        )

    logger.info(
        "Loaded observer tracks with pointing for %d observers.",
        n_observers,
    )

    return obs_tracks

def _save_star_projection_cache(
    obs_star_projections: Dict[str, Dict[str, Any]],
    output_path: Optional[str],
    logger: logging.Logger,
) -> str:
    """
    Save ``obs_star_projections`` to disk.

    This helper is called after the star projection stage has built the
    per-observer star projection cache in memory via
    :func:`build_star_projections_for_all_observers`. It performs three
    main tasks:

      * Resolves a default output path when ``output_path`` is None
        using :func:`_resolve_default_output_path`, which constructs
        a path of the form::

            NEBULA_OUTPUT_DIR / "STARS" / NEBULA_STAR_CATALOG.name /
                "obs_star_projections.pkl"

      * Ensures that the parent directory exists.

      * Serializes the ``obs_star_projections`` dictionary using
        :mod:`pickle` and logs a short summary.

    Parameters
    ----------
    obs_star_projections : dict
        Dictionary keyed by observer name that contains the star
        projection results for all observers. Each per-observer entry
        includes at least a ``"windows"`` list of per-window
        star-projection dicts.
    output_path : str or None
        Path to write ``obs_star_projections.pkl``. If None, a default
        is computed via :func:`_resolve_default_output_path`.
    logger : logging.Logger
        Logger for emitting informational messages.

    Returns
    -------
    str
        The absolute path where the file was written.
    """
    # If no explicit path is supplied, use the standard STARS/<catalog>
    # location defined by _resolve_default_output_path().
    if output_path is None:
        output_path = _resolve_default_output_path()

    # Ensure that the directory in which we are writing exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Log that we are writing the star projections to disk.
    logger.info("Writing obs_star_projections to '%s'.", output_path)

    # Serialize the obs_star_projections dictionary using pickle.
    with open(output_path, "wb") as f:
        pickle.dump(obs_star_projections, f)

    # Compute the number of observers contained in the dictionary for logging.
    n_observers = len(obs_star_projections)

    # Compute the total number of windows across all observers.
    total_windows = sum(
        len(entry.get("windows", [])) for entry in obs_star_projections.values()
    )

    # Compute the total number of on-detector stars across all windows.
    total_stars_on_detector = 0
    for entry in obs_star_projections.values():
        for w in entry.get("windows", []):
            total_stars_on_detector += int(w.get("n_stars_on_detector", 0))

    # Log a summary of what was saved.
    logger.info(
        "Saved obs_star_projections for %d observers (%d windows, %d stars on detector).",
        n_observers,
        total_windows,
        total_stars_on_detector,
    )

    return output_path

def _build_wcs_for_all_observers(
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build NebulaWCS objects for all observers.

    This function is a thin wrapper around the shared
    :func:`build_wcs_for_observer` helper from
    :mod:`Utility.SENSOR.NEBULA_WCS`. For each observer track, it
    invokes that builder and returns a mapping::

        wcs_map[obs_name] = NebulaWCS or list[NebulaWCS]

    where each entry can be:

      * a single :class:`NebulaWCS` instance for static pointing, or
      * a sequence of :class:`NebulaWCS` objects aligned with
        ``obs_track["t_mjd_utc"]`` for time-varying pointing.

    The subsequent :func:`_select_wcs_for_window` function then chooses
    the appropriate WCS snapshot per window based on the mid-window time.

    Parameters
    ----------
    obs_tracks : dict
        Dictionary keyed by observer name, each value being the observer
        track dict that includes at least:

          * ``"t_mjd_utc"`` : 1D float array of coarse times
          * ``"pointing_boresight_ra_deg"`` : scalar or 1D float array
          * ``"pointing_boresight_dec_deg"`` : scalar or 1D float array
          * ``"roll_deg"`` : scalar or 1D float array

        These fields are typically attached by the scheduling / pointing
        stages (e.g. NEBULA_SCHEDULE_PICKLER and NEBULA_POINTING_ANTISUN_GEO)
        before the pixel pipeline runs.
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, FOV, etc.). The exact
        dataclass definition lives in :mod:`Configuration.NEBULA_SENSOR_CONFIG`.
    logger : logging.Logger
        Logger for emitting informational messages.

    Returns
    -------
    dict
        A mapping from observer name to whatever object (or sequence of
        objects) :func:`build_wcs_for_observer` returns for that observer.
    """
    # Initialize an empty dictionary to hold the WCS objects per observer.
    wcs_map: Dict[str, Any] = {}

    # Loop over each observer track in the provided dictionary.
    for obs_name, obs_track in obs_tracks.items():
        # Log that we are building WCS objects for this observer.
        logger.info("Building WCS for observer '%s'.", obs_name)

        # Call the shared WCS builder with the observer track and sensor config.
        nebula_wcs_entry = build_wcs_for_observer(
            observer_track=obs_track,
            sensor_config=sensor_config,
        )

        # Store the resulting NebulaWCS (or list of NebulaWCS) in the map.
        wcs_map[obs_name] = nebula_wcs_entry

    # Return the mapping from observer name to WCS entry.
    return wcs_map

def _propagate_gaia_to_window_epoch(
    gaia_window: Dict[str, Any],
    t_mid_time: Time,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate Gaia star positions to the window mid-time epoch.

    This function takes the Gaia cone data for a single window and
    computes the apparent RA/Dec at the specified mid-time, using
    Astropy's :class:`SkyCoord` and :meth:`SkyCoord.apply_space_motion`.

    Proper-motion semantics
    -----------------------
    We adopt the following conventions:

      * Gaia proper motion components are stored (when available) as::

            pm_ra_masyr   # pmra (already cos(dec)-weighted)
            pm_dec_masyr  # pmdec

        These map directly to Astropy's::

            pm_ra_cosdec = pm_ra_masyr * (u.mas / u.yr)
            pm_dec       = pm_dec_masyr * (u.mas / u.yr)

      * Optional parallax and radial velocity fields (if present) are::

            parallax_mas
            radial_velocity_km_s

    Whether proper motion is used is controlled by
    ``NEBULA_STAR_QUERY.use_proper_motion`` (bool). If
    ``use_proper_motion`` is False, or the necessary astrometric fields
    are missing, the function simply returns the catalog RA/Dec as-is.

    Parameters
    ----------
    gaia_window : dict
        Gaia window entry containing at least:

          * ``"ra_deg"``, ``"dec_deg"`` : arrays of catalog RA/Dec
            in degrees

        and optionally:

          * ``"pm_ra_masyr"``, ``"pm_dec_masyr"``
          * ``"parallax_mas"``
          * ``"radial_velocity_km_s"``
    t_mid_time : astropy.time.Time
        Window mid-time at which to evaluate the star positions.

    Returns
    -------
    ra_deg_mid : np.ndarray
        Array of RA values at window mid-time in degrees.
    dec_deg_mid : np.ndarray
        Array of Dec values at window mid-time in degrees.
    """
    # Extract the catalog RA/Dec in degrees as numpy arrays. Let a missing
    # key raise KeyError rather than silently propagating empty arrays.
    ra_deg = np.asarray(gaia_window["ra_deg"], dtype=float)
    dec_deg = np.asarray(gaia_window["dec_deg"], dtype=float)

    # If the NEBULA_STAR_QUERY configuration disables proper motion,
    # return the catalog positions directly.
    use_pm = bool(getattr(NEBULA_STAR_QUERY, "use_proper_motion", False))
    if not use_pm:
        return ra_deg, dec_deg

    # Extract proper motion arrays if present; if missing, fall back
    # to catalog positions.
    pm_ra_arr = gaia_window.get("pm_ra_masyr", None)
    pm_dec_arr = gaia_window.get("pm_dec_masyr", None)
    if pm_ra_arr is None or pm_dec_arr is None:
        return ra_deg, dec_deg

    # Convert proper motions to numpy arrays of floats.
    pm_ra_masyr = np.asarray(pm_ra_arr, dtype=float)
    pm_dec_masyr = np.asarray(pm_dec_arr, dtype=float)

    # Extract parallax and radial velocity arrays if present.
    parallax_arr = gaia_window.get("parallax_mas", None)
    rv_arr = gaia_window.get("radial_velocity_km_s", None)

    # Build keyword arguments for SkyCoord construction. Gaia positions
    # are in the ICRS frame.
    coord_kwargs: Dict[str, Any] = {
        "ra": ra_deg * u.deg,
        "dec": dec_deg * u.deg,
        "pm_ra_cosdec": pm_ra_masyr * (u.mas / u.yr),
        "pm_dec": pm_dec_masyr * (u.mas / u.yr),
        "frame": "icrs",
    }

    # If parallax is available, pass it as a quantity to SkyCoord.
    if parallax_arr is not None:
        coord_kwargs["parallax"] = np.asarray(parallax_arr, dtype=float) * u.mas

    # If radial velocity is available, pass it as a quantity to SkyCoord.
    if rv_arr is not None:
        coord_kwargs["radial_velocity"] = (
            np.asarray(rv_arr, dtype=float) * (u.km / u.s)
        )

    # Determine the catalog reference epoch from NEBULA_STAR_CATALOG.
    ref_epoch = getattr(NEBULA_STAR_CATALOG, "reference_epoch", None)

    # Attempt to construct an Astropy Time for the reference epoch.
    try:
        # If ref_epoch is a numeric year, interpret it as Julian year.
        if isinstance(ref_epoch, (int, float)):
            obstime = Time(ref_epoch, format="jyear")
        else:
            # Otherwise, treat it as a time string that Time can parse.
            obstime = Time(str(ref_epoch))
    except Exception:  # noqa: BLE001
        obstime = None  # type: ignore[assignment]

    # If we could not determine a valid reference epoch, fall back to
    # catalog RA/Dec.
    if obstime is None:
        return ra_deg, dec_deg

    # Construct a SkyCoord at the catalog reference epoch with proper motions.
    coord = SkyCoord(obstime=obstime, **coord_kwargs)

    # Apply space motion to propagate to the window mid-time.
    coord_mid = coord.apply_space_motion(new_obstime=t_mid_time)

    # Extract RA and Dec at the mid-time epoch in degrees and return them.
    return coord_mid.ra.deg, coord_mid.dec.deg

def _make_empty_star_window_projection(
    window_entry: Dict[str, Any],
    gaia_status: str,
    reason: str,
) -> StarWindowProjection:
    """
    Construct an empty StarWindowProjection for non-success Gaia cases.

    This helper fills in the per-window metadata from the frames-with-sky
    window entry and sets all star-related counts to zero, with an empty
    ``"stars"`` dict. It is used when Gaia data is missing, in error,
    or when a query succeeds but returns zero rows for a given window.

    Parameters
    ----------
    window_entry : dict
        Window entry from frames-with-sky containing metadata such as
        ``window_index``, ``start_index``, ``end_index``, ``n_frames``,
        ``t_mid_utc``, ``t_mid_mjd_utc``, ``sky_center_ra_deg``,
        ``sky_center_dec_deg``, ``sky_radius_deg``, and
        ``sky_selector_status``.
    gaia_status : {"ok_empty", "error", "missing"}
        Status string describing the Gaia data situation for this window.
        This helper is only used for non-success cases; successful windows
        use ``gaia_status="ok"`` and are constructed separately.
    reason : str
        Human-readable explanation for the status (stored in
        ``gaia_error_message``).

    Returns
    -------
    StarWindowProjection
        A StarWindowProjection dict with zero stars and the given status.
    """
    # Optional sanity check: catch accidental misuse with "ok".
    # allowed_statuses = {"ok_empty", "error", "missing"}
    # if gaia_status not in allowed_statuses:
    #     raise ValueError(f"Unexpected gaia_status='{gaia_status}' for empty projection.")

    # Create the base projection dict, copying over basic window metadata.
    projection: StarWindowProjection = {
        # Copy window index from frames-with-sky entry.
        "window_index": int(window_entry.get("window_index")),
        # Copy start index (frame index) from frames-with-sky entry.
        "start_index": int(window_entry.get("start_index")),
        # Copy end index (frame index) from frames-with-sky entry.
        "end_index": int(window_entry.get("end_index")),
        # Copy number of frames in this window.
        "n_frames": int(window_entry.get("n_frames")),
        # Copy mid-time in UTC (could be datetime or string).
        "t_mid_utc": window_entry.get("t_mid_utc"),
        # Copy mid-time in MJD UTC.
        "t_mid_mjd_utc": float(window_entry.get("t_mid_mjd_utc")),
        # Copy sky center RA in degrees.
        "sky_center_ra_deg": float(window_entry.get("sky_center_ra_deg")),
        # Copy sky center Dec in degrees.
        "sky_center_dec_deg": float(window_entry.get("sky_center_dec_deg")),
        # Copy sky radius in degrees.
        "sky_radius_deg": float(window_entry.get("sky_radius_deg")),
        # Copy sky selector status string.
        "sky_selector_status": window_entry.get("sky_selector_status"),
        # Set Gaia status for this window.
        "gaia_status": gaia_status,
        # Attach a human-readable explanation for the Gaia status.
        "gaia_error_message": reason,
        # Initialize the number of Gaia stars in the cone as zero.
        "n_stars_input": 0,
        # Initialize the number of on-detector stars as zero.
        "n_stars_on_detector": 0,
        # No density information is available for empty/error windows.
        "star_density_on_detector_per_deg2": None,
        # Initialize an empty dict for per-star details.
        "stars": {},
    }

    # Return the constructed empty projection.
    return projection

def project_gaia_stars_for_window(
    obs_name: str,
    window_entry: Dict[str, Any],
    gaia_window_or_none: Optional[Dict[str, Any]],
    obs_track: Dict[str, Any],
    nebula_wcs_entry: Any,
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> StarWindowProjection:
    """
    Project Gaia stars for a single window onto the sensor.

    This function performs the per-window core logic:

        1. Handles missing or error Gaia cones by returning an empty
           StarWindowProjection with an appropriate gaia_status.

        2. For valid Gaia cones:

            * Extracts Gaia positions and magnitudes.
            * Builds an Astropy Time for the window mid-time.
            * Propagates the catalog RA/Dec to the mid-time epoch
              using :func:`_propagate_gaia_to_window_epoch`.
            * Selects a WCS snapshot for the window via
              :func:`_select_wcs_for_window`.
            * Projects RA/Dec at mid-time to pixel coordinates using
              project_radec_to_pixels (or NebulaWCS.world_to_pixel).
            * Applies a simple FOV mask:

                  0 <= x < sensor_config.n_cols
                  0 <= y < sensor_config.n_rows

            * Counts on-detector stars and computes a star density
              based on the Gaia query cone area:

                  density = n_stars_on_detector / (π * query_radius_deg²)

            * Builds the per-star dictionary keyed by Gaia source ID string.

    Parameters
    ----------
    obs_name : str
        Name of the observer (used for logging only).
    window_entry : dict
        Window entry from frames-with-sky, containing metadata and
        window mid-time.
    gaia_window_or_none : dict or None
        Matching Gaia window entry from the Gaia cache for this window
        (same window_index), or None if missing entirely.
    obs_track : dict
        Observer track dict containing at least "t_mjd_utc".
    nebula_wcs_entry : object or sequence
        WCS entry returned by :func:`build_wcs_for_observer`.
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, etc.).
    logger : logging.Logger
        Logger for emitting debug information.

    Returns
    -------
    StarWindowProjection
        A fully populated StarWindowProjection dict for this window.
    """

    # ---------------------------------------------------------------------
    # 1) Handle missing Gaia windows
    # ---------------------------------------------------------------------

    # If there is no Gaia window entry at all for this window_index,
    # we cannot project any stars. Treat this as a "missing" Gaia case
    # and return a structurally valid but empty StarWindowProjection.
    if gaia_window_or_none is None:
        return _make_empty_star_window_projection(
            window_entry=window_entry,
            gaia_status="missing",
            reason="No Gaia window for this window_index",
        )

    # At this point we know we have *some* Gaia window dict (could be ok/error).
    gaia_window = gaia_window_or_none

    # If Gaia recorded a non-"ok" status for this window (e.g. TAP error),
    # we treat it as an "error" at the projection level and return an empty
    # projection with an appropriate status + message.
    if gaia_window.get("status") != "ok":
        return _make_empty_star_window_projection(
            window_entry=window_entry,
            gaia_status="error",
            reason=gaia_window.get("error_message") or "Gaia status != 'ok'",
        )

    # Extract the number of Gaia rows reported for this window. Even if the
    # query "succeeded", n_rows may be zero (valid empty cone).
    n_rows_gaia = int(gaia_window.get("n_rows", 0))

    # If the query returned zero rows but status was "ok", this is a valid
    # *empty* result: there are simply no Gaia stars within the cone.
    # We distinguish this case as "ok_empty".
    if n_rows_gaia == 0:
        return _make_empty_star_window_projection(
            window_entry=window_entry,
            gaia_status="ok_empty",
            reason="Gaia query returned 0 rows",
        )

    # ---------------------------------------------------------------------
    # 2) Optional consistency check on window indices
    # ---------------------------------------------------------------------

    # Both frames-with-sky and Gaia cache should agree on window_index.
    # If they don't, log a warning but continue; this is a sanity check
    # to catch potential misalignment in upstream matching logic.
    if int(window_entry.get("window_index")) != int(gaia_window.get("window_index")):
        logger.warning(
            "Observer '%s': frames window_index=%s does not match Gaia window_index=%s.",
            obs_name,
            window_entry.get("window_index"),
            gaia_window.get("window_index"),
        )

    # ---------------------------------------------------------------------
    # 3) Extract required Gaia arrays
    # ---------------------------------------------------------------------

    # Convert Gaia columns to NumPy arrays. We treat these as required
    # fields; if any are missing, KeyError will surface and reveal a
    # schema break early rather than silently returning nonsense.
    gaia_source_id = np.asarray(gaia_window["gaia_source_id"], dtype=int)
    ra_deg = np.asarray(gaia_window["ra_deg"], dtype=float)
    dec_deg = np.asarray(gaia_window["dec_deg"], dtype=float)
    mag_G = np.asarray(gaia_window["mag_G"], dtype=float)

    # Number of Gaia stars in the cone, based on the RA array length.
    n_input = ra_deg.size

    # Cross-check that the number of rows we see in the arrays matches
    # the n_rows field recorded when the query was run. If there is a
    # mismatch, log a warning to help debug upstream issues.
    if n_input != n_rows_gaia:
        logger.warning(
            "Observer '%s', window %s: n_input=%d but Gaia n_rows=%d.",
            obs_name,
            window_entry.get("window_index"),
            n_input,
            n_rows_gaia,
        )

    # ---------------------------------------------------------------------
    # 4) Build the mid-window time as an Astropy Time
    # ---------------------------------------------------------------------

    # Prefer the numeric MJD representation if available, since the schema
    # makes t_mid_mjd_utc required and it ties directly into other MJD-based
    # time arrays (e.g. obs_track["t_mjd_utc"]).
    # if "t_mid_mjd_utc" in window_entry:
    #     t_mid = Time(float(window_entry["t_mid_mjd_utc"]), format="mjd", scale="utc")
    # else:
    #     # Fallback: if only t_mid_utc is present (datetime or ISO string),
    #     # let Astropy parse it into a Time object.
    #     t_mid = Time(window_entry.get("t_mid_utc"))
    # Use a robust helper that understands both the new schema
    # (t_mid_mjd_utc float) and older variants of t_mid_utc.
    # t_mid = _coerce_window_mid_time(window_entry, logger=logger)
    # ---------------------------------------------------------------------
    # 4) Determine the mid-window epoch
    # ---------------------------------------------------------------------
    # We want a single Astropy Time object, t_mid, representing the epoch
    # at which we will:
    #   * Propagate Gaia astrometry (proper motion / parallax / RV), and
    #   * Select the appropriate WCS snapshot for this window.
    #
    # Primary source of truth for the mid-window time is the Gaia window
    # entry produced by NEBULA_QUERY_GAIA. That cache stores both:
    #   - 't_mid_mjd_utc' (float, MJD UTC)
    #   - 't_mid_utc'     (ISO-8601 UTC string)
    #
    # For robustness with older caches / frames-with-sky pickles, we fall
    # back to any per-window mid-time stored in window_entry, and as a
    # last resort derive the mid-epoch from the coarse time grid contained
    # in obs_track['t_mjd_utc'].
    t_mid: Time
    
    # 4a) Try Gaia cache MJD mid-time first
    mid_mjd = None
    if gaia_window is not None:
        mid_mjd = gaia_window.get("t_mid_mjd_utc")
    
    # 4b) Fall back to frames-with-sky mid-time, if present
    if mid_mjd is None and "t_mid_mjd_utc" in window_entry:
        mid_mjd = window_entry.get("t_mid_mjd_utc")
    
    if mid_mjd is not None:
        try:
            t_mid = Time(float(mid_mjd), format="mjd", scale="utc")
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "project_gaia_stars_for_window: failed to parse "
                    "t_mid_mjd_utc=%r for observer=%s, window=%s (%s); "
                    "will fall back to t_mid_utc / coarse grid.",
                    mid_mjd,
                    obs_name,
                    window_entry.get("window_index"),
                    exc,
                )
            t_mid = None
    else:
        t_mid = None
    
    # 4c) If MJD path failed, try ISO UTC string from Gaia cache / frames
    if t_mid is None:
        mid_iso = None
        if gaia_window is not None:
            mid_iso = gaia_window.get("t_mid_utc")
        if mid_iso is None:
            mid_iso = window_entry.get("t_mid_utc")
    
        if isinstance(mid_iso, str):
            try:
                # Let Astropy detect the appropriate string format (iso/isot)
                t_mid = Time(mid_iso)
            except Exception as exc:
                if logger is not None:
                    logger.warning(
                        "project_gaia_stars_for_window: failed to parse "
                        "t_mid_utc=%r for observer=%s, window=%s (%s); "
                        "will fall back to coarse-grid mid index.",
                        mid_iso,
                        obs_name,
                        window_entry.get("window_index"),
                        exc,
                    )
                t_mid = None
        elif isinstance(mid_iso, Time):
            # Already an Astropy Time object
            t_mid = mid_iso
    
    # 4d) Last-resort: derive mid-epoch from the coarse time grid using
    #      the window's start/end coarse indices.
    if t_mid is None:
        start_idx = int(window_entry.get("start_index", 0))
        end_idx = int(window_entry.get("end_index", start_idx))
        # Prefer 't_mjd_utc' (current schema); fall back to older 't_mjd'.
        t_mjd_array = obs_track.get("t_mjd_utc") or obs_track.get("t_mjd")
    
        if t_mjd_array is None:
            raise ValueError(
                "project_gaia_stars_for_window: could not determine "
                "mid-window time for observer=%s, window=%s: no "
                "t_mid_mjd_utc/t_mid_utc in Gaia cache or frames, and "
                "obs_track has no t_mjd_utc/t_mjd array."
                % (obs_name, window_entry.get("window_index"))
            )
    
        t_mjd_array = np.asarray(t_mjd_array, dtype=float)
        mid_idx = max(0, min((start_idx + end_idx) // 2, t_mjd_array.size - 1))
        t_mid = Time(float(t_mjd_array[mid_idx]), format="mjd", scale="utc")
    
        if logger is not None:
            logger.warning(
                "project_gaia_stars_for_window: fell back to coarse-grid "
                "mid index %d for observer=%s, window=%s.",
                mid_idx,
                obs_name,
                window_entry.get("window_index"),
            )
    
    # Make sure the window has a consistent 't_mid_mjd_utc' field so that
    # downstream helpers (e.g. _select_wcs_for_window) can rely on it.
    window_entry["t_mid_mjd_utc"] = float(t_mid.mjd)
    
    # ---------------------------------------------------------------------
    # 5) Propagate Gaia astrometry to the mid-window epoch
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # 5) Propagate Gaia astrometry to the mid-window epoch
    # ---------------------------------------------------------------------

    # Use the helper that knows about NEBULA_STAR_QUERY.use_proper_motion
    # and NEBULA_STAR_CATALOG.reference_epoch. This will either:
    #   - return catalog RA/Dec unchanged (if PM disabled or missing), or
    #   - return RA/Dec propagated to t_mid via SkyCoord.apply_space_motion.
    ra_deg_mid, dec_deg_mid = _propagate_gaia_to_window_epoch(
        gaia_window=gaia_window,
        t_mid_time=t_mid,
    )

    # ---------------------------------------------------------------------
    # 6) Select the appropriate WCS snapshot for this window
    # ---------------------------------------------------------------------

    # _select_wcs_for_window chooses a NebulaWCS instance corresponding
    # to this window's mid-time:
    #   - If nebula_wcs_entry is a single WCS: returns that WCS.
    #   - If it's a list/tuple: uses obs_track["t_mjd_utc"] and t_mid_mjd_utc
    #     to pick the closest time index and returns nebula_wcs_entry[idx].
    nebula_wcs_mid = _select_wcs_for_window(
        nebula_wcs_entry=nebula_wcs_entry,
        obs_track=obs_track,
        window_entry=window_entry,
    )

    # ---------------------------------------------------------------------
    # 7) Convert RA/Dec at mid-time to pixel coordinates
    # ---------------------------------------------------------------------

    # If the shared WCS helper from NEBULA_WCS is available, use it; this
    # centralizes all WCS calling conventions and origin handling.
    if project_radec_to_pixels is not None:
        x_pix_mid, y_pix_mid = project_radec_to_pixels(
            nebula_wcs_mid,
            ra_deg_mid,
            dec_deg_mid,
        )
    else:
        # Fallback path: require that the NebulaWCS object itself exposes
        # a world_to_pixel method. If it does not, raise a clear error,
        # because we have no way to project stars to the detector.
        if not hasattr(nebula_wcs_mid, "world_to_pixel"):
            raise RuntimeError(
                "Neither 'project_radec_to_pixels' nor 'world_to_pixel' is available "
                "for RA/Dec -> pixel projection."
            )
        # Call the WCS object's world_to_pixel directly.
        x_pix_mid, y_pix_mid = nebula_wcs_mid.world_to_pixel(
            ra_deg_mid,
            dec_deg_mid,
        )

    # Ensure pixel coordinates are NumPy float arrays, which makes downstream
    # masking and storage consistent and predictable.
    x_pix_mid = np.asarray(x_pix_mid, dtype=float)
    y_pix_mid = np.asarray(y_pix_mid, dtype=float)

    # ---------------------------------------------------------------------
    # 8) Apply sensor FOV mask and compute star density
    # ---------------------------------------------------------------------

    # Get sensor geometry from the SensorConfig object. We support both
    # "n_rows"/"n_cols" and "rows"/"cols" attribute names to be robust
    # against small schema variations.
    n_rows = int(getattr(sensor_config, "n_rows", getattr(sensor_config, "rows", 0)))
    n_cols = int(getattr(sensor_config, "n_cols", getattr(sensor_config, "cols", 0)))

    # Build a boolean mask indicating which stars land inside the detector
    # rectangle in pixel coordinates:
    #   0 <= x < n_cols
    #   0 <= y < n_rows
    on_detector = (
        (x_pix_mid >= 0.0)
        & (x_pix_mid < float(n_cols))
        & (y_pix_mid >= 0.0)
        & (y_pix_mid < float(n_rows))
    )

    # Count how many stars are on the detector for this window.
    n_on = int(on_detector.sum())

    # Extract the Gaia query cone radius in degrees. This was set when
    # querying Gaia (sky_radius_deg + padding) and is used to compute
    # an approximate sky area.
    query_radius_deg = float(gaia_window.get("query_radius_deg", 0.0))

    # Compute the cone area in square degrees using A = π r^2.
    area_deg2 = np.pi * (query_radius_deg ** 2)

    # Compute the on-detector star density per square degree if the area
    # is non-zero; otherwise set to None (undefined).
    density = (n_on / area_deg2) if area_deg2 > 0.0 else None

    # ---------------------------------------------------------------------
    # 9) Extract optional astrometric arrays once (efficiency)
    # ---------------------------------------------------------------------
    # Initialize optional arrays to None. If the corresponding keys exist
    # in the Gaia window and are not None, we will populate them as NumPy
    # arrays, and then simply index them inside the per-star loop.
    pm_ra_arr = pm_dec_arr = parallax_arr = rv_arr = None

    # Proper motion in RA * cos(dec), in mas/yr, if available.
    if "pm_ra_masyr" in gaia_window and gaia_window["pm_ra_masyr"] is not None:
        pm_ra_arr = np.asarray(gaia_window["pm_ra_masyr"], dtype=float)

    # Proper motion in Dec, in mas/yr, if available.
    if "pm_dec_masyr" in gaia_window and gaia_window["pm_dec_masyr"] is not None:
        pm_dec_arr = np.asarray(gaia_window["pm_dec_masyr"], dtype=float)

    # Parallax in mas, if available.
    if "parallax_mas" in gaia_window and gaia_window["parallax_mas"] is not None:
        parallax_arr = np.asarray(gaia_window["parallax_mas"], dtype=float)

    # Radial velocity in km/s, if available.
    if "radial_velocity_km_s" in gaia_window and gaia_window["radial_velocity_km_s"] is not None:
        rv_arr = np.asarray(gaia_window["radial_velocity_km_s"], dtype=float)


    # ---------------------------------------------------------------------
    # 10) Build per-star entries for this window
    # ---------------------------------------------------------------------

    # Initialize the per-star dictionary; keys will be stringified Gaia
    # source IDs (e.g. "1234567890"), and values will be dicts with
    # astrometry, photometry, and pixel position for each star.
    stars: Dict[str, Dict[str, Any]] = {}

    # Iterate over each star in the cone and assemble its entry.
    for i in range(n_input):
        # Gaia source ID as integer.
        sid_int = int(gaia_source_id[i])
        # String alias used both as dict key and as a generic "source_id".
        sid_str = str(sid_int)

        # Build the core star entry containing:
        #   - identifiers
        #   - catalog and mid-time RA/Dec
        #   - G magnitude
        #   - pixel position at window mid-time
        #   - on-detector flag
        star_entry: Dict[str, Any] = {
            "gaia_source_id": sid_int,
            "source_id": sid_str,
            "source_type": "star",
            "ra_deg_catalog": float(ra_deg[i]),
            "dec_deg_catalog": float(dec_deg[i]),
            "ra_deg_mid": float(ra_deg_mid[i]),
            "dec_deg_mid": float(dec_deg_mid[i]),
            "mag_G": float(mag_G[i]),
            "x_pix_mid": float(x_pix_mid[i]),
            "y_pix_mid": float(y_pix_mid[i]),
            "on_detector": bool(on_detector[i]),
        }

        # If proper motion in RA is available, attach the value for this star.
        if pm_ra_arr is not None:
            star_entry["pm_ra_masyr"] = float(pm_ra_arr[i])

        # If proper motion in Dec is available, attach the value for this star.
        if pm_dec_arr is not None:
            star_entry["pm_dec_masyr"] = float(pm_dec_arr[i])

        # If parallax is available, attach the value for this star.
        if parallax_arr is not None:
            star_entry["parallax_mas"] = float(parallax_arr[i])

        # If radial velocity is available, attach the value for this star.
        if rv_arr is not None:
            star_entry["radial_velocity_km_s"] = float(rv_arr[i])

        # Store this star entry in the stars dict, keyed by the string source ID.
        stars[sid_str] = star_entry

    # Log a debug summary for this window: how many stars were in the cone,
    # and how many landed on the detector.
    logger.debug(
        "Observer '%s', window %s: n_input=%d, n_on_detector=%d.",
        obs_name,
        window_entry.get("window_index"),
        n_input,
        n_on,
    )

    # ---------------------------------------------------------------------
    # 11) Assemble the final StarWindowProjection dict for this window
    # ---------------------------------------------------------------------

    # Build the full per-window projection object, mirroring the schema
    # used by _make_empty_star_window_projection but filling in all the
    # star-related fields for the successful "ok" case.
    projection: StarWindowProjection = {
        # Window indexing and frame range.
        "window_index": int(window_entry.get("window_index")),
        "start_index": int(window_entry.get("start_index")),
        "end_index": int(window_entry.get("end_index")),
        "n_frames": int(window_entry.get("n_frames")),
        # Mid-time in UTC and MJD UTC.
        "t_mid_utc": window_entry.get("t_mid_utc"),
        "t_mid_mjd_utc": float(window_entry.get("t_mid_mjd_utc")),
        # Sky selector metadata (cone center and radius).
        "sky_center_ra_deg": float(window_entry.get("sky_center_ra_deg")),
        "sky_center_dec_deg": float(window_entry.get("sky_center_dec_deg")),
        "sky_radius_deg": float(window_entry.get("sky_radius_deg")),
        "sky_selector_status": window_entry.get("sky_selector_status"),
        # Gaia status and (lack of) error message for a successful window.
        "gaia_status": "ok",
        "gaia_error_message": None,
        # Star counts and density on the detector.
        "n_stars_input": int(n_input),
        "n_stars_on_detector": int(n_on),
        "star_density_on_detector_per_deg2": density,
        # Per-star data keyed by Gaia source ID string.
        "stars": stars,
    }

    # Return the populated StarWindowProjection for this window.
    return projection

def build_star_projections_for_observer(
    obs_name: str,
    frames_entry: Dict[str, Any],
    gaia_obs_entry: Dict[str, Any],
    obs_track: Dict[str, Any],
    nebula_wcs_entry: Any,
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build star projections for all windows of a single observer.

    For a given observer, this function:

        * Iterates over all windows in the frames-with-sky structure.

        * Uses a lookup dictionary to associate each window_index with
          its corresponding Gaia cones window.

        * Calls :func:`project_gaia_stars_for_window` for each window,
          collecting the resulting StarWindowProjection objects.

        * Maintains per-observer counters:

              - n_windows_total
              - n_windows_with_gaia
              - n_windows_error
              - total_stars_input
              - total_stars_on_detector

          as well as a list of densities for windows with non-None
          star_density_on_detector_per_deg2.

        * Computes density statistics across windows:

              median, p10, p90

        * Computes the time range covered by windows that have Gaia data.

        * Builds the final per-observer entry of the form:

              {
                  "observer_name": ...,
                  "rows": ...,
                  "cols": ...,
                  "catalog_name": ...,
                  "catalog_band": ...,
                  "mag_limit_sensor_G": ...,
                  "run_meta": {...},
                  "windows": [StarWindowProjection, ...],
              }

    Parameters
    ----------
    obs_name : str
        Name of the observer.
    frames_entry : dict
        Entry from frames-with-sky keyed by this observer, containing
        rows, cols, dt_frame_s, and a "windows" list.
    gaia_obs_entry : dict
        Gaia cones cache entry keyed by this observer, containing
        catalog metadata and a "windows" list of Gaia windows.
    obs_track : dict
        Observer track dict including "t_mjd_utc" and pointing fields.
    nebula_wcs_entry : object or sequence
        WCS entry returned by :func:`build_wcs_for_observer` for this observer.
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, etc.).
    logger : logging.Logger
        Logger for emitting summaries.

    Returns
    -------
    dict
        Per-observer star projection entry as described above.
    """
    # ------------------------------------------------------------------
    # 1) Extract windows and build Gaia lookup by window_index
    # ------------------------------------------------------------------

    # Make a local list of frames-with-sky windows for this observer.
    windows_frames: List[Dict[str, Any]] = list(frames_entry.get("windows", []))

    # Build a mapping from window_index -> Gaia window dict so we can
    # efficiently look up the Gaia cones entry for each frames window.
    gaia_by_idx: Dict[int, Dict[str, Any]] = {
        int(w.get("window_index")): w for w in gaia_obs_entry.get("windows", [])
    }

    # ------------------------------------------------------------------
    # 2) Initialize counters and collections for statistics
    # ------------------------------------------------------------------

    # Count of all frames-with-sky windows for this observer.
    n_windows_total = 0
    # Count of windows that have some Gaia window entry (status may be ok/error/empty).
    n_windows_with_gaia = 0
    # Count of windows whose final projection reports Gaia status "error" or "missing".
    n_windows_error = 0
    # Total number of Gaia stars across all windows (cone counts).
    total_stars_input = 0
    # Total number of stars that land on the detector across all windows.
    total_stars_on_detector = 0

    # Densities (per deg^2) for windows where density is defined.
    densities_on_detector: List[float] = []

    # Per-window StarWindowProjection dicts to be attached to this observer.
    window_projections: List[StarWindowProjection] = []

    # Mid-times (in MJD UTC) for windows that have a Gaia window entry,
    # used to compute an overall time range for this observer.
    times_mjd_with_gaia: List[float] = []

    # ------------------------------------------------------------------
    # 3) Loop over frames-with-sky windows and project stars per window
    # ------------------------------------------------------------------

    for window_entry in windows_frames:
        # Increment the total number of windows processed.
        n_windows_total += 1

        # Extract this window's index from the frames entry.
        widx = int(window_entry.get("window_index"))

        # Look up the corresponding Gaia window for this index, if it exists.
        gaia_window = gaia_by_idx.get(widx)

        # If a Gaia window entry exists, track that this window has Gaia context.
        if gaia_window is not None:
            n_windows_with_gaia += 1

        # Call the per-window core helper that:
        #   - Handles missing/error/empty Gaia cases.
        #   - Propagates Gaia astrometry.
        #   - Selects the appropriate WCS.
        #   - Projects stars to pixel coordinates.
        #   - Builds the StarWindowProjection dict.
        projection = project_gaia_stars_for_window(
            obs_name=obs_name,
            window_entry=window_entry,
            gaia_window_or_none=gaia_window,
            obs_track=obs_track,
            nebula_wcs_entry=nebula_wcs_entry,
            sensor_config=sensor_config,
            logger=logger,
        )

        # Append the resulting per-window projection into our list.
        window_projections.append(projection)

        # Update star-count statistics using fields from the projection.
        total_stars_input += int(projection.get("n_stars_input", 0))
        total_stars_on_detector += int(projection.get("n_stars_on_detector", 0))

        # If this window has a defined density, record it for percentile stats.
        density_val = projection.get("star_density_on_detector_per_deg2")
        if density_val is not None:
            densities_on_detector.append(float(density_val))

        # If Gaia status indicates error or missing, increment the error window count.
        if projection.get("gaia_status") in {"error", "missing"}:
            n_windows_error += 1

        # For windows that had a Gaia window entry at all (even if error/empty),
        # record the mid-time in MJD UTC for time-range computation.
        if gaia_window is not None:
            # t_mid_mjd_utc is required by our schema for StarWindowProjection.
            t_mid_mjd = float(projection.get("t_mid_mjd_utc"))
            times_mjd_with_gaia.append(t_mid_mjd)

    # ------------------------------------------------------------------
    # 4) Compute density statistics across windows (if available)
    # ------------------------------------------------------------------

    if densities_on_detector:
        # Convert to NumPy array to compute median and percentiles efficiently.
        dens_arr = np.asarray(densities_on_detector, dtype=float)
        dens_median = float(np.median(dens_arr))
        dens_p10 = float(np.percentile(dens_arr, 10.0))
        dens_p90 = float(np.percentile(dens_arr, 90.0))
    else:
        # If no densities were recorded, leave all stats as None.
        dens_median = None
        dens_p10 = None
        dens_p90 = None

    # ------------------------------------------------------------------
    # 5) Compute time range for windows that have Gaia entries
    # ------------------------------------------------------------------

    if times_mjd_with_gaia:
        # Build an Astropy Time array from MJD UTC values.
        t_arr = Time(times_mjd_with_gaia, format="mjd", scale="utc")
        # Convert the earliest and latest times to ISO strings for human readability.
        t_min_str = t_arr.min().iso
        t_max_str = t_arr.max().iso
        time_range_utc: Tuple[Optional[str], Optional[str]] = (t_min_str, t_max_str)
    else:
        # If there are no Gaia windows, encode the time range as (None, None).
        time_range_utc = (None, None)

    # ------------------------------------------------------------------
    # 6) Extract sensor and catalog metadata for this observer
    # ------------------------------------------------------------------

    # Prefer rows/cols from the frames entry but fall back to the SensorConfig
    # if necessary, to be robust to minor schema inconsistencies.
    rows = int(frames_entry.get("rows", getattr(sensor_config, "rows", 0)))
    cols = int(frames_entry.get("cols", getattr(sensor_config, "cols", 0)))

    # Pull catalog metadata from the Gaia observer entry, with sensible fallbacks.
    catalog_name = gaia_obs_entry.get("catalog_name", NEBULA_STAR_CATALOG.name)
    catalog_band = gaia_obs_entry.get("band", "G")
    mag_limit_sensor_G = float(gaia_obs_entry.get("mag_limit_sensor_G", np.nan))

    # ------------------------------------------------------------------
    # 7) Assemble run_meta summary for this observer
    # ------------------------------------------------------------------

    run_meta = {
        # Star projection stage version string.
        "version": STAR_PROJECTION_RUN_META_VERSION,
        # UTC creation time of this per-observer projection entry.
        "created_utc": datetime.utcnow().isoformat(),
        # Source file paths (filled by main() at the top level).
        "frames_source_file": None,
        "gaia_cones_file": None,
        "observer_tracks_file": None,
        # Time range covered by windows with Gaia entries (ISO strings or None).
        "time_range_utc": time_range_utc,
        # Window-level statistics.
        "n_windows_total": n_windows_total,
        "n_windows_with_gaia": n_windows_with_gaia,
        "n_windows_error": n_windows_error,
        # Star count statistics across all windows.
        "total_stars_input": total_stars_input,
        "total_stars_on_detector": total_stars_on_detector,
        # Density statistics across windows with defined density values.
        "density_stats_on_detector": {
            "per_deg2_median": dens_median,
            "per_deg2_p10": dens_p10,
            "per_deg2_p90": dens_p90,
        },
    }

    # ------------------------------------------------------------------
    # 8) Build and return the final per-observer entry
    # ------------------------------------------------------------------

    obs_star_entry: Dict[str, Any] = {
        # Observer name (key used in obs_star_projections).
        "observer_name": obs_name,
        # Sensor geometry for this observer.
        "rows": rows,
        "cols": cols,
        # Catalog metadata for the Gaia queries used.
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "mag_limit_sensor_G": mag_limit_sensor_G,
        # Metadata and statistics for this projection run.
        "run_meta": run_meta,
        # List of per-window StarWindowProjection dicts.
        "windows": window_projections,
    }

    return obs_star_entry

def build_star_projections_for_all_observers(
    frames_with_sky: Dict[str, Dict[str, Any]],
    gaia_cache: Dict[str, Dict[str, Any]],
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
    frames_source_file: Optional[str] = None,
    gaia_cones_file: Optional[str] = None,
    observer_tracks_file: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build star projections for all observers.

    This function orchestrates the per-observer star projection flow:

        * Builds WCS entries for all observers using
          :func:`_build_wcs_for_all_observers`.

        * Iterates over all observers present in frames_with_sky.

        * For each observer:

            - Checks whether the observer also exists in gaia_cache
              and obs_tracks.

            - If either is missing, logs a warning and skips the observer.

            - Otherwise, calls :func:`build_star_projections_for_observer`
              to obtain the per-observer star projection entry.

            - Inserts the entry into the obs_star_projections dict.

        * After constructing each per-observer entry, this function also
          fills in the run_meta["frames_source_file"], run_meta["gaia_cones_file"],
          and run_meta["observer_tracks_file"] fields using the paths passed
          from :func:`main`.

    Parameters
    ----------
    frames_with_sky : dict
        Dictionary keyed by observer name, as loaded from
        obs_target_frames_ranked_with_sky.pkl.
    gaia_cache : dict
        Dictionary keyed by observer name, as loaded from obs_gaia_cones.pkl.
    obs_tracks : dict
        Dictionary keyed by observer name, as loaded from
        observer_tracks_with_pointing.pkl.
    sensor_config : SensorConfig
        Active sensor configuration (rows, cols, etc.).
    logger : logging.Logger
        Logger for emitting summaries.
    frames_source_file : str or None, optional
        Path to the frames-with-sky source file for run_meta embedding.
    gaia_cones_file : str or None, optional
        Path to the Gaia cones source file for run_meta embedding.
    observer_tracks_file : str or None, optional
        Path to the observer tracks source file for run_meta embedding.

    Returns
    -------
    dict
        obs_star_projections dict keyed by observer name.
    """
    # ------------------------------------------------------------------
    # 1) Build WCS entries for all observers using their tracks
    # ------------------------------------------------------------------
    # The WCS builder derives NebulaWCS objects (static or time-varying)
    # from each observer's track and the active sensor configuration.
    wcs_map = _build_wcs_for_all_observers(
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
    )

    # Initialize the final mapping from observer name -> per-observer entry.
    obs_star_projections: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 2) Loop over observers that have frames-with-sky entries
    # ------------------------------------------------------------------
    for obs_name, frames_entry in frames_with_sky.items():
        # Ensure this observer has a Gaia cache entry; otherwise we cannot
        # project stars for its windows, so we log and skip.
        if obs_name not in gaia_cache:
            logger.warning(
                "Observer '%s' is present in frames_with_sky but not in gaia_cache; skipping.",
                obs_name,
            )
            continue

        # Ensure this observer has an associated track (and thus pointing).
        if obs_name not in obs_tracks:
            logger.warning(
                "Observer '%s' is present in frames_with_sky but not in obs_tracks; skipping.",
                obs_name,
            )
            continue

        # It is expected that wcs_map has the same keys as obs_tracks, but
        # we guard against mismatches for robustness.
        if obs_name not in wcs_map:
            logger.warning(
                "Observer '%s' is present in obs_tracks but missing from WCS map; skipping.",
                obs_name,
            )
            continue

        # Retrieve the Gaia cones entry, observer track, and WCS entry.
        gaia_obs_entry = gaia_cache[obs_name]
        obs_track = obs_tracks[obs_name]
        nebula_wcs_entry = wcs_map[obs_name]

        # ------------------------------------------------------------------
        # 3) Build per-observer star projections via helper
        # ------------------------------------------------------------------
        # This call iterates over windows, performs per-window projections,
        # and returns a single summary dict for this observer.
        obs_star_entry = build_star_projections_for_observer(
            obs_name=obs_name,
            frames_entry=frames_entry,
            gaia_obs_entry=gaia_obs_entry,
            obs_track=obs_track,
            nebula_wcs_entry=nebula_wcs_entry,
            sensor_config=sensor_config,
            logger=logger,
        )

        # ------------------------------------------------------------------
        # 4) Patch provenance (source file paths) into run_meta
        # ------------------------------------------------------------------
        # Pull out run_meta (created by build_star_projections_for_observer)
        # and fill in the file paths used to load inputs.
        run_meta = obs_star_entry.get("run_meta", {})
        run_meta["frames_source_file"] = frames_source_file
        run_meta["gaia_cones_file"] = gaia_cones_file
        run_meta["observer_tracks_file"] = observer_tracks_file
        obs_star_entry["run_meta"] = run_meta

        # Store this observer's star projections in the final dict.
        obs_star_projections[obs_name] = obs_star_entry

    # ------------------------------------------------------------------
    # 5) Return star projections for all observers
    # ------------------------------------------------------------------
    return obs_star_projections

# def _select_wcs_for_window(
#     nebula_wcs_entry: Any,
#     obs_track: Dict[str, Any],
#     window_entry: Dict[str, Any],
# ) -> Any:
#     """
#     Select the appropriate WCS snapshot for a given window.

#     Parameters
#     ----------
#     nebula_wcs_entry : object or sequence of objects
#         The WCS information associated with this observer. This is either:
#         - A single WCS-like instance (static pointing), or
#         - A list/tuple/array of WCS-like objects, one per coarse timestep.
#     obs_track : dict
#         Observer track dictionary from observer_tracks_with_pixels.pkl.
#         May or may not contain a 't_mjd_utc' or 't_mjd' array.
#     window_entry : dict
#         One window entry from frames-with-sky, as produced by
#         NEBULA_SKY_SELECTOR. Must contain at least 'start_index' and
#         'end_index'. Optionally contains 't_mid_mjd_utc'.

#     Returns
#     -------
#     wcs_mid : object
#         The WCS object to use at the mid-point of this window.
#     """

#     # ------------------------------------------------------------------
#     # 1) If this is already a single static WCS-like object, just use it
#     # ------------------------------------------------------------------
#     # Anything that is not a list/tuple/ndarray is treated as “one WCS”.
#     if not isinstance(nebula_wcs_entry, (list, tuple, np.ndarray)):
#         return nebula_wcs_entry

#     # ------------------------------------------------------------------
#     # 2) Otherwise, interpret as a sequence of WCS objects
#     # ------------------------------------------------------------------
#     wcs_list = list(nebula_wcs_entry)
#     if len(wcs_list) == 0:
#         raise ValueError("nebula_wcs_entry sequence is empty; cannot select WCS.")

#     # ------------------------------------------------------------------
#     # 3) Try to select by time if a coarse MJD grid is available
#     # ------------------------------------------------------------------
#     idx_mid: Optional[int] = None

#     t_grid_mjd = obs_track.get("t_mjd_utc", None)
#     if t_grid_mjd is None:
#         # Some older pipelines may have used 't_mjd' instead
#         t_grid_mjd = obs_track.get("t_mjd", None)

#     t_mid_mjd = window_entry.get("t_mid_mjd_utc", None)

#     if t_grid_mjd is not None and t_mid_mjd is not None:
#         # Use nearest coarse time index to mid-window epoch
#         t_coarse = np.asarray(t_grid_mjd, dtype=float)
#         t_mid_val = float(t_mid_mjd)
#         idx_mid = int(np.argmin(np.abs(t_coarse - t_mid_val)))

#     # ------------------------------------------------------------------
#     # 4) Fallback: select by coarse index mid-point if no MJD grid exists
#     # ------------------------------------------------------------------
#     if idx_mid is None:
#         start_idx = window_entry.get("start_index", None)
#         end_idx = window_entry.get("end_index", None)

#         if start_idx is not None and end_idx is not None:
#             # Midpoint of the coarse indices that make up this window
#             idx_mid = (int(start_idx) + int(end_idx)) // 2
#         else:
#             # Ultimate fallback: just use the very first WCS
#             idx_mid = 0

#     # Clamp index to valid range
#     if idx_mid < 0:
#         idx_mid = 0
#     if idx_mid >= len(wcs_list):
#         idx_mid = len(wcs_list) - 1

#     return wcs_list[idx_mid]

def _select_wcs_for_window(
    nebula_wcs_entry: Any,
    obs_track: Dict[str, Any],
    window_entry: Dict[str, Any],
) -> Any:
    """
    Select the appropriate WCS snapshot for a given window.

    Parameters
    ----------
    nebula_wcs_entry : NebulaWCS or astropy.wcs.WCS or sequence of NebulaWCS
        The WCS information associated with this observer. This is either:
        - A single NebulaWCS/WCS instance (static pointing), or
        - A list/tuple/array of NebulaWCS objects, one per coarse timestep.
    obs_track : dict
        Observer track dictionary from observer_tracks_with_pixels.pkl.
        May or may not contain a 't_mjd_utc' or 't_mjd' array.
    window_entry : dict
        One window entry from frames-with-sky, as produced by
        NEBULA_SKY_SELECTOR. Must contain at least 'start_index' and
        'end_index'. Optionally contains 't_mid_mjd_utc'.

    Returns
    -------
    wcs_mid : NebulaWCS or astropy.wcs.WCS
        The WCS object to use at the mid-point of this window.
    """

    # ------------------------------------------------------------------
    # 1) If this is already a single static WCS, just return it
    # ------------------------------------------------------------------
    if isinstance(nebula_wcs_entry, (NebulaWCS, WCS)):
        return nebula_wcs_entry

    # ------------------------------------------------------------------
    # 2) Otherwise, interpret as a sequence of WCS objects
    # ------------------------------------------------------------------
    if not isinstance(nebula_wcs_entry, (list, tuple, np.ndarray)):
        raise TypeError(
            "nebula_wcs_entry must be a NebulaWCS/WCS or a sequence of "
            "NebulaWCS objects; got type={!r}".format(type(nebula_wcs_entry))
        )

    wcs_list = list(nebula_wcs_entry)
    if len(wcs_list) == 0:
        raise ValueError("nebula_wcs_entry sequence is empty; cannot select WCS.")

    # ------------------------------------------------------------------
    # 3) Try to select by time if a coarse MJD grid is available
    # ------------------------------------------------------------------
    idx_mid: Optional[int] = None

    t_grid_mjd = obs_track.get("t_mjd_utc", None)
    if t_grid_mjd is None:
        # Some older pipelines may have used 't_mjd' instead
        t_grid_mjd = obs_track.get("t_mjd", None)

    t_mid_mjd = window_entry.get("t_mid_mjd_utc", None)

    if t_grid_mjd is not None and t_mid_mjd is not None:
        # Use nearest coarse time index to mid-window epoch
        t_coarse = np.asarray(t_grid_mjd, dtype=float)
        t_mid_val = float(t_mid_mjd)
        idx_mid = int(np.argmin(np.abs(t_coarse - t_mid_val)))

    # ------------------------------------------------------------------
    # 4) Fallback: select by coarse index mid-point if no MJD grid exists
    # ------------------------------------------------------------------
    if idx_mid is None:
        start_idx = window_entry.get("start_index", None)
        end_idx = window_entry.get("end_index", None)

        if start_idx is not None and end_idx is not None:
            # Midpoint of the coarse indices that make up this window
            idx_mid = (int(start_idx) + int(end_idx)) // 2
        else:
            # Ultimate fallback: just use the very first WCS
            idx_mid = 0

    # Clamp index to valid range
    if idx_mid < 0:
        idx_mid = 0
    if idx_mid >= len(wcs_list):
        idx_mid = len(wcs_list) - 1

    return wcs_list[idx_mid]

def main(
    sensor_config: Optional[SensorConfig] = None,
    frames_path: Optional[str] = None,
    gaia_cache_path: Optional[str] = None,
    obs_tracks_path: Optional[str] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Entry point for NEBULA_STAR_PROJECTION.

    This function ties together the star-projection stage for the entire
    simulation. It:

        1. Resolves the active sensor configuration (defaults to
           Configuration.NEBULA_SENSOR_CONFIG.ACTIVE_SENSOR if not
           provided).

        2. Loads the three required inputs from disk:

              * frames_with_sky:
                    obs_target_frames_ranked_with_sky.pkl

              * gaia_cache:
                    obs_gaia_cones.pkl

              * obs_tracks:
                    observer_tracks_with_pointing.pkl

           using the existing helper wrappers and NEBULA_QUERY_GAIA
           where possible.

        3. Calls :func:`build_star_projections_for_all_observers` to:

              * Build WCS objects for each observer (via NEBULA_WCS).
              * Project Gaia stars into pixel coordinates for each
                observer/window.
              * Compute per-window and per-observer star statistics.
              * Embed basic provenance paths into run_meta.

        4. Writes the resulting obs_star_projections dict to disk using
           :func:`_save_star_projection_cache` and returns the same dict.

    Parameters
    ----------
    sensor_config : SensorConfig or None, optional
        Sensor configuration to use for projection. If None, the function
        uses ACTIVE_SENSOR imported from Configuration.NEBULA_SENSOR_CONFIG.
        If both are unavailable, a RuntimeError is raised.
    frames_path : str or None, optional
        Path to obs_target_frames_ranked_with_sky.pkl. If None, a default
        is computed via :func:`_resolve_default_frames_path`. The effective
        path is recorded in run_meta["frames_source_file"].
    gaia_cache_path : str or None, optional
        Location of the Gaia cones cache. Two usage patterns are supported:

            * If None:
                  The function uses NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR and
                  NEBULA_STAR_CATALOG.name to infer the default STARS directory
                  and assumes the file name "obs_gaia_cones.pkl".

            * If a directory:
                  Treated as the STARS directory; the file is assumed to be
                  "<gaia_cache_path>/obs_gaia_cones.pkl".

            * If a file path:
                  Treated as the full path to the Gaia cache pickle. In this
                  case, the file is loaded directly with pickle rather than
                  via NEBULA_QUERY_GAIA.load_gaia_cache (to avoid assumptions
                  about file naming).

        The effective file path is recorded in
        run_meta["gaia_cones_file"].
    obs_tracks_path : str or None, optional
        Path to observer_tracks_with_pointing.pkl. If None, a default is
        computed via :func:`_resolve_default_obs_tracks_path`. The effective
        path is recorded in run_meta["observer_tracks_file"].
    output_path : str or None, optional
        Path where obs_star_projections.pkl should be written. If None,
        a default is computed via :func:`_resolve_default_output_path`.
        The chosen path is logged but not embedded into run_meta.
    logger : logging.Logger or None, optional
        Logger to use. If None, a simple module-level logger named
        "NEBULA_STAR_PROJECTION" is created.

    Returns
    -------
    dict
        obs_star_projections dictionary keyed by observer name. Each entry
        is the per-observer structure returned by
        :func:`build_star_projections_for_observer`, with run_meta fields
        patched to include the source file paths.
    """
    # --------------------------------------------------------------
    # 1) Initialize logger if the caller did not supply one
    # --------------------------------------------------------------
    if logger is None:
        # Get/create a logger specific to this module.
        logger = logging.getLogger("NEBULA_STAR_PROJECTION")

        # If no handlers exist yet, configure a basic stream handler.
        if not logger.handlers:
            # Create a handler that writes to stderr.
            handler = logging.StreamHandler()
            # Define a simple log format (time, level, name, message).
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            # Attach the formatter to the handler.
            handler.setFormatter(formatter)
            # Add the handler to the logger.
            logger.addHandler(handler)

        # Set a reasonable default logging level.
        logger.setLevel(logging.INFO)

    # --------------------------------------------------------------
    # 2) Resolve the active sensor configuration
    # --------------------------------------------------------------
    if sensor_config is None:
        # If no explicit sensor_config was passed, fall back to ACTIVE_SENSOR.
        if ACTIVE_SENSOR is None:
            # Without a sensor, we cannot build WCS or project stars.
            raise RuntimeError(
                "NEBULA_STAR_PROJECTION: no sensor_config provided and "
                "ACTIVE_SENSOR is None. Please supply a SensorConfig or "
                "define ACTIVE_SENSOR in NEBULA_SENSOR_CONFIG."
            )
        # Use the globally configured active sensor.
        sensor_config = ACTIVE_SENSOR

    # Log a brief summary of the sensor being used.
    logger.info(
        "NEBULA_STAR_PROJECTION: using sensor '%s' (%d x %d pixels).",
        getattr(sensor_config, "name", "<unknown>"),
        int(getattr(sensor_config, "rows", getattr(sensor_config, "n_rows", 0))),
        int(getattr(sensor_config, "cols", getattr(sensor_config, "n_cols", 0))),
    )

    # --------------------------------------------------------------
    # 3) Resolve and load frames_with_sky
    # --------------------------------------------------------------
    # If no frames_path was provided, compute a default using the helper.
    if frames_path is None:
        frames_path = _resolve_default_frames_path()
    # Normalize to an absolute path for provenance.
    frames_source_file = os.path.abspath(frames_path)

    # Use the existing wrapper (which delegates to NEBULA_TARGET_PHOTONS)
    # to load obs_target_frames_ranked_with_sky.pkl.
    frames_with_sky = _load_frames_with_sky_from_disk(
        frames_path=frames_source_file,
        logger=logger,
    )

    # --------------------------------------------------------------
    # 4) Resolve and load Gaia cones cache (gaia_cache)
    # --------------------------------------------------------------
    use_query_gaia_loader = False  # whether to call NEBULA_QUERY_GAIA.load_gaia_cache

    if gaia_cache_path is None:
        # No path provided: infer default STARS directory directly
        # from NEBULA_OUTPUT_DIR and the catalog name.
        catalog_name = getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")
        stars_dir = os.path.join(NEBULA_OUTPUT_DIR, "STARS", catalog_name)

        # The Gaia cache file is assumed to be named obs_gaia_cones.pkl.
        gaia_cones_file = os.path.join(stars_dir, "obs_gaia_cones.pkl")

        # In this default case, we can safely use NEBULA_QUERY_GAIA.load_gaia_cache.
        use_query_gaia_loader = True

    else:
        # A path was provided. It may be a directory (STARS dir) or a file.
        if os.path.isdir(gaia_cache_path):
            # Treat the argument as the STARS directory.
            stars_dir = gaia_cache_path
            # Assume standard file name inside that directory.
            gaia_cones_file = os.path.join(stars_dir, "obs_gaia_cones.pkl")
            # We can use NEBULA_QUERY_GAIA.load_gaia_cache in this case.
            use_query_gaia_loader = True
        else:
            # Treat the argument as the full file path to the Gaia cache.
            gaia_cones_file = gaia_cache_path
            # Derive a directory for logging/debugging only.
            stars_dir = os.path.dirname(gaia_cones_file) or "."
            # Since the file name may be non-standard, we will load it
            # directly with pickle instead of using load_gaia_cache().
            use_query_gaia_loader = False

    # Normalize Gaia cones file path for provenance.
    gaia_cones_file = os.path.abspath(gaia_cones_file)

    # Now load the gaia_cache using NEBULA_QUERY_GAIA if available, or
    # fall back to a direct pickle load.
    if use_query_gaia_loader and (NEBULA_QUERY_GAIA is not None) and hasattr(
        NEBULA_QUERY_GAIA, "load_gaia_cache"
    ):
        # Use the dedicated loader from NEBULA_QUERY_GAIA, which performs
        # schema and catalog consistency checks.
        gaia_cache = NEBULA_QUERY_GAIA.load_gaia_cache(
            stars_dir=stars_dir,
            logger=logger,
        )
    else:
        # Fall back to a simple pickle load for the Gaia cache file.
        if not os.path.exists(gaia_cones_file):
            raise FileNotFoundError(
                f"NEBULA_STAR_PROJECTION: Gaia cache file not found: {gaia_cones_file}"
            )

        logger.info(
            "NEBULA_STAR_PROJECTION: loading Gaia cache directly from '%s'.",
            gaia_cones_file,
        )
        with open(gaia_cones_file, "rb") as f:
            gaia_cache = pickle.load(f)

    # --------------------------------------------------------------
    # 5) Resolve and load observer_tracks_with_pointing
    # --------------------------------------------------------------
    # If no path was provided, compute a default using the helper.
    if obs_tracks_path is None:
        obs_tracks_path = _resolve_default_obs_tracks_path()
    # Normalize to an absolute path for provenance.
    observer_tracks_file = os.path.abspath(obs_tracks_path)

    # Use the existing wrapper (which delegates to your pointing pickler)
    # to load observer_tracks_with_pointing.pkl.
    obs_tracks = _load_observer_tracks_with_pointing_from_disk(
        obs_tracks_path=observer_tracks_file,
        logger=logger,
    )

    # --------------------------------------------------------------
    # 6) Build star projections for all observers
    # --------------------------------------------------------------
    # This call:
    #   * Builds WCS objects via _build_wcs_for_all_observers / NEBULA_WCS.
    #   * Iterates over observers and windows.
    #   * Projects Gaia stars onto the sensor for each window.
    #   * Computes per-window and per-observer star statistics.
    #   * Embeds the three source file paths into run_meta for each observer.
    obs_star_projections = build_star_projections_for_all_observers(
        frames_with_sky=frames_with_sky,
        gaia_cache=gaia_cache,
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
        frames_source_file=frames_source_file,
        gaia_cones_file=gaia_cones_file,
        observer_tracks_file=observer_tracks_file,
    )

    # --------------------------------------------------------------
    # 7) Save obs_star_projections to disk
    # --------------------------------------------------------------
    # Write the obs_star_projections dictionary to disk, computing a
    # default output path if needed. The helper logs a summary including
    # observers, windows, and on-detector star counts.
    final_output_path = _save_star_projection_cache(
        obs_star_projections=obs_star_projections,
        output_path=output_path,
        logger=logger,
    )

    # Log where the final star projection cache was written.
    logger.info(
        "NEBULA_STAR_PROJECTION: star projections written to '%s'.",
        final_output_path,
    )

    # --------------------------------------------------------------
    # 8) Return the in-memory obs_star_projections dict
    # --------------------------------------------------------------
    return obs_star_projections





# def _load_gaia_cache_from_disk(
#     gaia_cache_path: Optional[str],
#     logger: logging.Logger,
# ) -> Dict[str, Dict[str, Any]]:
#     """
#     Load obs_gaia_cones from disk.

#     This thin wrapper:

#         * Resolves a default Gaia cache path via NEBULA_PATH_CONFIG
#           and NEBULA_STAR_CATALOG if ``gaia_cache_path`` is None.

#         * Delegates the actual loading to NEBULA_QUERY_GAIA, which
#           owns the Gaia cones cache loading logic.

#     Parameters
#     ----------
#     gaia_cache_path : str or None
#         Path to 'obs_gaia_cones.pkl'. If None, a default is computed
#         via :func:`_resolve_default_gaia_cache_path`.
#     logger : logging.Logger
#         Logger for emitting informational messages.

#     Returns
#     -------
#     dict
#         Dictionary keyed by observer name, each entry being the Gaia cones
#         cache structure for that observer.

#     Raises
#     ------
#     RuntimeError
#         If NEBULA_QUERY_GAIA or its loader function cannot be found.
#     """
#     # If gaia_cache_path is None, compute a default using the path config helper.
#     if gaia_cache_path is None:
#         gaia_cache_path = _resolve_default_gaia_cache_path()

#     # Ensure that the NEBULA_QUERY_GAIA module was successfully imported.
#     if NEBULA_QUERY_GAIA is None:
#         raise RuntimeError(
#             "NEBULA_QUERY_GAIA could not be imported. "
#             "Please update the import path in NEBULA_STAR_PROJECTION.py "
#             "to point to your actual NEBULA_QUERY_GAIA module."
#         )

#     # Attempt to retrieve the Gaia cache loader function.
#     load_func = getattr(
#         NEBULA_QUERY_GAIA,
#         "load_obs_gaia_cones",
#         None,
#     )

#     # If the loader function is missing, raise with a helpful message.
#     if load_func is None:
#         raise RuntimeError(
#             "NEBULA_QUERY_GAIA.load_obs_gaia_cones is not defined. "
#             "Please either implement it or adjust this wrapper to call your "
#             "actual Gaia cones cache loader."
#         )

#     # Log that we are loading the Gaia cones cache from disk.
#     logger.info("Loading Gaia cones cache from '%s'.", gaia_cache_path)

#     # Call the loader function to obtain the Gaia cache dictionary.
#     gaia_cache = load_func(gaia_cache_path, logger=logger)

#     # Compute the number of observers contained in the Gaia cache.
#     n_observers = len(gaia_cache)

#     # Compute the total number of windows across all observers for logging.
#     total_windows = sum(
#         len(entry.get("windows", [])) for entry in gaia_cache.values()
#     )

#     # Compute the total number of stars across all windows, if n_rows is available.
#     total_stars = 0
#     for entry in gaia_cache.values():
#         for w in entry.get("windows", []):
#             total_stars += int(w.get("n_rows", 0))

#     # Log a summary of the Gaia cache contents.
#     logger.info(
#         "Loaded Gaia cache for %d observers (%d windows, %d stars total in cones).",
#         n_observers,
#         total_windows,
#         total_stars,
#     )

#     # Return the loaded Gaia cache dictionary.
#     return gaia_cache
# def _resolve_default_gaia_cache_path() -> str:
#     """
#     Resolve the default path to ``obs_gaia_cones.pkl``.

#     This helper assumes that the NEBULA pipeline has been run via
#     ``sim_test.py`` with ``RUN_GAIA_PIPELINE=True`` so that
#     :mod:`Utility.STARS.NEBULA_QUERY_GAIA` has written the Gaia cones
#     cache to disk.

#     In that workflow, :func:`NEBULA_QUERY_GAIA.main` writes
#     ``obs_gaia_cones.pkl`` under::

#         NEBULA_OUTPUT_DIR / "STARS" / NEBULA_STAR_CATALOG.name

#     This function simply reconstructs that default path using the
#     configured ``NEBULA_OUTPUT_DIR`` and the current
#     ``NEBULA_STAR_CATALOG.name``.

#     Returns
#     -------
#     str
#         Absolute path to the Gaia cones cache pickle
#         (``obs_gaia_cones.pkl``).
#     """
#     # Use the same directory layout as NEBULA_QUERY_GAIA:
#     #   NEBULA_OUTPUT_DIR / "STARS" / <catalog_name> / "obs_gaia_cones.pkl"
#     catalog_name = NEBULA_STAR_CATALOG.name
#     default_path = os.path.join(
#         NEBULA_OUTPUT_DIR,
#         "STARS",
#         catalog_name,
#         "obs_gaia_cones.pkl",
#     )
#     return default_path


# def _select_wcs_for_window(
#     nebula_wcs_entry: Any,
#     obs_track: Dict[str, Any],
#     window_entry: Dict[str, Any],
# ) -> Any:
#     """
#     Select the NebulaWCS snapshot to use for a given window.

#     WCS selection semantics
#     -----------------------
#     For v1, we assume that :func:`build_wcs_for_observer` returns either:

#       * a single NebulaWCS instance, valid for all times
#         (static pointing case), or

#       * a sequence (list, tuple, etc.) of NebulaWCS objects aligned
#         with the observer's coarse time grid ``obs_track["t_mjd_utc"]``.

#     For each window, we then:

#       * read ``t_mid_mjd_utc`` from the window entry,

#       * find the index ``i`` where ``obs_track["t_mjd_utc"][i]`` is
#         closest to ``t_mid_mjd_utc`` in absolute value, and

#       * return ``nebula_wcs_entry[i]`` when ``nebula_wcs_entry`` is a
#         sequence. If ``nebula_wcs_entry`` is a single object, we simply
#         return it for all windows.

#     This mirrors the coarse time / WCS selection used for target pixel
#     projection, ensuring that star and target coordinates land on the
#     same detector geometry.

#     Parameters
#     ----------
#     nebula_wcs_entry : object or sequence of objects
#         WCS entry returned by :func:`build_wcs_for_observer`. May be a
#         single NebulaWCS or a sequence aligned with the coarse time grid.
#     obs_track : dict
#         Observer track containing at least the ``"t_mjd_utc"`` key.
#     window_entry : dict
#         Window entry containing at least ``"t_mid_mjd_utc"``.

#     Returns
#     -------
#     object
#         The selected NebulaWCS instance to be used for this window.

#     Raises
#     ------
#     RuntimeError
#         If the coarse time grid is empty or if the computed index is
#         out of range for the WCS sequence.
#     """
#     # If nebula_wcs_entry is not a list or tuple, treat it as a single WCS
#     # instance that is valid for all windows.
#     if not isinstance(nebula_wcs_entry, (list, tuple)):
#         return nebula_wcs_entry

#     # Extract the coarse time grid from the observer track. Let a missing
#     # key raise KeyError instead of silently proceeding with an empty array.
#     t_coarse = np.asarray(obs_track["t_mjd_utc"], dtype=float)

#     # Ensure the window has a mid-time defined.
#     if "t_mid_mjd_utc" not in window_entry:
#         raise RuntimeError(
#             "Window entry is missing 't_mid_mjd_utc'; cannot select WCS."
#         )

#     # Extract the window mid-time in MJD UTC and cast it to float.
#     t_mid = float(window_entry["t_mid_mjd_utc"])

#     # If the coarse time grid is empty, we cannot select a WCS snapshot.
#     if t_coarse.size == 0:
#         raise RuntimeError(
#             "Observer track has empty 't_mjd_utc' array; cannot select WCS."
#         )

#     # Compute the absolute time difference between t_mid and each coarse time.
#     dt = np.abs(t_coarse - t_mid)

#     # Find the index of the minimum time difference (closest coarse snapshot).
#     idx = int(np.argmin(dt))

#     # Ensure that the index is within the range of the WCS sequence.
#     if idx < 0 or idx >= len(nebula_wcs_entry):
#         raise RuntimeError(
#             f"Selected WCS index {idx} out of range for nebula_wcs_entry "
#             f"of length {len(nebula_wcs_entry)}."
#         )

#     # Return the WCS snapshot at the selected index.
#     return nebula_wcs_entry[idx]

