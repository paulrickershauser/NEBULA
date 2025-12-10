"""
NEBULA_STAR_SLEW_PROJECTION
===========================

Purpose
-------
Star track builder for *non-sidereal* (slewing) observers.

This module takes:

    1. Per-window star projections from NEBULA_STAR_PROJECTION
       (obs_star_projections[obs_name]["windows"][...]),
       where each window has:
           - window_index, start_index, end_index, n_frames
           - t_mid_utc, t_mid_mjd_utc
           - sky selection metadata
           - a "stars" dict keyed by Gaia source ID string, with
             per-star RA/Dec and mid-window pixel position.

    2. Observer tracks with pointing vs coarse time, typically the same
       obs_tracks[obs_name] dictionaries used by NEBULA_WCS, carrying:
           - "t_mjd_utc" (coarse time grid)
           - "pointing_boresight_ra_deg"
           - "pointing_boresight_dec_deg"
           - "roll_deg"
           - (and other fields from NEBULA_PIXEL_PICKLER / schedule picklers)

    3. A SensorConfig describing the active sensor geometry.

It then builds, for each observer and window, a *time-resolved* star
track catalog that describes how each Gaia star moves across the
detector during that window as the observer slews, assuming the star's
RA/Dec is effectively fixed over the duration of the window.

The result is a dictionary:

    obs_star_slew_tracks[obs_name] = {
        "observer_name": ...,
        "rows": ...,
        "cols": ...,
        "catalog_name": ...,
        "catalog_band": ...,
        "run_meta": {...},
        "windows": [
            {
                "window_index": ...,
                "start_index": ...,
                "end_index": ...,
                "n_frames": ...,
                "coarse_indices": np.ndarray[int],
                "t_mjd_utc": np.ndarray[float],
                "stars": {
                    source_id_str: {
                        "gaia_source_id": int,
                        "source_id": str,
                        "source_type": "star",
                        "mag_G": float,
                        "ra_deg_ref": float,
                        "dec_deg_ref": float,
                        "coarse_indices": np.ndarray[int],
                        "t_mjd_utc": np.ndarray[float],
                        "x_pix": np.ndarray[float],
                        "y_pix": np.ndarray[float],
                        "on_detector": np.ndarray[bool],
                    },
                    ...
                },
            },
            ...
        ],
    }

This is intended to be consumed later by modules that convert star
tracks into photons per frame, or that visualize star smear across the
sensor during slewing motions.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging
import os
import pickle

import numpy as np

# Astropy time is optional here; we mainly care about MJD floats from obs_tracks.
from astropy.time import Time  # noqa: F401  # kept for future extensions

# ---------------------------------------------------------------------------
# NEBULA configuration imports
# ---------------------------------------------------------------------------

# Base output directory, etc.
from Configuration import NEBULA_PATH_CONFIG  # type: ignore[attr-defined]

# Sensor configuration and active sensor selection.
from Configuration.NEBULA_SENSOR_CONFIG import (  # type: ignore[attr-defined]
    SensorConfig,
    ACTIVE_SENSOR,
)

# Star catalog metadata (for provenance only).
from Configuration.NEBULA_STAR_CONFIG import (  # type: ignore[attr-defined]
    NEBULA_STAR_CATALOG,
)
#Pixel pipeline paths (used to locate default observer tracks).
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER

# WCS builder for turning pointing into NebulaWCS objects.
from Utility.SENSOR.NEBULA_WCS import (  # type: ignore[attr-defined]
    build_wcs_for_observer,
    project_radec_to_pixels,
)


# ---------------------------------------------------------------------------
# Type aliases and version tag
# ---------------------------------------------------------------------------

# Per-window, per-star track dictionary type.
StarSlewWindow = Dict[str, Any]

# Top-level type: obs_star_slew_tracks[obs_name] -> dict
ObsStarSlewTracks = Dict[str, Dict[str, Any]]

# Version tag for this module's run_meta.
STAR_SLEW_PROJECTION_VERSION: str = "0.1"
# ---------------------------------------------------------------------------
# Default path resolvers
# ---------------------------------------------------------------------------


def _resolve_default_star_projection_path() -> str:
    """
    Resolve the default path to ``obs_star_projections.pkl``.

    Returns
    -------
    str
        Absolute path under ``NEBULA_OUTPUT/STARS/<catalog_name>``.
    """

    catalog_name = getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")
    return os.path.join(
        NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR,
        "STARS",
        catalog_name,
        "obs_star_projections.pkl",
    )


def _resolve_default_obs_tracks_path() -> str:
    """
    Resolve the default path to ``observer_tracks_with_pixels.pkl``.

    Returns
    -------
    str
        Path used by :mod:`Utility.SAT_OBJECTS.NEBULA_PIXEL_PICKLER`.
    """

    return NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH


def _resolve_default_output_path() -> str:
    """
    Resolve the default output path for ``obs_star_slew_projections.pkl``.

    Returns
    -------
    str
        Absolute path under ``NEBULA_OUTPUT/STARS/<catalog_name>``.
    """

    catalog_name = getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")
    return os.path.join(
        NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR,
        "STARS",
        catalog_name,
        "obs_star_slew_projections.pkl",
    )

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger for this module, creating a default one if needed.

    Parameters
    ----------
    logger : logging.Logger or None
        Existing logger to reuse. If None, a module-local logger is
        created with a simple StreamHandler.

    Returns
    -------
    logging.Logger
        Logger instance configured for NEBULA_STAR_SLEW_PROJECTION.
    """
    if logger is not None:
        return logger

    logger = logging.getLogger("NEBULA_STAR_SLEW_PROJECTION")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# I/O helpers: star projection cache and observer tracks
# ---------------------------------------------------------------------------

def _load_obs_star_projections(
    star_projection_path: str,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Load obs_star_projections (per-window mid-time star projections) from disk.

    Parameters
    ----------
    star_projection_path : str
        Absolute path to obs_star_projections.pkl. This should be passed
        explicitly by sim_test or the caller.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    dict
        obs_star_projections dict keyed by observer name.

    Raises
    ------
    ValueError
        If star_projection_path is an empty string.
    FileNotFoundError
        If the file does not exist at star_projection_path.
    """
    if not star_projection_path:
        raise ValueError("star_projection_path must be a non-empty string.")

    logger.info("Loading obs_star_projections from '%s'.", star_projection_path)
    with open(star_projection_path, "rb") as f:
        obs_star_projections = pickle.load(f)

    logger.info(
        "Loaded star projections for %d observers.",
        len(obs_star_projections),
    )
    return obs_star_projections


def _load_obs_tracks(
    obs_tracks_path: str,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Load observer tracks with pointing information from disk.

    Parameters
    ----------
    obs_tracks_path : str
        Absolute path to the observer_tracks_with_pointing (or
        observer_tracks_with_pixels) pickle. This should be passed
        explicitly by sim_test.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    dict
        obs_tracks dict keyed by observer name, each entry containing
        coarse times (t_mjd_utc) and pointing fields.

    Raises
    ------
    ValueError
        If obs_tracks_path is an empty string.
    FileNotFoundError
        If the file does not exist at obs_tracks_path.
    """
    if not obs_tracks_path:
        raise ValueError("obs_tracks_path must be a non-empty string.")

    logger.info("Loading observer tracks from '%s'.", obs_tracks_path)
    with open(obs_tracks_path, "rb") as f:
        obs_tracks = pickle.load(f)

    logger.info("Loaded observer tracks for %d observers.", len(obs_tracks))
    return obs_tracks


def _save_star_slew_tracks(
    obs_star_slew_tracks: ObsStarSlewTracks,
    output_path: str,
    logger: logging.Logger,
) -> str:
    """
    Save obs_star_slew_tracks to disk.

    Parameters
    ----------
    obs_star_slew_tracks : dict
        Dictionary keyed by observer name describing per-window star
        tracks during slewing.
    output_path : str
        Destination path for obs_star_slew_tracks.pkl. Should be
        provided explicitly by sim_test.
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    str
        Absolute path where the file was written.

    Raises
    ------
    ValueError
        If output_path is an empty string.
    """
    if not output_path:
        raise ValueError("output_path must be a non-empty string.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Writing obs_star_slew_tracks to '%s'.", output_path)
    with open(output_path, "wb") as f:
        pickle.dump(obs_star_slew_tracks, f)

    logger.info(
        "Saved star slew tracks for %d observers.",
        len(obs_star_slew_tracks),
    )
    return output_path


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def _build_wcs_for_all_observers(
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build NebulaWCS entries for all observers, reusing NEBULA_WCS.

    Parameters
    ----------
    obs_tracks : dict
        Observer tracks keyed by observer name, each entry containing
        pointing fields that NEBULA_WCS.build_wcs_for_observer expects.
    sensor_config : SensorConfig
        Sensor configuration used to define the WCS geometry.
    logger : logging.Logger
        Logger for informational messages.

    Returns
    -------
    dict
        Mapping from observer name -> NebulaWCS or sequence of NebulaWCS,
        exactly as returned by :func:`build_wcs_for_observer`.
    """
    wcs_map: Dict[str, Any] = {}

    for obs_name, obs_track in obs_tracks.items():
        logger.info(
            "Building WCS sequence for observer '%s' (star slew projection).",
            obs_name,
        )
        wcs_entry = build_wcs_for_observer(
            observer_track=obs_track,
            sensor_config=sensor_config,
        )
        wcs_map[obs_name] = wcs_entry

    return wcs_map


def _select_wcs_for_coarse_index(
    nebula_wcs_entry: Any,
    coarse_index: int,
) -> Any:
    """
    Select the NebulaWCS corresponding to a given coarse index.

    Semantics
    ---------
    We assume that:

        - build_wcs_for_observer(...) has returned either:
            * a single WCS (static pointing case), or
            * a sequence (list/tuple/ndarray) of WCS objects aligned
              with the coarse time grid obs_track["t_mjd_utc"].

        - The 'coarse_index' we are given corresponds to the index
          in that same coarse time grid (e.g., the same index used
          for NEBULA_PIXEL_PICKLER / frame building).

    Parameters
    ----------
    nebula_wcs_entry : object or sequence
        Output of build_wcs_for_observer for a single observer.
    coarse_index : int
        Coarse time index for which we want a WCS.

    Returns
    -------
    object
        Selected WCS object for that coarse index.
    """
    # Static WCS: use the same WCS for all indices.
    if not isinstance(nebula_wcs_entry, (list, tuple, np.ndarray)):
        return nebula_wcs_entry

    # Dynamic WCS: index into the list/array.
    if coarse_index < 0 or coarse_index >= len(nebula_wcs_entry):
        raise IndexError(
            f"Coarse index {coarse_index} out of range for WCS entry "
            f"of length {len(nebula_wcs_entry)}."
        )

    return nebula_wcs_entry[coarse_index]


def _build_star_tracks_for_window_slew(
    obs_name: str,
    window_projection: Dict[str, Any],
    obs_track: Dict[str, Any],
    nebula_wcs_entry: Any,
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> StarSlewWindow:
    """
    Build per-frame star tracks for a single window during observer slewing.

    This function takes one StarWindowProjection entry (from
    NEBULA_STAR_PROJECTION) and uses the observer's WCS sequence to
    compute, for each star in that window, its (x_pix, y_pix) position
    at each coarse timestep inside the window.

    Geometry model
    --------------
    For v1, we assume:

        * Star positions are *fixed* at a single reference epoch
          within the window (e.g., the mid-time RA/Dec stored in
          window_projection["stars"][sid]["ra_deg_mid"] /
          "dec_deg_mid").

        * All smear across the detector is due to changing pointing
          (boresight RA/Dec and roll) as encoded in the WCS sequence.

    Steps
    -----
        1. Read start_index, end_index from the window_projection and
           construct the array of coarse indices:

               idx_window = np.arange(start_index, end_index + 1)

           For each coarse index, read obs_track["t_mjd_utc"][idx].

        2. For each star in window_projection["stars"]:

               - ra_ref = star["ra_deg_mid"]  (or "ra_deg_catalog" if needed)
               - dec_ref = star["dec_deg_mid"]

           Build a vector of RA/Dec repeated across all coarse indices
           (RA and Dec are effectively constant for this window).

        3. For each coarse index:

               - Select WCS via _select_wcs_for_coarse_index(...).
               - Project RA/Dec -> (x_pix, y_pix) using
                 project_radec_to_pixels or NebulaWCS.world_to_pixel.

           Collect per-frame positions into arrays x_pix[k], y_pix[k].

        4. Apply FOV mask using sensor rows/cols:

               0 <= x < n_cols, 0 <= y < n_rows

           Record on_detector[k] as a boolean time series.

    Parameters
    ----------
    obs_name : str
        Observer name (used for logging).
    window_projection : dict
        One StarWindowProjection entry from obs_star_projections[obs_name]
        describing a single window and its Gaia stars.
    obs_track : dict
        Observer track dict with at least "t_mjd_utc" (or legacy
        "t_mjd"), used to align coarse indices with timestamps.
    nebula_wcs_entry : object or sequence
        WCS entry for this observer (static or per-timestep sequence)
        returned by build_wcs_for_observer.
    sensor_config : SensorConfig
        Sensor configuration describing rows/cols.
    logger : logging.Logger
        Logger for debug output.

    Returns
    -------
    StarSlewWindow
        Dictionary describing per-star time series for this window,
        including per-frame coarse indices and times.

    Raises
    ------
    RuntimeError
        If t_mjd_utc is missing/empty, if the window index range is
        empty, or if sensor rows/cols cannot be determined, or if no
        projection method is available.
    IndexError
        If the window index range is out of bounds for t_mjd_utc.
    KeyError
        If required per-star fields (RA/Dec or gaia_source_id) are
        missing from the window_projection["stars"] entries.
    """
    # ------------------------------------------------------------------
    # 1) Extract coarse-index range for this window
    # ------------------------------------------------------------------
    # start_index and end_index are frame indices on the coarse time grid.
    start_index = int(window_projection.get("start_index"))
    end_index = int(window_projection.get("end_index"))

    # Build the inclusive index range [start_index, end_index].
    idx_window = np.arange(start_index, end_index + 1, dtype=int)

    # Guard against pathological windows with no indices (should not
    # happen in valid NEBULA windows, but better to fail clearly).
    if idx_window.size == 0:
        raise RuntimeError(
            f"Observer '{obs_name}' has empty index range "
            f"[{start_index}, {end_index}] for window "
            f"{window_projection.get('window_index')}."
        )

    # ------------------------------------------------------------------
    # 2) Extract t_mjd_utc (or legacy t_mjd) for all coarse times, then
    #    subset to this window
    # ------------------------------------------------------------------
    t_mjd_array = obs_track.get("t_mjd_utc")

    # Fall back to legacy 't_mjd' if the newer field is missing/empty.
    if t_mjd_array is None or (hasattr(t_mjd_array, "__len__") and len(t_mjd_array) == 0):
        t_mjd_array = obs_track.get("t_mjd")

    t_mjd_utc_all = np.asarray(t_mjd_array if t_mjd_array is not None else [], dtype=float)

    # If the coarse time grid is empty, we cannot build any tracks.
    if t_mjd_utc_all.size == 0:
        raise RuntimeError(
            f"Observer '{obs_name}' has empty 't_mjd_utc'/'t_mjd'; "
            "cannot build star slew tracks."
        )

    # Optional guard: ensure indices are within valid range [0, N-1].
    if start_index < 0 or end_index >= t_mjd_utc_all.size:
        raise IndexError(
            f"Observer '{obs_name}' window indices [{start_index}, {end_index}] "
            f"are out of bounds for t_mjd_utc size {t_mjd_utc_all.size}."
        )

    # Subset the coarse time grid to just this window.
    t_mjd_window = t_mjd_utc_all[idx_window]

    # ------------------------------------------------------------------
    # 3) Sensor geometry (rows, cols)
    # ------------------------------------------------------------------
    # Support both 'n_rows'/'n_cols' and 'rows'/'cols' attribute names.
    n_rows = int(getattr(sensor_config, "n_rows", getattr(sensor_config, "rows", 0)))
    n_cols = int(getattr(sensor_config, "n_cols", getattr(sensor_config, "cols", 0)))

    # Fail loudly if we cannot infer sensor geometry.
    if n_rows <= 0 or n_cols <= 0:
        raise RuntimeError(
            "SensorConfig must define positive 'n_rows'/'n_cols' (or 'rows'/'cols'); "
            f"got n_rows={n_rows}, n_cols={n_cols}."
        )

    # ------------------------------------------------------------------
    # 4) Build per-star tracks across the window
    # ------------------------------------------------------------------
    # Input stars are keyed by Gaia source ID string.
    stars_in = window_projection.get("stars", {})
    # Output container for per-star time series.
    stars_out: Dict[str, Dict[str, Any]] = {}

    # Loop over each star present in this window's mid-time projection.
    for sid_str, star in stars_in.items():
        # --------------------------------------------------------------
        # 4a) Reference RA/Dec for this window (e.g., mid-window RA/Dec)
        # --------------------------------------------------------------
        # Prefer mid-time RA/Dec; if not present, fall back to catalog.
        ra_ref_val = None
        dec_ref_val = None

        if "ra_deg_mid" in star and star["ra_deg_mid"] is not None:
            ra_ref_val = star["ra_deg_mid"]
        elif "ra_deg_catalog" in star and star["ra_deg_catalog"] is not None:
            ra_ref_val = star["ra_deg_catalog"]

        if "dec_deg_mid" in star and star["dec_deg_mid"] is not None:
            dec_ref_val = star["dec_deg_mid"]
        elif "dec_deg_catalog" in star and star["dec_deg_catalog"] is not None:
            dec_ref_val = star["dec_deg_catalog"]

        if ra_ref_val is None or dec_ref_val is None:
            raise KeyError(
                f"Star '{sid_str}' in observer '{obs_name}' window "
                f"{window_projection.get('window_index')} is missing both "
                "mid-window and catalog RA/Dec fields."
            )

        ra_ref = float(ra_ref_val)
        dec_ref = float(dec_ref_val)

        # Photometry and ID.
        mag_G = float(star.get("mag_G", np.nan))

        if "gaia_source_id" not in star or star["gaia_source_id"] is None:
            raise KeyError(
                f"Star '{sid_str}' in observer '{obs_name}' window "
                f"{window_projection.get('window_index')} is missing 'gaia_source_id'."
            )
        gaia_source_id = int(star["gaia_source_id"])

        # --------------------------------------------------------------
        # 4b) Allocate arrays for this star's track
        # --------------------------------------------------------------
        # One sample per coarse time index inside this window.
        x_pix = np.zeros_like(t_mjd_window, dtype=float)
        y_pix = np.zeros_like(t_mjd_window, dtype=float)
        on_detector = np.zeros_like(t_mjd_window, dtype=bool)

        # --------------------------------------------------------------
        # 4c) Loop over coarse indices and project RA/Dec -> pixels
        # --------------------------------------------------------------
        for i, coarse_idx in enumerate(idx_window):
            # Select the appropriate WCS snapshot for this coarse index.
            wcs_t = _select_wcs_for_coarse_index(
                nebula_wcs_entry=nebula_wcs_entry,
                coarse_index=int(coarse_idx),
            )

            # Project the (fixed) RA/Dec to pixel coordinates at this time.
            if project_radec_to_pixels is not None:
                # Use shared helper for RA/Dec -> pixel projection.
                x_i, y_i = project_radec_to_pixels(
                    wcs_t,
                    np.array([ra_ref], dtype=float),
                    np.array([dec_ref], dtype=float),
                )
            else:
                # Fallback: require that the WCS object exposes world_to_pixel.
                if not hasattr(wcs_t, "world_to_pixel"):
                    raise RuntimeError(
                        "Neither 'project_radec_to_pixels' nor 'world_to_pixel' "
                        "is available for RA/Dec -> pixel projection."
                    )
                x_i, y_i = wcs_t.world_to_pixel(
                    np.array([ra_ref], dtype=float),
                    np.array([dec_ref], dtype=float),
                )

            # Store scalar pixel positions for this timestep.
            x_pix[i] = float(x_i[0])
            y_pix[i] = float(y_i[0])

            # ----------------------------------------------------------
            # 4d) FOV mask for this timestep
            # ----------------------------------------------------------
            # On-detector if inside [0, n_cols) x [0, n_rows) in pixels.
            on_detector[i] = (
                (0.0 <= x_pix[i] < float(n_cols))
                and (0.0 <= y_pix[i] < float(n_rows))
            )

        # --------------------------------------------------------------
        # 4e) Pack per-star track entry
        # --------------------------------------------------------------
        star_track: Dict[str, Any] = {
            "gaia_source_id": gaia_source_id,
            "source_id": sid_str,
            "source_type": "star",
            "mag_G": mag_G,
            "ra_deg_ref": ra_ref,
            "dec_deg_ref": dec_ref,
            "coarse_indices": idx_window.copy(),
            "t_mjd_utc": t_mjd_window.copy(),
            "x_pix": x_pix,
            "y_pix": y_pix,
            "on_detector": on_detector,
        }

        # Attach this star's track to the output dict.
        stars_out[sid_str] = star_track

    # ------------------------------------------------------------------
    # 5) Assemble per-window output structure
    # ------------------------------------------------------------------
    window_out: StarSlewWindow = {
        "window_index": int(window_projection.get("window_index")),
        "start_index": start_index,
        "end_index": end_index,
        "n_frames": int(window_projection.get("n_frames")),
        "coarse_indices": idx_window,
        "t_mjd_utc": t_mjd_window,
        "stars": stars_out,
    }

    # Log a compact summary for debugging.
    logger.debug(
        "Observer '%s', window %d: %d stars with slew tracks.",
        obs_name,
        window_out["window_index"],
        len(stars_out),
    )

    return window_out


# ---------------------------------------------------------------------------
# Per-observer and all-observer orchestration
# ---------------------------------------------------------------------------

def build_star_slew_tracks_for_observer(
    obs_name: str,
    obs_star_entry: Dict[str, Any],
    obs_track: Dict[str, Any],
    nebula_wcs_entry: Any,
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build star slew tracks for all windows of a single observer.

    This function takes the per-observer star projection entry from
    NEBULA_STAR_PROJECTION (obs_star_projections[obs_name]) and, for
    each window, calls _build_star_tracks_for_window_slew to project
    each star through the observer's time-varying WCS sequence. The
    result is a per-observer container with one StarSlewWindow entry
    per window and a small run_meta block describing the slewing run.

    Parameters
    ----------
    obs_name : str
        Observer name.
    obs_star_entry : dict
        The obs_star_projections[obs_name] entry from NEBULA_STAR_PROJECTION,
        containing metadata and a "windows" list of StarWindowProjection
        dictionaries.
    obs_track : dict
        Observer track dictionary with 't_mjd_utc' and pointing fields.
    nebula_wcs_entry : object or sequence
        WCS entry for this observer returned by build_wcs_for_observer
        (either a single static WCS or a sequence aligned with t_mjd_utc).
    sensor_config : SensorConfig
        Sensor geometry configuration (rows/cols).
    logger : logging.Logger
        Logger for summaries and per-window progress.

    Returns
    -------
    dict
        Per-observer star slew track entry with the structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "catalog_name": str,
                "catalog_band": str,
                "run_meta": {
                    "version": str,
                    "n_windows": int,
                    "total_star_tracks": int,
                    "upstream_star_projection_meta": dict or None,
                },
                "windows": [StarSlewWindow, ...],
            }

    Raises
    ------
    RuntimeError
        If sensor rows/cols cannot be determined (both obs_star_entry
        and sensor_config fail to provide positive dimensions), or if
        any underlying call to _build_star_tracks_for_window_slew
        raises a RuntimeError (e.g., empty t_mjd_utc).
    IndexError
        If a window's index range is out of bounds for t_mjd_utc
        (propagated from _build_star_tracks_for_window_slew).
    KeyError
        If required per-star fields are missing from a window
        projection (also propagated from the per-window builder).
    """
    # ------------------------------------------------------------------
    # 1) Collect the list of windows to process for this observer
    # ------------------------------------------------------------------
    # This is the list of StarWindowProjection dicts created by
    # NEBULA_STAR_PROJECTION for this observer.
    windows_star: List[Dict[str, Any]] = list(obs_star_entry.get("windows", []))

    # Container for the per-window slew outputs and a track counter.
    slew_windows: List[StarSlewWindow] = []
    total_tracks = 0

    # ------------------------------------------------------------------
    # 2) Build slew tracks for each window
    # ------------------------------------------------------------------
    for window_projection in windows_star:
        window_index = int(window_projection.get("window_index"))
        logger.info(
            "Observer '%s': building star slew tracks for window %d.",
            obs_name,
            window_index,
        )

        # Delegate per-window geometry and projection logic.
        window_slew = _build_star_tracks_for_window_slew(
            obs_name=obs_name,
            window_projection=window_projection,
            obs_track=obs_track,
            nebula_wcs_entry=nebula_wcs_entry,
            sensor_config=sensor_config,
            logger=logger,
        )
        slew_windows.append(window_slew)
        total_tracks += len(window_slew.get("stars", {}))

    # ------------------------------------------------------------------
    # 3) Extract sensor dims and catalog metadata for provenance
    # ------------------------------------------------------------------
    # Prefer rows/cols recorded in the obs_star_entry; fall back to the
    # active SensorConfig if not present. If neither yields positive
    # values, treat this as a configuration error.
    rows_raw = obs_star_entry.get("rows", getattr(sensor_config, "rows", None))
    cols_raw = obs_star_entry.get("cols", getattr(sensor_config, "cols", None))

    # If the above did not find rows/cols, try n_rows/n_cols from the sensor.
    if rows_raw is None:
        rows_raw = getattr(sensor_config, "n_rows", 0)
    if cols_raw is None:
        cols_raw = getattr(sensor_config, "n_cols", 0)

    rows = int(rows_raw)
    cols = int(cols_raw)

    if rows <= 0 or cols <= 0:
        raise RuntimeError(
            "Failed to determine positive sensor dimensions for observer "
            f"'{obs_name}': rows={rows}, cols={cols}. "
            "Ensure obs_star_entry or SensorConfig defines rows/cols "
            "or n_rows/n_cols."
        )

    catalog_name = obs_star_entry.get(
        "catalog_name", getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG")
    )
    catalog_band = obs_star_entry.get(
        "catalog_band", getattr(NEBULA_STAR_CATALOG, "band", "G")
    )

    # ------------------------------------------------------------------
    # 4) Build run_meta for this slewing stage
    # ------------------------------------------------------------------
    run_meta = {
        "version": STAR_SLEW_PROJECTION_VERSION,
        "n_windows": len(slew_windows),
        "total_star_tracks": total_tracks,
        # Upstream provenance from NEBULA_STAR_PROJECTION, if present.
        "upstream_star_projection_meta": obs_star_entry.get("run_meta"),
    }

    # Log a compact summary for this observer.
    logger.info(
        "Observer '%s': built star slew tracks for %d windows "
        "(%d total star tracks).",
        obs_name,
        len(slew_windows),
        total_tracks,
    )

    # ------------------------------------------------------------------
    # 5) Assemble per-observer output structure
    # ------------------------------------------------------------------
    obs_slew_entry: Dict[str, Any] = {
        "observer_name": obs_name,
        "rows": rows,
        "cols": cols,
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "run_meta": run_meta,
        "windows": slew_windows,
    }

    return obs_slew_entry


def build_star_slew_tracks_for_all_observers(
    obs_star_projections: Dict[str, Dict[str, Any]],
    obs_tracks: Dict[str, Dict[str, Any]],
    sensor_config: SensorConfig,
    logger: logging.Logger,
) -> ObsStarSlewTracks:
    """
    Build star slew tracks for all observers present in obs_star_projections.

    This function orchestrates the slewing star-track build across all
    observers for which we have both:

        * A star projection entry from NEBULA_STAR_PROJECTION
          (obs_star_projections[obs_name]), and
        * A corresponding observer track with pointing information
          (obs_tracks[obs_name]).

    For each such observer, it:

        1. Uses _build_wcs_for_all_observers (and ultimately
           build_wcs_for_observer) to construct a static or time-varying
           WCS entry.
        2. Calls build_star_slew_tracks_for_observer(...) to build
           per-window star slewing tracks.

    Observers that are present in obs_star_projections but missing from
    obs_tracks or the WCS map are skipped with a warning. Likewise,
    observers present only in obs_tracks but not in obs_star_projections
    are implicitly ignored.

    Parameters
    ----------
    obs_star_projections : dict
        Output of NEBULA_STAR_PROJECTION, keyed by observer name. Each
        value is a per-observer dict with a "windows" list of
        StarWindowProjection entries and some metadata.
    obs_tracks : dict
        Observer tracks with pointing, keyed by observer name. Each
        entry must contain at least 't_mjd_utc' and the pointing fields
        expected by NEBULA_WCS.build_wcs_for_observer.
    sensor_config : SensorConfig
        Sensor configuration describing rows/cols/FOV. The same sensor
        is assumed for all observers in this call.
    logger : logging.Logger
        Logger for high-level summaries and warnings.

    Returns
    -------
    dict
        obs_star_slew_tracks dict keyed by observer name, where each
        value is the per-observer structure returned by
        build_star_slew_tracks_for_observer, i.e.:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "catalog_name": str,
                "catalog_band": str,
                "run_meta": {...},
                "windows": [StarSlewWindow, ...],
            }

    Raises
    ------
    RuntimeError
        If WCS construction fails for an observer, if sensor geometry
        is invalid for an observer, or if underlying calls to
        build_star_slew_tracks_for_observer/_build_star_tracks_for_window_slew
        encounter configuration errors.
    IndexError
        Propagated from per-window builders if a window index range is
        out of bounds for t_mjd_utc.
    KeyError
        Propagated from per-window builders if required star fields
        are missing from the projection cache.
    """
    # ------------------------------------------------------------------
    # 1) Build WCS entries for all observers present in obs_tracks
    # ------------------------------------------------------------------
    wcs_map = _build_wcs_for_all_observers(
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
    )

    # Container for all observers' slew outputs.
    obs_star_slew_tracks: ObsStarSlewTracks = {}

    # Optional: counters for a summary at the end.
    n_obs_processed = 0
    n_windows_total = 0
    n_tracks_total = 0

    # ------------------------------------------------------------------
    # 2) Loop over observers that have star projections
    # ------------------------------------------------------------------
    for obs_name, obs_star_entry in obs_star_projections.items():
        # Ensure this observer has corresponding tracks.
        if obs_name not in obs_tracks:
            logger.warning(
                "Observer '%s' present in obs_star_projections but missing "
                "from obs_tracks; skipping.",
                obs_name,
            )
            continue

        # Ensure we successfully built a WCS entry for this observer.
        if obs_name not in wcs_map:
            logger.warning(
                "Observer '%s' present in obs_tracks but missing from WCS map; "
                "skipping.",
                obs_name,
            )
            continue

        obs_track = obs_tracks[obs_name]
        nebula_wcs_entry = wcs_map[obs_name]

        # Build per-observer slew tracks.
        obs_slew_entry = build_star_slew_tracks_for_observer(
            obs_name=obs_name,
            obs_star_entry=obs_star_entry,
            obs_track=obs_track,
            nebula_wcs_entry=nebula_wcs_entry,
            sensor_config=sensor_config,
            logger=logger,
        )

        obs_star_slew_tracks[obs_name] = obs_slew_entry
        n_obs_processed += 1

        # Pull per-observer stats from run_meta if available.
        run_meta = obs_slew_entry.get("run_meta", {})
        n_windows_total += int(run_meta.get("n_windows", 0))
        n_tracks_total += int(run_meta.get("total_star_tracks", 0))

    # ------------------------------------------------------------------
    # 3) Final summary / sanity check
    # ------------------------------------------------------------------
    if n_obs_processed == 0:
        logger.warning(
            "build_star_slew_tracks_for_all_observers: no observers were "
            "processed. Check for mismatches between obs_star_projections "
            "and obs_tracks."
        )
    else:
        logger.info(
            "build_star_slew_tracks_for_all_observers: built star slew tracks "
            "for %d observers (%d windows, %d total star tracks).",
            n_obs_processed,
            n_windows_total,
            n_tracks_total,
        )

    return obs_star_slew_tracks

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main(
    sensor_config: Optional[SensorConfig] = None,
    star_projection_path: Optional[str] = None,
    obs_tracks_path: Optional[str] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> ObsStarSlewTracks:
    """
    High-level entry point for building star slew tracks for all observers.

    This function is intended to run *after* NEBULA_STAR_PROJECTION has
    produced ``obs_star_projections.pkl`` and after the pointing / schedule
    pipeline has produced an ``observer_tracks_with_pointing`` (or
    ``observer_tracks_with_pixels``) pickle.

    By default this function uses the standard NEBULA output layout to
    resolve paths when they are not provided:

        * star_projection_path ->
              NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl
        * obs_tracks_path ->
              Utility.SAT_OBJECTS.NEBULA_PIXEL_PICKLER.OBS_PIXEL_PICKLE_PATH
        * output_path ->
              NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_slew_projections.pkl

    These defaults make it possible to call ``main()`` from ``sim_test``
    without passing explicit file locations, while still allowing callers
    to override any path if needed.

    Parameters
    ----------
    sensor_config : SensorConfig or None, optional
        Sensor configuration to use. If None, ``ACTIVE_SENSOR`` is used.
        A RuntimeError is raised if neither is available.
    star_projection_path : str or None, optional
        Absolute path to ``obs_star_projections.pkl`` created by
        NEBULA_STAR_PROJECTION. This must be a non-empty string; if it
        is None or empty, a ValueError is raised.
    obs_tracks_path : str or None, optional
        Absolute path to the ``observer_tracks_with_pointing`` (or
        ``observer_tracks_with_pixels``) pickle that contains ``t_mjd_utc``
        and pointing fields. This must be a non-empty string; if it is
        None or empty, a ValueError is raised.
    output_path : str or None, optional
        Destination path for ``obs_star_slew_tracks.pkl``. If None, the
        slewing result is returned in memory only and nothing is written
        to disk. If provided, the directory is created as needed and the
        pickle is written there.
    logger : logging.Logger or None, optional
        Logger to use. If None, a module-local logger is created via
        :func:`_get_logger`.

    Returns
    -------
    dict
        ``obs_star_slew_tracks`` dict keyed by observer name, as returned
        by :func:`build_star_slew_tracks_for_all_observers`.

    Raises
    ------
    RuntimeError
        If ``sensor_config`` cannot be determined (both the argument and
        ``ACTIVE_SENSOR`` are None).
    ValueError
        If ``star_projection_path`` or ``obs_tracks_path`` is missing or
        an empty string.
    FileNotFoundError
        If the requested input files do not exist (propagated from the
        underlying load helpers).
    Other
        Any errors raised by :func:`build_star_slew_tracks_for_all_observers`
        or its per-observer/per-window helpers (e.g., IndexError, KeyError).
    """
    # ------------------------------------------------------------------
    # 1) Normalize / create logger and resolve sensor_config
    # ------------------------------------------------------------------
    logger = _get_logger(logger)

    # Prefer explicitly provided sensor_config; fall back to ACTIVE_SENSOR.
    sensor_config = sensor_config or ACTIVE_SENSOR
    if sensor_config is None:
        raise RuntimeError(
            "NEBULA_STAR_SLEW_PROJECTION.main: no SensorConfig available "
            "(sensor_config is None and ACTIVE_SENSOR is not defined)."
        )

    # Resolve default paths when not provided explicitly.
    star_projection_path = star_projection_path or _resolve_default_star_projection_path()
    obs_tracks_path = obs_tracks_path or _resolve_default_obs_tracks_path()
    if output_path is None:
        output_path = _resolve_default_output_path()

    logger.info(
        "NEBULA_STAR_SLEW_PROJECTION: starting star slew track build.\n"
        "  star_projection_path = %s\n"
        "  obs_tracks_path      = %s\n"
        "  output_path          = %s",
        star_projection_path,
        obs_tracks_path,
        output_path,
    )

    # ------------------------------------------------------------------
    # 2) Load upstream products from disk
    # ------------------------------------------------------------------
    obs_star_projections = _load_obs_star_projections(
        star_projection_path=star_projection_path,
        logger=logger,
    )
    obs_tracks = _load_obs_tracks(
        obs_tracks_path=obs_tracks_path,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 3) Build slew tracks for all observers
    # ------------------------------------------------------------------
    obs_star_slew_tracks = build_star_slew_tracks_for_all_observers(
        obs_star_projections=obs_star_projections,
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 4) Optionally save to disk
    # ------------------------------------------------------------------
    if output_path is not None:
        output_path = _save_star_slew_tracks(
            obs_star_slew_tracks=obs_star_slew_tracks,
            output_path=output_path,
            logger=logger,
        )
        logger.info(
            "NEBULA_STAR_SLEW_PROJECTION: finished; output saved to '%s'.",
            output_path,
        )
    else:
        logger.info(
            "NEBULA_STAR_SLEW_PROJECTION: finished; returning in-memory "
            "result without saving (output_path is None)."
        )

    return obs_star_slew_tracks


if __name__ == "__main__":
    # This module is intended to be called from a driver (e.g., sim_test)
    # with explicit paths. Running it directly without arguments will
    # raise errors. You can either:
    #
    #   * Edit this block to hard-code test paths for ad-hoc experiments, or
    #   * Import NEBULA_STAR_SLEW_PROJECTION in your driver and call main(...)
    #     with explicit star_projection_path / obs_tracks_path / output_path.
    raise SystemExit(
        "NEBULA_STAR_SLEW_PROJECTION is a library module. Import it and call "
        "main(...) from your driver (e.g., sim_test) with explicit paths."
    )

