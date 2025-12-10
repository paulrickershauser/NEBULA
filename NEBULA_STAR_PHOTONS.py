"""
NEBULA_STAR_PHOTONS
===================

This module converts *star projection* products into per-star photon
time series that mirror the existing **target photon** schema.

Inputs (all per-observer):
--------------------------
1) obs_star_projections
   - Loaded from the pickle produced by NEBULA_STAR_PROJECTION.
   - For each observer and window, provides:
       * mid-window star positions on the detector (x_pix_mid, y_pix_mid)
       * Gaia G-band magnitude (mag_G)
       * flags like on_detector, gaia_status, etc.

2) frames_with_sky
   - Loaded via NEBULA_TARGET_PHOTONS (same pickle used by
     NEBULA_PHOTON_FRAME_BUILDER).
   - For each observer and window, provides:
       * window segmentation (start_index, end_index, n_frames)
       * per-frame timing (t_utc, t_exp_s or equivalent)
       * sky selector metadata (sky_center_ra_deg, sky_radius_deg, etc.)

3) SensorConfig / ACTIVE_SENSOR
   - From Configuration.NEBULA_SENSOR_CONFIG.
   - Describes sensor geometry (rows, cols, pixel pitch, etc.)
     and is embedded as metadata in the star photon products.

Outputs:
--------
obs_star_photons : dict
    Top-level mapping keyed by observer name. For each observer:

        obs_star_photons[obs_name] = {
            "observer_name": str,
            "rows": int,
            "cols": int,
            "catalog_name": str,
            "catalog_band": str,        # e.g., "G"
            "run_meta": {...},
            "windows": [StarPhotonWindow, ...],
        }

    Each StarPhotonWindow contains per-window, per-star photon time
    series, aligned in spirit with the existing target photon pickles.

The resulting dict is typically serialized to:

    NEBULA_OUTPUT/FRAMES/obs_star_photons.pkl

and can be consumed by later sensor / event simulation stages.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Standard library imports
# ----------------------------------------------------------------------
from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging
import os
import pickle
from datetime import datetime

# ----------------------------------------------------------------------
# Third-party imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
# NEBULA configuration imports (must succeed; failures should be loud)
# ----------------------------------------------------------------------

# Base path configuration for NEBULA input/output directories.
from Configuration import NEBULA_PATH_CONFIG

# Sensor configuration: dataclass + active sensor definition.
from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig, ACTIVE_SENSOR

# Star catalog configuration: name, band metadata, etc.
from Configuration.NEBULA_STAR_CONFIG import NEBULA_STAR_CATALOG

# ----------------------------------------------------------------------
# NEBULA utility imports (must also succeed in a proper NEBULA run)
# ----------------------------------------------------------------------

# Radiometry / magnitude-to-photon-flux routines
# (C:\Users\prick\Desktop\Research\NEBULA\Utility\RADIOMETRY\NEBULA_FLUX.py)
from Utility.RADIOMETRY import NEBULA_FLUX

# ----------------------------------------------------------------------
# Module-wide constants and type aliases
# ----------------------------------------------------------------------

# Version tag for this star-photon stage (embedded in run_meta).
STAR_PHOTONS_RUN_META_VERSION: str = "0.1"

# Type alias for a per-window star photon dict.
StarPhotonWindow = Dict[str, Any]

# Type alias for the top-level mapping: observer_name -> per-observer entry.
ObsStarPhotons = Dict[str, Dict[str, Any]]


# ----------------------------------------------------------------------
# Logger helper
# ----------------------------------------------------------------------
def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Return a logger instance suitable for this module.

    Parameters
    ----------
    logger : logging.Logger or None, optional
        Existing logger to reuse. If None, a module-level logger
        named after this module (``__name__``) is returned.

        Logging configuration (handlers, levels, formatting) is assumed
        to be handled by the NEBULA driver script (e.g., sim_test.py)
        or by the top-level application; this helper does not modify
        global logging setup.

    Returns
    -------
    logging.Logger
        Logger to use inside NEBULA_STAR_PHOTONS.
    """
    # If the caller passed a logger, just use it.
    if logger is not None:
        return logger

    # Otherwise, return a standard module-level logger.
    return logging.getLogger(__name__)

def compute_star_photon_flux_from_mag(
    mag_G: np.ndarray,
    eta_eff: float = 1.0,
) -> np.ndarray:
    """
    Convert Gaia G-band magnitude(s) into photon flux at the aperture.

    This helper mirrors the radiometric conventions used in NEBULA_FLUX:
    it uses the Sun's G-band magnitude at 1 AU and an effective G-band
    solar irradiance F_SUN_G_1AU_W_M2 as the reference, then applies the
    standard magnitude–flux relation to recover the absolute energy flux
    and photon flux.

    Parameters
    ----------
    mag_G : array-like
        Gaia G-band apparent magnitudes for one or more stars. This can be
        a scalar, list, or NumPy array; it will be converted to a float
        array internally.

    eta_eff : float, optional
        Overall throughput / quantum-efficiency factor (0–1). This scales
        the photon flux after conversion from energy flux. Default is 1.0,
        i.e., no additional loss beyond what is already implicit in the
        reference irradiance.

    Returns
    -------
    np.ndarray
        Photon flux in units of photons m⁻² s⁻¹ at the aperture, with the
        same shape as ``mag_G`` broadcast to a NumPy array.

    Notes
    -----
    The conversion is:

        Δm = mag_G - GAIA_G_M_SUN_APP_1AU
        p  = 10^(-0.4 * Δm)
        F  = F_SUN_G_1AU_W_M2 * p
        Ṅ = F / GAIA_G_PHOTON_ENERGY_J * eta_eff

    where F_SUN_G_1AU_W_M2, GAIA_G_M_SUN_APP_1AU, GAIA_G_PHOTON_ENERGY_J,
    and NUM_EPS are taken from NEBULA_FLUX to ensure consistency with the
    target radiometry pipeline.
    """
    # Convert input magnitudes to a NumPy float array for vectorized math.
    mag_arr = np.asarray(mag_G, dtype=float)

    # Pull the reference G-band quantities from NEBULA_FLUX.
    m_sun_g = NEBULA_FLUX.GAIA_G_M_SUN_APP_1AU
    F_sun_g = NEBULA_FLUX.F_SUN_G_1AU_W_M2

    # Magnitude difference relative to the Sun in G band.
    delta_m = mag_arr - m_sun_g

    # Flux ratio p = F_star / F_sun,G using the standard magnitude–flux relation.
    p_flux_ratio = 10.0 ** (-0.4 * delta_m)

    # Convert dimensionless flux ratio into an absolute G-band energy flux
    # at the aperture (W m⁻²), using the same normalization as NEBULA_FLUX.
    flux_g_w_m2 = F_sun_g * p_flux_ratio

    # Convert energy flux to photon flux using the effective G-band photon energy.
    # Use NUM_EPS as a floor to avoid division by zero in pathological cases.
    photon_flux_g_m2_s = flux_g_w_m2 / max(
        NEBULA_FLUX.GAIA_G_PHOTON_ENERGY_J,
        NEBULA_FLUX.NUM_EPS,
    )

    # Apply overall instrument throughput / QE.
    photon_flux_g_m2_s *= float(eta_eff)

    return photon_flux_g_m2_s

def build_star_photon_timeseries_for_window_sidereal(
    obs_name: str,
    window_frames_entry: Dict[str, Any],
    window_star_projection: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    mode: str = "sidereal",
) -> StarPhotonWindow:
    """
    Build per-star photon time series for a single observer + window.

    This helper takes:
        - one window entry from the frames-with-sky structure
          (produced by NEBULA_PHOTON_FRAME_BUILDER / NEBULA_TARGET_PHOTONS), and
        - the matching star projection entry from obs_star_projections
          (produced by NEBULA_STAR_PROJECTION),

    and returns a StarPhotonWindow dictionary with per-star, per-frame
    photon flux time series.

    Current implementation
    ----------------------
    * Only the "sidereal" tracking mode is implemented.
    * For sidereal windows, each star's pixel coordinates (x_pix, y_pix)
      are assumed to be fixed across all frames in the window and are
      taken from the mid-window star projection (x_pix_mid, y_pix_mid).
    * Each star's Gaia G magnitude mag_G is treated as constant over the
      window. The corresponding photon flux at the aperture
      phi_ph_m2_s is computed once via compute_star_photon_flux_from_mag()
      and broadcast across frames.
    * Per-frame photon counts are then:

            flux_ph_m2_frame = phi_ph_m2_s * t_exp_s

      where t_exp_s is the exposure time of each frame.

    Expected input schema (minimal)
    --------------------------------
    window_frames_entry (from frames-with-sky) should provide:
        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_frames": int,
            "t_start_utc": ...,
            "t_end_utc": ...,
            "t_mid_utc": ... (optional),
            "t_mid_mjd_utc": float (optional),
            "sky_center_ra_deg": float (optional),
            "sky_center_dec_deg": float (optional),
            "sky_radius_deg": float (optional),
            "sky_selector_status": str (optional),
            "tracking_mode": str (optional, e.g., "sidereal" or "slew"),
            "frames": [
                {
                    "frame_index": int,
                    "t_utc": datetime or str,
                    "t_mjd_utc": float (optional),
                    "t_exp_s": float,
                    ...
                },
                ...
            ],
        }

    window_star_projection (from obs_star_projections) should provide:
        {
            "window_index": int,
            "t_mid_utc": ...,
            "t_mid_mjd_utc": float,
            "n_stars_input": int,
            "n_stars_on_detector": int,
            "stars": {
                "<gaia_source_id_str>": {
                    "gaia_source_id": int or str,
                    "source_id": str (optional),
                    "mag_G": float,
                    "x_pix_mid": float,
                    "y_pix_mid": float,
                    "on_detector": bool,
                    ...
                },
                ...
            },
            ...
        }

    Returned schema
    ---------------
    The returned StarPhotonWindow has the form:

        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_frames": int,
            "t_start_utc": ...,
            "t_end_utc": ...,
            "t_mid_utc": ...,
            "t_mid_mjd_utc": float or None,
            "sky_center_ra_deg": float or None,
            "sky_center_dec_deg": float or None,
            "sky_radius_deg": float or None,
            "sky_selector_status": str or None,
            "tracking_mode": str,
            "n_stars": int,
            "stars": {
                "<gaia_source_id_str>": {
                    "source_id": str,
                    "source_type": "star",
                    "gaia_source_id": int or str,
                    "t_utc": np.ndarray[object],
                    "t_exp_s": np.ndarray[float],
                    "x_pix": np.ndarray[float],
                    "y_pix": np.ndarray[float],
                    "phi_ph_m2_s": np.ndarray[float],
                    "flux_ph_m2_frame": np.ndarray[float],
                    "mag_G": np.ndarray[float],
                    "on_detector": np.ndarray[bool],
                },
                ...
            },
            "n_sources_total": int,  # == n_stars for now
        }

    Parameters
    ----------
    obs_name : str
        Name of the observer (used only for logging / error messages).

    window_frames_entry : dict
        Single-window entry from the frames-with-sky structure for this
        observer.

    window_star_projection : dict
        Matching single-window star projection entry from
        obs_star_projections[obs_name]["windows"].

    logger : logging.Logger, optional
        Logger for debug/info messages. If None, a local module logger
        obtained via _get_logger() is used.

    mode : {"sidereal", ...}, optional
        Tracking mode for this window. Currently only "sidereal" is
        supported; any other value results in a ValueError.

    Returns
    -------
    StarPhotonWindow
        Dictionary containing per-star photon time series for this
        observer + window.

    Raises
    ------
    ValueError
        If mode is not "sidereal".
    RuntimeError
        If the window has no frames, or required keys are missing.
    """
    # Resolve a logger to use inside this function.
    log = _get_logger(logger)

    # For now we only support sidereal handling here.
    if mode != "sidereal":
        raise ValueError(
            f"build_star_photon_timeseries_for_window: only 'sidereal' mode "
            f"is implemented, got mode={mode!r} for observer '{obs_name}'."
        )

    # ------------------------------------------------------------------
    # Extract and validate frame-level information for this window.
    # ------------------------------------------------------------------
    frames = window_frames_entry.get("frames", None)
    if not frames:
        raise RuntimeError(
            f"build_star_photon_timeseries_for_window: window "
            f"{window_frames_entry.get('window_index')} for observer "
            f"'{obs_name}' has no 'frames' entry."
        )

    # Number of frames in this window.
    n_frames = len(frames)

    # Collect per-frame times and exposure durations.
    t_utc_list: list[Any] = []
    t_exp_s_list: list[float] = []

    for f in frames:
        # Each frame must provide a time stamp and exposure duration.
        if "t_utc" not in f:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window: frame in window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' is missing 't_utc'."
            )
        if "t_exp_s" not in f:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window: frame in window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' is missing 't_exp_s'."
            )

        t_utc_list.append(f["t_utc"])
        t_exp_s_list.append(float(f["t_exp_s"]))

    # Convert per-frame lists to NumPy arrays for vectorized math.
    t_exp_s = np.asarray(t_exp_s_list, dtype=float)

    # ------------------------------------------------------------------
    # Extract and validate per-star projection information.
    # ------------------------------------------------------------------
    stars_proj: Dict[str, Any] = window_star_projection.get("stars", {})
    if stars_proj is None:
        stars_proj = {}

    # Dictionary to accumulate per-star time series.
    stars_timeseries: Dict[str, Dict[str, Any]] = {}

    # If there are no stars at all, we still return a valid (but empty)
    # StarPhotonWindow structure.
    if len(stars_proj) == 0:
        log.debug(
            "Observer '%s', window %s: no stars in projection; returning "
            "empty StarPhotonWindow.",
            obs_name,
            window_frames_entry.get("window_index"),
        )

    for star_key, star_entry in stars_proj.items():
        # Respect the on_detector flag: skip stars that never land
        # on the detector for this window, if such filtering is desired.
        on_det_flag = bool(star_entry.get("on_detector", True))
        if not on_det_flag:
            # You could choose to include them with on_detector=False
            # for every frame; for now we skip them entirely to keep
            # the star catalog compact.
            continue

        # Gaia source identifier. Prefer an explicit gaia_source_id field
        # from NEBULA_STAR_PROJECTION; fall back to the dict key if absent.
        gaia_source_id = star_entry.get("gaia_source_id", star_key)

        # Human-readable/short source identifier; fall back to the key.
        source_id = star_entry.get("source_id", str(star_key))


        # Mid-window pixel coordinates from the star projection stage.
        try:
            x_mid = float(star_entry["x_pix_mid"])
            y_mid = float(star_entry["y_pix_mid"])
        except KeyError as exc:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window: missing "
                f"x_pix_mid/y_pix_mid for star {star_key!r} in observer "
                f"'{obs_name}', window "
                f"{window_frames_entry.get('window_index')}."
            ) from exc

        # Gaia G magnitude; treat as constant over this window.
        if "mag_G" not in star_entry:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window: missing mag_G "
                f"for star {star_key!r} in observer '{obs_name}', window "
                f"{window_frames_entry.get('window_index')}."
            )

        mag_val = float(np.asarray(star_entry["mag_G"], dtype=float).ravel()[0])

        # Convert magnitude to photon flux [ph m^-2 s^-1] at the aperture.
        phi_ph_m2_s_scalar = float(compute_star_photon_flux_from_mag(mag_val))

        # Broadcast scalar quantities over all frames in this window.
        x_pix = np.full(n_frames, x_mid, dtype=float)
        y_pix = np.full(n_frames, y_mid, dtype=float)
        phi_ph_m2_s = np.full(n_frames, phi_ph_m2_s_scalar, dtype=float)
        mag_G_series = np.full(n_frames, mag_val, dtype=float)
        on_detector_series = np.full(n_frames, True, dtype=bool)

        # Per-frame photon counts: phi * exposure time.
        flux_ph_m2_frame = phi_ph_m2_s * t_exp_s

        # Assemble per-star time series record.
        star_rec: Dict[str, Any] = {
            "source_id": source_id,
            "source_type": "star",
            "gaia_source_id": gaia_source_id,
            "t_utc": np.asarray(t_utc_list, dtype=object),
            "t_exp_s": t_exp_s,
            "x_pix": x_pix,
            "y_pix": y_pix,
            "phi_ph_m2_s": phi_ph_m2_s,
            "flux_ph_m2_frame": flux_ph_m2_frame,
            "mag_G": mag_G_series,
            "on_detector": on_detector_series,
        }


        stars_timeseries[str(star_key)] = star_rec

    # ------------------------------------------------------------------
    # Build the window-level StarPhotonWindow wrapper.
    # ------------------------------------------------------------------
    window_index = window_frames_entry.get(
        "window_index", window_star_projection.get("window_index")
    )

    star_window: StarPhotonWindow = {
        "window_index": window_index,
        "start_index": window_frames_entry.get("start_index"),
        "end_index": window_frames_entry.get("end_index"),
        "n_frames": n_frames,
    
        # Prefer the photon-frame builder's naming, but fall back
        # gracefully if older field names are present instead.
        "t_start_utc": window_frames_entry.get(
            "t_start_utc", window_frames_entry.get("start_time")
        ),
        "t_end_utc": window_frames_entry.get(
            "t_end_utc", window_frames_entry.get("end_time")
        ),
    
        # For mid-times, use the frames view as canonical. If you later
        # decide that NEBULA_STAR_PROJECTION or NEBULA_STAR_SLEW_PROJECTION
        # stores a more precise mid-time, you can revisit this.
        "t_mid_utc": window_frames_entry.get("t_mid_utc"),
        "t_mid_mjd_utc": window_frames_entry.get("t_mid_mjd_utc"),
    
        "sky_center_ra_deg": window_frames_entry.get("sky_center_ra_deg"),
        "sky_center_dec_deg": window_frames_entry.get("sky_center_dec_deg"),
        "sky_radius_deg": window_frames_entry.get("sky_radius_deg"),
        "sky_selector_status": window_frames_entry.get("sky_selector_status"),
        "tracking_mode": window_frames_entry.get("tracking_mode", mode),
        "n_stars": len(stars_timeseries),
        "stars": stars_timeseries,
        "n_sources_total": len(stars_timeseries),
    }



    log.debug(
        "Observer '%s', window %s: built star photon time series for %d stars.",
        obs_name,
        window_index,
        star_window["n_stars"],
    )

    return star_window

def build_star_photon_timeseries_for_window_slew(
    obs_name: str,
    window_frames_entry: Dict[str, Any],
    window_star_slew_tracks: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    mode: str = "slew",
) -> StarPhotonWindow:
    """
    Build per-star photon time series for a single observer + window
    in **slew** mode.

    This helper takes:
        - one window entry from the frames-with-sky structure
          (produced by NEBULA_PHOTON_FRAME_BUILDER / NEBULA_TARGET_PHOTONS), and
        - the matching star *slew track* entry from NEBULA_STAR_SLEW_PROJECTION,

    and returns a StarPhotonWindow dictionary with per-star, per-frame
    photon flux time series.

    Conceptually this is the "moving stars" counterpart to
    build_star_photon_timeseries_for_window_sidereal:

      * Sidereal:
          - stars are assumed fixed on the detector for the duration of
            the window; we use (x_pix_mid, y_pix_mid) from
            NEBULA_STAR_PROJECTION and broadcast across frames.

      * Slew:
          - the boresight is sweeping relative to the celestial sphere,
            so stars move across the detector.
          - NEBULA_STAR_SLEW_PROJECTION gives us per-star tracks:
                coarse_indices[j]
                t_mjd_utc[j]
                x_pix[j], y_pix[j]
                on_detector[j]
            over the coarse time grid for that window.
          - We align those coarse indices with the per-frame
            "coarse_index" field in window_frames_entry["frames"] and
            build per-frame (x_pix, y_pix, on_detector) arrays.

    Expected input schema (minimal)
    --------------------------------
    window_frames_entry (from frames-with-sky) should provide:
        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_frames": int,
            "t_start_utc": ...,
            "t_end_utc": ...,
            "t_mid_utc": ... (optional),
            "t_mid_mjd_utc": float (optional),
            "sky_center_ra_deg": float (optional),
            "sky_center_dec_deg": float (optional),
            "sky_radius_deg": float (optional),
            "sky_selector_status": str (optional),
            "tracking_mode": "slew" (recommended),
            "frames": [
                {
                    "frame_index": int,
                    "coarse_index": int,
                    "t_utc": datetime or str,
                    "t_mjd_utc": float (optional),
                    "t_exp_s": float,
                    ...
                },
                ...
            ],
        }

    window_star_slew_tracks (from NEBULA_STAR_SLEW_PROJECTION) should provide:
        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_coarse": int,
            "coarse_indices": np.ndarray[int],
            "t_mjd_utc": np.ndarray[float],
            "n_stars_input": int,
            "n_stars_on_detector": int,
            "stars": {
                "<gaia_source_id_str>": {
                    "gaia_source_id": int or str,
                    "source_id": str (optional),
                    "mag_G": float,
                    "coarse_indices": np.ndarray[int],
                    "t_mjd_utc": np.ndarray[float],
                    "x_pix": np.ndarray[float],
                    "y_pix": np.ndarray[float],
                    "on_detector": np.ndarray[bool],
                    ...
                },
                ...
            },
        }

    Returned schema
    ---------------
    Matches the sidereal variant, so downstream code can treat them
    uniformly:

        {
            "window_index": int,
            "start_index": int,
            "end_index": int,
            "n_frames": int,
            "t_start_utc": ...,
            "t_end_utc": ...,
            "t_mid_utc": ...,
            "t_mid_mjd_utc": float or None,
            "sky_center_ra_deg": float or None,
            "sky_center_dec_deg": float or None,
            "sky_radius_deg": float or None,
            "sky_selector_status": str or None,
            "tracking_mode": "slew" or from frames entry,
            "n_stars": int,
            "stars": {
                "<gaia_source_id_str>": {
                    "source_id": str,
                    "source_type": "star",
                    "gaia_source_id": int or str,
                    "t_utc": np.ndarray[object],
                    "t_exp_s": np.ndarray[float],
                    "x_pix": np.ndarray[float],
                    "y_pix": np.ndarray[float],
                    "phi_ph_m2_s": np.ndarray[float],
                    "flux_ph_m2_frame": np.ndarray[float],
                    "mag_G": np.ndarray[float],
                    "on_detector": np.ndarray[bool],
                },
                ...
            },
            "n_sources_total": int,  # == n_stars for now
        }

    Parameters
    ----------
    obs_name : str
        Name of the observer (used only for logging / error messages).

    window_frames_entry : dict
        Single-window entry from the frames-with-sky structure for this
        observer.

    window_star_slew_tracks : dict
        Matching single-window star-slew entry from
        obs_star_slew_tracks[obs_name]["windows"] produced by
        NEBULA_STAR_SLEW_PROJECTION.

    logger : logging.Logger, optional
        Logger for debug/info messages. If None, a local module logger
        obtained via _get_logger() is used.

    mode : {"slew", ...}, optional
        Tracking mode label for this window. Currently we only accept
        "slew" here; any other value results in a ValueError.

    Returns
    -------
    StarPhotonWindow
        Dictionary containing per-star photon time series for this
        observer + window, with stars moving across the detector
        according to their slew tracks.

    Raises
    ------
    ValueError
        If mode is not "slew".
    RuntimeError
        If the window has no frames, coarse indices are missing, or
        required star-track keys are missing / inconsistent.
    """
    log = _get_logger(logger)

    # Enforce that this helper is only used for slew windows.
    if mode != "slew":
        raise ValueError(
            f"build_star_photon_timeseries_for_window_slew: only 'slew' mode "
            f"is implemented here, got mode={mode!r} for observer '{obs_name}'."
        )

    # ------------------------------------------------------------------
    # Extract and validate frame-level information for this window.
    # ------------------------------------------------------------------
    frames = window_frames_entry.get("frames", None)
    if not frames:
        raise RuntimeError(
            f"build_star_photon_timeseries_for_window_slew: window "
            f"{window_frames_entry.get('window_index')} for observer "
            f"'{obs_name}' has no 'frames' entry."
        )

    n_frames = len(frames)

    # Collect per-frame times, exposure durations, and coarse indices.
    t_utc_list: List[Any] = []
    t_exp_s_list: List[float] = []
    coarse_idx_list: List[int] = []

    for f in frames:
        if "t_utc" not in f:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window_slew: frame in window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' is missing 't_utc'."
            )
        if "t_exp_s" not in f:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window_slew: frame in window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' is missing 't_exp_s'."
            )
        if "coarse_index" not in f:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window_slew: frame in window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' is missing 'coarse_index'."
            )

        t_utc_list.append(f["t_utc"])
        t_exp_s_list.append(float(f["t_exp_s"]))
        coarse_idx_list.append(int(f["coarse_index"]))

    t_exp_s = np.asarray(t_exp_s_list, dtype=float)
    coarse_idx_frames = np.asarray(coarse_idx_list, dtype=int)

    # ------------------------------------------------------------------
    # Extract and validate per-star slew tracks.
    # ------------------------------------------------------------------
    stars_tracks: Dict[str, Any] = window_star_slew_tracks.get("stars", {})
    if stars_tracks is None:
        stars_tracks = {}

    stars_timeseries: Dict[str, Dict[str, Any]] = {}

    if len(stars_tracks) == 0:
        log.debug(
            "Observer '%s', window %s (slew): no stars in slew tracks; "
            "returning empty StarPhotonWindow.",
            obs_name,
            window_frames_entry.get("window_index"),
        )

    for star_key, star_entry in stars_tracks.items():
        # Gaia source identifier. Prefer explicit gaia_source_id but fall
        # back to the dict key if needed.
        gaia_source_id = star_entry.get("gaia_source_id", star_key)
        source_id = star_entry.get("source_id", str(star_key))

        # Required time / geometry arrays for this star track.
        try:
            star_coarse_idx = np.asarray(star_entry["coarse_indices"], dtype=int)
            x_track = np.asarray(star_entry["x_pix"], dtype=float)
            y_track = np.asarray(star_entry["y_pix"], dtype=float)
        except KeyError as exc:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window_slew: missing "
                f"'coarse_indices'/'x_pix'/'y_pix' for star {star_key!r} in "
                f"observer '{obs_name}', window "
                f"{window_frames_entry.get('window_index')}."
            ) from exc

        # on_detector is expected to be a boolean array; if absent,
        # assume the star is on-detector wherever it has a defined track.
        on_det_track = np.asarray(
            star_entry.get("on_detector", np.ones_like(x_track, dtype=bool)),
            dtype=bool,
        )

        # Basic consistency checks on per-star arrays.
        if not (
            star_coarse_idx.shape == x_track.shape == y_track.shape == on_det_track.shape
        ):
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window_slew: inconsistent "
                f"array lengths for star {star_key!r} in observer '{obs_name}', "
                f"window {window_frames_entry.get('window_index')}."
            )

        # If the star is never actually on the detector during this
        # window, skip it entirely to keep the catalog compact.
        if not np.any(on_det_track):
            continue

        # Gaia G magnitude; treat as constant over this window / track.
        if "mag_G" not in star_entry:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window_slew: missing mag_G "
                f"for star {star_key!r} in observer '{obs_name}', window "
                f"{window_frames_entry.get('window_index')}."
            )
        mag_val = float(np.asarray(star_entry["mag_G"], dtype=float).ravel()[0])

        # Convert magnitude to photon flux [ph m^-2 s^-1] at the aperture.
        phi_ph_m2_s_scalar = float(compute_star_photon_flux_from_mag(mag_val))

        # ------------------------------------------------------------------
        # Map star's coarse indices onto the per-frame coarse_index list.
        # ------------------------------------------------------------------
        # Start with "no data" defaults for each frame; we will fill in
        # entries wherever this star has a track sample.
        x_pix = np.full(n_frames, np.nan, dtype=float)
        y_pix = np.full(n_frames, np.nan, dtype=float)
        on_detector_series = np.zeros(n_frames, dtype=bool)

        # Build a dictionary: coarse_index -> track index, so we can
        # do O(1) lookups for each frame.
        index_map: Dict[int, int] = {}
        for idx_ci, ci in enumerate(star_coarse_idx):
            # If duplicate coarse indices ever appear, that would indicate
            # a bug in the slew projection stage.
            if int(ci) in index_map:
                raise RuntimeError(
                    f"build_star_photon_timeseries_for_window_slew: duplicate "
                    f"coarse_index {int(ci)} for star {star_key!r} in observer "
                    f"'{obs_name}', window "
                    f"{window_frames_entry.get('window_index')}."
                )
            index_map[int(ci)] = idx_ci

        # For each frame, see if this star has a defined track sample.
        for i_frame, ci_frame in enumerate(coarse_idx_frames):
            idx_track = index_map.get(int(ci_frame))
            if idx_track is None:
                # No track sample for this coarse index; leave this star
                # off-detector for this frame.
                continue

            x_pix[i_frame] = x_track[idx_track]
            y_pix[i_frame] = y_track[idx_track]
            on_detector_series[i_frame] = bool(on_det_track[idx_track])

        # If, after mapping, the star is never on-detector for any of the
        # frames, we can drop it.
        if not np.any(on_detector_series):
            continue

        # Broadcast scalar magnitude and photon flux across frames.
        phi_ph_m2_s = np.full(n_frames, phi_ph_m2_s_scalar, dtype=float)
        mag_G_series = np.full(n_frames, mag_val, dtype=float)

        # Per-frame photon counts: phi * exposure time.
        flux_ph_m2_frame = phi_ph_m2_s * t_exp_s

        # Assemble per-star time series record.
        star_rec: Dict[str, Any] = {
            "source_id": source_id,
            "source_type": "star",
            "gaia_source_id": gaia_source_id,
            "t_utc": np.asarray(t_utc_list, dtype=object),
            "t_exp_s": t_exp_s,
            "x_pix": x_pix,
            "y_pix": y_pix,
            "phi_ph_m2_s": phi_ph_m2_s,
            "flux_ph_m2_frame": flux_ph_m2_frame,
            "mag_G": mag_G_series,
            "on_detector": on_detector_series,
        }

        stars_timeseries[str(star_key)] = star_rec

    # ------------------------------------------------------------------
    # Build the window-level StarPhotonWindow wrapper (same shape as
    # the sidereal variant).
    # ------------------------------------------------------------------
    window_index = window_frames_entry.get(
        "window_index", window_star_slew_tracks.get("window_index")
    )

    star_window: StarPhotonWindow = {
        "window_index": window_index,
        "start_index": window_frames_entry.get("start_index"),
        "end_index": window_frames_entry.get("end_index"),
        "n_frames": n_frames,
        # Prefer the photon-frame builder's naming, but fall back
        # gracefully if older field names are present instead.
        "t_start_utc": window_frames_entry.get(
            "t_start_utc", window_frames_entry.get("start_time")
        ),
        "t_end_utc": window_frames_entry.get(
            "t_end_utc", window_frames_entry.get("end_time")
        ),
        # Treat frames-with-sky mid-times as canonical; you can revisit
        # this if you later decide that NEBULA_STAR_SLEW_PROJECTION has a
        # more precise mid-time definition.
        "t_mid_utc": window_frames_entry.get("t_mid_utc"),
        "t_mid_mjd_utc": window_frames_entry.get("t_mid_mjd_utc"),
        "sky_center_ra_deg": window_frames_entry.get("sky_center_ra_deg"),
        "sky_center_dec_deg": window_frames_entry.get("sky_center_dec_deg"),
        "sky_radius_deg": window_frames_entry.get("sky_radius_deg"),
        "sky_selector_status": window_frames_entry.get("sky_selector_status"),
        "tracking_mode": window_frames_entry.get("tracking_mode", mode),
        "n_stars": len(stars_timeseries),
        "stars": stars_timeseries,
        "n_sources_total": len(stars_timeseries),
    }

    log.debug(
        "Observer '%s', window %s (slew): built star photon time series for %d stars.",
        obs_name,
        window_index,
        star_window["n_stars"],
    )

    return star_window

def build_star_photon_timeseries_for_window(
    obs_name: str,
    window_frames_entry: Dict[str, Any],
    window_star_projection: Optional[Dict[str, Any]] = None,
    window_star_slew_entry: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> StarPhotonWindow:
    """
    Dispatch helper: build per-star photon time series for one window,
    choosing the appropriate implementation based on tracking_mode.

    This is the *single* entry point that higher-level code (e.g.,
    "build_star_photons_for_all_observers") should call.

    It inspects the window's tracking_mode (as annotated earlier by
    annotate_windows_with_tracking_mode in sim_test) and then:

        - If tracking_mode == "sidereal":
              -> calls build_star_photon_timeseries_for_window_sidereal(...)
                 using window_star_projection.

        - If tracking_mode == "slew":
              -> calls build_star_photon_timeseries_for_window_slew(...)
                 using window_star_slew_entry.

    Any missing inputs or unknown tracking_mode values are treated as
    *fatal* errors (fail-hard), since they indicate a mismatch between
    the windows, projection products, or earlier classification.

    Parameters
    ----------
    obs_name : str
        Name of the observer (for logging / error messages).

    window_frames_entry : dict
        Single-window entry from the frames-with-sky structure for this
        observer, i.e. one element of:
            frames_with_sky[obs_name]["windows"].

        Must contain a "tracking_mode" field set to "sidereal" or "slew"
        by annotate_windows_with_tracking_mode.

    window_star_projection : dict or None, optional
        Matching single-window entry from obs_star_projections for this
        observer, used when tracking_mode == "sidereal". If None in that
        case, a RuntimeError is raised.

    window_star_slew_entry : dict or None, optional
        Matching single-window entry from obs_star_slew_tracks for this
        observer, used when tracking_mode == "slew". If None in that
        case, a RuntimeError is raised.

    logger : logging.Logger, optional
        Logger for informational / debug messages. If None, a module-
        level logger obtained via _get_logger() is used.

    Returns
    -------
    StarPhotonWindow
        Dictionary containing per-star photon time series for this
        observer + window, with either sidereal or slew kinematics
        applied as appropriate.

    Raises
    ------
    RuntimeError
        If tracking_mode is missing, or the required star projection
        entry for the selected mode is None.

    ValueError
        If tracking_mode is not one of the recognized values
        ("sidereal", "slew").
    """
    log = _get_logger(logger)

    # ------------------------------------------------------------------
    # Determine tracking mode for this window.
    # ------------------------------------------------------------------
    mode_raw = window_frames_entry.get("tracking_mode", None)
    if mode_raw is None:
        raise RuntimeError(
            f"build_star_photon_timeseries_for_window: window "
            f"{window_frames_entry.get('window_index')} for observer "
            f"'{obs_name}' has no 'tracking_mode' field. "
            f"Did you run annotate_windows_with_tracking_mode first?"
        )

    mode = str(mode_raw).lower()

    # ------------------------------------------------------------------
    # Dispatch based on mode.
    # ------------------------------------------------------------------
    if mode == "sidereal":
        if window_star_projection is None:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window: sidereal window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' has no matching window_star_projection."
            )

        return build_star_photon_timeseries_for_window_sidereal(
            obs_name=obs_name,
            window_frames_entry=window_frames_entry,
            window_star_projection=window_star_projection,
            logger=log,
            mode="sidereal",
        )

    elif mode == "slew":
        if window_star_slew_entry is None:
            raise RuntimeError(
                f"build_star_photon_timeseries_for_window: slew window "
                f"{window_frames_entry.get('window_index')} for observer "
                f"'{obs_name}' has no matching window_star_slew_entry."
            )

        return build_star_photon_timeseries_for_window_slew(
            obs_name=obs_name,
            window_frames_entry=window_frames_entry,
            window_star_slew_entry=window_star_slew_entry,
            logger=log,
            mode="slew",
        )

    else:
        raise ValueError(
            f"build_star_photon_timeseries_for_window: unknown tracking_mode "
            f"{mode_raw!r} for observer '{obs_name}', window "
            f"{window_frames_entry.get('window_index')}."
        )

def build_star_photons_for_observer(
    obs_name: str,
    frames_for_obs: Dict[str, Any],
    star_projections_for_obs: Dict[str, Any],
    star_slew_tracks_for_obs: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build star photon time series for all windows of a single observer.

    This is the main per-observer helper for NEBULA_STAR_PHOTONS. It takes:

      * the frames-with-sky view for one observer (``frames_for_obs``),
      * the corresponding sidereal star projections for that observer
        (``star_projections_for_obs`` from NEBULA_STAR_PROJECTION), and
      * optionally, the per-frame star tracks for slewing windows
        (``star_slew_tracks_for_obs`` from NEBULA_STAR_SLEW_PROJECTION),

    and returns a per-observer star-photon catalog:

        {
            "observer_name": str,
            "rows": int,
            "cols": int,
            "catalog_name": str,
            "catalog_band": str,
            "run_meta": {...},
            "windows": [StarPhotonWindow, ...],
        }

    For each window in ``frames_for_obs["windows"]``:

      1. Determine the tracking mode from
             window["tracking_mode"]  (annotated earlier by sim_test).

      2. Use ``window_index`` to look up the matching star information:

             * For ``tracking_mode == "sidereal"``:
                   star_projections_for_obs["windows"] is used, and
                   build_star_photon_timeseries_for_window_sidereal(...)
                   is called.

             * For ``tracking_mode == "slew"``:
                   star_slew_tracks_for_obs["windows"] is used, and
                   build_star_photon_timeseries_for_window_slew(...)
                   is called.

      3. Collect the resulting StarPhotonWindow dict into the output
         list ``windows`` for this observer.

    STRICT / fail-loud behaviour
    ----------------------------
    This function is intentionally strict to catch pipeline mismatches:

      * If a window has an unrecognized tracking_mode, a ValueError is raised.
      * If a sidereal window is missing a matching entry in
        ``star_projections_for_obs["windows"]``, a RuntimeError is raised.
      * If a slew window is missing either ``star_slew_tracks_for_obs`` or
        the matching entry in ``star_slew_tracks_for_obs["windows"]``,
        a RuntimeError is raised.

    Parameters
    ----------
    obs_name : str
        Name of the observer whose windows are being processed.

    frames_for_obs : dict
        Frames-with-sky structure for this observer, as produced by
        NEBULA_PHOTON_FRAME_BUILDER / NEBULA_TARGET_PHOTONS. Expected
        minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "windows": [
                    {
                        "window_index": int,
                        "tracking_mode": str,  # "sidereal" or "slew"
                        "frames": [...],
                        ...
                    },
                    ...
                ],
                ...
            }

    star_projections_for_obs : dict
        Per-observer star projection product from NEBULA_STAR_PROJECTION.
        Expected minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "run_meta": {...},
                "windows": [
                    {
                        "window_index": int,
                        "n_stars_input": int,
                        "n_stars_on_detector": int,
                        "stars": { ... },
                        ...
                    },
                    ...
                ],
            }

        There should be one StarWindowProjection entry per window_index,
        even if it contains zero stars.

    star_slew_tracks_for_obs : dict or None, optional
        Per-observer star tracks for slewing windows from
        NEBULA_STAR_SLEW_PROJECTION. Expected minimal structure:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "run_meta": {...},
                "windows": [
                    {
                        "window_index": int,
                        "coarse_indices": [...],
                        "stars": {
                            "<gaia_source_id_str>": {
                                "gaia_source_id": int or str,
                                "source_id": str,
                                "x_pix": np.ndarray[float],
                                "y_pix": np.ndarray[float],
                                "on_detector": np.ndarray[bool],
                                "mag_G": float,
                                ...
                            },
                            ...
                        },
                        ...
                    },
                    ...
                ],
            }

        If None, any window in slewing mode will cause a RuntimeError.

    logger : logging.Logger or None, optional
        Logger for informational / debug messages. If None, a module-level
        logger from _get_logger() is used.

    Returns
    -------
    dict
        Per-observer star photon catalog with the schema:

            {
                "observer_name": str,
                "rows": int,
                "cols": int,
                "catalog_name": str,
                "catalog_band": str,
                "run_meta": {...},
                "windows": [StarPhotonWindow, ...],
            }

        where each element of "windows" is the result of either
        build_star_photon_timeseries_for_window_sidereal(...) or
        build_star_photon_timeseries_for_window_slew(...).

    Raises
    ------
    RuntimeError
        If a required star window (sidereal or slew) is missing, or if
        frames_for_obs does not have the expected "windows" structure.

    ValueError
        If a window has an unsupported tracking_mode.
    """
    # Resolve a logger for this function.
    log = _get_logger(logger)

    # ------------------------------------------------------------------
    # Basic validation of input structures.
    # ------------------------------------------------------------------
    # Ensure the frames_for_obs object has a windows list.
    windows_frames = frames_for_obs.get("windows", None)
    if windows_frames is None:
        raise RuntimeError(
            f"build_star_photons_for_observer: frames_for_obs for observer "
            f"'{obs_name}' is missing a 'windows' entry."
        )

    # Extract the list of sidereal star projection windows.
    sidereal_windows = star_projections_for_obs.get("windows", None)
    if sidereal_windows is None:
        raise RuntimeError(
            f"build_star_photons_for_observer: star_projections_for_obs for "
            f"observer '{obs_name}' is missing a 'windows' entry."
        )

    # Build a lookup: window_index -> StarWindowProjection for sidereal mode.
    sidereal_by_index: Dict[int, Dict[str, Any]] = {}
    for w in sidereal_windows:
        idx = int(w.get("window_index"))
        if idx in sidereal_by_index:
            raise RuntimeError(
                f"build_star_photons_for_observer: duplicate sidereal "
                f"window_index={idx} for observer '{obs_name}'."
            )
        sidereal_by_index[idx] = w

    # If slew tracks are provided, build a similar lookup for them.
    slew_by_index: Dict[int, Dict[str, Any]] = {}
    if star_slew_tracks_for_obs is not None:
        slew_windows = star_slew_tracks_for_obs.get("windows", [])
        for w in slew_windows:
            idx = int(w.get("window_index"))
            if idx in slew_by_index:
                raise RuntimeError(
                    f"build_star_photons_for_observer: duplicate slew "
                    f"window_index={idx} for observer '{obs_name}'."
                )
            slew_by_index[idx] = w

    # ------------------------------------------------------------------
    # Loop over all windows for this observer and build star photons.
    # ------------------------------------------------------------------
    star_windows: List[StarPhotonWindow] = []

    for window_frames_entry in windows_frames:
        # Each window must have a window_index so we can match it.
        if "window_index" not in window_frames_entry:
            raise RuntimeError(
                f"build_star_photons_for_observer: a window for observer "
                f"'{obs_name}' is missing 'window_index'."
            )

        window_index = int(window_frames_entry["window_index"])

        # Tracking mode should already have been annotated earlier
        # (e.g., by annotate_windows_with_tracking_mode in sim_test).
        mode = window_frames_entry.get("tracking_mode", "sidereal")

        if mode == "sidereal":
            # Look up the matching StarWindowProjection.
            window_star_proj = sidereal_by_index.get(window_index)
            if window_star_proj is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: no sidereal star "
                    f"projection found for observer '{obs_name}', "
                    f"window_index={window_index}."
                )

            # Delegate the actual time series construction to the sidereal helper.
            star_window = build_star_photon_timeseries_for_window_sidereal(
                obs_name=obs_name,
                window_frames_entry=window_frames_entry,
                window_star_projection=window_star_proj,
                logger=log,
                mode="sidereal",
            )

        elif mode == "slew":
            # For slewing windows we require the slew tracks product.
            if star_slew_tracks_for_obs is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: encountered a 'slew' "
                    f"window (index={window_index}) for observer '{obs_name}', "
                    f"but star_slew_tracks_for_obs is None."
                )

            # Look up the matching StarSlewWindow.
            window_slew = slew_by_index.get(window_index)
            if window_slew is None:
                raise RuntimeError(
                    f"build_star_photons_for_observer: no slew star tracks "
                    f"found for observer '{obs_name}', window_index={window_index}."
                )

            # Delegate to the slew helper (which uses per-frame x_pix, y_pix).
            star_window = build_star_photon_timeseries_for_window_slew(
                obs_name=obs_name,
                window_frames_entry=window_frames_entry,
                window_star_slew=window_slew,
                logger=log,
                mode="slew",
            )

        else:
            # Any other tracking mode is currently unsupported.
            raise ValueError(
                f"build_star_photons_for_observer: unsupported tracking_mode={mode!r} "
                f"for observer '{obs_name}', window_index={window_index}."
            )

        # Append the per-window star photon structure to our list.
        star_windows.append(star_window)

    # ------------------------------------------------------------------
    # Build the per-observer wrapper structure.
    # ------------------------------------------------------------------
    # Observer name: prefer frames_for_obs, but fall back to the argument.
    observer_name = frames_for_obs.get("observer_name", obs_name)

    # Sensor geometry: prefer what frames_for_obs already encoded, but
    # fall back to ACTIVE_SENSOR in case those fields are missing.
    rows = int(frames_for_obs.get("rows", getattr(ACTIVE_SENSOR, "rows", 0)))
    cols = int(frames_for_obs.get("cols", getattr(ACTIVE_SENSOR, "cols", 0)))

    # Catalog metadata from NEBULA_STAR_CONFIG.
    catalog_name = getattr(
        NEBULA_STAR_CATALOG, "name", getattr(NEBULA_STAR_CATALOG, "catalog_name", "Gaia")
    )
    catalog_band = getattr(NEBULA_STAR_CATALOG, "band", "G")

    # Minimal run_meta; can be extended later with file paths, etc.
    run_meta: Dict[str, Any] = {
        "star_photons_version": STAR_PHOTONS_RUN_META_VERSION,
        "builder": "NEBULA_STAR_PHOTONS.build_star_photons_for_observer",
        "observer_name": observer_name,
        "n_windows_input": len(windows_frames),
        "n_windows_output": len(star_windows),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",}

    # Assemble the final per-observer star photon catalog.
    obs_star_photons: Dict[str, Any] = {
        "observer_name": observer_name,
        "rows": rows,
        "cols": cols,
        "catalog_name": catalog_name,
        "catalog_band": catalog_band,
        "run_meta": run_meta,
        "windows": star_windows,
    }

    log.info(
        "build_star_photons_for_observer: built star photon catalog for "
        "observer '%s' with %d windows.",
        observer_name,
        len(star_windows),
    )

    return obs_star_photons

def build_star_photons_for_all_observers(
    obs_target_frames: Dict[str, Any],
    obs_star_projections: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> ObsStarPhotons:
    """
    Build star photon time series for *all* observers present in the
    target-photon frames dictionary.

    This is the main in-memory dispatcher for NEBULA_STAR_PHOTONS:
    given per-observer windowed photon frames (from NEBULA_TARGET_PHOTONS /
    NEBULA_PHOTON_FRAME_BUILDER) and per-observer star projections
    (from NEBULA_STAR_PROJECTION and/or NEBULA_STAR_SLEW_PROJECTION),
    it calls :func:`build_star_photons_for_observer` for each observer and
    aggregates the results.

    Parameters
    ----------
    obs_target_frames : dict
        Per-observer target photon frames, typically the
        ``obs_target_frames_ranked`` structure produced by
        :func:`NEBULA_TARGET_PHOTONS.build_obs_target_frames_for_all_observers`.

        Expected minimal structure:

        .. code-block:: python

            obs_target_frames = {
                "<observer_name>": {
                    "observer_name": str,
                    "windows": [
                        {
                            "window_index": int,
                            "start_index": int,
                            "end_index": int,
                            "n_frames": int,
                            "tracking_mode": str,  # "sidereal" or "slew"
                            "frames": [...],
                            ...
                        },
                        ...
                    ],
                    ...
                },
                ...
            }

    obs_star_projections : dict
        Per-observer star projection products, typically the output of
        :func:`NEBULA_STAR_PROJECTION.build_star_projections_for_all_observers`
        (and, in the future, NEBULA_STAR_SLEW_PROJECTION if you choose to
        separate slewing projections).

        Expected minimal structure:

        .. code-block:: python

            obs_star_projections = {
                "<observer_name>": {
                    "observer_name": str,
                    "windows": [
                        {
                            "window_index": int,
                            "t_mid_utc": ...,
                            "t_mid_mjd_utc": float,
                            "n_stars_input": int,
                            "n_stars_on_detector": int,
                            "stars": {
                                "<gaia_source_id_str>": {
                                    "gaia_source_id": int or str,
                                    "mag_G": float,
                                    "x_pix_mid": float,
                                    "y_pix_mid": float,
                                    "on_detector": bool,
                                    ...
                                },
                                ...
                            },
                            ...
                        },
                        ...
                    ],
                    ...
                },
                ...
            }

        The observer keys (``"<observer_name>"``) must match those in
        ``obs_target_frames`` for star photons to be constructed.

    logger : logging.Logger, optional
        Logger for informational / diagnostic messages. If ``None``,
        a module-level logger obtained via :func:`_get_logger` is used.

    Returns
    -------
    ObsStarPhotons
        Dictionary mapping each observer name to its star photon product:

        .. code-block:: python

            obs_star_photons = {
                "<observer_name>": {
                    "observer_name": str,
                    "rows": int,
                    "cols": int,
                    "catalog_name": str,
                    "catalog_band": str,
                    "run_meta": dict,
                    "windows": [StarPhotonWindow, ...],
                },
                ...
            }

        Each ``StarPhotonWindow`` is produced by
        :func:`build_star_photons_for_observer`, which in turn uses
        :func:`build_star_photon_timeseries_for_window_sidereal` and
        :func:`build_star_photon_timeseries_for_window_slew` depending on
        ``tracking_mode``.

    Raises
    ------
    RuntimeError
        If an observer present in ``obs_target_frames`` has no matching
        entry in ``obs_star_projections``. This is a fail-hard design:
        star photons are only considered valid if both target frames and
        star projections exist for the same observer.
    """
    # Resolve a logger for this function.
    log = _get_logger(logger)

    # Container for the per-observer star photon products.
    obs_star_photons: ObsStarPhotons = {}

    # Loop over every observer that has target photon frames.
    for obs_name, frames_for_obs in obs_target_frames.items():
        # Look up the matching star projection entry.
        star_proj_for_obs = obs_star_projections.get(obs_name)

        if star_proj_for_obs is None:
            # Fail hard: if we have target frames but no star projections
            # for this observer, something went wrong upstream.
            raise RuntimeError(
                "build_star_photons_for_all_observers: no star projection "
                f"entry found for observer '{obs_name}'. "
                "Ensure NEBULA_STAR_PROJECTION (and/or NEBULA_STAR_SLEW_PROJECTION) "
                "has been run for the same set of observers."
            )

        # Delegate the actual per-observer construction to the helper.
        obs_entry = build_star_photons_for_observer(
            obs_name=obs_name,
            frames_for_obs=frames_for_obs,
            star_projections_for_obs=star_proj_for_obs,
            logger=log,
        )

        # Store the result in the top-level mapping.
        obs_star_photons[obs_name] = obs_entry

        # Log a compact summary for this observer.
        n_windows = len(obs_entry.get("windows", []))
        log.info(
            "NEBULA_STAR_PHOTONS: observer '%s' -> built star photons for %d window(s).",
            obs_name,
            n_windows,
        )

    # Optionally, log a one-line summary across all observers.
    log.info(
        "NEBULA_STAR_PHOTONS: completed star photon construction for %d observer(s).",
        len(obs_star_photons),
    )

    return obs_star_photons

def run_star_photons_pipeline_from_pickles(
    frames_with_sky_path: Optional[str] = None,
    star_projection_sidereal_path: Optional[str] = None,
    star_projection_slew_path: Optional[str] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> ObsStarPhotons:
    """
    High-level driver: load existing pickles, build star photons,
    and write obs_star_photons.pkl.

    This function is intentionally *read-only* with respect to upstream
    stages: it assumes that:

        1) NEBULA_TARGET_PHOTONS / NEBULA_PHOTON_FRAME_BUILDER have
           already produced:

               NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl

        2) NEBULA_STAR_PROJECTION has produced sidereal star projections:

               NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl

        3) NEBULA_STAR_SLEW_PROJECTION has (optionally) produced
           non-sidereal (slew) star projections:

               NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_slew_projections.pkl

    It then:

        * loads these three pickles,
        * calls build_star_photons_for_all_observers(...) to construct
          per-observer, per-window star photon time series, and
        * writes:

               NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_photons.pkl

    Parameters
    ----------
    frames_with_sky_path : str or None, optional
        Path to the ranked target-frames pickle (frames-with-sky structure),
        typically:

            NEBULA_OUTPUT/TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl

        If None, this default location is used.

    star_projection_sidereal_path : str or None, optional
        Path to the sidereal star projection pickle produced by
        NEBULA_STAR_PROJECTION, typically:

            NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_projections.pkl

        If None, this default location is used.

    star_projection_slew_path : str or None, optional
        Path to the non-sidereal (slew) star projection pickle produced by
        NEBULA_STAR_SLEW_PROJECTION, typically:

            NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_slew_projections.pkl

        If None, the function will look for this file at the default
        location and, if missing, proceed with an empty slew-projection
        dict (i.e., sidereal-only stars). If your frames contain windows
        with tracking_mode == "slew", your downstream logic can decide
        whether to treat missing slew projections as an error.

    output_path : str or None, optional
        Path where the resulting obs_star_photons pickle will be written.
        If None, the default is:

            NEBULA_OUTPUT/STARS/<catalog_name>/obs_star_photons.pkl

    logger : logging.Logger or None, optional
        Logger for status / debug messages. If None, a module-level
        logger obtained via _get_logger() is used.

    Returns
    -------
    ObsStarPhotons
        The in-memory obs_star_photons mapping keyed by observer name.

    Raises
    ------
    FileNotFoundError
        If the ranked target-frames pickle or the sidereal star
        projection pickle cannot be found at the resolved paths.
    RuntimeError
        If downstream helper functions detect inconsistent data.
    """
    # Resolve a logger to use internally.
    log = _get_logger(logger)

    # ------------------------------------------------------------------
    # Resolve default paths if caller did not supply them explicitly.
    # ------------------------------------------------------------------
    nebula_output_dir = NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR
    catalog_name = getattr(
        NEBULA_STAR_CATALOG, "name", getattr(NEBULA_STAR_CATALOG, "catalog_name", "UNKNOWN_CATALOG")
    )

    # Ranked target frames (frames-with-sky) from NEBULA_TARGET_PHOTONS.
    if frames_with_sky_path is None:
        frames_with_sky_dir = os.path.join(
            nebula_output_dir,
            "TARGET_PHOTON_FRAMES",
        )
        frames_with_sky_path = os.path.join(
            frames_with_sky_dir,
            "obs_target_frames_ranked.pkl",
        )

    # Sidereal star projections from NEBULA_STAR_PROJECTION.
    if star_projection_sidereal_path is None:
        stars_dir = os.path.join(
            nebula_output_dir,
            "STARS",
            f"{catalog_name}",
        )
        star_projection_sidereal_path = os.path.join(
            stars_dir,
            "obs_star_projections.pkl",
        )

    # Slew star projections from NEBULA_STAR_SLEW_PROJECTION (optional).
    if star_projection_slew_path is None:
        stars_dir = os.path.join(
            nebula_output_dir,
            "STARS",
            f"{catalog_name}",
        )
        star_projection_slew_path = os.path.join(
            stars_dir,
            "obs_star_slew_projections.pkl",
        )

    # Output path for obs_star_photons.
    if output_path is None:
        stars_dir = os.path.join(
            nebula_output_dir,
            "STARS",
            f"{catalog_name}",
        )
        output_path = os.path.join(
            stars_dir,
            "obs_star_photons.pkl",
        )

    log.info(
        "NEBULA_STAR_PHOTONS: using frames_with_sky_path=%s",
        frames_with_sky_path,
    )
    log.info(
        "NEBULA_STAR_PHOTONS: using star_projection_sidereal_path=%s",
        star_projection_sidereal_path,
    )
    log.info(
        "NEBULA_STAR_PHOTONS: using star_projection_slew_path=%s",
        star_projection_slew_path,
    )
    log.info(
        "NEBULA_STAR_PHOTONS: output will be written to %s",
        output_path,
    )

    # ------------------------------------------------------------------
    # Load input pickles from disk.
    # ------------------------------------------------------------------
    if not os.path.exists(frames_with_sky_path):
        raise FileNotFoundError(
            f"run_star_photons_pipeline_from_pickles: frames_with_sky_path "
            f"does not exist: {frames_with_sky_path!r}"
        )

    if not os.path.exists(star_projection_sidereal_path):
        raise FileNotFoundError(
            f"run_star_photons_pipeline_from_pickles: sidereal star "
            f"projection pickle not found at: "
            f"{star_projection_sidereal_path!r}.\n"
            f"Did you run NEBULA_STAR_PROJECTION?"
        )

    # Load ranked target frames (frames-with-sky).
    with open(frames_with_sky_path, "rb") as f:
        frames_with_sky = pickle.load(f)

    # Load sidereal star projections.
    with open(star_projection_sidereal_path, "rb") as f:
        obs_star_projections_sidereal = pickle.load(f)

    # Load slew star projections if present; otherwise, fall back to an
    # empty dict. Downstream logic can decide how strict to be about
    # missing slew projections when there are slew windows.
    if os.path.exists(star_projection_slew_path):
        with open(star_projection_slew_path, "rb") as f:
            obs_star_projections_slew = pickle.load(f)
        log.info(
            "NEBULA_STAR_PHOTONS: loaded slew star projections from '%s'.",
            star_projection_slew_path,
        )
    else:
        obs_star_projections_slew = {}
        log.info(
            "NEBULA_STAR_PHOTONS: no slew star projections found at '%s'; "
            "proceeding with sidereal-only star projections.",
            star_projection_slew_path,
        )

    # ------------------------------------------------------------------
    # Build star photon time series for all observers.
    # ------------------------------------------------------------------
    obs_star_photons = build_star_photons_for_all_observers(
        frames_with_sky,
        obs_star_projections_sidereal,
        obs_star_projections_slew,
        logger=log,
    )

    # ------------------------------------------------------------------
    # Write the resulting obs_star_photons to disk.
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(obs_star_photons, f)

    log.info(
        "NEBULA_STAR_PHOTONS: wrote obs_star_photons for %d observers to '%s'.",
        len(obs_star_photons),
        output_path,
    )

    return obs_star_photons

# ----------------------------------------------------------------------
# Script guard
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # This module is intended to be used as a library, called from a
    # driver (e.g., sim_test) once all upstream pickles exist.
    #
    # Typical usage pattern:
    #
    #   from Utility.STARS import NEBULA_STAR_PHOTONS as NSP
    #
    #   obs_star_photons = NSP.run_star_photons_pipeline_from_pickles(
    #       frames_path=...,
    #       star_projection_path=...,
    #       star_slew_tracks_path=...,   # optional / for slewing mode
    #       output_path=...,             # optional, to save a pickle
    #       logger=logger,
    #   )
    #
    # For now, running this file directly without arguments will just
    # exit with a message. If you want an ad-hoc test harness, you can
    # edit this block locally to hard-code paths and call your pipeline
    # function.
    raise SystemExit(
        "NEBULA_STAR_PHOTONS is a library module. Import it and call your "
        "pipeline function (e.g., run_star_photons_pipeline_from_pickles(...)) "
        "from a driver such as sim_test."
    )
