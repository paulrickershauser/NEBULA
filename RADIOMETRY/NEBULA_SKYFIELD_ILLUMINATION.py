"""
NEBULA_SKYFIELD_ILLUMINATION.py

Skyfield-based solar illumination for a single observer–target pair.

Given:
    - observer_track : SatelliteTrack or dict with NEBULA fields
    - target_track   : SatelliteTrack or dict with NEBULA fields

both on (roughly) the same time grid and expressed in Earth-centered
TEME / "ECI" coordinates, this module:

  * loads the JPL DE440s planetary ephemeris from a NEBULA-local path,
  * computes, at each timestep:
        - whether the target is in sunlight (is_sunlit),
        - the Sun–target–observer phase angle (radians),
        - the corresponding illuminated fraction of a Lambertian sphere,
  * optionally attaches these arrays directly to the *target* track.

Definitions follow the Skyfield API:

  - is_sunlit:
      Skyfield's ICRF.is_sunlit(ephemeris) test for Earth-shadow.

  - phase_angle:
      The angle Sun–target–observer: 0 rad = fully lit face,
      π rad = fully backlit.

  - fraction_illuminated:
      Fraction of the apparent disc that is illuminated for a Lambertian
      sphere: f = (1 + cos(phase_angle)) / 2, same as Skyfield's
      fraction_illuminated() for a spherical target.

All distances are kilometers, angles are radians, times are Python
datetime objects stored on the tracks.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standard library imports
# --------------------------------------------------------------------------- #

from dataclasses import dataclass
from typing import Any, Optional
import logging
import os

# --------------------------------------------------------------------------- #
# Third-party imports
# --------------------------------------------------------------------------- #

import numpy as np

from skyfield.api import load, Distance, Velocity
from skyfield.positionlib import ICRF
from skyfield.sgp4lib import TEME
from skyfield.constants import AU_KM  # if you ever need it later
from pathlib import Path

# --------------------------------------------------------------------------- #
# NEBULA configuration imports
# --------------------------------------------------------------------------- #

from Configuration.NEBULA_ENV_CONFIG import R_EARTH  # noqa: F401

# --------------------------------------------------------------------------- #
# Local configuration: default ephemeris path (project-relative)
# --------------------------------------------------------------------------- #

# Project root = .../NEBULA (two levels above this file: Utility/RADIOMETRY)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default DE440s location: <project_root>/Input/NEBULA_EPHEMERIS/de440s.bsp
# This makes NEBULA portable: as long as that relative folder exists in the
# zipped project, the ephemeris will be found regardless of where the user
# unpacks NEBULA.
EPHEMERIS_PATH_DEFAULT = str(
    PROJECT_ROOT / "Input" / "NEBULA_EPHEMERIS" / "de440s.bsp"
)



# --------------------------------------------------------------------------- #
# Track helpers
# --------------------------------------------------------------------------- #


def _get_track_field(track: Any, field_name: str, required: bool = True):
    """
    Safely extract a field from a NEBULA "track" object.

    Supports both:
        - dict-like tracks produced by NEBULA_SAT_PICKLER, and
        - object-like tracks (e.g., SatelliteTrack instances).
    """
    if isinstance(track, dict):
        if field_name in track:
            return track[field_name]
        if required:
            raise KeyError(f"Track dict missing required field '{field_name}'")
        return None

    if hasattr(track, field_name):
        return getattr(track, field_name)
    if required:
        raise AttributeError(f"Track object missing required field '{field_name}'")
    return None


def _set_track_field(track: Any, field_name: str, value: Any) -> None:
    """
    Attach or overwrite a field on a NEBULA "track" object.

    Works for both dict-like and object-like tracks.
    """
    if isinstance(track, dict):
        track[field_name] = value
    else:
        setattr(track, field_name, value)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class IlluminationResult:
    """
    Container for per-timestep illumination outputs.

    Attributes
    ----------
    times : np.ndarray
        Time stamps used for the evaluation (aligned observer/target grid).
    is_sunlit : np.ndarray, bool, shape (N,)
        True if the target is in direct sunlight (not in Earth eclipse).
    phase_angle_rad : np.ndarray, float, shape (N,)
        Sun–target–observer phase angle in radians.
        0 rad   → observer fully on the lit side.
        π rad   → observer fully on the dark side.
    fraction_illuminated : np.ndarray, float, shape (N,)
        Fraction of the target's apparent disc that is illuminated, under
        the Lambertian-sphere assumption:
            f = (1 + cos(phase_angle_rad)) / 2
        This ignores Earth shadowing; multiply by is_sunlit if you want
        effective illumination.
    """
    times: np.ndarray
    is_sunlit: np.ndarray
    phase_angle_rad: np.ndarray
    fraction_illuminated: np.ndarray


# --------------------------------------------------------------------------- #
# Ephemeris loader
# --------------------------------------------------------------------------- #


def load_de440s_ephemeris(
    ephemeris_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Load the JPL DE440s planetary ephemeris from NEBULA's local copy.

    Parameters
    ----------
    ephemeris_path : str or None, optional
        Path to "de440s.bsp". If None, uses EPHEMERIS_PATH_DEFAULT.
    logger : logging.Logger or None, optional
        Optional logger for informational messages.

    Returns
    -------
    eph : skyfield.jpllib.SpiceKernel
        Loaded DE440s ephemeris object.

    Raises
    ------
    FileNotFoundError
        If the ephemeris file does not exist at the requested path.
    """
    path = ephemeris_path or EPHEMERIS_PATH_DEFAULT

    if not os.path.isfile(path):
        msg = (
            f"DE440s ephemeris not found at '{path}'. "
            "Update EPHEMERIS_PATH_DEFAULT in NEBULA_SKYFIELD_ILLUMINATION.py "
            "or pass ephemeris_path explicitly."
        )
        if logger is not None:
            logger.error(msg)
        raise FileNotFoundError(msg)

    eph = load(path)
    if logger is not None:
        logger.info("Loaded DE440s ephemeris from: %s", path)
    return eph


# --------------------------------------------------------------------------- #
# Core illumination routine for one observer–target pair
# --------------------------------------------------------------------------- #


def compute_illumination_timeseries_for_pair(
    observer_track: Any,
    target_track: Any,
    eph=None,
    ephemeris_path: Optional[str] = None,
    store_on_target: bool = True,
    logger: Optional[logging.Logger] = None,
) -> IlluminationResult:
    """
    Compute solar illumination for a single observer–target pair.

    Steps:
        1. Build a Skyfield Time array from the track times.
        2. Load DE440s ephemeris from NEBULA's local path (or use `eph`).
        3. Build a Skyfield ICRF position for the target using TEME states,
           then call ICRF.is_sunlit(eph) to test Earth eclipse.
        4. Compute the Sun–target–observer phase angle at each timestep.
        5. Compute the illuminated fraction f = (1 + cos(α)) / 2.
        6. Optionally attach these arrays to the target track.

    Parameters
    ----------
    observer_track : dict or SatelliteTrack-like
        Track representing the observer. Must have:
            - 'times'    : length-N array of datetime objects
            - 'r_eci_km' : array (N, 3) positions [km]
            - 'v_eci_km_s': array (N, 3) velocities [km/s]
    target_track : dict or SatelliteTrack-like
        Track representing the target satellite, with same fields.
    eph : skyfield.jpllib.SpiceKernel or None, optional
        Pre-loaded DE440s ephemeris object. If None, this function loads it
        from EPHEMERIS_PATH_DEFAULT or `ephemeris_path`.
    ephemeris_path : str or None, optional
        Explicit path to "de440s.bsp". Only used if `eph` is None.
    store_on_target : bool, optional
        If True, attach:
            - 'illum_is_sunlit'
            - 'illum_phase_angle_rad'
            - 'illum_fraction_illuminated'
        to `target_track`.
    logger : logging.Logger or None, optional
        Optional logger for warnings and info.

    Returns
    -------
    IlluminationResult
        Dataclass with times, is_sunlit, phase_angle_rad, and
        fraction_illuminated arrays.
    """
    # -------------------------- Times & ephemeris -------------------------- #

    ts = load.timescale()
    times_obs = np.asarray(_get_track_field(observer_track, "times"))
    times_tar = np.asarray(_get_track_field(target_track, "times"))

    n_obs = int(times_obs.shape[0])
    n_tar = int(times_tar.shape[0])
    n_steps = min(n_obs, n_tar)

    if (n_obs != n_tar) and (logger is not None):
        logger.warning(
            "Observer/target time arrays differ in length (%d vs %d); "
            "truncating to n_steps = %d",
            n_obs,
            n_tar,
            n_steps,
        )

    times = times_obs[:n_steps]

    if (
        n_obs == n_tar
        and logger is not None
        and not np.all(times_obs == times_tar)
    ):
        logger.warning(
            "Observer and target times have same length but different values; "
            "using observer times as the common grid."
        )

    t = ts.from_datetimes(times)

    if eph is None:
        eph = load_de440s_ephemeris(ephemeris_path=ephemeris_path, logger=logger)

    earth = eph["earth"]
    sun = eph["sun"]

    # ---------------------- Extract states from tracks --------------------- #

    r_obs = np.asarray(_get_track_field(observer_track, "r_eci_km"), dtype=float)[
        :n_steps, :
    ]
    v_obs = np.asarray(_get_track_field(observer_track, "v_eci_km_s"), dtype=float)[
        :n_steps, :
    ]
    r_tar = np.asarray(_get_track_field(target_track, "r_eci_km"), dtype=float)[
        :n_steps, :
    ]
    v_tar = np.asarray(_get_track_field(target_track, "v_eci_km_s"), dtype=float)[
        :n_steps, :
    ]

    if r_obs.shape != r_tar.shape:
        raise ValueError(
            f"Observer/target position shapes differ: {r_obs.shape} vs {r_tar.shape}"
        )

    # --------------------------- Sun position ------------------------------ #

    astrometric_sun = earth.at(t).observe(sun)
    r_sun_eci_km = astrometric_sun.position.km.T  # shape (N, 3)

    # -------------------------- is_sunlit (Skyfield) ---------------------- #

    dist_tar = Distance(km=r_tar.T)
    vel_tar = Velocity(km_per_s=v_tar.T)

    # Transform from TEME to Skyfield's internal ICRF frame
    pos_tar_icrf = ICRF.from_time_and_frame_vectors(t, TEME, dist_tar, vel_tar)

    # IMPORTANT: tell Skyfield this is an Earth-centered orbit (NAIF id 399)
    # Without this, is_sunlit() raises "cannot tell whether this position is sunlit"
    # because it doesn’t know which body is casting the shadow.
    pos_tar_icrf.center = 399

    # Now Skyfield can safely perform the Earth-shadow test
    is_sunlit_arr = np.asarray(pos_tar_icrf.is_sunlit(eph), dtype=bool)

    # ------------------- Phase angle & illuminated fraction ---------------- #

    # Target → Sun and target → observer vectors
    vec_target_to_sun = r_sun_eci_km - r_tar
    vec_target_to_obs = r_obs - r_tar

    norm_ts = np.linalg.norm(vec_target_to_sun, axis=1)
    norm_to = np.linalg.norm(vec_target_to_obs, axis=1)
    dot_ts_to = np.einsum("ij,ij->i", vec_target_to_sun, vec_target_to_obs)

    denom = norm_ts * norm_to
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_phase = np.where(denom > 0.0, dot_ts_to / denom, np.nan)

    cos_phase = np.clip(cos_phase, -1.0, 1.0)
    phase_angle_rad = np.arccos(cos_phase)

    fraction_illum = 0.5 * (1.0 + np.cos(phase_angle_rad))

    result = IlluminationResult(
        times=np.array(times, copy=True),
        is_sunlit=is_sunlit_arr,
        phase_angle_rad=phase_angle_rad,
        fraction_illuminated=fraction_illum,
    )

    # ---------------------- Attach to target track ------------------------ #

    if store_on_target:
        _set_track_field(target_track, "illum_is_sunlit", result.is_sunlit)
        _set_track_field(target_track, "illum_phase_angle_rad", result.phase_angle_rad)
        _set_track_field(
            target_track,
            "illum_fraction_illuminated",
            result.fraction_illuminated,
        )
        if logger is not None and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Attached illumination fields to target track: "
                "illum_is_sunlit, illum_phase_angle_rad, "
                "illum_fraction_illuminated"
            )


    return result


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #


def attach_skyfield_illumination_to_target(
    observer_track: Any,
    target_track: Any,
    eph=None,
    ephemeris_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> IlluminationResult:
    """
    Convenience wrapper: compute illumination and store it on the target.
    """
    return compute_illumination_timeseries_for_pair(
        observer_track=observer_track,
        target_track=target_track,
        eph=eph,
        ephemeris_path=ephemeris_path,
        store_on_target=True,
        logger=logger,
    )
